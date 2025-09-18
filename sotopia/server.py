import asyncio
import itertools
import logging
from typing import Literal, Sequence, Type, cast

import gin
import rich
from beartype import beartype

from sotopia.agents import (
    Agents,
    HumanAgent,
    LLMAgent,
    RedisAgent,
    ScriptWritingAgent,
)
from sotopia.agents.base_agent import BaseAgent
from sotopia.database import EpisodeLog
from sotopia.envs import ParallelSotopiaEnv
from sotopia.envs.evaluators import (
    EvaluationForTwoAgents,
    ReachGoalLLMEvaluator,
    RuleBasedTerminatedEvaluator,
    SotopiaDimensions,
    unweighted_aggregate_evaluate,
)
from sotopia.generation_utils.generate import LLM_Name, agenerate_script
from sotopia.goal_visibility import GoalVisibilitySystem
from sotopia.messages import AgentAction, Message, Observation
from sotopia.messages.message_classes import (
    ScriptBackground,
    ScriptEnvironmentResponse,
)
from sotopia.samplers import BaseSampler, EnvAgentCombo


@beartype
def run_sync_server(
    model_name_dict: dict[str, LLM_Name],
    action_order: Literal["simultaneous", "round-robin", "random"],
    agents_info: dict[str, dict[str, str]] | None = None,
    partial_background_file: str | None = None,
    full_background_file: str | None = None,
    mode: str | None = None,
) -> list[tuple[str, str, Message]]:
    # Create Environment and agents
    # This step will be moved to outside this function

    env = ParallelSotopiaEnv(
        model_name=model_name_dict["env"],
        action_order=action_order,
        evaluators=[
            RuleBasedTerminatedEvaluator(),
        ],
    )
    if partial_background_file:
        environment_messages = env.reset(
            options={"partial_background_file": partial_background_file}
        )
    elif full_background_file:
        environment_messages = env.reset(
            options={"full_background_file": full_background_file}
        )
    else:
        environment_messages = env.reset()
    agents = Agents()
    agents_model_names = [model_name_dict["agent1"], model_name_dict["agent2"]]
    for agent_name, agent_model in zip(env.agents, agents_model_names):
        if agent_model == "human":
            agents[agent_name] = HumanAgent(agent_name)
        elif mode == "speak":
            raise NotImplementedError(
                "Deprecated. The original Speaker Agent is not implemented in the async context."
            )
        else:
            agents[agent_name] = LLMAgent(agent_name, model_name=agent_model)
    agents.reset()

    messages: list[tuple[str, str, Message]] = []

    # Main Event Loop
    done = False
    for agent_name in env.agents:
        messages.append(("Environment", agent_name, environment_messages[agent_name]))

    while not done:
        # gather agent messages
        agent_messages: dict[str, AgentAction] = dict()
        for agent_name in env.agents:
            if agents_info is not None:
                agents[agent_name].goal = agents_info[agent_name]["goal"]
            agent_messages[agent_name] = agents[agent_name].act(
                environment_messages[agent_name]
            )
            messages.append((agent_name, "Environment", agent_messages[agent_name]))

        # send agent messages to environment
        environment_messages, _, terminated, ___, ____ = env.step(agent_messages)
        for agent_name in env.agents:
            messages.append(
                ("Environment", agent_name, environment_messages[agent_name])
            )
        done = all(terminated.values())

    return messages


@gin.configurable
async def arun_one_episode(
    env: ParallelSotopiaEnv,
    agent_list: Sequence[BaseAgent[Observation, AgentAction]],
    omniscient: bool = False,
    script_like: bool = False,
    json_in_script: bool = False,
    tag: str | None = None,
    push_to_db: bool = False,
    verbose: bool = False,
    vis_goal: str = "none",
) -> list[tuple[str, str, Message]]:
    agents = Agents({agent.agent_name: agent for agent in agent_list})
    environment_messages = env.reset(agents=agents, omniscient=omniscient)
    agents.reset()
    
    # Create goal visibility system
    goal_visibility_system = GoalVisibilitySystem(vis_goal_mode=vis_goal, verbose=verbose)
    
    # Set agent order for proper SToM assignment
    agent_names = list(env.agents)
    goal_visibility_system.set_agent_order(agent_names)
    
    for agent in agent_list:
        # Pass goal_visibility_system to LLMAgent if it's an LLMAgent
        if isinstance(agent, LLMAgent):
            agent.goal_visibility_system = goal_visibility_system

    messages: list[list[tuple[str, str, Message]]] = []

    # Main Event Loop
    done = False
    messages.append(
        [
            ("Environment", agent_name, environment_messages[agent_name])
            for agent_name in env.agents
        ]
    )
    # set goal for agents
    for index, agent_name in enumerate(env.agents):
        agents[agent_name].goal = env.profile.agent_goals[index]
    
    # Initialize SToM for agents based on who has extra visibility
    scenario_description = env.profile.scenario if hasattr(env.profile, 'scenario') else "Social interaction scenario"
    
    if len(agent_names) >= 2:
        # Check each agent and initialize SToM only if they have extra visibility
        for agent_name in agent_names:
            if goal_visibility_system.agent_has_stom(agent_name):
                partner_name = agent_names[1] if agent_name == agent_names[0] else agent_names[0]
                await goal_visibility_system.initialize_stom_for_agent(
                    agent_name=agent_name,
                    partner_name=partner_name, 
                    scenario_description=scenario_description,
                    conversation_context=""
                )
                if verbose:
                    print(f"🧠 SToM: Initialized for {agent_name} (has extra visibility)")
    rewards: list[list[float]] = []
    reasons: list[str] = []
    while not done:
        # gather agent messages
        agent_messages: dict[str, AgentAction] = dict()
        
        # Save current action mask before astep (will be updated after astep)
        current_action_mask = env.action_mask.copy() if hasattr(env, 'action_mask') else None
        
        actions = await asyncio.gather(
            *[
                agents[agent_name].aact(environment_messages[agent_name])
                for agent_name in env.agents
            ]
        )
        if script_like:
            # manually mask one message
            agent_mask = env.action_mask
            for idx in range(len(agent_mask)):
                print("Current mask: ", agent_mask)
                if agent_mask[idx] == 0:
                    print("Action not taken: ", actions[idx])
                    actions[idx] = AgentAction(action_type="none", argument="")
                else:
                    print("Current action taken: ", actions[idx])

        # actions = cast(list[AgentAction], actions)
        for idx, agent_name in enumerate(env.agents):
            agent_messages[agent_name] = actions[idx]
            messages[-1].append((agent_name, "Environment", agent_messages[agent_name]))

        # Update SToM after agent actions (for agents with extra visibility)
        if vis_goal in ["agent1", "agent2", "both", "stom"]:
            agent_names = list(env.agents)
            if len(agent_names) >= 2:
                # Build conversation context from messages (filter out "did nothing" actions)
                all_messages = [
                    f"{msg[0]}: {msg[2].to_natural_language()}" 
                    for msg_group in messages 
                    for msg in msg_group
                    if msg[0] != "Environment"
                ]
                
                # Filter out "did nothing" messages
                filtered_messages = []
                for msg in all_messages:
                    msg_lower = msg.lower()
                    if not ("did nothing" in msg_lower or "no action" in msg_lower or msg.endswith(": ")):
                        filtered_messages.append(msg)
                
                conversation_context = "\n".join(filtered_messages)
                
                # Update each agent's SToM about their partner's action (only if they have extra visibility)
                current_turn = len(messages) - 1  # Current turn number (0-indexed)
                for idx, agent_name in enumerate(agent_names):
                    if goal_visibility_system.agent_has_stom(agent_name):  # Only update if agent has extra visibility
                        partner_name = agent_names[1 - idx]  # Get the other agent
                        partner_idx = 1 - idx  # Partner's index in the agent list
                        
                        if partner_name in agent_messages:
                            # Check if partner's action was masked (using the action mask from BEFORE astep)
                            partner_action_was_masked = False
                            if current_action_mask and hasattr(env, 'action_order'):
                                if env.action_order in ["round-robin", "random"]:
                                    partner_action_was_masked = not current_action_mask[partner_idx]
                            
                            # Only update SToM if partner took a real action (not masked and not "did nothing")
                            partner_action = agent_messages[partner_name]
                            partner_action_str = partner_action.to_natural_language()
                            
                            # Check if it's a meaningful action (not "did nothing", "left the conversation", or similar)
                            is_meaningful_action = True
                            if partner_action_str:
                                action_lower = partner_action_str.lower().strip()
                                if ("did nothing" in action_lower or 
                                    "no action" in action_lower or 
                                    "left the conversation" in action_lower or
                                    action_lower in ["", "none"] or
                                    action_lower.endswith(": ") or
                                    action_lower.endswith(":")):
                                    is_meaningful_action = False
                            else:
                                is_meaningful_action = False
                            
                            if not partner_action_was_masked and is_meaningful_action:
                                await goal_visibility_system.update_stom_for_agent(
                                    agent_name=agent_name,
                                    partner_name=partner_name,
                                    new_partner_action=partner_action_str,
                                    conversation_context=conversation_context,
                                    turn_number=current_turn
                                )
                            elif verbose:
                                if partner_action_was_masked:
                                    print(f"🧠 SToM: Skipping update - {partner_name} action was masked in {env.action_order} mode")
                                else:
                                    print(f"🧠 SToM: Skipping update - {partner_name} took non-meaningful action: '{partner_action_str}'")

        # send agent messages to environment
        (
            environment_messages,
            rewards_in_turn,
            terminated,
            ___,
            info,
        ) = await env.astep(agent_messages)
        messages.append(
            [
                ("Environment", agent_name, environment_messages[agent_name])
                for agent_name in env.agents
            ]
        )
        # print("Environment message: ", environment_messages)
        # exit(0)
        rewards.append([rewards_in_turn[agent_name] for agent_name in env.agents])
        reasons.append(
            " ".join(info[agent_name]["comments"] for agent_name in env.agents)
        )
        done = all(terminated.values())

    # TODO: clean up this part
    epilog = EpisodeLog(
        environment=env.profile.pk,
        agents=[agent.profile.pk for agent in agent_list],
        tag=tag,
        models=[env.model_name, agent_list[0].model_name, agent_list[1].model_name],
        messages=[
            [(m[0], m[1], m[2].to_natural_language()) for m in messages_in_turn]
            for messages_in_turn in messages
        ],
        reasoning=info[env.agents[0]]["comments"],
        rewards=[info[agent_name]["complete_rating"] for agent_name in env.agents],
        rewards_prompt=info["rewards_prompt"]["overall_prompt"],
    )
    rich.print(epilog.rewards_prompt)
    agent_profiles, conversation = epilog.render_for_humans()
    for agent_profile in agent_profiles:
        rich.print(agent_profile)
    for message in conversation:
        rich.print(message)

    if push_to_db:
        try:
            epilog.save()
        except Exception as e:
            logging.error(f"Failed to save episode log: {e}")
    # flatten nested list messages
    return list(itertools.chain(*messages))


@gin.configurable
@beartype
async def run_async_server(
    sampler: BaseSampler[Observation, AgentAction] = BaseSampler(),
    action_order: Literal["simutaneous", "round-robin", "random"] = "round-robin",
    model_dict: dict[str, LLM_Name] = {},
    env_agent_combo_list: list[EnvAgentCombo[Observation, AgentAction]] = [],
    omniscient: bool = False,
    script_like: bool = False,
    json_in_script: bool = False,
    tag: str | None = None,
    push_to_db: bool = False,
    using_async: bool = True,
    verbose: bool = False,
    vis_goal: str = "none",
) -> list[list[tuple[str, str, Message]]]:
    """
    Doc incomplete

    Args:
        omniscient (bool): Whether the agent knows the goal of the other, default to False
        script_like (bool): Whether we generate the turn in script like manner, default to False
        json_in_script (bool): Whether we requires the script generator to return json (Only valid when script_like is True), default to False

    Note: env_agent_combo_list is optional. When it defaults to [], sampler is used
    else the sampler is not used. Please pass in BaseSampler or simply not specify it when using this option.
    """

    assert not (push_to_db and tag is None), "please provide a tag when push to db"
    assert (
        model_dict or env_agent_combo_list
    ), "please provide model_dict or env_agent_combo_list"

    # Create Environment and agents
    # This step will be moved to outside this function

    def get_agent_class(
        model_name: str,
    ) -> Type[BaseAgent[Observation, AgentAction]]:
        if model_name == "human":
            return HumanAgent
        elif model_name == "redis":
            return RedisAgent
        elif script_like and not json_in_script:
            return ScriptWritingAgent
        else:
            return LLMAgent

    if env_agent_combo_list:
        assert (
            type(sampler) is BaseSampler
        ), "No sampler should be used when `env_agent_combo_list` is not empty"
        env_agent_combo_iter = iter(env_agent_combo_list)
    else:
        env_params = {
            "model_name": model_dict["env"],
            "action_order": action_order,
            "evaluators": [
                RuleBasedTerminatedEvaluator(max_turn_number=20, max_stale_turn=2),
            ],
            "terminal_evaluators": [
                ReachGoalLLMEvaluator(
                    model_dict["env"],
                    EvaluationForTwoAgents[SotopiaDimensions],
                ),
            ],
        }
        agents_model_dict = {
            "agent1": model_dict["agent1"],
            "agent2": model_dict["agent2"],
        }
        env_agent_combo_iter = sampler.sample(
            agent_classes=[
                get_agent_class(model_name) for model_name in agents_model_dict.values()
            ],
            n_agent=len(agents_model_dict),
            env_params=env_params,
            agents_params=[
                {"model_name": model_name, "verbose": verbose} if model_name != "human" else {"verbose": verbose}
                for model_name in agents_model_dict.values()
            ],
        )
    episode_futures = [
        arun_one_episode(
            env=env_agent_combo[0],
            agent_list=env_agent_combo[1],
            omniscient=omniscient,
            script_like=script_like,
            json_in_script=json_in_script,
            tag=tag,
            push_to_db=push_to_db,
            verbose=verbose,
            vis_goal=vis_goal,
        )
        for env_agent_combo in env_agent_combo_iter
    ]

    batch_results = (
        await asyncio.gather(*episode_futures)
        if using_async
        else [await i for i in episode_futures]
    )

    return cast(list[list[tuple[str, str, Message]]], batch_results)


async def arun_one_script(
    env: ParallelSotopiaEnv,
    agent_list: Sequence[BaseAgent[Observation, AgentAction]],
    model_dict: dict[str, LLM_Name],
    omniscient: bool = False,
    tag: str | None = None,
    push_to_db: bool = False,
) -> list[tuple[str, str, Message]]:
    """
    Generate script for one episode
    Args:
        omniscient (bool): Whether the agent knows the goal of the other
    """

    agents = Agents({agent.agent_name: agent for agent in agent_list})
    env.reset(agents=agents, omniscient=omniscient)

    agent_names = [agent.agent_name for agent in agent_list]
    assert len(agent_names) == 2, f"only support 2 agents, current: {agent_names}"

    script_background = env.inbox[0][1]
    assert isinstance(script_background, ScriptBackground)
    story, prompt = await agenerate_script(
        model_name=model_dict["env"],
        background=script_background,
        agent_names=agent_names,
    )
    messages, agent_messages = story
    env_message = [("Environment", script_background)]
    agent_messages = env_message + agent_messages

    evaluator = ReachGoalLLMEvaluator(
        model_name="gpt-4",
        response_format_class=EvaluationForTwoAgents[SotopiaDimensions],
    )
    response = unweighted_aggregate_evaluate(
        list(
            itertools.chain(
                *await asyncio.gather(
                    *[
                        sing_evaluator.__acall__(
                            turn_number=-1,
                            messages=agent_messages,
                        )
                        for sing_evaluator in [evaluator]
                    ]
                )
            )
        )
    )
    info: dict[str, dict[str, str | ScriptEnvironmentResponse | float | None]] = {
        script_background.p1_name: {
            "comments": response.comments or "",
            "complete_rating": response.p1_rate or 0,  # type: ignore
        },
        script_background.p2_name: {
            "comments": response.comments or "",
            "complete_rating": response.p2_rate or 0,  # type: ignore
        },
        "rewards_prompt": {"overall_prompt": evaluator.prompt or ""},
    }
    epilog = EpisodeLog(
        environment=env.profile.pk,
        agents=[agent.profile.pk for agent in agent_list],
        tag=tag,
        models=[model_dict["env"], model_dict["agent1"], model_dict["agent2"]],
        messages=[
            [(m[0], m[1], m[2].to_natural_language()) for m in messages_in_turn]
            for messages_in_turn in messages
        ],
        reasoning=str(info[env.agents[0]]["comments"])
        + str(info[env.agents[1]]["comments"]),
        rewards=[info[agent_name]["complete_rating"] for agent_name in env.agents],
        rewards_prompt=info["rewards_prompt"]["overall_prompt"],
    )
    print("Reward prompt: ")
    rich.print(epilog.rewards_prompt)
    agent_profiles, conversation = epilog.render_for_humans()
    print("Agent profiles: ")
    for agent_profile in agent_profiles:
        rich.print(agent_profile)
    for message in conversation:
        rich.print(message)

    if push_to_db:
        try:
            epilog.save()
        except Exception as e:
            logging.error(f"Failed to save episode log: {e}")
    # flatten nested list messages
    return list(itertools.chain(*messages))


async def aevaluate_one_episode(
    episode: EpisodeLog,
    model: str = "gpt-4",
    tag: str | None = None,
    push_to_db: bool = False,
) -> None:
    history = episode.rewards_prompt.replace("Prompt after formatting:", "").split(
        ",\nBased on previous interactions"
    )[0]
    evaluator = ReachGoalLLMEvaluator(
        model_name=model,
        response_format_class=EvaluationForTwoAgents[SotopiaDimensions],
    )
    response = unweighted_aggregate_evaluate(
        list(
            itertools.chain(
                *await asyncio.gather(
                    *[
                        single_evaluator.__acall__(
                            turn_number=-1,
                            history=history,
                            messages=None,
                            temperature=0.0,
                        )
                        for single_evaluator in [evaluator]
                    ]
                )
            )
        )
    )
    info: dict[str, dict[str, str | ScriptEnvironmentResponse | float | None]] = {
        episode.agents[0]: {
            "comments": response.comments or "",
            "complete_rating": response.p1_rate or 0,  # type: ignore
        },
        episode.agents[1]: {
            "comments": response.comments or "",
            "complete_rating": response.p2_rate or 0,  # type: ignore
        },
    }
    assert isinstance(episode.models, list)
    epilog = EpisodeLog(
        environment=episode.environment,
        agents=episode.agents,
        tag=tag,
        models=[model, episode.models[1], episode.models[2]],
        messages=episode.messages,
        reasoning=str(info[episode.agents[0]]["comments"])
        + str(info[episode.agents[1]]["comments"]),
        rewards=[info[agent_name]["complete_rating"] for agent_name in episode.agents],
        rewards_prompt="TBD",
    )
    # rich.print(history)
    # rich.print(epilog.rewards)

    if push_to_db:
        try:
            epilog.save()
        except Exception as e:
            logging.error(f"Failed to save episode log: {e}")
