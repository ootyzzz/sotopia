import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import cast

from sotopia.agents import BaseAgent
from sotopia.database import AgentProfile
from sotopia.generation_utils.generate import (
    LLM_Name,
    agenerate_action,
    agenerate_goal,
    agenerate_script,
)
from sotopia.messages import AgentAction, Observation
from sotopia.messages.message_classes import ScriptBackground


async def ainput(prompt: str = "") -> str:
    with ThreadPoolExecutor(1, "ainput") as executor:
        return (
            await asyncio.get_event_loop().run_in_executor(executor, input, prompt)
        ).rstrip()


class LLMAgent(BaseAgent[Observation, AgentAction]):
    def __init__(
        self,
        agent_name: str | None = None,
        uuid_str: str | None = None,
        agent_profile: AgentProfile | None = None,
        model_name: str = "gpt-4o-mini",
        script_like: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            agent_name=agent_name,
            uuid_str=uuid_str,
            agent_profile=agent_profile,
        )
        self.model_name = model_name
        self.script_like = script_like
        self.verbose = verbose
        self.goal_visibility_system = None  # Will be set by server if needed

    @property
    def goal(self) -> str:
        if self._goal is not None:
            return self._goal
        else:
            raise Exception("Goal is not set.")

    @goal.setter
    def goal(self, goal: str) -> None:
        self._goal = goal

    def act(
        self,
        _obs: Observation,
    ) -> AgentAction:
        raise Exception("Sync act method is deprecated. Use aact instead.")

    async def aact(self, obs: Observation) -> AgentAction:
        self.recv_message("Environment", obs)

        if self._goal is None:
            self._goal = await agenerate_goal(
                self.model_name,
                background=self.inbox[0][
                    1
                ].to_natural_language(),  # Only consider the first message for now
                verbose=self.verbose,
            )

        if len(obs.available_actions) == 1 and "none" in obs.available_actions:
            return AgentAction(action_type="none", argument="")
        else:
            
            # Build base history
            base_history = "\n".join(f"{y.to_natural_language()}" for x, y in self.inbox)
            
            # Add SToM section if this agent has extra visibility
            enhanced_history = base_history
            if self.goal_visibility_system and hasattr(self.goal_visibility_system, 'agent_has_stom'):
                if self.goal_visibility_system.agent_has_stom(self.agent_name):
                    # Find partner name from multiple sources
                    partner_name = None
                    
                    # Method 1: Look for other agent names in the conversation
                    if len(self.inbox) > 0:
                        for sender, message in self.inbox:
                            if sender != "Environment" and sender != self.agent_name:
                                partner_name = sender
                                break
                    
                    # Method 2: Get partner from agent_order if available
                    if not partner_name and hasattr(self.goal_visibility_system, 'agent_order'):
                        agent_order = self.goal_visibility_system.agent_order
                        if len(agent_order) >= 2:
                            for agent in agent_order:
                                if agent != self.agent_name:
                                    partner_name = agent
                                    break
                    
                    # Method 3: If no conversation yet, try to get from SToM distributions  
                    if not partner_name and hasattr(self.goal_visibility_system, 'stom_system'):
                        # Get all stored distributions and find the one for this agent
                        distributions = getattr(self.goal_visibility_system.stom_system, 'intention_distributions', {})
                        for key in distributions:
                            if key.startswith(f"{self.agent_name}_about_"):
                                partner_name = key.split("_about_")[1]
                                break
                    
                    if partner_name:
                        stom_section = self.goal_visibility_system.get_stom_section(self.agent_name, partner_name)
                        if stom_section:
                            enhanced_history = f"{base_history}\n\n{stom_section}"
                    elif self.verbose:
                        print(f"\n⚠️  SToM: No partner found for {self.agent_name} to reason about (Turn #{obs.turn_number})\n")
            
            action = await agenerate_action(
                self.model_name,
                history=enhanced_history,
                turn_number=obs.turn_number,
                action_types=obs.available_actions,
                agent=self.agent_name,
                goal=self.goal,
                script_like=self.script_like,
                verbose=self.verbose,
            )
            # Temporary fix for mixtral-moe model for incorrect generation format
            if "Mixtral-8x7B-Instruct-v0.1" in self.model_name:
                current_agent = self.agent_name
                if f"{current_agent}:" in action.argument:
                    print("Fixing Mixtral's generation format")
                    action.argument = action.argument.replace(f"{current_agent}: ", "")
                elif f"{current_agent} said:" in action.argument:
                    print("Fixing Mixtral's generation format")
                    action.argument = action.argument.replace(
                        f"{current_agent} said: ", ""
                    )

            return action


class ScriptWritingAgent(LLMAgent):
    def __init__(
        self,
        agent_name: str | None = None,
        uuid_str: str | None = None,
        agent_profile: AgentProfile | None = None,
        model_name: str = "gpt-4o-mini",
        agent_names: list[str] = [],
        background: ScriptBackground | None = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            agent_name=agent_name,
            uuid_str=uuid_str,
            agent_profile=agent_profile,
            verbose=verbose,
        )
        self.model_name = model_name
        self.agent_names = agent_names
        assert background is not None, "background cannot be None"
        self.background = background

    async def aact(self, obs: Observation) -> AgentAction:
        self.recv_message("Environment", obs)
        message_to_compose = [y for idx, (x, y) in enumerate(self.inbox) if idx != 0]

        history = "\n".join(f"{y.to_natural_language()}" for y in message_to_compose)

        action, prompt = await agenerate_script(
            model_name=self.model_name,
            background=self.background,
            agent_names=self.agent_names,
            history=history,
            agent_name=self.agent_name,
            single_step=True,
            verbose=self.verbose,
        )
        returned_action = cast(AgentAction, action[1][0][1])
        return returned_action


class HumanAgent(BaseAgent[Observation, AgentAction]):
    """
    A human agent that takes input from the command line.
    """

    def __init__(
        self,
        agent_name: str | None = None,
        uuid_str: str | None = None,
        agent_profile: AgentProfile | None = None,
    ) -> None:
        super().__init__(
            agent_name=agent_name,
            uuid_str=uuid_str,
            agent_profile=agent_profile,
        )
        self.model_name: LLM_Name = "human"

    @property
    def goal(self) -> str:
        if self._goal is not None:
            return self._goal
        goal = input("Goal: ")
        return goal

    @goal.setter
    def goal(self, goal: str) -> None:
        self._goal = goal

    def act(self, obs: Observation) -> AgentAction:
        self.recv_message("Environment", obs)

        print("Available actions:")
        for i, action in enumerate(obs.available_actions):
            print(f"{i}: {action}")

        action_type = obs.available_actions[int(input("Action type: "))]
        argument = input("Argument: ")

        return AgentAction(action_type=action_type, argument=argument)

    async def aact(self, obs: Observation) -> AgentAction:
        self.recv_message("Environment", obs)

        print("Available actions:")
        for i, action in enumerate(obs.available_actions):
            print(f"{i}: {action}")

        if obs.available_actions != ["none"]:
            action_type_number = await ainput(
                "Action type (Please only input the number): "
            )
            try:
                action_type_number = int(action_type_number)  # type: ignore
            except TypeError:
                print("Please input a number.")
                action_type_number = await ainput(
                    "Action type (Please only input the number): "
                )
                action_type_number = int(action_type_number)  # type: ignore
            assert isinstance(action_type_number, int), "Please input a number."
            action_type = obs.available_actions[action_type_number]
        else:
            action_type = "none"
        if action_type in ["speak", "non-verbal communication"]:
            argument = await ainput("Argument: ")
        else:
            argument = ""

        return AgentAction(action_type=action_type, argument=argument)


class Agents(dict[str, BaseAgent[Observation, AgentAction]]):
    def reset(self) -> None:
        for agent in self.values():
            agent.reset()

    def act(self, obs: dict[str, Observation]) -> dict[str, AgentAction]:
        return {
            agent_name: agent.act(obs[agent_name]) for agent_name, agent in self.items()
        }
