from typing import Dict, Optional
from enum import Enum
from .stom import SToMSystem


class VisGoalMode(Enum):
    NONE = "none"
    BOTH = "both"
    AGENT1 = "agent1"  # agent1 has extra visibility and can see agent2's goal -> only agent1 uses SToM
    AGENT2 = "agent2"  # agent2 has extra visibility and can see agent1's goal -> only agent2 uses SToM
    STOM = "stom"  # Legacy explicit SToM mode


class GoalVisibilitySystem:
    def __init__(self, vis_goal_mode: str, model_name: str = "gpt-4o-2024-08-06", verbose: bool = False):
        self.mode = VisGoalMode(vis_goal_mode)  # 使用枚举确保类型安全
        self.model_name = model_name
        self.verbose = verbose
        
        # SToM系统始终初始化，但只在有agent开启额外视野时使用
        self.stom_system = SToMSystem(model_name=model_name, verbose=verbose)
        
        # 记录agent顺序，用于确定哪个是agent1/agent2
        self.agent_order = []
    
    def set_agent_order(self, agent_names: list):
        """设置agent顺序，用于确定哪个是agent1/agent2"""
        self.agent_order = agent_names.copy()
    
    def agent_has_stom(self, agent_name: str) -> bool:
        """判断指定agent是否应该使用SToM（基于额外视野）"""
        if self.mode == VisGoalMode.NONE:
            return False
        elif self.mode == VisGoalMode.BOTH:
            return True
        elif self.mode == VisGoalMode.AGENT1:
            # vis_goal="agent1" 意味着agent2能看到agent1的goal，所以agent2使用SToM
            return len(self.agent_order) > 1 and agent_name == self.agent_order[1]
        elif self.mode == VisGoalMode.AGENT2:
            # vis_goal="agent2" 意味着agent1能看到agent2的goal，所以agent1使用SToM
            return len(self.agent_order) > 0 and agent_name == self.agent_order[0]
        elif self.mode == VisGoalMode.STOM:
            return True  # Legacy mode
        return False
    
    async def initialize_stom_for_agent(
        self,
        agent_name: str,
        partner_name: str,
        scenario_description: str,
        conversation_context: str = ""
    ):
        """为agent初始化对partner的意图分布（如果该agent有额外视野）"""
        if self.agent_has_stom(agent_name) and self.stom_system:
            return await self.stom_system.initialize_intentions(
                agent_name=agent_name,
                partner_name=partner_name,
                scenario_description=scenario_description,
                conversation_context=conversation_context
            )
        return None
    
    async def update_stom_for_agent(
        self,
        agent_name: str,
        partner_name: str,
        new_partner_action: str,
        conversation_context: str,
        turn_number: int = None
    ):
        """根据partner的新行动更新意图分布（如果该agent有额外视野）"""
        if self.agent_has_stom(agent_name) and self.stom_system:
            return await self.stom_system.update_intentions(
                agent_name=agent_name,
                partner_name=partner_name,
                new_partner_action=new_partner_action,
                conversation_context=conversation_context,
                turn_number=turn_number
            )
        return None
    
    def get_stom_section(self, agent_name: str, partner_name: str) -> Optional[str]:
        """获取给定agent关于partner的SToM推理内容（如果该agent有额外视野）"""
        if self.agent_has_stom(agent_name) and self.stom_system:
            return self.stom_system.get_stom_section(agent_name, partner_name)
        return None
    
    def get_stom_section(self, agent_name: str, partner_name: str) -> str:
        """获取SToM部分的文本（用于模板）"""
        if self.mode in [VisGoalMode.AGENT2, VisGoalMode.STOM] and self.stom_system:
            distribution = self.stom_system.get_intentions(agent_name, partner_name)
            if distribution:
                return distribution.to_stom_section()
        return ""
    
    def get_visible_goals(self, agent_name: str, all_goals: Dict[str, str]) -> Dict[str, str]:
        if self.mode == VisGoalMode.BOTH:
            visible_goals = all_goals
        elif self.mode == VisGoalMode.NONE:
            visible_goals = {agent_name: all_goals.get(agent_name, "")}
        elif self.mode == VisGoalMode.AGENT1:
            # Only agent1's goal is visible to all agents
            agent1_name = list(all_goals.keys())[0] if all_goals else None
            if agent1_name:
                visible_goals = {agent1_name: all_goals.get(agent1_name, "")}
            else:
                visible_goals = {agent_name: all_goals.get(agent_name, "")}
        elif self.mode == VisGoalMode.AGENT2:
            # Only agent2's goal is visible to all agents, plus SToM for agent1
            agent2_name = list(all_goals.keys())[1] if len(all_goals) > 1 else None
            if agent2_name:
                visible_goals = {agent2_name: all_goals.get(agent2_name, "")}
            else:
                visible_goals = {agent_name: all_goals.get(agent_name, "")}
        elif self.mode == VisGoalMode.STOM:
            # 在SToM模式下，agent只能看到自己的goal
            visible_goals = {agent_name: all_goals.get(agent_name, "")}
        else:
            raise ValueError(f"Unsupported vis_goal mode: {self.mode}")
        
        return visible_goals