from typing import Dict
from enum import Enum

class VisGoalMode(Enum):
    NONE = "none"
    BOTH = "both"
    AGENT1 = "agent1"
    AGENT2 = "agent2"

class GoalVisibilitySystem:
    def __init__(self, vis_goal_mode: str):
        self.mode = VisGoalMode(vis_goal_mode)  # 使用枚举确保类型安全
        self.intention_model = None  # 预留 IM 扩展
    
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
            # Only agent2's goal is visible to all agents
            agent2_name = list(all_goals.keys())[1] if len(all_goals) > 1 else None
            if agent2_name:
                visible_goals = {agent2_name: all_goals.get(agent2_name, "")}
            else:
                visible_goals = {agent_name: all_goals.get(agent_name, "")}
        else:
            raise ValueError(f"Unsupported vis_goal mode: {self.mode}")
        
        print(f"DEBUG - Agent {agent_name} visibility mode: {self.mode}, visible goals: {visible_goals}")
        return visible_goals