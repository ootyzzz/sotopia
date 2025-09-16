from typing import Dict, Tuple
from enum import Enum

def debug_print(message: str, verbose: bool = False) -> None:
    """统一的调试输出函数"""
    if verbose:
        print(f"DEBUG - {message}")

class VisGoalMode(Enum):
    NONE = "none"
    BOTH = "both"
    AGENT1 = "agent1"
    AGENT2 = "agent2"

class GoalVisibilitySystem:
    def __init__(self, vis_goal_mode: str, verbose: bool = False):
        self.mode = VisGoalMode(vis_goal_mode)  # 使用枚举确保类型安全
        self.verbose = verbose
        self.intention_model = None  # 预留 IM 扩展
    
    def get_agent_goals_and_stom(self, agent_name: str, all_goals: Dict[str, str]) -> Tuple[str, str]:
        """
        返回代理的可见目标和SToM信息
        Returns: (own_goal, stom_section)
        """
        # 每个代理始终只能看到自己的目标
        own_goal = all_goals.get(agent_name, "")
        
        # 根据vis_goal模式生成SToM section
        stom_section = self._generate_stom_section(agent_name, all_goals)
        
        debug_print(f"Agent {agent_name} visibility mode: {self.mode}", self.verbose)
        debug_print(f"Agent {agent_name} own goal: {own_goal}", self.verbose)
        debug_print(f"Agent {agent_name} SToM section: {stom_section}", self.verbose)
        
        return own_goal, stom_section
    
    def _generate_stom_section(self, agent_name: str, all_goals: Dict[str, str]) -> str:
        """生成SToM section内容"""
        if self.mode == VisGoalMode.NONE:
            return ""
        
        # 获取其他代理的目标
        other_agents = [name for name in all_goals.keys() if name != agent_name]
        
        if self.mode == VisGoalMode.BOTH:
            # 显示所有其他代理的目标
            stom_goals = []
            for other_agent in other_agents:
                goal = all_goals.get(other_agent, "")
                stom_goals.append(f"{other_agent}: {goal}")
            
            if stom_goals:
                stom_content = "\n".join(stom_goals)
                return self._format_stom_section(stom_content)
            
        elif self.mode == VisGoalMode.AGENT1:
            # 只显示agent1的目标
            agent1_name = list(all_goals.keys())[0] if all_goals else None
            if agent1_name and agent1_name != agent_name:
                goal = all_goals.get(agent1_name, "")
                stom_content = f"{agent1_name}: {goal}"
                return self._format_stom_section(stom_content)
                
        elif self.mode == VisGoalMode.AGENT2:
            # 只显示agent2的目标
            agent2_name = list(all_goals.keys())[1] if len(all_goals) > 1 else None
            if agent2_name and agent2_name != agent_name:
                goal = all_goals.get(agent2_name, "")
                stom_content = f"{agent2_name}: {goal}"
                return self._format_stom_section(stom_content)
        
        return ""
    
    def _format_stom_section(self, stom_content: str) -> str:
        """格式化SToM section"""
        return f"""
=== THEORY OF MIND (SToM) SECTION ===
Based on your Stochastic Theory of Mind (SToM) module reasoning, you have inferred the following about other participants' goals:

{stom_content}
Your performance will be marked based on your own goal achievement. Do bargin on the number if you feel you goal is contradicting with others. For example, if you can't ask for an extra 5 min, try asking for 3 min.
Use this SToM information strategically in your decision-making. Remember: your SToM inferences are private to you, just as others' SToM inferences are private to them. Other participants cannot see your SToM reasoning.
=========================================="""

    # 保持向后兼容的方法
    def get_visible_goals(self, agent_name: str, all_goals: Dict[str, str]) -> Dict[str, str]:
        """向后兼容方法 - 现在只返回自己的目标"""
        return {agent_name: all_goals.get(agent_name, "")}