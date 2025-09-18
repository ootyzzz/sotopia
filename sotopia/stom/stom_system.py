import asyncio
from typing import Optional, Tuple
from .data_structures import IntentionDistribution
from .intention_model import IntentionModel
from .likelihood_model import LikelihoodModel
import gin


@gin.configurable
class SToMSystem:
    """
    随机心智理论系统 - 统一管理意图建模和更新
    
    这个系统结合了意图模型(IM)和似然模型(LHM)，实现：
    1. 初始化：使用IM生成初始意图分布
    2. 更新：使用LHM根据新观察进行贝叶斯更新
    """
    
    def __init__(self, model_name: str = "gpt-4o-2024-08-06", verbose: bool = False,
                 temperature: float = 0.0, prompt_template: str = None):
        self.model_name = model_name
        self.verbose = verbose
        self.intention_model = IntentionModel(model_name, verbose, prompt_template, temperature)
        self.likelihood_model = LikelihoodModel(model_name, verbose, temperature)
        
        # 存储每个agent对其partner的意图分布
        self.intention_distributions = {}
    
    async def initialize_intentions(
        self,
        agent_name: str,
        partner_name: str,
        scenario_description: str,
        conversation_context: str = ""
    ) -> IntentionDistribution:
        """
        为agent初始化对partner的意图分布
        
        Args:
            agent_name: 观察者agent的名字
            partner_name: 被观察的partner的名字
            scenario_description: 场景描述
            conversation_context: 对话上下文
            
        Returns:
            IntentionDistribution: 初始化的意图分布
        """
        if self.verbose:
            print(f"🧠 SToM: Initializing intentions for {agent_name} about {partner_name}")
        
        # Build a mock final_prompt for the IntentionModel
        # This simulates what the agent would see in their prompt
        final_prompt = f"""Scenario: {scenario_description}

Agent: {agent_name}
Partner: {partner_name}

{conversation_context if conversation_context else "Conversation just starting..."}

Your task is to interact with {partner_name} to achieve your social goal."""
        
        distribution = await self.intention_model.initialize_intentions(
            agent_name=agent_name,
            partner_name=partner_name,
            final_prompt=final_prompt
        )
        
        # 存储分布
        key = f"{agent_name}_about_{partner_name}"
        self.intention_distributions[key] = distribution
        
        if self.verbose:
            print(f"🧠 SToM [INIT]: {agent_name} → {partner_name}")
            print(f"🧠 SToM [INIT]: Initial distribution stored")
            # Don't print the full section here to reduce noise
        
        return distribution
    
    async def update_intentions(
        self,
        agent_name: str,
        partner_name: str,
        new_partner_action: str,
        conversation_context: str,
        turn_number: int = None
    ) -> Optional[IntentionDistribution]:
        """
        根据partner的新行动更新意图分布
        
        Args:
            agent_name: 观察者agent的名字
            partner_name: 执行行动的partner的名字
            new_partner_action: partner的新行动
            conversation_context: 当前对话上下文
            turn_number: 当前turn数（用于logging）
            
        Returns:
            Optional[IntentionDistribution]: 更新后的分布，如果没有初始化则返回None
        """
        key = f"{agent_name}_about_{partner_name}"
        
        # 检查是否已有意图分布初始化
        if key not in self.intention_distributions:
            if self.verbose:
                turn_info = f" (Turn #{turn_number})" if turn_number is not None else ""
                print(f"🧠 SToM{turn_info}: No existing distribution for {key}, skipping update")
            return None
        
        current_distribution = self.intention_distributions[key]
        
        turn_info = f" [Turn #{turn_number}]" if turn_number is not None else ""
        if self.verbose:
            print(f"🧠 SToM{turn_info}: {agent_name} updating beliefs about {partner_name}")
            print(f"🧠 SToM{turn_info}: Observing: '{new_partner_action[:80]}{'...' if len(new_partner_action) > 80 else ''}'")
        
        # 计算似然值
        likelihoods = await self.likelihood_model.compute_likelihoods(
            current_distribution=current_distribution,
            agent_name=agent_name,
            partner_name=partner_name,
            new_partner_action=new_partner_action,
            conversation_context=conversation_context
        )
        
        if self.verbose:
            print(f"🧠 SToM{turn_info}: [LIKELIHOOD] Computed likelihoods for each intention")
            for i, (intention, likelihood) in enumerate(zip(current_distribution.intentions, likelihoods)):
                print(f"    Intent {i+1}: P(action|intent) = {likelihood:.3f}")
        
        # 进行贝叶斯更新
        current_distribution.update_probabilities(likelihoods)
        
        if self.verbose:
            print(f"🧠 SToM{turn_info}: [POSTERIOR] Updated belief distribution:")
            for i, (intention, prob) in enumerate(zip(current_distribution.intentions, current_distribution.probabilities)):
                print(f"    Intent {i+1}: P(intent|action) = {prob:.3f} - {intention[:60]}{'...' if len(intention) > 60 else ''}")
        
        return current_distribution
    
    def get_intentions(self, agent_name: str, partner_name: str) -> Optional[IntentionDistribution]:
        """获取agent对partner的当前意图分布"""
        key = f"{agent_name}_about_{partner_name}"
        return self.intention_distributions.get(key)
    
    def get_stom_section(self, agent_name: str, partner_name: str) -> Optional[str]:
        """获取agent对partner的SToM推理内容的字符串表示"""
        distribution = self.get_intentions(agent_name, partner_name)
        if distribution:
            return distribution.to_stom_section()
        return None
    
    def has_intentions(self, agent_name: str, partner_name: str) -> bool:
        """检查是否已为agent初始化对partner的意图分布"""
        key = f"{agent_name}_about_{partner_name}"
        return key in self.intention_distributions
    
    def clear_intentions(self, agent_name: str = None, partner_name: str = None):
        """清除意图分布（可选择性清除）"""
        if agent_name is None and partner_name is None:
            # 清除所有
            self.intention_distributions.clear()
            if self.verbose:
                print("🧠 SToM: Cleared all intention distributions")
        else:
            # 清除特定的
            keys_to_remove = []
            for key in self.intention_distributions:
                if agent_name and not key.startswith(f"{agent_name}_about_"):
                    continue
                if partner_name and not key.endswith(f"_about_{partner_name}"):
                    continue
                keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.intention_distributions[key]
                if self.verbose:
                    print(f"🧠 SToM: Cleared intention distribution for {key}")
    
    def _is_non_action(self, action: str) -> bool:
        """
        检查是否为无效行动（应该跳过SToM更新的行动）
        
        Args:
            action: 行动的字符串表示
            
        Returns:
            bool: True如果是无效行动，False如果是有效行动
        """
        if not action or action.strip() == "":
            return True
        
        # 标准化行动文本
        action_lower = action.lower().strip()
        
        # 检查常见的无行动模式
        non_action_patterns = [
            "did nothing",
            "no action", 
            "none",
            "wait",
            "waiting",
            "[no action]",
            "no response",
            "silence",
            "said: [did nothing]",
            "action_type: none",
            "argument: ",
            "action_type: none, argument:",
            # 针对AgentAction对象的检查
            "action_type=none",
            "argument=",
            "action_type='none'",
            "argument=''",
        ]
        
        for pattern in non_action_patterns:
            if pattern in action_lower:
                return True
        
        # 检查是否只包含空白字符或标点
        cleaned = action_lower.replace(" ", "").replace(".", "").replace(",", "").replace(":", "").replace("'", "").replace('"', "")
        if cleaned == "" or cleaned == "none":
            return True
        
        # 特别检查AgentAction的默认无效状态
        if "action_type" in action_lower and "none" in action_lower and "argument" in action_lower:
            # 类似 "AgentAction(action_type='none', argument='')" 的情况
            return True
            
        return False