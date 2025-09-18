from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
import time
import math


@dataclass
class IntentionDistribution:
    """表示对伙伴意图的概率分布"""
    intentions: List[Tuple[str, float]]  # (intention_description, probability)
    confidence: float  # 置信度 (0-1)
    update_count: int = 0  # 更新次数
    last_update_info: str = ""  # 最后更新的信息
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """确保概率分布归一化"""
        total_prob = sum(prob for _, prob in self.intentions)
        if total_prob > 0:
            self.intentions = [(desc, prob / total_prob) for desc, prob in self.intentions]
        
        # 计算置信度（基于Shannon熵）
        self._calculate_confidence()
    
    def _calculate_confidence(self):
        """基于 Shannon 熵的贝叶斯置信度 (Bayesian Confidence)"""
        if not self.intentions:
            self.confidence = 0.0
            return

        # 提取概率，并做 ε-平滑避免 log(0)
        epsilon = 1e-12
        probs = [max(prob, epsilon) for _, prob in self.intentions]
        total = sum(probs)
        probs = [p / total for p in probs]

        # Shannon 熵
        entropy = -sum(p * math.log(p) for p in probs)

        # 最大熵：均匀分布时
        max_entropy = math.log(len(probs)) if len(probs) > 1 else 1.0

        # 归一化到 [0, 1]
        self.confidence = max(0.0, 1.0 - entropy / max_entropy)
    
    def update_probabilities(self, likelihoods: List[float]):
        """基于likelihood更新概率分布（贝叶斯更新）"""
        if len(likelihoods) != len(self.intentions):
            raise ValueError("Likelihood数量必须与意图数量一致")
        
        # 贝叶斯更新: P(θ|obs) ∝ P(obs|θ) * P(θ)
        new_probs = []
        for i, (desc, prior_prob) in enumerate(self.intentions):
            posterior = likelihoods[i] * prior_prob
            new_probs.append(posterior)
        
        # 归一化
        total = sum(new_probs)
        if total > 0:
            new_probs = [p / total for p in new_probs]
        else:
            # 如果所有likelihood都是0，保持原分布
            new_probs = [prob for _, prob in self.intentions]
        
        # 更新分布
        self.intentions = [(desc, prob) for (desc, _), prob in zip(self.intentions, new_probs)]
        self.update_count += 1
        self._calculate_confidence()
    
    def get_most_likely_intention(self) -> Tuple[str, float]:
        """获取最可能的意图"""
        if not self.intentions:
            return ("Unknown", 0.0)
        return max(self.intentions, key=lambda x: x[1])
    
    def to_stom_section(self) -> str:
        """转换为SToM section格式"""
        if not self.intentions:
            return ""
        
        # 构建意图分析内容
        stom_content = []
        for desc, prob in self.intentions:
            percentage = prob * 100
            stom_content.append(f"• {desc}: {percentage:.1f}%")
        
        most_likely, confidence_level = self.get_most_likely_intention()
        
        confidence_text = "HIGH" if self.confidence > 0.7 else "MED" if self.confidence > 0.4 else "LOW"
        
        return f"""
=== 🧠 THEORY OF MIND (SToM) SECTION ===
Based on your Stochastic Theory of Mind (SToM) module reasoning, you have inferred the following about other participants' goals:

{chr(10).join(stom_content)}

Overall Confidence: {confidence_text} ({self.confidence:.2f})
Most Likely: {most_likely} ({confidence_level*100:.1f}%)

Use this SToM information strategically in your decision-making. When confidence is low, consider asking questions to gather more information. When confidence is high, act decisively based on your inferences.

Remember: your SToM inferences are private to you, just as others' SToM inferences are private to them. Other participants cannot see your SToM reasoning.
=========================================="""


@dataclass 
class SToMState:
    """管理单个agent的SToM状态"""
    agent_name: str
    partner_name: str
    distribution: IntentionDistribution = None
    is_initialized: bool = False
    conversation_history: List[str] = field(default_factory=list)
    
    def add_conversation_turn(self, turn: str):
        """添加对话轮次"""
        self.conversation_history.append(turn)
    
    def get_context_for_lhm(self) -> str:
        """获取用于LHM的上下文"""
        return "\n".join(self.conversation_history)