import asyncio
from typing import List
from sotopia.generation_utils.generate import agenerate, StrOutputParser
from .data_structures import IntentionDistribution


class LikelihoodModel:
    """似然模型 - 根据新的对话内容更新意图概率"""
    
    def __init__(self, model_name: str = "gpt-4o-2024-08-06", verbose: bool = False):
        self.model_name = model_name
        self.verbose = verbose
    
    async def compute_likelihoods(
        self,
        current_distribution: IntentionDistribution,
        agent_name: str,
        partner_name: str,
        new_partner_action: str,
        conversation_context: str
    ) -> List[float]:
        """
        计算新行动对每个意图假设的似然值
        
        Returns:
            List[float]: 每个意图的似然值，顺序对应current_distribution.intentions
        """
        
        # 构建意图列表
        intention_list = []
        for i, (desc, prob) in enumerate(current_distribution.intentions):
            intention_list.append(f"{i+1}. {desc} (Current: {prob:.2f})")
        
        template = """You are an expert in social psychology and behavioral analysis. Your task is to evaluate how likely a person's recent action is, given different possible intentions they might have.

**Scenario:**
- {agent_name} is interacting with {partner_name}
- You have several hypotheses about {partner_name}'s possible intentions
- {partner_name} just took a new action
- You need to evaluate how consistent this action is with each intention hypothesis

**Current Intention Hypotheses for {partner_name}:**
{intention_list}

**Conversation Context:**
{conversation_context}

**{partner_name}'s Latest Action:**
{new_partner_action}

**Your Task:**
For each intention hypothesis listed above, estimate the likelihood that {partner_name} would take this specific action if they truly had that intention. 

Consider:
- How well does this action align with achieving that intention?
- Is this action strategically consistent with that goal?
- Does the timing and context make sense for that intention?

**Output Format:**
Provide only the likelihood values (0.0 to 1.0) for each intention, one per line:
<likelihood_1>
<likelihood_2>
<likelihood_3>
...

Example:
0.85
0.20
0.60

Your likelihood assessments:"""

        try:
            if self.verbose:
                print(f"🧠 SToM [LHM]: Computing likelihoods for {partner_name}'s action")
                print(f"🧠 SToM [LHM]: Action: '{new_partner_action[:100]}{'...' if len(new_partner_action) > 100 else ''}'")
                print(f"🧠 SToM [LHM]: Evaluating against {len(current_distribution.intentions)} intentions")
                
                # Show the complete prompt being sent to LHM
                formatted_prompt = template.format(
                    agent_name=agent_name,
                    partner_name=partner_name,
                    intention_list="\n".join(intention_list),
                    conversation_context=conversation_context,
                    new_partner_action=new_partner_action
                )
                print(f"🧠 SToM [LHM INPUT]: Complete prompt being sent to likelihood model:")
                print(f"🧠 SToM [LHM INPUT]: {formatted_prompt}")
            
            result = await agenerate(
                model_name=self.model_name,
                template=template,
                input_values={
                    "agent_name": agent_name,
                    "partner_name": partner_name,
                    "intention_list": "\n".join(intention_list),
                    "conversation_context": conversation_context,
                    "new_partner_action": new_partner_action
                },
                output_parser=StrOutputParser(),
                verbose=True,  # Always show LHM output
                temperature=0.0  # 确保输出稳定
            )
            
            if self.verbose:
                print(f"🧠 SToM [LHM OUTPUT]: Raw likelihood model response:")
                print(f"🧠 SToM [LHM OUTPUT]: {result}")
            
            # 解析似然值
            likelihoods = self._parse_likelihood_output(result, len(current_distribution.intentions))
            
            if self.verbose:
                print(f"🧠 SToM [PARSED]: Parsed likelihoods: {[f'{l:.3f}' for l in likelihoods]}")
            
            return likelihoods
            
        except Exception as e:
            if self.verbose:
                print(f"🧠 SToM: Error computing likelihoods: {e}")
            
            # 返回均匀似然值（不改变分布）
            num_intentions = len(current_distribution.intentions)
            return [1.0] * num_intentions
    
    def _parse_likelihood_output(self, output: str, expected_count: int) -> List[float]:
        """解析LHM输出为似然值列表"""
        likelihoods = []
        lines = output.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):  # 忽略空行和注释
                try:
                    # 尝试解析为浮点数
                    likelihood = float(line)
                    # 确保在合理范围内
                    likelihood = max(0.0, min(1.0, likelihood))
                    likelihoods.append(likelihood)
                except ValueError:
                    # 如果不能解析为数字，跳过这一行
                    continue
        
        # 确保数量正确
        if len(likelihoods) != expected_count:
            if self.verbose:
                print(f"🧠 SToM: Expected {expected_count} likelihoods, got {len(likelihoods)}. Using uniform values.")
            # 返回均匀似然值
            return [1.0] * expected_count
        
        return likelihoods