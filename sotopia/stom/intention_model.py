import asyncio
from typing import List, Tuple
from sotopia.generation_utils.generate import agenerate, StrOutputParser
from .data_structures import IntentionDistribution


class IntentionModel:
    """意图模型 - 负责初始化意图类别和概率分布"""
    
    def __init__(self, model_name: str = "gpt-4o-2024-08-06", verbose: bool = False):
        self.model_name = model_name
        self.verbose = verbose
    
    async def initialize_intentions(
        self, 
        agent_name: str, 
        partner_name: str, 
        final_prompt: str
    ) -> IntentionDistribution:
        """
        基于agent的final prompt初始化伙伴意图分布
        """
        template = """You are an Intention Model (IM).
Given a scenario description and my goal, infer a probability distribution over my partner's possible hidden intentions.

Requirements:
1. The output must be a complete, non-overlapping distribution, and the probabilities must sum to exactly 1.
2. Use the following fixed top-level categories:
   [Conflict] Contradicts my goal
   [Neutral] Neither supports nor opposes my goal
   [Cooperative] Supports or agrees with my goal
   [Irrelevant] Completely unrelated to the task
3. For [Conflict]: expand into about 3 most probable specific subcases (not including [Conflict]-other).  
   For [Neutral], [Cooperative], and [Irrelevant]: expand into 1–2 common subcases; all remaining cases should be grouped under "other."
4. Ordering: list the most likely [Conflict] subcases first, then [Neutral], then [Cooperative], then [Irrelevant].
5. Strict output format:
   [Category] specific intention description probability
6. Probabilities must be decimals between 0 and 1 and normalized to sum exactly to 1.
7. Do not provide explanations or commentary. Only output the distribution.

**Context for Analysis:**
- Scenario involves {agent_name} and {partner_name}
- You need to infer {partner_name}'s possible intentions from {agent_name}'s perspective

**Input Information:**
{final_prompt}

**Required Output Format:**
[Conflict] specific description probability
[Conflict] specific description probability
[Conflict] specific description probability
[Conflict] other probability
[Neutral] specific description probability
[Neutral] other probability
[Cooperative] specific description probability
[Cooperative] other probability
[Irrelevant] specific description probability
[Irrelevant] other probability

Your distribution:"""

        try:
            if self.verbose:
                print(f"🧠 SToM: Initializing intentions for {partner_name} from {agent_name}'s perspective")
            
            result = await agenerate(
                model_name=self.model_name,
                template=template,
                input_values={
                    "agent_name": agent_name,
                    "partner_name": partner_name,
                    "final_prompt": final_prompt
                },
                output_parser=StrOutputParser(),
                verbose=self.verbose,
                temperature=0.0  # 确保输出稳定
            )
            
            # 解析输出
            intentions = self._parse_intention_output(result)
            
            distribution = IntentionDistribution(
                intentions=intentions,
                confidence=0.0,  # 将在__post_init__中计算
                last_update_info="Initial analysis"
            )
            
            if self.verbose:
                print(f"🧠 SToM: Initialized {len(intentions)} intentions with confidence {distribution.confidence:.2f}")
                for desc, prob in intentions:
                    print(f"   • {desc}: {prob:.2f}")
            
            return distribution
            
        except Exception as e:
            if self.verbose:
                print(f"🧠 SToM: Error initializing intentions: {e}")
            
            # 返回默认分布
            return IntentionDistribution(
                intentions=[
                    ("Unknown intention - analysis failed", 1.0)
                ],
                confidence=0.0,
                last_update_info=f"Error during initialization: {str(e)}"
            )
    
    def _parse_intention_output(self, output: str) -> List[Tuple[str, float]]:
        """解析IM输出为意图列表 - 支持新的结构化格式"""
        intentions = []
        lines = output.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 新格式: [Category] description probability
            if line.startswith('[') and ']' in line:
                try:
                    # 找到类别和描述的分界点
                    category_end = line.find(']')
                    if category_end == -1:
                        continue
                        
                    category = line[1:category_end]  # 提取类别名
                    remainder = line[category_end + 1:].strip()
                    
                    # 从右边找最后一个空格，分离描述和概率
                    parts = remainder.rsplit(' ', 1)
                    if len(parts) != 2:
                        continue
                        
                    description = parts[0].strip()
                    prob_str = parts[1].strip()
                    
                    # 构建完整的意图描述 (包含类别信息)
                    full_description = f"[{category}] {description}"
                    
                    # 解析概率
                    prob = float(prob_str)
                    intentions.append((full_description, prob))
                    
                except (ValueError, IndexError) as e:
                    if self.verbose:
                        print(f"🧠 SToM: Failed to parse structured line '{line}': {e}")
                    continue
            
            # 兼容旧格式: "Intention: <description> | Probability: <0.XX>"
            elif 'Intention:' in line and 'Probability:' in line:
                try:
                    parts = line.split('|')
                    if len(parts) >= 2:
                        intention_part = parts[0].replace('Intention:', '').strip()
                        prob_part = parts[1].replace('Probability:', '').strip()
                        
                        # 清理格式标记
                        intention_part = intention_part.strip('*').strip()
                        if intention_part and intention_part[0].isdigit():
                            intention_part = intention_part.split('.', 1)[1].strip() if '.' in intention_part else intention_part
                        intention_part = intention_part.strip('*').strip()
                        
                        prob_part = prob_part.strip('*').strip()
                        prob = float(prob_part)
                        intentions.append((intention_part, prob))
                        
                except (ValueError, IndexError) as e:
                    if self.verbose:
                        print(f"🧠 SToM: Failed to parse legacy line '{line}': {e}")
                    continue
        
        # 如果解析失败，返回默认意图
        if not intentions:
            if self.verbose:
                print(f"🧠 SToM: No intentions parsed from output, using default")
            intentions = [("Could not determine specific intentions", 1.0)]
        
        # 确保概率和为1
        total_prob = sum(prob for _, prob in intentions)
        if total_prob > 0:
            intentions = [(desc, prob / total_prob) for desc, prob in intentions]
        else:
            if self.verbose:
                print(f"🧠 SToM: Total probability was 0, using uniform distribution")
            intentions = [(desc, 1.0/len(intentions)) for desc, _ in intentions]
        
        if self.verbose:
            print(f"🧠 SToM: Successfully parsed {len(intentions)} intentions")
            for i, (desc, prob) in enumerate(intentions):
                print(f"    {i+1}. {desc}: {prob:.3f}")
        
        return intentions