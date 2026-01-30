import openai
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class PromptOptimizer:
    """
    负责调用 LLM 进行提示词优化
    """
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: str = "gpt-4"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def optimize(self, 
                 original_prompt: str, 
                 rules: List[Dict], 
                 opinions: List[str], 
                 custom_feedback: str = "") -> str:
        """
        根据规则和意见优化提示词
        """
        if not self.api_key or not self.api_key.strip():
            return original_prompt + "\n\n(注意: 未配置 API Key，无法进行 AI 优化)"

        # DeepSeek 接口地址通常需要显式指定
        base_url = self.base_url or "https://api.deepseek.com"
        
        try:
            client = openai.OpenAI(api_key=self.api_key, base_url=base_url)
            
            rules_str = "\n".join([f"- {r['name']}: {r['description']}" for r in rules])
            opinions_str = "\n".join([f"- {o}" for o in opinions])
            
            system_prompt = "你是一个顶级的提示词工程（Prompt Engineering）专家，擅长将普通提示词转化为高性能、结构化的指令。"
            
            user_prompt = f"""请根据以下信息，对给出的【原始提示词】进行重写和优化。

【原始提示词】
{original_prompt}

【优化规则】
{rules_str if rules else "保持原有风格，进行通用质量提升。"}

【用户改进建议/意见】
{opinions_str if opinions else "无"}
{f"自定义反馈: {custom_feedback}" if custom_feedback else ""}

【任务要求】
1. 严格遵循上述优化规则和改进建议。
2. 保持提示词的核心逻辑和业务目标不变（企业违约风险评估）。
3. 增强提示词的结构化程度，使其更易于大模型理解和执行。
4. 优化后的提示词应包含清晰的角色定义、任务描述、约束条件和输出格式。
5. 直接返回优化后的提示词内容，不要包含任何解释性文字、前导语或 Markdown 代码块标识符。

优化后的提示词："""

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            optimized_content = response.choices[0].message.content.strip()
            # 清理 Markdown 代码块
            if optimized_content.startswith("```"):
                lines = optimized_content.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                optimized_content = "\n".join(lines).strip()
                
            return optimized_content
            
        except Exception as e:
            logger.error(f"提示词优化失败: {e}")
            return f"优化失败: {str(e)}"
