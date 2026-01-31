import json
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class PromptManager:
    """
    管理提示词模板、优化规则和用户意见
    """
    def __init__(self, storage_path: Optional[str] = None):
        if storage_path is None:
            # 默认存储在项目根目录的 config 文件夹下
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            self.storage_path = os.path.join(base_dir, "config", "prompt_design.json")
        else:
            self.storage_path = storage_path
        self._ensure_storage_exists()
        self.data = self._load_data()

    def _ensure_storage_exists(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        if not os.path.exists(self.storage_path):
            default_data = {
                "templates": [
                    {
                        "id": "default_risk",
                        "name": "企业违约风险评估模板",
                        "description": "基础的企业违约风险评估 Prompt，包含角色、任务和框架。",
                        "content": """【角色】
您是一位具备金融风控专业知识的智能分析师，擅长结合企业定性及定量的指标数据对企业违约风险进行评估。

【任务指令】
请依据下方的【风险分析框架】和输入中提供的【企业指标数据报告】，按以下步骤对该企业信用风险进行综合判断：
1. 依据【风险分析框架】对企业经营、债务、信用等维度进行分析；
2. 通过指标间的交叉验证，判断是否存在框架未覆盖的风险点；
3. 综合评估企业在未来3个月内的违约风险。

【风险分析框架】
（1）企业经营稳定性：结合企业属性、行业及收支流水，判断经营是否良好及具备偿债能力。
（2）企业债务和流动性：结合短借长投、债务结构，判断是否过度举债。
（3）其它风险信号：关注机器学习评分、行内评级及外部征信数据。

【强制约束】
1. 必须建立双向验证机制：当关键信号与指标数据矛盾时，需启动二次核查；
2. 严格区分事实指标与推测结论。

【输出要求】
根据以上分析，如果认为该企业在未来3个月内不会违约，请输出“否”；如果存在违约风险，请输出“是”。请直接输出结果，不要包含其他分析过程。

【企业指标数据报告】
{data_report}"""
                    }
                ],
                "rules": [
                    {"id": "style_prof", "name": "专业金融风格", "category": "语言风格", "description": "使用更加专业、严谨的金融风控术语。"},
                    {"id": "struct_cot", "name": "思维链(CoT)增强", "category": "结构优化", "description": "引导模型输出详细的推理过程，而不仅仅是结论。"},
                    {"id": "content_logic", "name": "逻辑一致性校验", "category": "内容增强", "description": "增加对数据逻辑矛盾的识别和处理指令。"},
                    {"id": "scene_bank", "name": "银行信贷场景适配", "category": "特定场景", "description": "针对银行信贷业务特点，加强对还款意愿的评估。"}
                ],
                "opinions": [
                    {"id": "op_more_data", "content": "增加对现金流指标的关注"},
                    {"id": "op_shorter", "content": "精简输出，只保留关键风险点"},
                    {"id": "op_clear_step", "content": "步骤描述需要更清晰"},
                    {"id": "op_add_ref", "content": "引用具体的数据指标作为支撑"}
                ],
                "feature_maps": [],
                "history": []
            }
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(default_data, f, indent=4, ensure_ascii=False)

    def _load_data(self) -> Dict[str, Any]:
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if "feature_maps" not in data:
                    data["feature_maps"] = []
                return data
        except Exception as e:
            logger.error(f"加载 Prompt 数据失败: {e}")
            return {"templates": [], "rules": [], "opinions": [], "feature_maps": [], "history": []}

    # --- Feature Map Methods ---
    def get_feature_maps(self) -> List[Dict]:
        return self.data.get("feature_maps", [])

    def update_feature_maps(self, feature_maps: List[Dict]):
        self.data["feature_maps"] = feature_maps
        self._save_data()

    def _save_data(self):
        try:
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存 Prompt 数据失败: {e}")

    # --- Template Methods ---
    def get_templates(self) -> List[Dict]:
        return self.data.get("templates", [])

    def add_template(self, template: Dict):
        self.data["templates"].append(template)
        self._save_data()

    # --- Rule Methods ---
    def get_rules(self) -> List[Dict]:
        return self.data.get("rules", [])

    def add_rule(self, rule: Dict):
        if "id" not in rule:
            rule["id"] = f"rule_{len(self.data['rules']) + 1}"
        self.data["rules"].append(rule)
        self._save_data()

    def delete_rule(self, rule_id: str):
        self.data["rules"] = [r for r in self.data["rules"] if r["id"] != rule_id]
        self._save_data()

    # --- Opinion Methods ---
    def get_opinions(self) -> List[Dict]:
        return self.data.get("opinions", [])

    def add_opinion(self, opinion: Dict):
        if "id" not in opinion:
            opinion["id"] = f"op_{len(self.data['opinions']) + 1}"
        self.data["opinions"].append(opinion)
        self._save_data()

    # --- History/Version Control ---
    def save_version(self, original_id: str, original_content: str, optimized_content: str, metadata: Dict):
        entry = {
            "id": f"v_{len(self.data['history']) + 1}",
            "original_id": original_id,
            "original_content": original_content,
            "optimized_content": optimized_content,
            "metadata": metadata,
            "timestamp": metadata.get("timestamp")
        }
        self.data["history"].append(entry)
        self._save_data()
        return entry

    def get_latest_history(self) -> Optional[Dict]:
        if not self.data.get("history"):
            return None
        return self.data["history"][-1]
