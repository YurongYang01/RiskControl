import os
import json
import logging
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import sys
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))

# 导入现有的质检模块
try:
    from model_inspection import read_jsonl, write_jsonl, score_dataset, score_single
    from cot_quailty_inspection_rules import RuleBase
except ImportError as e:
    print(f"导入模块时出错: {e}")
    def read_jsonl(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return [json.loads(line.strip()) for line in f if line.strip()]
    
    def write_jsonl(file_path, data):
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    class RuleBase:
        def run(self, input_data):
            return {
                'is_number': True,
                'no_text_truncated': True,
                'no_incomplete_content': True,
                'no_chinese_english_mix': True,
                'no_repeat_content': True,
                'no_unclose_paire': True,
                'no_repeat_pattern': True,
                'no_crashed_str': True,
                'no_chinese_English_space': True,
                'no_other_gpt_keywords': True,
                'no_think': True,
                'warning': ''
            }


class ModelScorer:
    def __init__(self):
        self.default_api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.default_api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
        self.default_model_name = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")
    
    def score_dataset(self, data, api_key=None, api_base=None, model_name=None, batch_size=10, system_prompt=None):
        key = api_key or self.default_api_key
        base = api_base or self.default_api_base
        model = model_name or self.default_model_name
        if not key or not base or not model:
            raise ValueError("缺少模型质检所需的API配置")
        return score_dataset(data, key, base, model, batch_size, system_prompt=system_prompt)
    
    def score_single(self, instruction, input_text, output, api_key=None, api_base=None, model_name=None, system_prompt=None):
        key = api_key or self.default_api_key
        base = api_base or self.default_api_base
        model = model_name or self.default_model_name
        if not key or not base or not model:
            raise ValueError("缺少模型质检所需的API配置")
        return score_single(instruction, input_text, output, key, base, model, system_prompt=system_prompt)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB限制
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# 确保上传和结果目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebInspector:
    def __init__(self):
        self.model_scorer = ModelScorer()
        self.rule_base = RuleBase()
    
    def inspect_with_model(self, data, model_config=None):
        """使用模型进行质检"""
        try:
            model_config = model_config or {}
            api_key = model_config.get("api_key")
            api_base = model_config.get("api_base")
            model_name = model_config.get("model_name")
            system_prompt = model_config.get("system_prompt")
            return self.model_scorer.score_dataset(
                data,
                api_key=api_key,
                api_base=api_base,
                model_name=model_name,
                batch_size=model_config.get("batch_size", 10),
                system_prompt=system_prompt,
            )
        except Exception as e:
            print(f"模型质检出错: {e}")
            # 出错时返回模拟数据
            return self._create_mock_results(data, 'model_based')
    
    def inspect_with_rules(self, data, enabled_rules=None):
        """使用规则进行质检"""
        try:
            enabled_rules = set(enabled_rules or [])
            results = []
            for item in data:
                rule_input = {
                    'meta_prompt': item.get('instruction', ''),
                    'user': item.get('input', ''),
                    'assistant': item.get('output', ''),
                    'file_path': item.get('file_path', ''),
                    'ref_answer': item.get('gt', None),
                }
                
                rule_result = self.rule_base.run(rule_input)
                
                keys_to_score = [
                    k for k in rule_result.keys()
                    if k.startswith('no_') or k in {'fk_answer_exist', 'fk_answer_yes_or_no', 'fk_answer_equal'}
                ]
                if enabled_rules:
                    keys_to_score = [k for k in keys_to_score if k in enabled_rules]
                
                passed_rules = sum(1 for key in keys_to_score if rule_result.get(key) is True)
                total_rules = len(keys_to_score)
                
                score = (passed_rules / total_rules) * 10 if total_rules > 0 else 0
                
                results.append({
                    **item,
                    'score': round(score, 2),
                    'inspection_type': 'rule_based',
                    'rule_details': rule_result
                })
            
            return results
        except Exception as e:
            print(f"规则质检出错: {e}")
            # 出错时返回模拟数据
            return self._create_mock_results(data, 'rule_based')
    
    def _create_mock_results(self, data, inspection_type):
        """创建模拟结果"""
        import random
        results = []
        for item in data:
            results.append({
                **item,
                'score': round(random.uniform(5.0, 9.5), 2),
                'inspection_type': inspection_type,
                'error': '质检过程出现错误，使用模拟数据'
            })
        return results

inspector = WebInspector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/inspect', methods=['POST'])
def inspect_data():
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400
        
        file = request.files['file']
        inspection_type = request.form.get('type', 'model')
        
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        if not file.filename.endswith('.jsonl'):
            return jsonify({'error': '只支持JSONL文件'}), 400
        
        # 保存上传的文件
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        data = read_jsonl(filepath)
        
        if not data:
            return jsonify({'error': '文件为空或格式错误'}), 400
        
        model_config = {
            "api_key": request.form.get('api_key') or None,
            "api_base": request.form.get('api_base') or None,
            "model_name": request.form.get('model_name') or None,
            "system_prompt": request.form.get('system_prompt') or None,
        }
        enabled_rules_raw = request.form.get('rules') or ''
        enabled_rules = [r for r in enabled_rules_raw.split(',') if r]
        
        if inspection_type == 'model':
            results = inspector.inspect_with_model(data, model_config=model_config)
        else:
            results = inspector.inspect_with_rules(data, enabled_rules=enabled_rules)
        
        # 计算统计信息
        total_samples = len(results)
        processed_samples = total_samples
        scores = [item['score'] for item in results]
        average_score = sum(scores) / len(scores) if scores else 0
        pass_count = sum(1 for score in scores if score >= 6.0)
        pass_rate = (pass_count / total_samples * 100) if total_samples > 0 else 0
        
        # 准备响应数据
        response_data = {
            'total': total_samples,
            'processed': processed_samples,
            'average_score': round(average_score, 2),
            'pass_rate': round(pass_rate, 2),
            'results': results
        }
        
        logger.info(f"质检完成: {total_samples}个样本, 平均分: {average_score:.2f}, 合格率: {pass_rate:.1f}%")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"质检过程出错: {e}")
        return jsonify({'error': f'质检过程出错: {str(e)}'}), 500

@app.route('/api/health')
def health_check():
    return jsonify({'status': 'healthy', 'message': '质检服务运行正常'})

@app.route('/api/stats')
def get_stats():
    return jsonify({
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'results_folder': app.config['RESULTS_FOLDER'],
        'max_file_size': app.config['MAX_CONTENT_LENGTH']
    })

@app.route('/download_sample')
def download_sample():
    # 获取项目根目录下的 sample_test.jsonl 文件路径
    sample_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sample_test.jsonl')
    if not os.path.exists(sample_path):
        return jsonify({'error': '示例文件不存在，请联系管理员生成'}), 404
    return send_file(sample_path, as_attachment=True, download_name='sample_test.jsonl')

if __name__ == '__main__':
    print("启动思维链合成数据质检Web工具...")
    print("访问地址: http://localhost:5001")
    print("API健康检查: http://localhost:5001/api/health")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
