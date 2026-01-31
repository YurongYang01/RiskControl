from flask import Blueprint, request, jsonify, Response, current_app
from app.services.risk_cot.inference_engine import InferenceEngine
from app.services.risk_cot.prompt_manager import PromptManager
import threading
import os
import json
import time
from werkzeug.utils import secure_filename

distillation_bp = Blueprint('distillation', __name__, url_prefix='/api/distillation')

# 我们复用 InferenceEngine 的逻辑，但在 distillation 模块中进行特定的配置
distillation_engine = InferenceEngine()
prompt_manager = PromptManager()

@distillation_bp.route('/run', methods=['POST'])
def run_distillation():
    try:
        data = request.json
        # 默认使用 alpaca_mock_risk_data_ae39e2eb.jsonl
        input_file = data.get('input_file', 'alpaca_mock_risk_data_ae39e2eb.jsonl')
        
        # 确保输入文件路径正确
        if not os.path.isabs(input_file):
            input_file = os.path.join(current_app.config['DATA_FOLDER'], input_file)
            
        if not os.path.exists(input_file):
            return jsonify({'status': 'error', 'message': f'Input file not found: {input_file}'}), 404
            
        # 获取最新优化后的提示词
        latest_design = prompt_manager.get_latest_history()
        optimized_prompt = latest_design.get('optimized_content') if latest_design else None
        
        if not optimized_prompt:
            # 如果没有优化后的提示词，尝试使用默认模板
            templates = prompt_manager.get_templates()
            if templates:
                optimized_prompt = templates[0].get('content')
            else:
                return jsonify({'status': 'error', 'message': 'No optimized prompt or template found'}), 400
            
        # 生成输出文件名
        base_name = os.path.basename(input_file).replace('.jsonl', '')
        timestamp = int(time.time())
        output_file = os.path.join(current_app.config['RESULTS_FOLDER'], f"distillation_{base_name}_{timestamp}.jsonl")
        
        # 核心配置项后端写死
        DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'sk-c26320ac5d7147c2af0f1b18885578e4')
        
        config = {
            'input_file': input_file,
            'output_file': output_file,
            'api_key_before': DEEPSEEK_API_KEY,
            'api_key_after': DEEPSEEK_API_KEY,
            'model_before': 'deepseek-reasoner',
            'model_after': 'deepseek-reasoner',
            'workers': 3,
            'base_url_before': "https://api.deepseek.com/chat/completions",
            'base_url_after': "https://api.deepseek.com/chat/completions",
            'is_distillation': True,
            'optimized_prompt': optimized_prompt # 传递优化后的提示词模板
        }
        
        if distillation_engine.get_status()['status'] == 'running':
             return jsonify({'status': 'error', 'message': 'Task is already running'}), 409
        
        # 启动后台线程
        thread = threading.Thread(target=distillation_engine.run, args=(config,))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'success', 
            'message': 'Distillation comparison task started',
            'output_file': output_file
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@distillation_bp.route('/status')
def stream_status():
    def generate():
        while True:
            status = distillation_engine.get_status()
            yield f"data: {json.dumps(status)}\n\n"
            if status['status'] in ['completed', 'stopped', 'error']:
                break
            time.sleep(1)
    return Response(generate(), mimetype='text/event-stream')

@distillation_bp.route('/stop', methods=['POST'])
def stop_distillation():
    distillation_engine.stop()
    return jsonify({'status': 'success', 'message': 'Stop signal sent'})
