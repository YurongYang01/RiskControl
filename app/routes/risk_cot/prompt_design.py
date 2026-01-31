from flask import Blueprint, render_template, request, jsonify, current_app
import os
from app.services.risk_cot.prompt_manager import PromptManager
from app.services.risk_cot.prompt_optimizer import PromptOptimizer
import datetime
import uuid

prompt_design_bp = Blueprint('prompt_design', __name__, url_prefix='/api/prompt_design')
manager = PromptManager()

@prompt_design_bp.route('/templates', methods=['GET'])
def get_templates():
    return jsonify({
        'status': 'success',
        'templates': manager.get_templates()
    })

@prompt_design_bp.route('/rules', methods=['GET', 'POST'])
def handle_rules():
    if request.method == 'POST':
        rule = request.json
        manager.add_rule(rule)
        return jsonify({'status': 'success', 'message': '规则已添加'})
    return jsonify({
        'status': 'success',
        'rules': manager.get_rules()
    })

@prompt_design_bp.route('/rules/<rule_id>', methods=['DELETE'])
def delete_rule(rule_id):
    manager.delete_rule(rule_id)
    return jsonify({'status': 'success', 'message': '规则已删除'})

@prompt_design_bp.route('/opinions', methods=['GET', 'POST'])
def handle_opinions():
    if request.method == 'POST':
        opinion = request.json
        manager.add_opinion(opinion)
        return jsonify({'status': 'success', 'message': '意见已添加'})
    return jsonify({
        'status': 'success',
        'opinions': manager.get_opinions()
    })

@prompt_design_bp.route('/optimize', methods=['POST'])
def optimize_prompt():
    data = request.json
    original_prompt = data.get('original_prompt')
    selected_rules = data.get('rules', [])
    selected_opinions = data.get('opinions', [])
    custom_feedback = data.get('custom_feedback', "")
    
    api_key = data.get('api_key') or current_app.config.get('OPENAI_API_KEY')
    base_url = data.get('base_url') or current_app.config.get('OPENAI_BASE_URL')
    model = data.get('model', 'gpt-4')

    optimizer = PromptOptimizer(api_key=api_key, base_url=base_url, model=model)
    optimized_content = optimizer.optimize(
        original_prompt=original_prompt,
        rules=selected_rules,
        opinions=selected_opinions,
        custom_feedback=custom_feedback
    )

    return jsonify({
        'status': 'success',
        'optimized_prompt': optimized_content
    })

@prompt_design_bp.route('/save', methods=['POST'])
def save_optimized_prompt():
    data = request.json
    original_id = data.get('original_id')
    original_content = data.get('original_content')
    optimized_content = data.get('content')
    metadata = {
        'rules': data.get('rules', []),
        'opinions': data.get('opinions', []),
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    version = manager.save_version(original_id, original_content, optimized_content, metadata)
    return jsonify({
        'status': 'success',
        'message': '模板已保存',
        'version': version
    })

@prompt_design_bp.route('/latest', methods=['GET'])
def get_latest_design():
    latest = manager.get_latest_history()
    if not latest:
        return jsonify({'status': 'error', 'message': 'No history found'}), 404
    return jsonify({
        'status': 'success',
        'latest': latest
    })

@prompt_design_bp.route('/feature_maps', methods=['GET', 'POST'])
def handle_feature_maps():
    if request.method == 'POST':
        feature_maps = request.json.get('feature_maps', [])
        manager.update_feature_maps(feature_maps)
        return jsonify({'status': 'success', 'message': '特征映射已更新'})
    return jsonify({
        'status': 'success',
        'feature_maps': manager.get_feature_maps()
    })

@prompt_design_bp.route('/feature_maps/system', methods=['GET'])
def get_system_feature_maps():
    try:
        project_root = current_app.config['PROJECT_ROOT']
        feat_file = os.path.join(project_root, 'config', '全部特征.txt')
        if not os.path.exists(feat_file):
            return jsonify({'status': 'error', 'message': '系统特征文件不存在'}), 404
        
        content = ""
        with open(feat_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if len(lines) > 1: # Skip header
                for line in lines[1:]:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        content += f"{parts[0]},{parts[1]}\n"
        
        return jsonify({
            'status': 'success',
            'content': content
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
