
import json
import os
import logging
import requests
import time
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, List, Any

logger = logging.getLogger(__name__)

class InferenceEngine:
    """
    推理引擎服务：封装 DeepSeek R1 的批量推理逻辑
    """
    
    def __init__(self):
        self._stop_event = threading.Event()
        self._status = {
            "status": "idle", # idle, running, stopped, completed, error
            "total": 0,
            "processed": 0,
            "current_file": None,
            "output_file": None,
            "error": None,
            "latest_results": []
        }
        self._lock = threading.Lock()

    def get_status(self) -> Dict:
        """获取当前运行状态"""
        return self._status

    def stop(self):
        """停止任务"""
        self._stop_event.set()
        with self._lock:
            if self._status["status"] == "running":
                self._status["status"] = "stopped"

    def run(self, config: Dict):
        """
        运行推理任务
        config: {
            'api_key': str,
            'input_file': str,
            'output_file': str,
            'model': str,
            'workers': int
        }
        """
        self._stop_event.clear()
        
        with self._lock:
            self._status = {
                "status": "running",
                "total": 0,
                "processed": 0,
                "current_file": config.get('input_file'),
                "output_file": config.get('output_file'),
                "error": None,
                "latest_results": []
            }

        try:
            self._execute_inference(config)
            with self._lock:
                if self._status["status"] == "running":
                    self._status["status"] = "completed"
        except Exception as e:
            logger.error(f"推理任务异常: {e}")
            with self._lock:
                self._status["status"] = "error"
                self._status["error"] = str(e)

    def _execute_inference(self, config: Dict):
        input_file = config['input_file']
        output_file = config['output_file']
        api_key = config['api_key']
        workers = config.get('workers', 5)
        model = config.get('model', 'deepseek-reasoner')
        base_url = config.get('base_url', "https://api.deepseek.com/chat/completions")
        orig_prompt = config.get('original_prompt')
        opt_prompt = config.get('optimized_prompt')
        is_distillation = config.get('is_distillation', False)

        # 1. 加载数据
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"输入文件不存在: {input_file}")

        all_data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        all_data.append(json.loads(line))
                    except:
                        pass
        
        # 2. 断点续传
        prompt_suffix = f"{orig_prompt}{opt_prompt}" if orig_prompt else ("distillation" if is_distillation else "")
        processed_hashes = self._load_processed_hashes(output_file, prompt_suffix)
        tasks = []
        for item in all_data:
            if self._get_data_hash(item, prompt_suffix) not in processed_hashes:
                tasks.append(item)
        
        with self._lock:
            self._status["total"] = len(all_data)
            self._status["processed"] = len(all_data) - len(tasks)

        if not tasks:
            return

        # 3. 确保输出目录
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # 4. 多线程处理
        with ThreadPoolExecutor(max_workers=workers) as executor:
            if is_distillation:
                model_before = config.get('model_before')
                model_after = config.get('model_after')
                base_url_before = config.get('base_url_before')
                base_url_after = config.get('base_url_after')
                api_key_before = config.get('api_key_before', api_key)
                api_key_after = config.get('api_key_after', api_key)
                
                future_to_item = {
                    executor.submit(self._process_distillation_item, item, api_key_before, api_key_after, 
                                   model_before, model_after, base_url_before, base_url_after, opt_prompt): item 
                    for item in tasks
                }
            else:
                future_to_item = {
                    executor.submit(self._process_item, item, api_key, model, base_url, orig_prompt, opt_prompt): item 
                    for item in tasks
                }
            
            for future in as_completed(future_to_item):
                if self._stop_event.is_set():
                    break
                
                try:
                    result_item = future.result()
                    if result_item:
                        # 写入结果
                        with open(output_file, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(result_item, ensure_ascii=False) + '\n')
                        
                        with self._lock:
                            self._status["processed"] += 1
                            # Keep last 5 results for preview
                            self._status["latest_results"].append(result_item)
                            if len(self._status["latest_results"]) > 5:
                                self._status["latest_results"].pop(0)
                except Exception as e:
                    logger.error(f"Future error: {e}")

    def _process_distillation_item(self, item: Dict, api_key_before: str, api_key_after: str, 
                                  model_before: str, model_after: str, 
                                  base_url_before: str, base_url_after: str,
                                  optimized_prompt: Optional[str] = None) -> Optional[Dict]:
        try:
            input_text = item.get("input", "")
            
            # 如果有优化后的提示词模板，则使用它拼接特征数据
            if optimized_prompt:
                try:
                    instruction = optimized_prompt.format(data_report=input_text)
                    input_text = "" # 内容已经包含在 instruction 中了
                except:
                    instruction = f"{optimized_prompt}\n\n{input_text}"
                    input_text = ""
            else:
                instruction = item.get("instruction", "你是一个风控专家，请分析以下数据并给出结论。")
            
            # 并行调用两个模型
            res_before = self._call_llm(instruction, input_text, api_key_before, model_before, base_url_before)
            res_after = self._call_llm(instruction, input_text, api_key_after, model_after, base_url_after)

            item["output_before"] = self._format_output(res_before)
            item["output_after"] = self._format_output(res_after)
            item["output"] = item["output_after"]
            return item
        except Exception as e:
            logger.error(f"处理蒸馏对比数据失败: {e}")
            return None

    def _format_output(self, res: Dict) -> str:
        reasoning = res.get('reasoning', '').strip()
        content = res.get('content', '').strip()
        if reasoning:
            return f"<think>\n{reasoning}\n</think>\n\n{content}"
        return content

    def _process_item(self, item: Dict, api_key: str, model: str, base_url: str, 
                     original_prompt: Optional[str] = None, 
                     optimized_prompt: Optional[str] = None) -> Optional[Dict]:
        try:
            input_text = item.get("input", "")
            
            # 模式 1: 对比模式
            if original_prompt and optimized_prompt:
                # 构造指令
                def format_prompt(tpl, data):
                    try:
                        return tpl.format(data_report=data)
                    except:
                        return f"{tpl}\n\n{data}"

                inst_orig = format_prompt(original_prompt, input_text)
                inst_opt = format_prompt(optimized_prompt, input_text)

                # 并行调用或顺序调用（这里简单起见顺序调用，或者可以再开线程）
                res_orig = self._call_llm(inst_orig, "", api_key, model, base_url)
                res_opt = self._call_llm(inst_opt, "", api_key, model, base_url)

                item["output_original"] = f"<think>{res_orig['reasoning']}</think> <answer>{res_orig['content']}</answer>"
                item["output_optimized"] = f"<think>{res_opt['reasoning']}</think> <answer>{res_opt['content']}</answer>"
                item["output"] = item["output_optimized"] # 兼容旧版，默认显示优化的
                return item

            # 模式 2: 普通模式
            instruction = item.get("instruction", "")
            if not instruction and not input_text:
                return None

            result = self._call_llm(instruction, input_text, api_key, model, base_url)
            
            if result['content'] or result['reasoning']:
                item["output"] = f"<think>{result['reasoning']}</think> <answer>{result['content']}</answer>"
                return item
            return None
        except Exception as e:
            logger.error(f"处理单条数据失败: {e}")
            return None

    def _call_llm(self, instruction: str, user_input: str, api_key: str, model: str, base_url: str) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{instruction}\n\n{user_input}".strip()},
        ]
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": 0.6
        }
        
        reasoning_text = ""
        content_text = ""
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(base_url, headers=headers, json=payload, stream=True, timeout=120)
                if response.status_code != 200:
                    if response.status_code == 429:
                        time.sleep(2 * (attempt + 1))
                        continue
                    raise Exception(f"API Error: {response.text}")
                
                for line in response.iter_lines():
                    if not line: continue
                    decoded = line.decode('utf-8').strip()
                    if decoded.startswith("data:"):
                        data_str = decoded[5:].strip()
                        if data_str == "[DONE]": break
                        try:
                            data = json.loads(data_str)
                            delta = data['choices'][0]['delta']
                            if 'reasoning_content' in delta and delta['reasoning_content']:
                                reasoning_text += delta['reasoning_content']
                            if 'content' in delta and delta['content']:
                                content_text += delta['content']
                        except:
                            pass
                
                return {"reasoning": reasoning_text, "content": content_text}
                
            except Exception as e:
                logger.warning(f"API调用尝试 {attempt+1} 失败: {e}")
                time.sleep(2)
        
        return {"reasoning": "", "content": ""}

    def _get_data_hash(self, item: Dict, suffix: str = "") -> str:
        content = f"{item.get('instruction', '')}{item.get('input', '')}{suffix}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def _load_processed_hashes(self, output_file: str, suffix: str = "") -> set:
        processed = set()
        if not os.path.exists(output_file):
            return processed
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    if not line.strip(): continue
                    item = json.loads(line)
                    processed.add(self._get_data_hash(item, suffix))
                except:
                    continue
        return processed
