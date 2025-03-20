"""
This module provides implementations of embedding models
and their interactions with various APIs.
"""
import os
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import json
import copy
import textwrap
import requests
from zhipuai import ZhipuAI
from ollama import Client
from tqdm import tqdm  # 添加tqdm库导入
from src.utils import show

class Embedding(ABC):
    """Abstract base class for embedding."""

    @abstractmethod
    def create(self,
               inputs: str | List[str],
               ) -> Tuple[List[float], int, bool]:
        """Create an embedding for the given text."""



class ZhipuEmbedding(Embedding):
    """ZhipuAI embedding."""
    def __init__(self, api_key: str, model: str,
                 dimensions: Optional[int] = None):
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.client = ZhipuAI(api_key=api_key)

    def create(self,
               inputs: str | List[str],
               show_progress: bool = True,
               ) -> Tuple[List[float], int]:
        """Create an embedding for the given text."""
        inputs = copy.deepcopy(inputs)
        max_batch_size = 64
        max_batch_tokens = 8000
        max_single_tokens = 3072 
        if isinstance(inputs, list):
            embeddings = []
            total_tokens = 0
            i = 0
            
            # 使用tqdm创建进度条
            pbar = tqdm(total=len(inputs), disable=not show_progress, desc="创建嵌入向量")
            
            while i < len(inputs):
                batch_inputs = []
                batch_token_count = 0
                each_token_count = []
                # 收集输入直到达到最大批量大小或最大 token 数
                while i < len(inputs) and len(batch_inputs) < max_batch_size:
                    token_count = self.tokenizer_count(inputs[i])
                    if token_count > max_single_tokens:
                        inputs[i] = textwrap.shorten(inputs[i], width=len(inputs[i])*max_single_tokens//token_count, placeholder='')
                        token_count = max_single_tokens
                    if batch_token_count + token_count > max_batch_tokens:
                        break
                    batch_inputs.append(inputs[i])
                    batch_token_count += token_count
                    each_token_count.append(token_count)
                    i += 1

                try:
                    if self.dimensions is not None:
                        rsp = self.client.embeddings.create(
                            model=self.model,
                            input=batch_inputs,
                            dimensions=self.dimensions,
                            timeout=60,
                        )
                    else:
                        rsp = self.client.embeddings.create(
                            model=self.model,
                            input=batch_inputs,
                            timeout=60,
                        )
                    embeddings.extend([data.embedding for data in rsp.data])
                    total_tokens += rsp.usage.total_tokens
                    
                    # 更新进度条
                    pbar.update(len(batch_inputs))
                    
                except Exception as e:
                    print(f"\nZhipuEmbedding【{self.model}】发生异常：{str(e)}")
                    raise RuntimeError(f"Error creating embeddings: {e}") from e
            
            # 关闭进度条
            pbar.close()
            return embeddings, total_tokens
        try:
            token_count = self.tokenizer_count(inputs)
            if token_count > max_single_tokens:
                inputs = textwrap.shorten(inputs, width=len(inputs)*max_single_tokens//token_count, placeholder='')
                token_count = max_single_tokens
            if self.dimensions is not None:
                rsp = self.client.embeddings.create(
                    model=self.model,
                    input=inputs,
                    dimensions=self.dimensions,
                    timeout=60,
                )
            else:
                rsp = self.client.embeddings.create(
                    model=self.model,
                    input=inputs,
                    timeout=60,
                )
            return [data.embedding for data in rsp.data], rsp.usage.total_tokens
        except Exception as e:
            print(f"\nZhipuEmbedding【{self.model}】发生异常：{str(e)}")
            raise RuntimeError(f"Error creating embeddings: {e}") from e

    def tokenizer_count(self, text) -> int:
        """
        Calculate the number of tokens in the given text using the tokenizer API.

        Parameters:
        text (str): The input text to be tokenized.

        Returns:
        int: The number of tokens in the input text.
        """
        if os.getenv('ENABLE_TOKENIZER_COUNT', '0') == '0':
            return 0 # just for run on line
        # API endpoint and headers
        url = "https://open.bigmodel.cn/api/paas/v4/tokenizer"
        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json"
        }
        # Request payload
        payload = {
            "model": "glm-4-flash",  # 使用当前实例的模型
            "messages": [
                {
                    "role": "user",
                    "content": text
                }
            ]
        }
        
        # 添加重试逻辑
        max_retries = 3
        retry_delay = 2  # 重试延迟时间（秒）
        
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
                # 检查响应是否成功
                if response.status_code == 200:
                    data = response.json()
                    return data["usage"]["prompt_tokens"] * 100 // 87
                else:
                    # 如果这是最后一次尝试，则抛出错误
                    if attempt == max_retries:
                        response.raise_for_status()
                    # 否则继续重试
                    show(f"请求返回状态码 {response.status_code}，正在进行第 {attempt+1}/{max_retries} 次重试...")
                    
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                # 如果这是最后一次尝试，则抛出错误
                if attempt == max_retries:
                    raise RuntimeError(f"连接超时，已重试 {max_retries} 次: {e}") from e
                
                # 打印重试信息
                show(f"请求超时，正在进行第 {attempt+1}/{max_retries} 次重试...")
                
            # 在重试前等待一段时间（可以使用指数退避策略）
            if attempt < max_retries:
                import time
                time.sleep(retry_delay * (2 ** attempt))  # 指数退避
        
        return None  # 所有重试都失败时

class OllamaEmbedding(Embedding):
    """Ollama embedding."""
    def __init__(self, host: str, model: str):
        self.host = host
        self.model = model
        self.client = Client(host)

    def create(self,
               inputs: str | List[str],
               show_progress: bool = True,
               ) -> Tuple[List[List[float]], int]:
        """Create an embedding for the given text using Ollama."""
        inputs = copy.deepcopy(inputs)
        max_batch_size = 64
        
        if isinstance(inputs, list):
            embeddings = []
            total_tokens = 0
            i = 0
            
            # 使用tqdm创建进度条
            pbar = tqdm(total=len(inputs), disable=not show_progress, desc="创建嵌入向量")
            
            while i < len(inputs):
                # 每次最多处理64个输入
                batch_inputs = inputs[i:i + max_batch_size]
                try:
                    rsp = self.client.embed(
                        model=self.model,
                        input=batch_inputs
                    )
                    embeddings.extend(rsp.embeddings)
                    total_tokens += rsp.prompt_eval_count
                    
                    # 更新进度条
                    pbar.update(len(batch_inputs))
                    
                except Exception as e:
                    print(f"\nOllamaEmbedding【{self.model}】发生异常：{str(e)}")
                    raise RuntimeError(f"Error creating embeddings: {e}") from e
                i += max_batch_size
            
            # 关闭进度条
            pbar.close()
            return embeddings, total_tokens
            
        # 单个字符串输入的情况
        try:
            rsp = self.client.embed(
                model=self.model,
                input=inputs
            )
            return rsp.embeddings, rsp.prompt_eval_count
        except Exception as e:
            print(f"\nOllamaEmbedding【{self.model}】发生异常：{str(e)}")
            raise RuntimeError(f"Error creating embeddings: {e}") from e
