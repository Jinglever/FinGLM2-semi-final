"""
This module provides implementations of agent
and their interactions with various APIs.
"""

import os
import copy
from dataclasses import dataclass, field
from typing import Optional, Callable, Tuple, List, Dict
from src.llm import LLM, DEBUG_OPTION_PRINT_TOOL_CALL_RESULT
from src.log import get_logger

@dataclass
class AgentConfig:
    """Configuration settings for the Agent class."""
    llm: LLM
    name: str
    role: str
    constraint: Optional[str] = None
    output_format: Optional[str] = None
    knowledge: Optional[str] = None
    tools: Optional[List[Dict]] = None
    funcs: Optional[List[Callable]] = None
    retry_limit: int = 5
    enable_history: bool = True
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: Optional[bool] = None
    debug_tool_call_result: bool = True
    system_prompt_kv: Optional[Dict] = field(default_factory=dict)
    pre_process: Optional[Callable[['Agent', dict], None]] = None
    post_process: Optional[Callable[[str], str]] = None
    max_history_num: int = 30

    def deepcopy(self):
        """Custom deepcopy method to handle non-deepcopyable attributes."""
        new_config = copy.copy(self)  # Start with a shallow copy
        new_config.llm = self.llm.clone()
        new_config.system_prompt_kv = copy.deepcopy(self.system_prompt_kv)
        return new_config

class Agent():
    """Represents an agent that interacts with various APIs using a language model."""

    def __init__(self, config: AgentConfig):
        self.cfg = config.deepcopy()
        if self.cfg.system_prompt_kv is None:
            self.cfg.system_prompt_kv = {}
        self.cfg.pre_process = config.pre_process
        self.cfg.post_process = config.post_process
        self.history = []
        self.usage_tokens = 0 # 总共使用的token数量
        self.options = {}
        if config.temperature is not None:
            self.options["temperature"] = config.temperature
        if config.top_p is not None:
            self.options["top_p"] = config.top_p
        if config.funcs is not None:
            self.funcs = {func.__name__: func for func in config.funcs}
        else:
            self.funcs = None

    def clone(self) -> 'Agent':
        """Creates a clone of the current agent instance."""
        return Agent(config=self.cfg)

    def clear_history(self):
        """Clears the agent's conversation history and resets token counts."""
        self.history = []
        self.usage_tokens = 0

    def add_system_prompt_kv(self, kv: dict):
        """Sets the system prompt key-value pairs for the agent."""
        for k, v in kv.items():
            self.cfg.system_prompt_kv[k] = v

    def del_system_prompt_kv(self, key: str):
        """Deletes the specified key from the system prompt key-value pairs for the agent."""
        if key in self.cfg.system_prompt_kv:
            del self.cfg.system_prompt_kv[key]

    def clear_system_prompt_kv(self):
        """
        Clear the agent's additional system prompt settings
        """
        self.cfg.system_prompt_kv = {}

    def get_system_prompt(self):
        """Generates and returns the system prompt based on the agent's attributes."""
        system_prompt = f"## 角色描述\n{self.cfg.role}"
        if self.cfg.constraint is not None:
            system_prompt += f"\n\n## 约束要求\n{self.cfg.constraint}"
        if self.cfg.output_format is not None:
            system_prompt += f"\n\n## 输出格式\n{self.cfg.output_format}"
        if self.cfg.knowledge is not None:
            system_prompt += f"\n\n## 知识库\n{self.cfg.knowledge}"
        for key, value in self.cfg.system_prompt_kv.items():
            system_prompt += f"\n\n## {key}\n{value}"
        return system_prompt

    def chat(self, messages: list[dict]) -> Tuple[str, int]:
        """Attempts to generate a response from the language model, retrying if necessary.
        return:
            - str: assistant's answer
            - int: usage_tokens
        """
        debug_mode = os.getenv("DEBUG", "0") == "1"
        show_llm_input_msg = os.getenv("SHOW_LLM_INPUT_MSG", "0") == "1"
        logger = get_logger()

        if self.cfg.pre_process is not None:
            self.cfg.pre_process(self, messages)
        usage_tokens = 0
        is_exception_from_llm = False
        for attempt in range(self.cfg.retry_limit):
            if attempt > 0:
                if debug_mode:
                    print(f"\n重试第 {attempt} 次...\n")
                logger.info("\n重试第 %d 次...\n", attempt)
            response = ""
            try:
                msgs = (
                    messages if attempt == 0
                    else messages + [
                        {"role": "assistant", "content": response},
                        {"role": "user", "content": "请修正后重试"}
                    ] if not is_exception_from_llm else messages
                )
                if show_llm_input_msg:
                    if debug_mode:
                        print("\n\n>>>>> 【" + msgs[-1]["role"] + "】 Said:\n" + msgs[-1]["content"])
                    logger.debug("\n\n>>>>> 【%s】 Said:\n%s", msgs[-1]["role"], msgs[-1]["content"])
                if debug_mode:
                    print(f"\n\n>>>>> Agent【{self.cfg.name}】 Said:")
                logger.debug("\n\n>>>>> Agent【%s】 Said:\n", self.cfg.name)
                is_exception_from_llm = True
                response, token_count, ok = self.cfg.llm.generate_response(
                    system=self.get_system_prompt(),
                    messages=msgs,
                    tools=self.cfg.tools,
                    funcs=self.funcs,
                    options=self.options,
                    stream=self.cfg.stream,
                    debug_options={DEBUG_OPTION_PRINT_TOOL_CALL_RESULT: self.cfg.debug_tool_call_result},
                )
                is_exception_from_llm = False
                usage_tokens += token_count
                self.usage_tokens += token_count
                if ok and self.cfg.post_process is not None:
                    response = self.cfg.post_process(response)
            except Exception as e:
                print(f"\nAgent【{self.cfg.name}】chat发生异常：{str(e)}")
                logger.debug("\nAgent【%s】chat发生异常：%s", self.cfg.name, str(e))
                ok = False
                response += f"\n发生异常：{str(e)}"
            if ok:  # 如果生成成功，退出重试
                break
        else:
            response, token_count = f"发生异常：{response}", 0  # 如果所有尝试都失败，返回默认值
            return response, token_count

        if self.cfg.enable_history:
            self.history = messages + [{"role": "assistant", "content": response}]
            if len(self.history) > self.cfg.max_history_num:
                half = len(self.history) // 2 + 1
                # 浓缩一半的history
                if debug_mode:
                    print(f"\n\n>>>>> Agent【{self.cfg.name}】 Compress History:")
                logger.debug("\n\n>>>>> Agent【%s】 Compress History:\n", self.cfg.name)
                try:
                    compressed_msg, token_count, ok = self.cfg.llm.generate_response(
                        system="请你把所有历史对话浓缩成一段话，必须保留重要的信息，不要换行，不要有任何markdown格式",
                        messages=self.history[:half],
                        stream=self.cfg.stream,
                    )
                    usage_tokens += token_count
                    self.usage_tokens += token_count
                    if ok:
                        self.history = [{"role": "assistant", "content": compressed_msg}] +\
                            self.history[half:]
                except Exception as e:
                    print(f"\nAgent【{self.cfg.name}】压缩history发生异常：{str(e)}")
                    logger.debug("\nAgent【%s】压缩history发生异常：%s", self.cfg.name, str(e))
        return response, usage_tokens


    def answer(self, message: str) -> Tuple[str, int]:
        """Generates a response to a user's message using the agent's history.
        return:
            - str: assistant's answer
            - int: usage_tokens
        """
        messages = self.history + [{"role": "user", "content": message}]
        return self.chat(messages = messages)
