"""
This module initializes various language models.
"""
import os
from src.llm import OllamaLLM, ZhipuLLM, OpenAILLM, extract_answer_from_r1

llm_glm_4_plus = ZhipuLLM(
    api_key=os.getenv("ZHIPU_API_KEY"),
    model="glm-4-plus",
    default_stream=False,
)
llm_glm_4_air = ZhipuLLM(
    api_key=os.getenv("ZHIPU_API_KEY"),
    model="glm-4-air",
    default_stream=True,
)
llm_glm_4_flash = ZhipuLLM(
    api_key=os.getenv("ZHIPU_API_KEY"),
    model="glm-4-flash",
    default_stream=True,
)
llm_deepseek_r1 = OllamaLLM(
    host=os.getenv("OLLAMA_HOST"),
    model="deepseek-r1:14b",
    post_process=extract_answer_from_r1,
    default_stream=True,
)
llm_gpt_4o_mini = OpenAILLM(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",
    base_url=os.getenv("OPENAI_BASE_URL")
)
llm_gpt_4o = OpenAILLM(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o",
    base_url=os.getenv("OPENAI_BASE_URL")
)
llm_deepseek_v3 = OpenAILLM(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="deepseek/deepseek-chat",
    base_url=os.getenv("OPENAI_BASE_URL"),
    default_stream=True,
)
