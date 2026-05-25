from typing import List

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel
from src.llm.chat_chain import ChatExecutor

load_dotenv()

if __name__ == '__main__':
    class Star(BaseModel):
        name: str
        age: int
        gender: str

    class StarList(BaseModel):
        stars: List[Star]

    jop = JsonOutputParser(pydantic_object=StarList)
    executor = (
        ChatExecutor.Builder()
        .system("你是一个百事通")
        .human("请给我列出{num}个中国的{type}明星")
        .template_param({"num": 3, "type": "电影"})
        .parser(jop)
        .build("gpt-4o-mini")
    )
    res = executor.execute()
    print(res)


