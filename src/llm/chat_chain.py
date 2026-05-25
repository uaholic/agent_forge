from typing import Dict, Any

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate

class BaseModelExecutor:
    def __init__(self, model: str, output_parser: BaseOutputParser, messages, params: Dict[str, Any] | None = None):
        self.llm = init_chat_model(
            model=model,
            model_provider="openai"
        )
        self.output_parser = output_parser
        self.cpt = ChatPromptTemplate.from_messages(messages)
        self.params = params or {}

    def execute(self):
        chain = self.cpt | self.llm | self.output_parser
        return chain.invoke(self.params)


class ChatExecutor(BaseModelExecutor):
    def __init__(self, model: str, output_parser: BaseOutputParser, messages, params: Dict[str, Any] | None = None):
        super().__init__(model, output_parser, messages, params)

    class Builder:
        PARSER_TEMPLATE_PARAM_STR = "format_instructions"

        def __init__(self):
            self.messages = []
            self.params = {}
            self.output_parser: BaseOutputParser = None

        def system(self, content):
            self.messages.append({
                "role": "system",
                "content": content
            })
            return self

        def human(self, content):
            self.messages.append({
                "role": "human",
                "content": content
            })
            return self

        def template_param(self, params: Dict[str, str]):
            if self.params:
                raise RuntimeError(f"当前已存在模板参数：{self.params}请勿重复设置")
            self.params = params
            return self

        def parser(self, output_parser: BaseOutputParser):
            if self.output_parser:
                raise RuntimeError(f"当前已设置格式化输出：{self.output_parser}请勿重复设置")
            self.output_parser = output_parser
            return self

        def build(self, model: str):
            if self.output_parser:
                self.system(f"{{{self.PARSER_TEMPLATE_PARAM_STR}}}")
                self.params.update({self.PARSER_TEMPLATE_PARAM_STR: self.output_parser.get_format_instructions()})
            else:
                self.output_parser = StrOutputParser()
            return ChatExecutor(model, self.output_parser, self.messages, self.params)
