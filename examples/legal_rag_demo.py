from dotenv import load_dotenv

from src.embeddings.bge import bge_m3
from src.llm.chat_chain import ChatExecutor
from src.vectorstores.milvus import search_for_mixed, create_milvus_client

load_dotenv()

def generate_answer(client, query):
    m3_resp = bge_m3([query])
    search_content = search_for_mixed(
        client,
        m3_resp,
        "my_collection",
        ['id', 'text', 'metadata'],
        vector_anns_field="vector",
        sparse_anns_field="sparse",
    )
    content = "\n".join([item['entity']['text'] for item in search_content[0]])
    res = (
        ChatExecutor.Builder()
        .system("你是一个专业的法律咨询师")
        .human(f"请根据用户的回答问题，参考我给你的内容，进行用户问题解答，用的问题是{query},参考的内容是{content}")
        .build("gpt-4o-mini")
    ).execute()
    return res

if __name__ == '__main__':
    print(generate_answer(create_milvus_client("test", "study"), "被撤销死亡宣告意味着什么"))


