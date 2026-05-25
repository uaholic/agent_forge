from pymilvus import MilvusClient
from pymilvus import AnnSearchRequest, RRFRanker  # 排序器，就是一个排算法

def create_milvus_client(db_name, token, host="127.0.0.1", port=19530):
    client = MilvusClient(
        uri=f"http://{host}:{port}",
        db_name=db_name,
        token=token,
    )
    return client

def base_search(client, embed_query, collection_name, output_fields, anns_field, limit=10):
    return client.search(
        collection_name=collection_name,
        data=embed_query,  # 用户问的问题进行向量化之后的稠密向量
        limit=limit,  # 限制返回多少条数据
        output_fields=output_fields,  # 返回的数据字段有哪些 ，一般都是标量字段
        anns_field=anns_field,  # 拿着用户的稠密向量去表里面，跟哪个字段进行比对
    )

def search_for_vector(client, m3_resp, collection_name, output_fields, anns_field, limit=10):
    return base_search(client, m3_resp['dense_vecs'], collection_name, output_fields, anns_field, limit)

def search_for_sparse(client, m3_resp, collection_name, output_fields, anns_field, limit=10):
    return base_search(client, m3_resp['lexical_weights'], collection_name, output_fields, anns_field, limit)

def search_for_mixed(client, m3_resp, collection_name, output_fields, vector_anns_field, sparse_anns_field, limit=10):
    vector_req = AnnSearchRequest(
        data=m3_resp['dense_vecs'],
        anns_field=vector_anns_field,
        limit=limit,
        param={
            "metric_type": "L2",
        }
    )

    sparse_req = AnnSearchRequest(
        data=m3_resp['lexical_weights'],
        anns_field=sparse_anns_field,
        limit=limit,
        param={
            "metric_type": "IP",
        }
    )

    res = client.hybrid_search(
        collection_name=collection_name,
        reqs=[vector_req, sparse_req],
        ranker=RRFRanker(),
        limit=limit,
        output_fields=output_fields,
    )
    return res
