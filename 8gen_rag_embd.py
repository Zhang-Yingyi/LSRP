import os
from sentence_transformers import SentenceTransformer
import srsly
import numpy as np
import faiss
from tqdm import tqdm



dir_path = "./result/rag_idx/"
out_dir = "./result/rag_idx/"
with_pivacy = True


rag_path = dir_path+'privacy.json'
data = srsly.read_json(rag_path)
queries = ['query_1']
passages = ["样例文档-1", "样例文档-2"]
instruction = "Represent this sentence for searching relevant passages:"

test_path = "./data/CoGentest.json"
test_data = srsly.read_json(test_path)

model = SentenceTransformer('BAAI/bge-large-en-v1.5')

def generate_embeddings(data):
    embeddings = []
    ids = []
    for item in tqdm(data):
        item_id = item['id']
        text = item['text']
        if text:
            embedding = model.encode([text])[0]
            embeddings.append(embedding)
            ids.append(item_id)
    return np.array(embeddings, dtype="float32"), ids

# 构建 FAISS 数据库
def build_rag_database(embeddings, ids, output_path="faiss_database"):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 创建 FAISS 索引
    dimension = embeddings.shape[1]  # 嵌入向量的维度
    index = faiss.IndexFlatL2(dimension)  # 使用 L2 距离（欧氏距离）
    index.add(embeddings)

    # 保存索引
    faiss.write_index(index, os.path.join(output_path, "index.faiss"))

    # 保存 ID 对应关系
    id_file = os.path.join(output_path, "ids.json")
    srsly.write_json(id_file, ids)

    print(f"FAISS Database saved at {output_path}")

# 查询接口
def query_rag_database(query_text, model, database_path="faiss_database", top_k=50):
    # 加载 FAISS 索引
    index = faiss.read_index(os.path.join(database_path, "index.faiss"))
    ids = srsly.read_json(os.path.join(database_path, "ids.json"))

    # 生成查询向量
    query_embedding = model.encode([query_text])[0].astype("float32")
    distances, indices = index.search(np.array([query_embedding]), top_k)
    
    # 返回 Top K 结果
    results = [(ids[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
    return results

def rag_topk(query_text,index,ids,top_k=50):
    query_embedding = model.encode([query_text])[0].astype("float32")
    distances, indices = index.search(np.array([query_embedding]), top_k)
    # print(indices)
    # results = [(ids[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
    results = []
    for i,idx in enumerate(indices[0]):
        item = data[ids[idx]]
        item['distance'] = float(distances[0][i])
        # print(item)
        results.append(item)
    # results = [data[ids[idx]] for idx in indices[0]]
    # results = [(ids[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
    return results

# 主函数
if __name__ == "__main__":
    # # 生成嵌入
    embeddings, ids = generate_embeddings(data)

    # 构建 RAG 数据库
    output_path = out_dir
    build_rag_database(embeddings, ids, output_path)
    index = faiss.read_index(os.path.join(dir_path, "index.faiss"))
    ids = srsly.read_json(os.path.join(dir_path, "ids.json"))
    # 查询示例
    for item in tqdm(test_data):
        if with_pivacy:
            query = item["additional_profile"] + item["conversations"][0]['value']
        else:
            query = item["conversations"][0]['value']
        results = rag_topk(query, index,ids)
        item['rag_result'] = results
    
    save_path = out_dir+'test_with_rag.json'
    srsly.write_json(save_path, test_data)
    print("###End###")