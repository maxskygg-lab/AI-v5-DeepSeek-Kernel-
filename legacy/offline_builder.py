import arxiv
import time
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
# 借用你 app.py 里的函数来获取引用量
from app import smart_fetch_citations 

def build_offline_database():
    print("1. 正在从 ArXiv 批量拉取文献 (模拟搭建本地新数据库)...")
    # 假设你的专业方向是 education robot，拉取 500 篇
    search = arxiv.Search(
        query="education robot",
        max_results=500,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    results = [{"obj": r, "citations": 0} for r in search.results()]
    
    print("2. 正在获取 Semantic Scholar 真实引用量...")
    # 这里记得填入你的 ss_key，如果没有就不传
    id2c = smart_fetch_citations(results, ss_key="") 
    
    print("3. 正在组装文档并进行语义计算 (Embedding)...")
    docs = []
    for item in results:
        res = item["obj"]
        citations = id2c.get(res.entry_id, 0)
        
        # 将标题和摘要合并，作为语义计算的文本实体
        page_content = f"Title: {res.title}\nAbstract: {res.summary}"
        
        # 将结构化数据存入 metadata，留给最后一步“排序”使用
        metadata = {
            "title": res.title,
            "authors": ", ".join([a.name for a in res.authors]),
            "year": res.published.year,
            "citations": citations,
            "arxiv_id": res.entry_id,
            "url": res.entry_id
        }
        docs.append(Document(page_content=page_content, metadata=metadata))
    
    print("4. 正在保存至本地 FAISS 向量数据库...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    
    # 关键：将算好的高维向量和元数据永久保存到硬盘！
    db.save_local("my_semantic_paper_db")
    print("✅ 建库成功！已在当前目录生成 'my_semantic_paper_db' 文件夹。")

if __name__ == "__main__":
    build_offline_database()
