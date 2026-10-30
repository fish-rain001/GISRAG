"""
my_agentic_rag - Agentic RAG Library

Public API:
- configure(data_dir, collection_name): 一键配置数据目录和 Milvus 集合名
- get_embedding_service(): Get embedding service (BGE + BM25)
- build_rag_graph(): Build RAG processing graph
- reset_vectorstore(): Reset vectorstore singleton (for switching data sources)
"""

from .embedding import get_embedding_service
from .rag_graph import build_rag_graph
from .vectorstore import set_data_dir, reset_vectorstore


def configure(data_dir: str, collection_name: str = None):
    """
    调用方在使用 RAG 之前调用此函数，配置数据目录和数据库集合名。

    示例:
        from my_agentic_rag import configure, build_rag_graph
        from my_agentic_rag.vectorstore import get_vectorstore

        configure(data_dir="/path/to/my/docs", collection_name="my_project_docs")
        vs = get_vectorstore()
        vs.initialize()  # 首次入库，后续主项目自行判断是否跳过

    每个项目使用不同的 collection_name，数据互不干扰。
    """
    from . import config
    from .milvus_client import get_milvus_client

    set_data_dir(data_dir)
    if collection_name:
        config.MILVUS_COLLECTION = collection_name
        reset_vectorstore()
        # 重置 milvus 客户端，让它用新的集合名重新连接
        import my_agentic_rag.milvus_client as mc
        mc._milvus_client = None


__all__ = [
    "configure",
    "get_embedding_service",
    "build_rag_graph",
    "reset_vectorstore",
]
