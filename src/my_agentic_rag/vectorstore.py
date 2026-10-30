"""
============================================================
代码故事（从程序启动到回答用户问题的完整过程）
============================================================
程序启动后，当需要检索文档时...
【vectorstore.py】的【retrieve()】被调用...
它连接 Milvus，使用稠密+稀疏混合检索...

最终返回最相关的文档列表。
============================================================
"""

"""
============================================================
面试版：向量存储抽象层（纯 Milvus 模式）
============================================================
Milvus 模式：稠密+稀疏混合检索 + Rerank + Auto-merging
============================================================
"""

from typing import List, Dict, Any, Optional

from pymilvus import connections

from . import config
from .rag_utils import retrieve_documents
from .document_loader import get_document_loader


class VectorStore:
    """
    向量存储管理器（纯 Milvus 模式）
    """

    def __init__(self):
        self._initialized = False

    def initialize(self, data_dir=None) -> None:
        """初始化向量数据库"""
        if data_dir is None:
            global _data_dir_override
            data_dir = _data_dir_override if _data_dir_override else config.DATA_DIR

        if data_dir is None:
            raise RuntimeError(
                "未设置数据目录。请在使用前调用 set_data_dir('/path/to/your/data') "
                "注入项目专属的数据目录。"
            )

        # 连接 Milvus
        connections.connect(
            host=config.MILVUS_HOST,
            port=config.MILVUS_PORT,
            alias="default",
            timeout=10,
            user=config.MILVUS_USER if config.MILVUS_USER else None,
            password=config.MILVUS_PASSWORD if config.MILVUS_PASSWORD else None,
            secure=True,
        )
        print("Milvus 连接成功")

        # 入库
        loader = get_document_loader()
        chunks = loader.load_and_chunk_all(data_dir)
        loader.persist_chunks(chunks)
        print(f"Milvus 向量数据库初始化完成，共 {len(chunks)} 个块")
        self._initialized = True

    def load_and_index(self, data_dir=None) -> None:
        """初始化（兼容旧接口）"""
        self.initialize(data_dir)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """检索最相关的文档（混合检索 + Rerank + Auto-merging）"""
        if not self._initialized:
            self.initialize()

        result = retrieve_documents(query, top_k)
        docs = result["docs"]

        formatted = []
        for doc in docs:
            formatted.append({
                "content": doc.get("text", ""),
                "metadata": {
                    "source": doc.get("source", "unknown"),
                    "chunk_level": doc.get("chunk_level", 0),
                    "chunk_id": doc.get("chunk_id", ""),
                    "parent_chunk_id": doc.get("parent_chunk_id", ""),
                },
                "score": doc.get("score", 0.0)
            })

        return formatted


# 全局单例
_vectorstore: Optional[VectorStore] = None
_data_dir_override: Optional[str] = None


def set_data_dir(path: str) -> None:
    """
    由调用方（主项目）注入数据目录。

    同时更新：
    - BM25 状态文件路径 → {data_dir}/bm25_state.json
    - 重置 embedding 服务，下次调用时自动加载新路径的状态文件

    每个项目的 BM25 统计独立存储，互不干扰。
    """
    global _data_dir_override
    _data_dir_override = path
    # BM25 状态文件跟着数据目录走
    from pathlib import Path
    from . import config
    config.BM25_STATE_PATH = Path(path) / "bm25_state.json"
    # 重置 embedding 服务，让它从新路径加载
    from .embedding import _embedding_service
    _embedding_service = None


def reset_vectorstore() -> None:
    """重置 vectorstore 单例（切换数据源时使用）"""
    global _vectorstore
    _vectorstore = None


def get_vectorstore() -> VectorStore:
    """获取 VectorStore 单例"""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = VectorStore()
    return _vectorstore
