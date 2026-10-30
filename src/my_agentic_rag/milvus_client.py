"""
============================================================
代码故事（从程序启动到回答用户问题的完整过程）
============================================================
当用户提问时...
【milvus_client.py】的【hybrid_retrieve()】被调用...
它接收两个向量：
1. dense_embedding（稠密向量）- 用于语义相似性检索
2. sparse_embedding（稀疏向量）- 用于关键词匹配检索

它会：
1. 分别在 Milvus 中检索稠密向量和稀疏向量
2. 使用 RRF（倒数排名融合）算法合并两个结果
3. 返回融合后的 top_k 结果
============================================================
"""

"""
============================================================
面试版：Milvus 混合检索
============================================================
Milvus 是一个生产级向量数据库...
本模块实现稠密+稀疏的混合检索...
核心是 RRF（Reciprocal Rank Fusion）算法...
将不同检索方法的结果按排名融合...
============================================================
"""

from typing import List, Dict, Any, Optional
import numpy as np

from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    AnnSearchRequest,
    RRFRanker,
)

from . import config
from .embedding import get_embedding_service


class MilvusClient:
    """
    Milvus 向量数据库客户端

    支持：
    - 创建集合（稠密向量 + 稀疏向量 + 层级字段）
    - 插入文档
    - 混合检索（稠密 + 稀疏 RRF 融合）
    """

    def __init__(self):
        self._collection: Optional[Collection] = None
        self._embedding_service = get_embedding_service()
        self._connect()

    def _connect(self) -> None:
        """连接到 Milvus（支持本地或云端）"""
        if config.MILVUS_USER and config.MILVUS_PASSWORD:
            # 云端连接（Zilliz Cloud）
            connections.connect(
                host=config.MILVUS_HOST,
                port=config.MILVUS_PORT,
                user=config.MILVUS_USER,
                password=config.MILVUS_PASSWORD,
                alias="default",
                secure=True
            )
        else:
            # 本地连接
            connections.connect(
                host=config.MILVUS_HOST,
                port=config.MILVUS_PORT,
                alias="default"
            )

    def _create_collection(self) -> Collection:
        """创建集合"""
        # 字段定义
        fields = [
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=512, is_primary=True),
            FieldSchema(name="parent_chunk_id", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="chunk_level", dtype=DataType.INT64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
            # 稠密向量字段
            FieldSchema(name="dense_embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
            # 稀疏向量字段
            FieldSchema(name="sparse_embedding", dtype=DataType.SPARSE_FLOAT_VECTOR),
        ]

        schema = CollectionSchema(fields=fields, description="Agentic RAG Documents")

        collection = Collection(name=config.MILVUS_COLLECTION, schema=schema)

        # 创建索引
        # 稠密向量索引 (HNSW)
        dense_index_params = {
            "metric_type": "IP",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 256}
        }
        collection.create_index(
            field_name="dense_embedding",
            index_params=dense_index_params
        )

        # 稀疏向量索引 (SPARSE_INVERTED_INDEX)
        sparse_index_params = {
            "metric_type": "IP",
            "index_type": "SPARSE_INVERTED_INDEX",
            "params": {"drop_ratio_build": 0.2}
        }
        collection.create_index(
            field_name="sparse_embedding",
            index_params=sparse_index_params
        )

        return collection

    def get_collection(self) -> Collection:
        """获取集合，不存在则创建"""
        try:
            collection = Collection(name=config.MILVUS_COLLECTION)
            collection.load()
            self._collection = collection
            return collection
        except Exception:
            self._collection = self._create_collection()
            self._collection.load()
            return self._collection

    def clear_collection(self) -> None:
        """
        删除集合中的所有数据（用 DELETE 而非 DROP）

        用于重新初始化：旧数据格式变了（如 source 字段从 doc_NNN 改为真实文件名），
        直接 INSERT 会导致新旧数据混合。先清除再插入确保数据干净。

        为什么不用 DROP：
         Zilliz Cloud Serverless 的 DROP 操作容易超时（DEADLINE_EXCEEDED），
        用 DELETE 表达式 "chunk_id != ''" 删除所有行更轻量但效果相同。

        注意：这个操作不可逆，会丢失集合中的所有向量数据。
        """
        collection = self.get_collection()
        try:
            # 删除所有实体（chunk_id 是主键，只要不为空串就是有效数据）
            collection.delete('chunk_id != ""')
            collection.flush()
            print(f"  已清空集合: {collection.name}")
        except Exception as e:
            print(f"  清空失败: {e}，尝试跳过")

    def insert(self, chunks: List[Dict[str, Any]]) -> None:
        """
        批量插入文档块

        chunks 格式：
        [{
            "chunk_id": str,
            "parent_chunk_id": str,
            "chunk_level": int,
            "text": str,
            "source": str,
            "dense_embedding": List[float],
            "sparse_embedding": Dict[int, float]
        }]
        """
        collection = self.get_collection()

        entities = [
            [c["chunk_id"] for c in chunks],
            [c.get("parent_chunk_id", "") for c in chunks],
            [c["chunk_level"] for c in chunks],
            [c["text"] for c in chunks],
            [c["source"] for c in chunks],
            [c["dense_embedding"] for c in chunks],
            [c["sparse_embedding"] for c in chunks],
        ]

        print(f"  正在写入 {len(chunks)} 个块到 Milvus...")
        collection.insert(entities)
        collection.flush()
        print(f"  写入完成")

    def hybrid_retrieve(
        self,
        dense_embedding: List[float],
        sparse_embedding: Dict[int, float],
        top_k: int = 5,
        rrf_k: int = 60,
        filter_expr: str = ""
    ) -> List[Dict[str, Any]]:
        """
        混合检索：稠密 + 稀疏 RRF 融合

        1. 稠密向量搜索
        2. 稀疏向量搜索
        3. RRF 融合两个结果
        """
        collection = self.get_collection()

        candidate_k = top_k * config.CANDIDATE_MULTIPLIER

        # 稠密向量搜索请求
        # 注意：pymilvus 2.6 的 AnnSearchRequest 不接受 output_fields，
        # 输出字段统一在 hybrid_search() 上指定。
        dense_search = AnnSearchRequest(
            data=[dense_embedding],
            anns_field="dense_embedding",
            param={"metric_type": "IP", "params": {"ef": 64}},
            limit=candidate_k,
            expr=filter_expr if filter_expr else None,
        )

        # 稀疏向量搜索请求
        sparse_search = AnnSearchRequest(
            data=[sparse_embedding],
            anns_field="sparse_embedding",
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
            limit=candidate_k,
            expr=filter_expr if filter_expr else None,
        )

        # RRF 融合
        ranker = RRFRanker(k=rrf_k)

        results = collection.hybrid_search(
            reqs=[dense_search, sparse_search],
            rerank=ranker,
            limit=top_k,
            output_fields=["chunk_id", "parent_chunk_id", "chunk_level", "text", "source"]
        )

        # 格式化结果
        docs = []
        for hits in results:
            for hit in hits:
                docs.append({
                    "chunk_id": hit.entity.get("chunk_id"),
                    "parent_chunk_id": hit.entity.get("parent_chunk_id"),
                    "chunk_level": hit.entity.get("chunk_level"),
                    "text": hit.entity.get("text"),
                    "source": hit.entity.get("source"),
                    "score": hit.score
                })

        return docs

    def dense_retrieve(
        self,
        dense_embedding: List[float],
        top_k: int = 5,
        filter_expr: str = ""
    ) -> List[Dict[str, Any]]:
        """
        仅稠密向量检索（备用）
        """
        collection = self.get_collection()

        results = collection.search(
            data=[dense_embedding],
            anns_field="dense_embedding",
            param={"metric_type": "IP", "params": {"ef": 64}},
            limit=top_k,
            expr=filter_expr if filter_expr else None,
            output_fields=["chunk_id", "parent_chunk_id", "chunk_level", "text", "source"]
        )

        docs = []
        for hits in results:
            for hit in hits:
                docs.append({
                    "chunk_id": hit.entity.get("chunk_id"),
                    "parent_chunk_id": hit.entity.get("parent_chunk_id"),
                    "chunk_level": hit.entity.get("chunk_level"),
                    "text": hit.entity.get("text"),
                    "source": hit.entity.get("source"),
                    "score": hit.score
                })

        return docs

    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """
        按 chunk_id 批量查询块内容（用于 Auto-merging 拉取父块）

        Args:
            chunk_ids: 要查询的块的 chunk_id 列表

        Returns:
            块信息列表，每个包含 chunk_id, parent_chunk_id, chunk_level, text, source

        原理：
            Milvus 的 query 接口支持 IN 表达式，以主键批量获取实体。
            这比逐条查高效得多，一次网络往返拿到所有父块。

        坑点：
            - chunk_id 是 VARCHAR 类型，IN 子句里每个值要加引号
            - 如果 chunk_ids 为空，直接返回空列表，避免无效查询
            - pymilvus query 可能返回超过 offset+limit 的结果，
              如果需要大批量查询应分页；AC 场景量小不担心
        """
        if not chunk_ids:
            return []

        collection = self.get_collection()

        # 构建 IN 表达式：chunk_id in ["id1", "id2", ...]
        # 关键：反斜杠需要转义，否则 Milvus 将 \u、\n 等误解析为转义字符
        # 例如 "antfu\unocss.md" → "antfu\\unocss.md"
        escaped_ids = [cid.replace("\\", "\\\\") for cid in chunk_ids]
        ids_str = ", ".join(f'"{cid}"' for cid in escaped_ids)
        expr = f"chunk_id in [{ids_str}]"

        try:
            results = collection.query(
                expr=expr,
                output_fields=["chunk_id", "parent_chunk_id", "chunk_level", "text", "source"],
            )
        except Exception:
            return []

        return [
            {
                "chunk_id": r.get("chunk_id"),
                "parent_chunk_id": r.get("parent_chunk_id"),
                "chunk_level": r.get("chunk_level"),
                "text": r.get("text"),
                "source": r.get("source"),
            }
            for r in results
        ]


# 全局单例
_milvus_client: Optional[MilvusClient] = None


def get_milvus_client() -> MilvusClient:
    """获取 MilvusClient 单例"""
    global _milvus_client
    if _milvus_client is None:
        _milvus_client = MilvusClient()
    return _milvus_client
