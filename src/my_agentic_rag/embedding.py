"""
============================================================
代码故事（从程序启动到回答用户问题的完整过程）
============================================================
程序启动后，当需要处理文档时...
【embedding.py】被调用，首先检查是否有持久化的 BM25 状态...
如果有，则加载之前的状态...
当需要检索时：
1. 调用【get_dense_embedding()】获取 BGE 稠密向量
2. 调用【get_sparse_embedding()】获取 BM25 稀疏向量
这两个向量会被传给 Milvus 进行混合检索...
============================================================
"""

"""
============================================================
面试版：稠密向量 + BM25 稀疏向量
============================================================
Embedding 服务同时提供两种向量：
1. 稠密向量：BGE 模型生成的连续向量，捕捉语义相似性
2. 稀疏向量：BM25 算法生成的词权重向量，捕捉关键词匹配

两种向量互补：稠密向量擅长语义相似，BM25 擅长关键词精确匹配
============================================================
"""

import re
import json
import math
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

from . import config


class EmbeddingService:
    """
    文本向量化服务

    同时生成两种向量：
    - 稠密向量：BGE 模型生成的连续向量（语义相似）
    - 稀疏向量：BM25 算法生成的词权重向量（关键词匹配）
    """

    def __init__(self, state_path: Optional[Path] = None):
        # BGE 稠密向量嵌入模型
        self._embedder = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        # BM25 参数
        self.k1 = 1.5
        self.b = 0.75

        # BM25 词表和统计信息
        self._vocab: Dict[str, int] = {}
        self._doc_freq: Counter = Counter()
        self._total_docs = 0
        self._sum_token_len = 0
        self._avg_doc_len = 1.0

        # 状态持久化路径
        self._state_path = state_path or config.BM25_STATE_PATH

        # 加载持久化状态
        self._load_state()

    # ================================================================
    # 分词（中英文混合）
    # ================================================================
    def tokenize(self, text: str) -> List[str]:
        """
        中英文混合分词

        - 中文：按字符粒度
        - 英文：按单词粒度
        - 标点：忽略
        """
        text = text.lower()

        # 中文字符
        chinese_pattern = re.compile(r"[一-鿿]")
        # 英文单词
        english_pattern = re.compile(r"[a-zA-Z]+")
        # 数字
        number_pattern = re.compile(r"[0-9]+")

        tokens = []

        # 逐字符扫描
        i = 0
        while i < len(text):
            char = text[i]

            if chinese_pattern.match(char):
                tokens.append(char)
                i += 1
            elif english_pattern.match(char):
                # 收集完整英文单词
                word = ""
                while i < len(text) and english_pattern.match(text[i]):
                    word += text[i]
                    i += 1
                if word:
                    tokens.append(word)
            elif number_pattern.match(char):
                # 收集完整数字
                num = ""
                while i < len(text) and number_pattern.match(text[i]):
                    num += text[i]
                    i += 1
                if num:
                    tokens.append(num)
            else:
                i += 1

        return tokens

    # ================================================================
    # BM25 稀疏向量
    # ================================================================
    def get_sparse_embedding(self, text: str) -> Dict[int, float]:
        """
        获取 BM25 稀疏向量

        返回格式：{token_index: bm25_score, ...}
        只返回非零权重的词
        """
        tokens = self.tokenize(text)
        if not tokens:
            return {}

        doc_len = len(tokens)
        self._avg_doc_len = self._sum_token_len / max(self._total_docs, 1)

        scores = {}
        for token in set(tokens):
            if token not in self._vocab:
                continue

            token_idx = self._vocab[token]
            freq = tokens.count(token)

            # BM25 公式
            n = self._total_docs
            df = self._doc_freq.get(token, 0)

            if df == 0:
                continue

            # IDF
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1)

            # BM25 score
            score = idf * (freq * (self.k1 + 1)) / (
                freq + self.k1 * (1 - self.b + self.b * doc_len / self._avg_doc_len)
            )

            if score > 0:
                scores[token_idx] = score

        return scores

    # ================================================================
    # 稠密向量
    # ================================================================
    def get_dense_embedding(self, text: str) -> List[float]:
        """获取 BGE 稠密向量"""
        return self._embedder.embed_query(text)

    def get_dense_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量获取 BGE 稠密向量"""
        return self._embedder.embed_documents(texts)

    # ================================================================
    # 同时获取稠密和稀疏向量
    # ================================================================
    def get_all_embeddings(self, texts: List[str]) -> Tuple[List[List[float]], List[List[Dict[int, float]]]]:
        """
        同时获取稠密和稀疏向量

        返回：(dense_embeddings, sparse_embeddings)
        """
        dense_embeddings = self.get_dense_embeddings(texts)
        sparse_embeddings = [self.get_sparse_embedding(text) for text in texts]
        return dense_embeddings, sparse_embeddings

    # ================================================================
    # 构建索引（用于文档写入前的词表构建）
    # ================================================================
    def build_index(self, texts: List[str]) -> None:
        """
        从文档集合构建 BM25 词表和统计信息

        在写入文档到 Milvus 前调用
        """
        n = len(texts)
        print(f"  正在构建 BM25 索引（{n} 篇文档）...")

        # 合并所有文档收集词表
        all_tokens = []
        for i, text in enumerate(texts):
            tokens = self.tokenize(text)
            all_tokens.extend(tokens)
            if (i + 1) % 500 == 0:
                print(f"    分词进度: {i+1}/{n}")

        # 构建词表
        vocab_set = set(all_tokens)
        self._vocab = {token: idx for idx, token in enumerate(sorted(vocab_set))}
        print(f"  词表大小: {len(self._vocab)} 个词")

        # 统计文档频率
        unique_tokens_per_doc = [set(self.tokenize(text)) for text in texts]
        for tokens in unique_tokens_per_doc:
            for token in tokens:
                self._doc_freq[token] += 1

        self._total_docs = n
        self._sum_token_len = sum(len(self.tokenize(text)) for text in texts)
        self._avg_doc_len = self._sum_token_len / max(self._total_docs, 1)

        # 持久化
        self._persist_state()
        print(f"  BM25 索引构建完成")

    # ================================================================
    # 状态持久化
    # ================================================================
    def _load_state(self) -> None:
        """加载 BM25 状态"""
        if self._state_path and self._state_path.exists():
            try:
                with open(self._state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)

                self._vocab = state.get("vocab", {})
                self._doc_freq = Counter(state.get("doc_freq", {}))
                self._total_docs = state.get("total_docs", 0)
                self._sum_token_len = state.get("sum_token_len", 0)
                self._avg_doc_len = state.get("avg_doc_len", 1.0)
            except Exception:
                pass

    def _persist_state(self) -> None:
        """保存 BM25 状态"""
        if not self._state_path:
            return

        self._state_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "version": 1,
            "total_docs": self._total_docs,
            "sum_token_len": self._sum_token_len,
            "avg_doc_len": self._avg_doc_len,
            "vocab": self._vocab,
            "doc_freq": dict(self._doc_freq),
        }

        with open(self._state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)


# 全局单例
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """获取 EmbeddingService 单例"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
