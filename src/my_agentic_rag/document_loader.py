"""
============================================================
代码故事（从程序启动到回答用户问题的完整过程）
============================================================
程序启动后，如果需要初始化向量数据库...
【document_loader.py】的【load_and_chunk_all()】被调用...

它会：
1. 扫描 data 目录下的所有 .md 文件
2. 使用 RecursiveCharacterTextSplitter（LangChain 工业级分块器）对每个文档进行三级分块：
   - L1: 1200+ 字符的父块（无重叠，供 Auto-merging 替换用）
   - L2: 600+ 字符的中间块（50 字符重叠）
   - L3: 300+ 字符的叶子块（100 字符重叠，保证检索命中率）
3. 同时计算每个块的稠密向量和稀疏向量
4. 将所有块写入 Milvus 数据库

关键改进（v2.0）：
  - 从手写 _split_text 升级为 RecursiveCharacterTextSplitter，按分隔符优先级切分
  - 新增 chunk_overlap（L3=100, L2=50），解决"拦腰截断"问题
  - 中英文混合分隔符列表，适配双语知识库
============================================================
"""

"""
============================================================
面试版：RecursiveCharacterTextSplitter + chunk_overlap
============================================================

【为什么不用固定长度硬切分？】
硬切分（每 N 个字符一刀）是"实验室做法"，在生产环境会导致：
  - 一句话被拦腰截断："Milvus 2.5 支持混" | "合检索和 RRF 融合"
  - 检索时模型读到残缺信息，无法正确理解
  - 召回率直接受影响（视频教学称这一步 +15%）

【RecursiveCharacterTextSplitter 怎么工作？】
降级优先策略，按分隔符列表逐级尝试：
  ["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""]
      ↑       ↑      ↑                                  ↑       ↑      ↑
   段落优先  换行   中文标点                           英文标点  空格  兜底字符级

1. 先用最大分隔符（\n\n 段落）切分 → 如果某段落在 chunk_size 内，直接保留
2. 超长的段落 → 降级用换行（\n）切
3. 还是超长 → 降级用中文句号（。）切
4. ...逐级降级，直到空格，最后按字符硬切（兜底）

这就是"降级策略（Graceful Degradation）"——优先在自然语义边界上切分，
只在不得已时才硬切。类似于视频播放的自动降画质、CSS 字体回退。

【chunk_overlap 为什么重要？】
假如没有 overlap，两个相邻块：
  Chunk A: "Milvus 2.5 版本支持混合检索"
  Chunk B: "和 RRF 融合算法"
  → 检索"混合检索和 RRF"时，A 和 B 各有一半，但都不完整

有 100 字符的 overlap 后：
  Chunk A: "Milvus 2.5 版本支持混合检索和 RRF"
  Chunk B: "支持混合检索和 RRF 融合算法"
  → 检索命中任意一块都是完整的，不会漏信息

overlap 本质是"接力赛交棒"——两棒之间重叠一小段，保证不会掉棒。

【重叠多少合适？】
- L3（300 token）= 100 重叠 → ~30% 重叠，保证高召回
- L2（600 token）= 50 重叠 → ~8% 重叠，保持上下文连贯即可
- L1（1200 token）= 0 重叠 → 父块用于全部替换，不需要重叠
- 重叠太大 → 存储膨胀，而且相邻块差异太小，检索重复率高
- 重叠太小 → 交界处信息丢失

【线上落地注意事项】
- 分块大小不是拍脑袋定的，应该用 RAGAS 评估不同 chunk_size 的召回率
- 如果知识库以短文档为主，需要调小 chunk_size
- 如果知识库是长文/论文，可以适当调大
============================================================
"""

import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional

from bs4 import BeautifulSoup
from docx import Document
import pdfplumber
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .embedding import get_embedding_service
from .milvus_client import get_milvus_client

# 中英文混合分隔符列表（按优先级从高到低）
# 设计意图：优先在自然语义边界上切分——段落 > 换行 > 句子结束标点 > 其他标点 > 空格 > 字符硬切
# 注意：中文标点在前（因为知识库以中文为主），英文标点在后兜底
_SEPARATORS = [
    "\n\n",   # 段落分隔（最高优先级）
    "\n",     # 换行
    "。",     # 中文句号
    "！",     # 中文感叹号
    "？",     # 中文问号
    "；",     # 中文分号
    "，",     # 中文逗号
    ".",      # 英文句号
    "!",      # 英文感叹号
    "?",      # 英文问号
    ";",      # 英文分号
    ",",      # 英文逗号
    " ",      # 空格
    "",       # 字符级硬切（最后兜底）
]


def _extract_text(filepath: Path, ext: str) -> str:
    """按扩展名分派到对应提取器，统一返回纯文本"""
    if ext == ".html":
        return _extract_html(filepath)
    if ext == ".docx":
        return _extract_docx(filepath)
    if ext == ".pdf":
        return _extract_pdf(filepath)
    if ext == ".xlsx":
        return _extract_xlsx(filepath)
    return filepath.read_text(encoding="utf-8")


def _extract_html(filepath: Path) -> str:
    """HTML：BeautifulSoup 去标签，显式 UTF-8 读取（避免 Windows GBK 乱码）"""
    raw = filepath.read_text(encoding="utf-8")
    return BeautifulSoup(raw, "lxml").get_text("\n", strip=True)


def _extract_docx(filepath: Path) -> str:
    """Word：提取段落与表格文本"""
    doc = Document(str(filepath))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def _extract_pdf(filepath: Path) -> str:
    """PDF：pdfplumber 按页提取"""
    pages = []
    with pdfplumber.open(str(filepath)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages)


def _extract_xlsx(filepath: Path) -> str:
    """Excel：pandas 按 sheet 转文本"""
    sheets = pd.read_excel(str(filepath), sheet_name=None)
    parts = []
    for name, df in sheets.items():
        if df.empty:
            continue
        parts.append(f"## {name}\n{df.to_string(index=False)}")
    return "\n\n".join(parts)


class DocumentLoader:
    """
    文档加载 + 三级分块（v2.0：RecursiveCharacterTextSplitter + overlap）

    与 v1.0 的区别：
    - v1.0: 手写 _split_text，按段落和句号简单切分，无 overlap
    - v2.0: 使用 LangChain RecursiveCharacterTextSplitter，支持分隔符降级 + chunk_overlap

    为什么这个升级重要：
    - RecursiveCharacterTextSplitter 是 LangChain 中最广泛使用的分块器
    - 降级策略保证优先在段落/句子边界切分，保持语义完整
    - overlap 防止关键信息在块边界丢失
    - 这是 RAG 准确率从 60% 提升到 85% 的第一步（视频称这一步直接 +15%）
    """
    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.milvus_client = get_milvus_client()

    def load_documents(self, data_dir: Path) -> List[str]:
        """
        加载 data 目录下所有支持的文件（递归子目录）

        支持的文件类型见 config.SUPPORTED_EXTENSIONS：
        .md, .txt, .py, .js, .ts, .vue, .json, .yaml, .html, .css, .csv,
        .docx, .pdf, .xlsx 等
        """
        return [text for text, _ in self.load_documents_with_source(data_dir)]

    def load_documents_with_source(self, data_dir: Path) -> List[tuple]:
        """
        加载 data 目录下所有支持的文件，返回 (文本, 相对路径) 列表

        按扩展名分派到对应提取器：
        - .html → BeautifulSoup 去标签
        - .docx → python-docx 提取段落与表格
        - .pdf  → pdfplumber 按页提取
        - .xlsx → pandas 按 sheet 转文本
        - 其余   → 按 UTF-8 纯文本读取

        相对路径示例: 'md/antfu.md', 'word/使用说明.docx'
        """
        data_dir = Path(data_dir)
        result = []
        for ext in config.SUPPORTED_EXTENSIONS:
            for filepath in data_dir.rglob(f"*{ext}"):
                try:
                    text = _extract_text(filepath, ext)
                except Exception:
                    continue
                if not text or not text.strip():
                    continue
                rel_path = filepath.relative_to(data_dir)
                # 统一用 / 分隔符，避免 Windows 反斜杠在 Milvus 表达式中被误解析
                rel_str = str(rel_path).replace("\\", "/")
                # 跳过 README / manifest / 待下载清单等说明文件
                if rel_path.name in config.DATA_EXCLUDE_FILES or rel_str in config.DATA_EXCLUDE_PATHS:
                    continue
                result.append((text, rel_str))
        return result

    def _get_splitter(self, chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
        """
        创建 RecursiveCharacterTextSplitter 实例

        Args:
            chunk_size: 块大小（字符数）
            chunk_overlap: 相邻块重叠字符数

        Returns:
            配置好的 RecursiveCharacterTextSplitter

        设计要点：
        - length_function=len：中文字符计数（一个中文 ≈ 1 token），比 tiktoken 更适合中文
        - separators 按优先级排列，确保优先在自然边界切分
        - is_separator_regex=False：分隔符是纯字符串匹配，不是正则

        坑点：
        - 不应该用 tiktoken 做中文分词——tiktoken 对中文不友好，
          一个中文字可能被切成 2+ 个 token，导致 chunk_size 不可预测
        - 简繁混排的知识库需要在 separator 列表里同时考虑
        """
        return RecursiveCharacterTextSplitter(
            separators=_SEPARATORS,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def _create_chunk(
        self,
        text: str,
        chunk_level: int,
        parent_chunk_id: str,
        source: str,
        chunk_index: int
    ) -> Dict[str, Any]:
        """创建单个块，包含文本和向量"""
        chunk_id = f"{source}_{chunk_level}_{chunk_index}_{uuid.uuid4().hex[:8]}"

        dense_embedding = self.embedding_service.get_dense_embedding(text)
        sparse_embedding = self.embedding_service.get_sparse_embedding(text)

        return {
            "chunk_id": chunk_id,
            "parent_chunk_id": parent_chunk_id,
            "chunk_level": chunk_level,
            "text": text,
            "source": source,
            "dense_embedding": dense_embedding,
            "sparse_embedding": sparse_embedding,
        }

    def chunk_document(self, text: str, source: str) -> List[Dict[str, Any]]:
        """
        对单个文档进行三级分块（v2.0）

        流程：
        1. L3 叶子块：RecursiveCharacterTextSplitter(300 字符, overlap=100) → 最细粒度检索
        2. L2 中间块：RecursiveCharacterTextSplitter(600 字符, overlap=50) → 合并中间层
        3. L1 父块：RecursiveCharacterTextSplitter(1200 字符, overlap=0) → 完整上下文

        父子关系建立：
        - L3 → L2：按索引比例映射
        - L2 → L1：按索引比例映射

        为什么三级而不是一级？
        - 只用小块检索 → 找到后喂给 LLM 的信息太少，缺乏上下文
        - 只用大块检索 → 向量匹配精度降低（大块语义太杂）
        - 三级方案 → 小块做精准匹配，大块做上下文补充（"小搜大给"策略）
        """
        all_chunks = []

        # ------------------------------------------------------------------
        # L3 叶子块：最细粒度，检索命中率最高的层级
        # chunk_overlap=100 保证关键句子不会在块边界被截断
        # ------------------------------------------------------------------
        l3_splitter = self._get_splitter(config.L3_CHUNK_SIZE, config.L3_CHUNK_OVERLAP)
        l3_texts = l3_splitter.split_text(text)
        l3_chunks = []
        for i, t in enumerate(l3_texts):
            chunk = self._create_chunk(t, config.CHUNK_LEVEL_L3, "", source, i)
            l3_chunks.append(chunk)

        # ------------------------------------------------------------------
        # L2 中间块：中等粒度，overlap=50 保持上下文连贯
        # ------------------------------------------------------------------
        l2_splitter = self._get_splitter(config.L2_CHUNK_SIZE, config.L2_CHUNK_OVERLAP)
        l2_texts = l2_splitter.split_text(text)
        l2_chunks = []
        for i, t in enumerate(l2_texts):
            chunk = self._create_chunk(t, config.CHUNK_LEVEL_L2, "", source, i)
            l2_chunks.append(chunk)

        # ------------------------------------------------------------------
        # L1 父块：最大粒度，overlap=0 保持上下文干净
        # 父块在 Auto-merging 时用于替换多个命中子块，提供完整段落
        # ------------------------------------------------------------------
        l1_splitter = self._get_splitter(config.L1_CHUNK_SIZE, config.L1_CHUNK_OVERLAP)
        l1_texts = l1_splitter.split_text(text)
        l1_chunks = []
        for i, t in enumerate(l1_texts):
            chunk = self._create_chunk(t, config.CHUNK_LEVEL_L1, "", source, i)
            l1_chunks.append(chunk)

        # 建立父子关系
        # L3 → L2：按索引比例映射
        # 原理：L3 块数 / L2 块数 = 每个 L2 下有 ≈N 个 L3 子块
        l3_per_l2 = max(1, len(l3_chunks) // max(1, len(l2_chunks)))
        for i, l3_chunk in enumerate(l3_chunks):
            l2_index = min(i // l3_per_l2, len(l2_chunks) - 1)
            l3_chunk["parent_chunk_id"] = l2_chunks[l2_index]["chunk_id"]

        # L2 → L1：按索引比例映射
        l2_per_l1 = max(1, len(l2_chunks) // max(1, len(l1_chunks)))
        for i, l2_chunk in enumerate(l2_chunks):
            l1_index = min(i // l2_per_l1, len(l1_chunks) - 1)
            l2_chunk["parent_chunk_id"] = l1_chunks[l1_index]["chunk_id"]

        all_chunks.extend(l1_chunks)
        all_chunks.extend(l2_chunks)
        all_chunks.extend(l3_chunks)

        return all_chunks

    def load_and_chunk_all(self, data_dir: Path) -> List[Dict[str, Any]]:
        """
        加载目录下所有文档并分块

        返回所有层级的块列表（source 字段使用相对路径，如 'vue/pinia.md'）
        """
        docs_with_source = self.load_documents_with_source(data_dir)

        # 限制文档数量（MAX_LOAD_DOCS=0 表示不限制）
        max_docs = config.MAX_LOAD_DOCS
        if max_docs > 0 and len(docs_with_source) > max_docs:
            print(f"限制加载 {max_docs}/{len(docs_with_source)} 个文档")
            docs_with_source = docs_with_source[:max_docs]

        # 用原始文档全文建 BM25 词表（必须在 chunk_document 之前，
        # 这样 _create_chunk 里的 get_sparse_embedding 才能查到词 ID）
        raw_texts = [text for text, _ in docs_with_source]
        self.embedding_service.build_index(raw_texts)

        all_chunks = []
        for doc_text, source in docs_with_source:
            chunks = self.chunk_document(doc_text, source)
            all_chunks.extend(chunks)

        print(f"文档加载完成，共 {len(docs_with_source)} 个文档，{len(all_chunks)} 个块")
        return all_chunks

    def persist_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """将块写入 Milvus"""
        if not chunks:
            return

        self.milvus_client.insert(chunks)
        print(f"已写入 {len(chunks)} 个块到 Milvus")


# 全局单例
_document_loader: Optional[DocumentLoader] = None


def get_document_loader() -> DocumentLoader:
    """获取 DocumentLoader 单例"""
    global _document_loader
    if _document_loader is None:
        _document_loader = DocumentLoader()
    return _document_loader
