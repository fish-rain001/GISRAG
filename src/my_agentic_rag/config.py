"""
============================================================
代码故事（从程序启动到回答用户问题的完整过程）
============================================================
程序启动后，【config.py】被加载，读取各类配置：
- DeepSeek API 配置（用于 LLM 调用）
- Milvus 向量数据库配置（用于存储和检索文档向量）
- Rerank 配置（本地重排模型，用于重排检索结果）
- BM25 参数配置（用于稀疏向量检索）
============================================================
"""

"""
============================================================
面试版：配置管理
============================================================
采用集中式配置设计：
1. DeepSeek API：LLM 调用
2. Milvus：向量数据库（稠密+稀疏混合检索）
3. Rerank（本地重排模型）：结果重排
4. BM25 参数：稀疏向量检索调优
============================================================
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 项目根目录
BASE_DIR = Path(__file__).parent

load_dotenv(BASE_DIR.parent.parent / ".env")

# 文档目录（由调用方通过 set_data_dir() 注入，此处仅为占位）
DATA_DIR = None

# ================================================================
# DeepSeek API 配置（LLM）
# ================================================================
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic")
MODEL_ID = os.getenv("MODEL_ID", "deepseek-v4-pro")

# ================================================================
# Milvus 向量数据库配置
# ================================================================
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "agentic_rag_docs")
MILVUS_USER = os.getenv("MILVUS_USER", "")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "")

# ================================================================
# Rerank 配置（本地重排模型）
# ================================================================
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

# ================================================================
# BM25 稀疏向量配置
# ================================================================
BM25_STATE_PATH = BASE_DIR.parent.parent / "data" / "bm25_state.json"

# ================================================================
# 三级分块配置
# ================================================================
CHUNK_LEVEL_L1 = 1  # 父块层级
CHUNK_LEVEL_L2 = 2  # 中间块层级
CHUNK_LEVEL_L3 = 3  # 叶子块层级
L1_CHUNK_SIZE = 1200  # L1 块大小（token）
L2_CHUNK_SIZE = 600  # L2 块大小（token）
L3_CHUNK_SIZE = 300  # L3 块大小（token）
LEAF_RETRIEVE_LEVEL = CHUNK_LEVEL_L3  # 检索从 L3 开始
# 重叠窗口：相邻块之间共享的字符数，防止关键信息在切分边界被截断
# L3 检索块重叠最大（保证检索命中率），L2 适中，L1 父块不重叠（保持上下文干净）
L3_CHUNK_OVERLAP = 100
L2_CHUNK_OVERLAP = 50
L1_CHUNK_OVERLAP = 0

# ================================================================
# Auto-merging 配置
# ================================================================
AUTO_MERGE_THRESHOLD = 2  # 合并阈值：子块数量 >= threshold 时合并到父块

# ================================================================
# Embedding 模型配置
# ================================================================
EMBEDDING_MODEL = "BAAI/bge-m3"  # BGE 稠密向量模型

# ================================================================
# RAG 参数
# ================================================================
RETRIEVAL_TOP_K = 5       # 初始检索数量
RERANK_TOP_K = 3          # 重排后返回数量
CANDIDATE_MULTIPLIER = 3  # 候选文档倍数（用于混合检索）
RRF_K = 60                # RRF 融合参数
# 文档加载配置
SUPPORTED_EXTENSIONS = [
    ".md", ".txt", ".rst",          # 文档
    ".py", ".js", ".ts", ".jsx", ".tsx", ".vue", ".svelte",  # 代码
    ".json", ".yaml", ".yml", ".toml", ".xml",  # 配置
    ".html", ".css", ".scss", ".less",  # 前端
    ".csv", ".ini", ".cfg", ".env",  # 数据/配置
    ".docx", ".pdf", ".xlsx",       # 办公文档（Word/PDF/Excel）
]
MAX_LOAD_DOCS = 50        # 最多加载文档数 (0=全部)
# 数据目录里的说明文件/清单不是语料，不参与入库
DATA_EXCLUDE_FILES = {"README.md", "manifest.json", "MANUAL_DOWNLOAD.md", "bm25_state.json"}
DATA_EXCLUDE_PATHS = {"metadata/gb_standards.json"}
# 查询改写语义校验：改写后的文本与原问题的向量余弦相似度低于此阈值时，视为改歪，丢弃
# HyDE 生成的是技术文档（风格变化大），用较宽松的阈值
# Step-back 生成的是抽象问题（语义偏移大），也用较宽松的阈值
SEMANTIC_VALIDATION_THRESHOLD = 0.65

# ================================================================
# Agent / ArcMap MCP 配置
# ================================================================
# ArcMap MCP 是 stdio 服务，必须用它自己的 Python 3 虚拟环境启动
ARCMAP_MCP_PYTHON = os.getenv(
    "ARCMAP_MCP_PYTHON",
    r"C:\Users\gorde\AppData\Local\arcmap-mcp\venv\Scripts\python.exe",
)
ARCMAP_MCP_SERVER = os.getenv(
    "ARCMAP_MCP_SERVER",
    r"C:\mcp\arcmap-mcp\src\arcmap_mcp_server.py",
)
# Agent 最多执行几轮工具调用，防止无限循环
AGENT_MAX_TOOL_ROUNDS = int(os.getenv("AGENT_MAX_TOOL_ROUNDS", "6"))
# 单个工具结果超过该字符数就落盘，只把摘要交给 LLM
TOOL_RESULT_MAX_CHARS = int(os.getenv("TOOL_RESULT_MAX_CHARS", "50000"))
