"""
单独运行此脚本以初始化向量数据库（无需进入问答循环）

用法：
  方式1（推荐）: python -m my_agentic_rag.init_vectorstore
  方式2:        python src/my_agentic_rag/init_vectorstore.py
"""
import sys
import os
from pathlib import Path

# 确保项目根目录在 Python 路径中（支持直接运行）
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from .vectorstore import get_vectorstore
except ImportError:
    # 直接运行时相对导入不可用，用绝对导入
    from my_agentic_rag.vectorstore import get_vectorstore


def main():
    print("=" * 60)
    print("初始化向量数据库")
    print("=" * 60)

    data_dir = os.environ.get("DATA_DIR", "data")
    print(f"数据目录: {data_dir}")

    # 先清空旧数据，避免新旧混合（如 source 字段格式变更）
    try:
        from .milvus_client import get_milvus_client
    except ImportError:
        from my_agentic_rag.milvus_client import get_milvus_client
    get_milvus_client().clear_collection()

    # 初始化（加载文档 -> 三级分块 -> 写入 Milvus）
    vectorstore = get_vectorstore()
    vectorstore.initialize(data_dir)

    print("初始化完成！")


if __name__ == "__main__":
    main()
