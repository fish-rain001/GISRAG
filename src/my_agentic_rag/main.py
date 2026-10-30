r"""
============================================================
Agentic RAG 入口（支持直接运行和作为库调用）

直接运行:
  cd my_agentic_rag
  python src\my_agentic_rag\main.py

作为库调用:
  from my_agentic_rag.main import main
  main()
============================================================
"""
import asyncio
import sys
from pathlib import Path

# 确保项目根目录和 src 目录都在 Python 路径中
_PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
_SRC_ROOT = str(Path(__file__).parent.parent)
for _p in (_SRC_ROOT, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from .vectorstore import get_vectorstore, set_data_dir, reset_vectorstore
    from .rag_graph import run_rag
    from .agent.graph import run_agent
except ImportError:
    from my_agentic_rag.vectorstore import get_vectorstore, set_data_dir, reset_vectorstore
    from my_agentic_rag.rag_graph import run_rag
    from my_agentic_rag.agent.graph import run_agent


def initialize_vectorstore(data_dir: str = None, force: bool = False):
    """初始化向量数据库（首次运行或 force=True 时才真正入库）"""
    print("=" * 60)
    print("初始化向量数据库")
    print("=" * 60)

    if data_dir is None:
        data_dir = str(Path(_PROJECT_ROOT) / "data")

    print(f"数据目录: {data_dir}")
    set_data_dir(data_dir)

    # 检查 Milvus 中是否已有数据
    try:
        from .milvus_client import get_milvus_client
    except ImportError:
        from my_agentic_rag.milvus_client import get_milvus_client

    client = get_milvus_client()
    collection = client.get_collection()

    if collection.num_entities > 0 and not force:
        print(f"向量库已有 {collection.num_entities} 条数据，跳过入库")
        print("如需重新入库，输入 'reset' 命令")
        # 标记 vectorstore 已就绪（否则 retrieve() 会触发自动初始化）
        get_vectorstore()._initialized = True
        print("初始化完成！\n")
        return

    # 清空旧数据并重新入库
    if force:
        client.clear_collection()

    vectorstore = get_vectorstore()
    vectorstore.initialize(data_dir)

    print("初始化完成！\n")


def print_result(result: dict):
    """打印答案、路由和工具调用记录。"""
    print("\n" + "=" * 60)
    print("答案:")
    print("=" * 60)
    print(result.get("answer", ""))
    print("=" * 60)

    if result.get("intent"):
        print(f"\n路由: {result['intent']}")
    if result.get("tool_log"):
        print(f"\n工具调用:")
        for i, log in enumerate(result["tool_log"], 1):
            preview = (log.get("preview") or "").replace("\n", " ")
            print(f"  {i}. {log.get('tool')} [{log.get('status')}] {preview[:100]}")
    if result.get("mcp_error"):
        print(f"\n提示: ArcMap MCP 未连接（{result['mcp_error']}），本次只用了 RAG。")

    relevant_docs = result.get("relevant_docs") or []
    if relevant_docs:
        print(f"\n参考了 {len(relevant_docs)} 个文档:")
        for i, doc in enumerate(relevant_docs, 1):
            source = doc.get("metadata", {}).get("source", "unknown")
            print(f"  {i}. {source}")


def run_once(question: str) -> dict:
    """单次问答，方便演示和脚本调用。"""
    initialize_vectorstore()
    try:
        result = asyncio.run(run_agent(question))
    except Exception as e:
        print(f"\nAgent 运行失败，回退到纯 RAG：{e}")
        result = run_rag(question)
        result.setdefault("intent", "knowledge")
        result.setdefault("tool_log", [])
        result.setdefault("mcp_error", "")
    print_result(result)
    return result


def chat_loop():
    """交互式问答循环"""
    print("=" * 60)
    print("Agentic RAG 测绘成果质检助手")
    print("=" * 60)
    print("输入问题进行咨询，或输入 'quit' 退出")
    print("输入 'reset' 重新初始化向量数据库")
    print("=" * 60)

    while True:
        try:
            question = input("\n你: ").strip()

            if not question:
                continue

            if question.lower() == "quit":
                print("再见！")
                break

            if question.lower() == "reset":
                reset_vectorstore()
                initialize_vectorstore(force=True)
                continue

            # 运行 Agent（失败时回退到纯 RAG）
            print("\n正在思考...")
            try:
                result = asyncio.run(run_agent(question))
            except Exception as e:
                print(f"\nAgent 运行失败，回退到纯 RAG：{e}")
                result = run_rag(question)
                result.setdefault("intent", "knowledge")
                result.setdefault("tool_log", [])
                result.setdefault("mcp_error", "")

            print_result(result)

        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    if len(sys.argv) > 1:
        run_once(" ".join(sys.argv[1:]))
    else:
        initialize_vectorstore()
        chat_loop()


if __name__ == "__main__":
    main()
