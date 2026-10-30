# 测绘成果智能质检助手

Agentic RAG + ArcMap 工具调用的测绘成果质检助手。

- **规范知识**：检索测绘法律法规、技术大纲、实施细则，回答"要求是什么、依据哪一条"。
- **成果事实**：通过 ArcMap MCP 读取 mxd/shp/图层的真实信息（版本、断源、字段、要素数）。
- **结论生成**：把"事实"和"依据"分开呈现，生成带条款出处的检查结论。

> 它不是"AI 自动验收系统"。精度、现势性、图面符号等需要外业或人工判断的项目，工具只做初筛。

## 架构

```
用户问题
   │
   ▼
[router]  knowledge / operation / mixed
   │
   ├── knowledge ──► 原五节点 Agentic RAG（检索 → 评分 → 生成）
   │
   └── operation/mixed ──► [agent] ⇄ [tools]
                              │        ├─ search_standards（复用 RAG）
                              │        └─ arcmap_*（ArcMap MCP）
                              ▼
                           [finalize] 事实 + 依据 + 结论
```

设计取舍：

- **RAG 是工具，不是前置步骤**：模型自己决定何时查规范、查什么。
- **工具层取自 mini-agent**：`tools/base.py`、`tools/registry.py` 1:1 复制；不搬 `AgentLoop`。
- **循环用 LangGraph 重写**：两个节点 + 条件边，避免引入 mini-agent 的 compact/hooks/permissions 等子系统。
- **只暴露只读工具**：首期 11 个 ArcMap 工具全部是"查"，不开放 `execute_arcpy` 和会改数据的工具。

## 快速开始

```bash
# 1. 安装依赖
pip install -e .

# 2. 配置 .env（参考 .env.example）
#    ANTHROPIC_API_KEY / MILVUS_* / ARCMAP_MCP_*

# 3. 抓取公开语料（可选，已抓好的在 data/）
python tools/fetch_corpus.py

# 4. 重建向量索引
python src/my_agentic_rag/init_vectorstore.py

# 5. 启动问答
python src/my_agentic_rag/main.py                 # 交互模式
python src/my_agentic_rag/main.py "你的问题"      # 单次问答
```

## 语料

`data/` 当前包含 13 个公开文件，约 22 万字符：

- `regulations/`：测绘法、地图管理条例、基础测绘条例、测绘成果管理条例、测绘质量监督管理办法、测绘资质管理办法
- `guidelines/`：实景三维中国建设技术大纲、新型基础测绘体系建设试点技术大纲、福建省三调技术实施细则、云南村庄规划数据库质检细则、自然资源调查监测体系构建总体方案
- `metadata/`：18 条现行国标元数据（标准号、名称、实施日期、官方详情链接）
- `standards/`：预留给需要手动下载的标准全文

国家标准全文公开系统只提供在线预览，核心国标（GB/T 20257.1、GB/T 7930、GB 50026、CJJ/T 8、CH/T 1004、TD/T 1055 等）没有自动抓取。完整清单和官方渠道见 [data/MANUAL_DOWNLOAD.md](data/MANUAL_DOWNLOAD.md)。下载后放进 `data/standards/` 即可被自动收录。

## 工具

### search_standards

复用 `rag_graph.run_rag`，返回答案 + 来源文件。适合规范要求、术语定义、限差、依据类问题。

### arcmap_*（白名单 11 个）

| 工具 | 用途 | 需要 ArcMap 打开 |
|---|---|---|
| `describe_mxd` | 读取 .mxd 声明的版本 | 否 |
| `audit_folder` | 批量扫描文件夹内所有 .mxd：版本、断源、定义查询、出图范围 | 否 |
| `ping` | 检查 ArcMap 桥接 | 是 |
| `get_arcmap_info` | 当前文档、数据框、比例尺 | 是 |
| `list_layers` | 列出图层 | 是 |
| `get_layer_info` | 图层详情 | 是 |
| `describe_data` | 读取 shp/gdb/栅格信息 | 是 |
| `list_fields` | 列出字段 | 是 |
| `count_features` | 统计要素数 | 是 |
| `get_unique_values` | 字段唯一值 | 是 |
| `list_broken_data_sources` | 列出断源数据源 | 是 |

在线工具要求 ArcMap 已打开，并在 arcmap-mcp 工具条上点击"开始"；否则返回 `PUENTE CAIDO`。

## 演示问题

```text
# 只查规范
GB/T 24356 对测绘成果质量检查的基本规定是什么？

# 只查数据（不需要打开 ArcMap）
扫描 D:/成果/2026 这个目录里的 mxd，告诉我有没有断源图层。

# 混合：规范 + 数据
检查 D:/成果/2026 的 mxd，按 GB/T 24356 判断这些断源图层属于什么错漏等级。
```

## 目录结构

```
src/my_agentic_rag/
  rag_graph.py          # 原五节点 Agentic RAG（未改动）
  agent/
    graph.py            # 路由 + agent ⇄ tools + finalize
    rag_tool.py         # search_standards
    mcp_stdio.py        # arcmap-mcp 的 stdio 客户端
    arcmap_tools.py     # MCP 工具 → Tool，白名单
    tools/
      base.py           # 1:1 取自 mini-agent
      registry.py       # 1:1 取自 mini-agent
data/                   # 语料 + 元数据 + 清单
tools/fetch_corpus.py   # 语料抓取脚本
```

## 性能说明

- 首次查询要加载 `BAAI/bge-m3`（约 2.2GB）和 `BAAI/bge-reranker-v2-m3`（约 2.1GB），CPU 上可能需要几分钟。
- **演示时建议用交互模式**：模型只加载一次，后续问题快很多；单次问答模式每次都会重新加载。
- 混合检索已经修复为真正的 dense + sparse RRF（pymilvus 2.6 的 `AnnSearchRequest` 不接受 `output_fields`，旧代码会静默回退到纯稠密检索）。

## 测试数据

没有自己的项目数据时，可以用 ArcMap 自带模板做离线检查：

```text
D:/arcmap/Desktop10.8/Templates/ExportWebMapTemplates
```

`describe_mxd` 和 `audit_folder` 不需要打开 ArcMap。

## 已知限制

- 核心国标全文未包含，需按 `data/MANUAL_DOWNLOAD.md` 手动获取。
- 精度、现势性、图面符号、影像解译正确性不在自动检查范围内。
- 在线 ArcMap 工具依赖 arcMap 10.x + Add-in；离线审计不依赖。
- 首期不开放任何写操作（`add_layer`、`save_mxd`、`execute_arcpy` 等）。

## 面试时的讲法

> 测绘质检不能只靠语言模型背规范，也不能只跑脚本。我的项目把"规范依据"和"数据事实"放进同一个 agent 循环：RAG 负责检索条款，ArcMap MCP 负责读取成果，LangGraph 负责决定什么时候用哪个，最后生成带条款出处的质检结论。
