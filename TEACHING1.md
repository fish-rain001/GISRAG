## 目录

1. [第一层：这个项目是什么？](#第一层这个项目是什么)
2. [第二层：贯穿实例 — 追一条数据从磁盘到 LLM 的全过程](#第二层贯穿实例--追一条数据从磁盘到-llm-的全过程)
   - [阶段一（入库）第 1-4 步](#阶段一入库把-vitemd-变成-milvus-里的向量)
   - [阶段二（检索）第 5-8 步](#阶段二检索用户提问--向量--混合搜索--答案)
3. [外围支撑：Config、VectorStore、Main、Evaluate](#外围支撑configvectorstoremainevaluate)

---

## 第一层：这个项目是什么？

### 一句话定义

**my-agentic-rag 是一个生产级 Agentic RAG 系统** —— 用 LangGraph 编排多路检索 + LLM 自判断路由，配合 Milvus 混合检索（稠密 + 稀疏）实现远超传统 RAG 的检索质量。

### 为什么这个项目对你很重要？

它包含了 **RAG 系统从入门到生产的几乎所有核心概念**：

| 概念 | 对应模块 | 面试权重 |
|------|----------|----------|
| 混合检索（Dense + Sparse） | `milvus_client.py` | ⭐⭐⭐⭐⭐ |
| 多路检索增强（HyDE + Step-back + 子查询） | `rag_graph.py` | ⭐⭐⭐⭐⭐ |
| Rerank 重排序 | `rag_utils.py` | ⭐⭐⭐⭐⭐ |
| Auto-merging（层级合并） | `rag_utils.py` | ⭐⭐⭐⭐ |
| 三级分块 + RecursiveCharacterTextSplitter | `document_loader.py` | ⭐⭐⭐⭐ |
| LangGraph 条件路由 | `rag_graph.py` | ⭐⭐⭐⭐⭐ |
| 语义校验（反幻觉改写） | `rag_graph.py` | ⭐⭐⭐⭐ |
| 证据锚定生成（Grounded Generation） | `prompts.py` + `rag_graph.py` | ⭐⭐⭐⭐ |
| RAGAS 评估 | `evaluate.py` | ⭐⭐⭐⭐ |
| BM25 稀疏向量 | `embedding.py` | ⭐⭐⭐⭐ |

**学透这一个项目，你面试能讲的东西就非常多了。**

### 外部依赖一览

```
langchain + langchain_anthropic    ← LLM 调用 + Anthropic 兼容接口
langgraph                          ← 状态图编排 + 条件路由
langchain_huggingface              ← BGE 嵌入模型加载
pymilvus                           ← Milvus 向量数据库客户端
sentence-transformers              ← BGE 嵌入 + 重排（CrossEncoder）
ragas                              ← RAG 系统评估框架
python-dotenv                      ← 读取 .env 配置
beautifulsoup4 + lxml              ← HTML 去标签
python-docx                        ← Word 文档解析
pdfplumber                         ← PDF 解析
pandas + openpyxl                  ← Excel 解析
numpy                              ← 向量运算
```

### 术语约定

这三个词在文档里反复出现，先统一一下：

| 术语 | 含义 | 代码对应 |
|------|------|---------|
| **检索（Retrieve）** | 从向量数据库中找到最相关的文档块 | `vectorstore.retrieve()` |
| **评分（Grade）** | LLM 判断文档是否真的与问题相关 | `grade_documents()` |
| **生成（Generate）** | LLM 根据上下文生成最终答案 | `generate()` |

```
一次 RAG 问答的完整生命周期：

用户提问: "Vite + Vue 3 + Pinia 搭建项目怎么做？"
  │
  ├── Step 1: retrieve_initial() — 四路检索
  │     ├── 路径1: 原始查询检索
  │     ├── 路径2: HyDE 假设性文档检索（语义校验通过才用）
  │     ├── 路径3: Step-back 退步问题检索（语义校验通过才用）
  │     └── 路径4: 子查询拆解检索（复杂问题时触发）
  │     → 合并去重，得到候选文档列表
  │
  ├── Step 2: grade_documents() — LLM 语义评分
  │     每个文档调一次 LLM 判断是否与问题相关
  │     → 有相关文档 → 路由到 generate
  │     → 无相关文档 → 路由到 rewrite_question
  │
  ├── Step 3: rewrite_question() — 失败分类路由
  │     判断"为什么没搜到"
  │     → 模糊问题 → 重写
  │     → 知识缺失 → 拒答
  │     → 跨文档 → 跳过重写
  │
  ├── Step 4: retrieve_expanded() — 扩展检索
  │     用重写后的问题再次检索 + LLM 评分
  │
  └── Step 5: generate() — 证据锚定生成
        每个声明必须标注文档编号
        → 返回最终答案
```

---

## 第二层：贯穿实例 — 追一条数据从磁盘到 LLM 的全过程

```
┌──────────────────────────────────────────────────────────────┐
│                     main.py（入口）                            │
│              initialize_vectorstore() + chat_loop()            │
└──────────────────────┬───────────────────────────────────────┘
                       │
          ┌────────────┼────────────────┐
          │            │                │
          ▼            ▼                ▼
┌─────────────┐ ┌──────────────┐ ┌──────────────┐
│   Config    │ │  VectorStore │ │  RAG Graph   │
│   配置管理   │ │  向量存储层   │ │  LangGraph   │
└──────┬──────┘ └──────┬───────┘ └──────┬───────┘
       │               │                │
       │    ┌──────────┴──────────┐     │
       │    │                     │     │
       ▼    ▼                     ▼     ▼
┌──────────────────────────────────────────────────────────────┐
│                    数据层 & 检索层                              │
│                                                               │
│  ┌──────────────┐  ┌───────────────┐  ┌───────────────────┐  │
│  │ Embedding    │  │ Milvus Client │  │ Document Loader   │  │
│  │ 稠密+稀疏向量 │  │ 混合检索 RRF  │  │ 三级分块+入库      │  │
│  └──────┬───────┘  └───────┬───────┘  └────────┬──────────┘  │
│         │                  │                    │              │
│         └──────────────────┼────────────────────┘              │
│                            │                                   │
│                     ┌──────┴───────┐                           │
│                     │  RAG Utils   │                           │
│                     │ Rerank+Merge │                           │
│                     └──────────────┘                           │
└──────────────────────────────────────────────────────────────┘
         │                                    │
         ▼                                    ▼
┌──────────────┐                    ┌──────────────────┐
│   Prompts    │                    │    Evaluate      │
│  提示词模板   │                    │   RAGAS 评估      │
└──────────────┘                    └──────────────────┘
```

### 启动流程

项目分两个独立阶段运行：

```
═══════════════════════════════════════════════════════════════════
阶段一：入库（一次性）
═══════════════════════════════════════════════════════════════════

python src/my_agentic_rag/init_vectorstore.py
  │
  └── init_vectorstore.main()
        ├── get_milvus_client().clear_collection()  ← 清空旧数据
        └── get_vectorstore().initialize(data_dir)
              │
              ├── 连接 Milvus
              │
              ├── loader = get_document_loader()  ← 获取 DocumentLoader 单例
              │       └── 内部持有 self.embedding_service = get_embedding_service()
              │                          self.milvus_client = get_milvus_client()
              │
              ├── chunks = loader.load_and_chunk_all(data_dir)
              │     ├── load_documents_with_source() → 扫描 data/ 目录
              │     ├── build_index(原始文档全文) → 建 vocab → 写 bm25_state.json
              │     │
              │     └── for 每个文档: chunk_document(text, source)
              │           ├── RecursiveCharacterTextSplitter 三级切出文本片段
              │           └── _create_chunk():
              │                 ├── embedding_service.get_dense_embedding(text)
              │                 └── embedding_service.get_sparse_embedding(text)
              │                    ↑ vocab 已建好，稀疏向量一次算对
              │           返回 [{text, dense_embedding, sparse_embedding, ...}, ...]
              │
              └── loader.persist_chunks(chunks)
                    └── milvus_client.insert(chunks)
                          └── 批量写入 Milvus

═══════════════════════════════════════════════════════════════════
阶段二：问答（可反复运行）
═══════════════════════════════════════════════════════════════════

python src/my_agentic_rag/main.py
  │
  ├── initialize_vectorstore()
  │     │
  │     ├── set_data_dir(data_dir)       ← 记下数据目录路径
  │     ├── client = get_milvus_client()  ← 获取 Milvus 单例
  │     ├── collection = client.get_collection()  ← 连接 + 获取集合
  │     ├── if num_entities > 0:         ← 库里已有数据？
  │     │     标记 _initialized=True，直接返回  ← 上次入库过了，跳过！
  │     │
  │     └── 如果库里是空的 → 走阶段一的入库流程（兜底，一般不触发）
  │
  └── chat_loop()
        └── while True:
              ├── input("你: ") → 等待用户提问
              ├── run_rag(question):
              │     ├── build_rag_graph() → 编译 LangGraph 图
              │     └── graph.invoke(initial_state) → 执行五节点流程
              └── 打印答案 + 引用来源
```

**关键点：入库和问答从来不在同一次运行中发生。** `init_vectorstore.py` 只做入库，跑完就退出。`main.py` 先检查 Milvus 里有没有数据，有就直接跳过入库进交互循环。`bm25_state.json` 在入库时写好，问答阶段 `EmbeddingService` 启动时从磁盘加载，拿来就算。

**设计模式：全局单例** — 各个服务组件（EmbeddingService、MilvusClient、DocumentLoader、VectorStore）均通过 `get_xxx()` 工厂函数获取同一个实例，各模块通过 `config.py` 共享配置。

---

### 贯穿实例：追一条数据从磁盘到 LLM 的全过程

先看各文件的调用关系，再看具体每一步。

```
文件调用链（箭头 = "调用"）：

init_vectorstore.py ──→ vectorstore.py ──→ document_loader.py ──→ embedding.py
                    │                  │                        └─→ bm25_state.json (磁盘)
                    │                  │
                    │                  └────────────→ milvus_client.py ──→ Milvus (远程数据库)
                    │
main.py ──→ vectorstore.py ──→ rag_utils.py ──→ embedding.py
          │                              │    └─→ milvus_client.py
          │                              │
          └──→ rag_graph.py ──→ vectorstore.py
                          │    └─→ prompts.py
                          │
evaluate.py ──→ rag_graph.py
```

下面以 `data/antfu/vite.md` 为例，分两个阶段跟踪每一步数据在哪个文件里、传给了谁。

> **阶段一跑的是 `init_vectorstore.py`，阶段二跑的是 `main.py`。两个阶段是两次独立的程序运行。**

---

#### 阶段一（入库）：把 vite.md 变成 Milvus 里的向量

**调用入口：** `init_vectorstore.py` → `vectorstore.py`

---

**第 1 步：加载文件 → `document_loader.py`**

所属文件：`document_loader.py` — `load_documents_with_source()`

`vectorstore.py` 调用 `loader.load_and_chunk_all(data_dir)`，内部第一步就是加载。`load_documents_with_source()` 遍历 `config.SUPPORTED_EXTENSIONS`，按扩展名分派到对应的提取器（`_extract_text`），把 5 种格式统一转成纯文本：

```python
# document_loader.py → 按扩展名分派到专用提取器
docs_with_source = [
    ("Vite is a next-generation frontend build tool...", "md/vite.md"),      # .md   → UTF-8 纯文本
    ("Python 教程 — 课前甜点...", "html/tutorial.html"),                      # .html → BeautifulSoup 去标签
    ("全国地方遥感应用协会...", "word/使用说明.docx"),                          # .docx → python-docx（段落+表格）
    ("Supporting Information for...", "pdf/paper.pdf"),                     # .pdf  → pdfplumber 按页提取
    ("cluster1 | 2021-01-13 | ...", "excel/result.xlsx"),                   # .xlsx → pandas 按 sheet 转文本
]
# 数据类型: List[tuple[str, str]]  即 [(文本, 相对路径), ...]
```

五种格式各自用专用解析库提取，但对外统一返回纯文本，下游的分块、BM25 建词表、入库逻辑完全无感。

数据在 `document_loader.py` 内部，不传给其他文件。

---

**第 2 步：建 BM25 词表 → `document_loader.py` → `embedding.py`**

所属文件：`document_loader.py` — `load_and_chunk_all()` 内部

```python
# document_loader.py 内部：
raw_texts = [text for text, _ in docs_with_source]
# 提取纯文本列表: ["Vite is a next-generation...", "Pinia 是 Vue...", ...]

self.embedding_service.build_index(raw_texts)
#         │                                  │
#         │  document_loader.py              │  embedding.py
#         │  传入: List[str] (22 个原始文档全文)
#         │                                  ▼
#         │                          tokenize → 建 vocab (5891 个词)
#         │                          → 统计 doc_freq
#         │                          → 写 data/bm25_state.json
#         │  返回: None (写磁盘,无返回值)
```

**数据传递：** `document_loader.py` 把 `List[str]` 传给 `embedding.py`，`embedding.py` 处理后写磁盘，不返回数据。

---

**第 3 步：分块 + 计算向量 → `document_loader.py` → `embedding.py`**

所属文件：`document_loader.py` — `chunk_document()` → `_create_chunk()`

`_create_chunk()` 在 `document_loader.py` 中，但它内部调 `embedding.py` 的两个方法：

```python
# document_loader.py → _create_chunk(text="Vite is a next-generation...")
#                                        ↑ str (300 字符的 L3 文本片段)

# ① 稠密向量 — 调 embedding.py
dense_embedding = self.embedding_service.get_dense_embedding(text)
#   document_loader.py  ──传入 str──→  embedding.py
#                                     BGE 模型推理
#   document_loader.py  ←──返回 List[float] (1024维) ──
# → [0.023, -0.145, 0.678, -0.032, ...]

# ② 稀疏向量 — 调 embedding.py（vocab 在第 2 步已建好）
sparse_embedding = self.embedding_service.get_sparse_embedding(text)
#   document_loader.py  ──传入 str──→  embedding.py
#                                     tokenize → 查 vocab → 查 doc_freq → BM25
#   document_loader.py  ←──返回 Dict[int, float] ──
# → {5774: 1.52, 3239: 0.87, 1500: 0.62, ...}
#      ↑词ID   ↑BM25权重
```

两种向量拿回来后，`document_loader.py` 把文本、向量、来源路径打包：

```python
# document_loader.py → _create_chunk() 返回:
{
    "chunk_id":       "antfu/vite.md_3_0_a1b2c3d4",
    "parent_chunk_id": "antfu/vite.md_2_0_e5f6g7h8",
    "chunk_level":     3,
    "text":           "Vite is a next-generation frontend build tool...",
    "source":         "antfu/vite.md",
    "dense_embedding": [0.023, -0.145, 0.678, ...],    # 来自 embedding.py
    "sparse_embedding": {5774: 1.52, 3239: 0.87, ...}, # 来自 embedding.py
}
# 数据类型: Dict[str, Any]
```

所有 22 个文件共产生 2457 个这样的字典，全部存在 `document_loader.py` 的局部变量 `all_chunks` 里。

---

**第 4 步：写入 Milvus → `document_loader.py` → `milvus_client.py`**

所属文件：`document_loader.py` — `persist_chunks()`

`vectorstore.py` 拿到 `load_and_chunk_all()` 返回的 2457 个 chunk 后，调 `persist_chunks()`：

```python
# vectorstore.py:
chunks = loader.load_and_chunk_all(data_dir)  # 2457 个 Dict
loader.persist_chunks(chunks)
#       │
#       │  document_loader.py → persist_chunks():
#       │  self.milvus_client.insert(chunks)
#       │         │
#       │         │  document_loader.py  ──传入 List[Dict] (2457条)──→  milvus_client.py
#       │         │                                                      │
#       │         │                                                      ▼
#       │         │                                              连接 Milvus
#       │         │                                              collection.insert(entities)
#       │         │                                              → 写入远程数据库
#       │         │  返回: None
```

`init_vectorstore.py` 进程退出。数据现在存在 Milvus 远程数据库里。

---

#### 阶段二（检索）：多路检索 → 评分 → 路由 → 生成

**调用入口：** `main.py` → `rag_graph.py` → `vectorstore.py` → `rag_utils.py`

> 这次启动 `main.py`，`embedding.py` 从 `bm25_state.json` 加载了 vocab 和 doc_freq。

---

**第 5 步：多路检索（4 路并行） → `rag_graph.py` → `rag_utils.py` → `embedding.py` + `milvus_client.py`**

所属文件：`rag_graph.py` — `retrieve_initial()` 节点

不是只拿用户原问题搜一次。Agentic RAG 的核心：**生成三种变体各自检索，最后合并去重。**

```
用户问题: "Vite 是什么构建工具？"

路径1: 原始查询直接检索                                       [必跑]
   "Vite 是什么构建工具？" → get_dense/sparse → Milvus hybrid → 5个文档

路径2: HyDE（假设性文档）                                      [可选]

   为什么：用户问的是疑问句，知识库里全是陈述句。语义相同但句式不同，
   向量距离会偏大。HyDE 让 LLM 先写一段"伪文档"，陈述句和陈述句的
   向量距离更近。

   所属文件：`rag_graph.py` — `_generate_hypothetical_document()`

   ```python
   # rag_graph.py → _generate_hypothetical_document(llm, question):
   prompt = HYDE_PROMPT.format(question=question)
   #   HYDE_PROMPT 从 prompts.py import，是一个 str 模板
   response = llm.invoke(prompt)
   #   rag_graph.py  ──传入 str──→  LLM (DeepSeek)
   #   rag_graph.py  ←──返回 str──  "Vite 是一个新一代前端构建工具，基于原生 ESM..."
   hyde_doc = response.strip()
   # 失败返回 ""，不阻塞主流程
   ```

   生成后做语义校验，防止 LLM 改歪：

   ```python
   # rag_graph.py → _validate_semantic_similarity(original, rewritten):
   orig_vec = get_embedding_service().get_dense_embedding(original)    # 原问题向量
   rewr_vec = get_embedding_service().get_dense_embedding(rewritten)   # 改写文本向量
   similarity = sum(a * b for a, b in zip(orig_vec, rewr_vec))         # 余弦相似度
   #   BGE 的 normalize_embeddings=True，向量已 L2 归一化 → 余弦相似度 = 点积
   if similarity < 0.65:
       丢弃（LLM 把"怎么退款"改成了"怎么赚钱"——南辕北辙）
   else:
       纳入合并 → 用 hyde_doc 调 retrieve_documents()  → Milvus → 5个文档
   ```

路径3: Step-back（退步问题）                                    [可选]

   为什么：具体问题可能搜不到直接答案，抽象成通用概念更容易命中。

   所属文件：`rag_graph.py` — `_generate_step_back_question()`

   ```python
   # rag_graph.py → _generate_step_back_question(llm, question):
   prompt = STEP_BACK_QUESTION_PROMPT.format(question=question)
   #   STEP_BACK_QUESTION_PROMPT 从 prompts.py import
   response = llm.invoke(prompt)
   #   rag_graph.py  ──传入 str──→  LLM
   #   rag_graph.py  ←──返回 str──  "什么是前端构建工具"
   step_back_q = response.strip()
   ```

   同样经 `_validate_semantic_similarity()` 语义校验，通过后调 `retrieve_documents(step_back_q)` 检索。

路径4: 子查询拆解                                              [可选]

   为什么：复杂问题涉及多个独立主题，一次检索涵盖不全。

   所属文件：`rag_graph.py` — `_decompose_question()`

   ```python
   # rag_graph.py → _decompose_question(llm, question):
   prompt = DECOMPOSE_QUERY_PROMPT.format(question=question)
   response = llm.invoke(prompt)
   text = response.strip()

   if "SIMPLE" in text.upper():
       return []   # 问题够简单，不拆 —— "Vite 是什么构建工具？" 走这个分支

   # 否则拆成子问题，例如 "Vite+Vue 3+Pinia 搭建" →
   #   ["Vite 项目配置步骤", "Vue 3 组件写法规范", "Pinia store 定义方式"]
   # 每个子问题各自调 retrieve_documents() → 向量 → Milvus
   # 失败返回 []，不阻塞主流程
   ```

四路结果合并去重：

   所属文件：`rag_graph.py` — `_merge_dedupe_docs()`

   ```python
   # rag_graph.py → _merge_dedupe_docs(docs_list):
   #   docs_list = [[原始检索的5条], [HyDE的5条], [Step-back的5条], [子查询的N条]]
   seen = {}
   for docs in docs_list:
       for doc in docs:
           sig = doc["content"][:200]                 # 前200字符做去重指纹
           if sig in seen:
               if doc["score"] > seen[sig]["score"]:  # 重复的保留分数更高的
                   seen[sig] = doc
           else:
               seen[sig] = doc
   merged = sorted(seen.values(), key=lambda d: d["score"], reverse=True)
   # → 15 个候选文档，按 score 降序 → 写入 state["docs"]
   ```

**`doc["score"]` 是怎么来的：** 每条 chunk 的 score 由 Milvus 在 `hybrid_retrieve()` 中计算，一路透传到这里。

   ```
   Milvus 内部 RRF倒数排名融合
     → hit.score ──→ milvus_client.py: doc["score"] = hit.score
                  ──→ rag_utils.py: 原样保留（Rerank 加了 rerank_score 但不覆盖 score）
                  ──→ rag_graph.py: auto_merge 合并时取子块 score 的最大值
                  ──→ _merge_dedupe_docs: 用 score 比较去重 + 排序
   ```
   
   score 的值本质是 **RRF 融合分数**：`1/(60+稠密榜名次) + 1/(60+稀疏榜名次)`。两路排名都靠前的 chunk 分数高，只有一路排前面的分数中等，两路都靠后的分数低。详情见第 5 步的 Milvus 混合检索部分。

**"去重指纹"是什么：** 取文档内容前 200 个字符作为标识。

   ```
   四路检索可能搜到同一篇文档的不同 chunk，比如：
     原始查询 → chunk_L3_0 的 text = "Vite is a next-generation frontend build tool..."
     HyDE    → chunk_L3_0 的 text = "Vite is a next-generation frontend build tool..."
     ↑ 完全相同的 chunk，被两路独立搜到了
   
   用 text[:200] 做指纹 → 两路结果指纹相同 → 视为重复 → 只保留 score 更高的那条。
   
   为什么不用 chunk_id == chunk_id 判断重复？
   因为不同路可能搜到同一篇文档的不同 chunk（L3_0 vs L3_1），
   chunk_id 不同但内容高度重叠。用前 200 字符做近似去重，
   比严格 chunk_id 匹配能多去一些"半重复"。
   ```
```

**代码追踪：每一路检索最终都落到同样的底层调用。** 以路径1（原始查询）为例：

```python
# rag_graph.py → retrieve_initial(state):
#   state["question"] = "Vite 是什么构建工具？"
#   → 内部调 vectorstore.retrieve(question) → rag_utils.retrieve_documents(question):

# rag_utils.py → retrieve_documents(query="Vite 是什么构建工具？"):

# ① 获取查询的稠密向量 — 调 embedding.py
dense_embedding = embedding_service.get_dense_embedding(query)
#   rag_utils.py  ──传入 str──→  embedding.py → BGE 模型
#   rag_utils.py  ←──返回 List[float] (1024维) ──

# ② 获取查询的稀疏向量 — 调 embedding.py
sparse_embedding = embedding_service.get_sparse_embedding(query)
#   rag_utils.py  ──传入 str──→  embedding.py
#                               tokenize → 查 vocab → 查 doc_freq → BM25
#   rag_utils.py  ←──返回 Dict[int, float] ──

# ③ 混合检索 — 调 milvus_client.py
retrieved = milvus_client.hybrid_retrieve(
    dense_embedding, sparse_embedding,
    top_k=5, filter_expr="chunk_level == 3",
)
#   rag_utils.py  ──传入 (List[float], Dict[int,float])──→  milvus_client.py
#                                                            Milvus 双路搜索 + RRF
#   rag_utils.py  ←──返回 List[Dict] (5条) ──

# 路径2/3/4 同理：HyDE/Step-back/子查询各自调 LLM 生成文本后，
# 走同样的 retrieve_documents() → embedding → hybrid_retrieve 链路。
# 四路结果 → _merge_dedupe_docs() 合并去重 → state["docs"] = [候选文档列表]
```

---

**第 6 步：Rerank 重排 → `rag_utils.py` → 本地 CrossEncoder**

所属文件：`rag_utils.py` — `rerank_documents()`

第 5 步的混合检索已经筛出了候选文档。但 BGE 向量检索是"粗筛"——只能判断文本整体方向是否接近，不够精细。Rerank 用更强的 Cross-encoder 模型，逐对计算 (query, doc) 的相关度，做"精排"。本项目用本地模型 `BAAI/bge-reranker-v2-m3`，不依赖外部 API。

```
BGE 检索（粗筛）                    bge-reranker-v2-m3（精排）
─────────────────                   ─────────────────
单文本 → 向量 → 一次编码             (query, doc) 逐对 → 分数
处理 2457 个文档，取前 15            处理 15 个候选，取前 3
速度快，适合全库扫描                 速度慢，但候选池已很小
```

```python
# rag_utils.py → rerank_documents(query="Vite 是什么构建工具？", docs=15条, top_k=3):

# ① 本地加载 Cross-encoder 模型（单例懒加载，首次加载后常驻内存）
_reranker = CrossEncoder(config.RERANK_MODEL, device="cpu")   # "BAAI/bge-reranker-v2-m3"

# ② 逐对计算 (query, doc) 的相关性分数
pairs = [[query, doc["text"]] for doc in docs]                # 15 个 (query, doc) 对
scores = _reranker.predict(pairs)                             # 返回 15 个相关度分数

# ③ 按分数从高到低排序，取 top-3，加上 rerank_score
scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
reranked = [{**doc, "rerank_score": float(s)} for doc, s in scored[:3]]
return reranked, {"rerank_applied": True, "rerank_model": "BAAI/bge-reranker-v2-m3"}
```

**和 Embedding 检索的区别：**

| | Embedding 检索（第 5 步） | Rerank 重排（第 6 步） |
|---|---|---|
| 模型 | BGE（轻量，适合大规模） | bge-reranker-v2-m3 Cross-encoder（重，逐对计算） |
| 输入 | 单文本 → 向量 | (query, doc) 对 → 分数 |
| 处理量 | 2457 个 chunk 全库扫描 | 15 个候选精细比对 |
| 成本 | 一次编码 | 15 次逐对比较 |
| 输出 | score（RRF 融合分数） | rerank_score（语义相关度） |

模型权重首次运行时从 HuggingFace Hub 下载到本地缓存（约 2.2GB），之后全程本地推理，无额度限制。如果模型加载失败——返回原始 docs 不变，不阻塞主流程。

---

**第 7 步：Auto-merging → `rag_utils.py` → `milvus_client.py`**

所属文件：`rag_utils.py` — `auto_merge_documents()`

```python
# rag_utils.py → auto_merge_documents(docs=3条):
# 发现 2 个子块的 parent_chunk_id 相同 → 触发合并

parent_chunks = milvus_client.get_chunks_by_ids(["antfu/vite.md_2_0_..."])
#   rag_utils.py  ──传入 List[str]──→  milvus_client.py → collection.query(...)
#   rag_utils.py  ←──返回 List[Dict] (父块完整数据) ──

# 用父块替换 2 个子块:
# 合并前: [L3块(300字), L3块(300字), L3块(300字)]
# 合并后: [L2父块(600字), L3块(300字)]
```

---

**第 8 步：评分 → 条件路由 → 生成 → `rag_graph.py` + `prompts.py`**

所属文件：`rag_graph.py` — `grade_documents()` / `rewrite_question()` / `retrieve_expanded()` / `generate()`

这里不是一条直线，而是 **LangGraph 条件路由**——LLM 评分后根据结果走不同分支。

**8a. 评分（Pydantic 结构化输出）：**

```python
# rag_graph.py → grade_documents(state):
class GradeResult(BaseModel):
    relevant: bool
    reason: str = ""

for doc in state["docs"]:    # 3 个合并后的文档
    structured_llm = llm.with_structured_output(GradeResult)
    result: GradeResult = structured_llm.invoke(
        GRADE_DOCUMENTS_PROMPT.format(question, doc["content"])
    )
    if result.relevant:
        relevant_docs.append(doc)

# → relevant_docs = [父块, L3块]  → 有相关文档 → route = "generate"
```

不用字符串匹配（`"相关" in response`），因为"不相关"里也包含"相关"二字。Pydantic 强制 LLM 输出 `{"relevant": true/false}` 的 JSON。

**8b. 如果步骤 8a 发现所有文档都不相关 → 失败分类路由：**

```python
# rag_graph.py → rewrite_question(state):
# 先让 LLM 判断"为什么没搜到"

CLASSIFY_FAILURE_PROMPT → LLM 分析:
  if "missing_knowledge":   # 知识库里真没有
      route = "generate"     # → 直接进 8c，输出"未找到相关信息"
  elif "multi_hop":          # 需跨文档综合
      route = "generate"     # → 子查询已在第 5 步做了
  else:                      # 模糊问题 or 检索策略问题
      REWRITE_QUERY_PROMPT → LLM 重写问题
      route = "rewrite"      # → 进 retrieve_expanded 节点

# rag_graph.py → retrieve_expanded(state):
# 用重写后的问题再次检索 + LLM 评分 → 进 generate
```
用户问："Vite、Vitest 和 Vue 3 三者之间是什么关系？"

这个问题：
  一篇文档讲 Vite 是什么（构建工具）
  一篇文档讲 Vitest 是什么（测试框架，由 Vite 驱动）
  一篇文档讲 Vue 3 是什么（前端框架）
  → 没有任何一篇文档直接回答"三者关系"
  → 答案是分散在三篇里的，需要"跳"三次（multi-hop）才能拼出完整答案

检索结果：
  搜"三者关系" → 没有文档直接命中 → 全部判为不相关 → 进 rewrite_question
  分类 LLM 判断：不是"知识库缺了"（三篇都在），而是"需要跨文档综合" → 返回 "multi_hop_query"

处理逻辑：
  if "multi_hop": route = "generate"
  → 不重写问题（重写也没用，还是搜不到"三者关系"这篇不存在的文章）
  → 子查询拆解已经做了：第 5 步里 decompose 会拆成
      "Vite 的作用" / "Vitest 的作用" / "Vue 3 的作用" 各自检索
  → 子查询的三路结果已经在 state["docs"] 里了
  → 直接进 generate，LLM 读这三篇然后综合回答
**8c. 证据锚定生成：**

```python
# rag_graph.py → generate(state):
# 构建上下文: [文档1] (来源: antfu/vite.md) + 文档内容

from .prompts import GROUNDED_GENERATE_PROMPT
# 关键规则（写在 GROUNDED_GENERATE_PROMPT 模板里）:
#   1. 每个事实性声明必须标注 [文档N]
#   2. 禁止从元数据推断事实关系（不能从"文件名"推理出"作者"）
#   3. 禁止使用文档外知识
#   4. 找不到证据就说"未找到足够证据"

prompt = GROUNDED_GENERATE_PROMPT.format(context=context, question=question)
response = llm.invoke(prompt)
# → "Vite 是一个新一代前端构建工具 [文档1]，由原生 ESM 驱动 [文档1]。
#    Vite 8 版本基于 Rolldown 打包器 [文档1]。关于 Vite 和 Webpack 的对比，
#    未在检索到的文档中找到足够证据。"

state["answer"] = response
# rag_graph.py → main.py: 打印给用户
```

---

**LangGraph 条件路由总览：**

```
retrieve_initial (4路检索)
  │
  ▼
grade_documents (LLM 逐个评分)
  │
  ├── 有相关文档 ──→ generate (证据锚定生成) ──→ 答案
  │
  └── 全部不相关
        │
        ▼
      rewrite_question (LLM 分类失败原因)
        │
        ├── missing_knowledge ──→ generate ("未找到相关信息")
        ├── multi_hop         ──→ generate (子查询结果已有)
        └── 其他               ──→ retrieve_expanded (重写+再搜) ──→ generate
```

---

**各文件在贯穿实例中出场次数统计：**

```
document_loader.py   ████████ (第1,2,3,4步) ← 入库核心
embedding.py         ██████   (第2,3,5步)   ← 向量提供者
milvus_client.py     █████    (第4,5,7步)   ← 数据库读写
rag_graph.py         ██████   (第5,8步)     ← 编排+多路检索+路由+生成
rag_utils.py         █████    (第5,6,7步)   ← 检索流水线
prompts.py           ███      (第5,8步)     ← HyDE/Step-back/评分/生成模板
vectorstore.py       ██       (入口)        ← 薄调度层
```

---

**完整数据流总结：**

```
阶段一：入库 (init_vectorstore.py)            阶段二：检索 (main.py chat_loop)
─────────────────────────────────────       ─────────────────────────────────
                                            启动: EmbeddingService 加载
vite.md (磁盘)                               bm25_state.json → vocab 就绪
  │                                            │
  ▼                                            │
load_documents_with_source()                   │
  │                                            │
  ▼                                            │
build_index() → bm25_state.json 落地           │
  │                                            │
  ▼                                            │
chunk_document() → 三级分块                     │
  │                                            │
  ▼                                            │
_create_chunk():                               │
  get_dense_embedding → 1024维                │
  get_sparse_embedding → {词ID:权重}           │
  │                                            │
  ▼                                            │
persist_chunks():                              用户: "Vite 是什么构建工具？"
  insert() → 2457 条写入 Milvus                  │
                                                 ▼
  进程退出 ✓                              retrieve_initial() [rag_graph.py]
                                              ├── 原始查询检索
                                              ├── HyDE → 语义校验 → 检索
                                              ├── Step-back → 语义校验 → 检索
                                              └── 子查询拆解 → 各自检索
                                              → 四路合并去重 → 15 候选
                                              │
                                              ▼
                                            rerank_documents()
                                              本地 CrossEncoder 精排 → top-3
                                              │
                                              ▼
                                            auto_merge_documents()
                                              命中同一父块 → 拉取父块替换
                                              │
                                              ▼
                                            grade_documents()
                                              LLM 逐个语义判断
                                              │
                                    ┌─────────┴─────────┐
                                    │                   │
                                 有相关               全部不相关
                                    │                   │
                                    ▼                   ▼
                              generate()        rewrite_question()
                               证据锚定            LLM 分类失败原因
                                    │            ├── 缺知识 → generate
                                    │            └── 模糊 → 重写+扩展检索 → generate
                                    ▼                   ▼
                              答案: "Vite 是... [文档1]"
```

## 完整的 RAG 流程回顾

```
┌─────────────────────────────────────────────────────────────────┐
│                    一次 RAG 问答的数据流                           │
└─────────────────────────────────────────────────────────────────┘

用户: "Vite + Vue 3 + Pinia 搭建项目怎么做？"
  │
  ▼
main.py: chat_loop() → run_rag(question)
  │
  ▼
rag_graph.py: build_rag_graph() → graph.invoke(initial_state)
  │
  ├── Step 1: retrieve_initial()
  │     ├── embedding.py: get_dense_embedding() + get_sparse_embedding()
  │     ├── milvus_client.py: hybrid_retrieve() + RRF
  │     ├── rag_utils.py: rerank_documents() + auto_merge_documents()
  │     ├── prompts.py: HYDE_PROMPT + STEP_BACK_QUESTION_PROMPT + DECOMPOSE_QUERY_PROMPT
  │     └── _merge_dedupe_docs() → 15 个候选文档
  │
  ├── Step 2: grade_documents()
  │     ├── 每个文档调一次 LLM 语义判断
  │     └── 相关则保留，不相关则丢弃
  │     └── GradeResult(is_relevant=True/False)
  │     → 3 个相关文档 → route = "generate"
  │
  ├── Step 5: generate()
  │     ├── 构建上下文（[文档1] ... [文档2] ... [文档3]）
  │     └── prompts.GROUNDED_GENERATE_PROMPT → LLM 生成
  │     → "建议使用 [文档1] Nuxt 作为全栈框架 + [文档2] Pinia 做状态管理 + [文档3] UnoCSS..."
  │
  ▼
main.py: print(answer) → 用户看到结果
```

---

### 外围支撑：Config — 集中式配置管理

**一条直线：** `.env` 文件 → `dotenv` 加载 → `config.py` 模块变量 → 各模块 `from . import config` 使用

**涉及文件：** [.env.example](.env.example) → [src/my_agentic_rag/config.py](src/my_agentic_rag/config.py)

```python
from dotenv import load_dotenv
load_dotenv(BASE_DIR.parent.parent / ".env")

# LLM 配置
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic")
MODEL_ID = os.getenv("MODEL_ID", "deepseek-v4-pro")

# Milvus 配置
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "agentic_rag_docs")

# 分块配置
L1_CHUNK_SIZE = 1200   # 父块（无重叠）
L2_CHUNK_SIZE = 600    # 中间块（50 重叠）
L3_CHUNK_SIZE = 300    # 叶子块（100 重叠）

# RAG 参数
RETRIEVAL_TOP_K = 5
RERANK_TOP_K = 3
CANDIDATE_MULTIPLIER = 3
RRF_K = 60
SEMANTIC_VALIDATION_THRESHOLD = 0.65
```

> 集中式配置：所有可调参数在一个文件里。贯穿实例的每一步背后都有这里的影子——`L3_CHUNK_SIZE` 控制分块粒度，`RRF_K` 控制混合检索的排名权重，`SEMANTIC_VALIDATION_THRESHOLD` 控制 HyDE/Step-back 改写质量的校验门槛。

---

### 外围支撑：VectorStore — 向量存储抽象层

**一条直线：** `get_vectorstore()` 获取单例 → `initialize()` 加载+分块+入库 → `retrieve()` 混合检索+Rerank+Auto-merging

**涉及文件：** [src/my_agentic_rag/vectorstore.py](src/my_agentic_rag/vectorstore.py)

VectorStore 是一个薄抽象层。它不关心底层是 Milvus 还是 ChromaDB，只暴露 `initialize()` 和 `retrieve()` 两个接口。内部组装 DocumentLoader + MilvusClient + RAG Utils，统一输出 `content + metadata + score` 格式。

---

### 外围支撑：Main + Init — 入口与初始化

**涉及文件：** [src/my_agentic_rag/main.py](src/my_agentic_rag/main.py) → [src/my_agentic_rag/init_vectorstore.py](src/my_agentic_rag/init_vectorstore.py)

- `init_vectorstore.py` — 一次性入库脚本，清空旧数据 → 调用 VectorStore.initialize()
- `main.py` — 交互问答入口，先检查 Milvus 有无数据（有就跳过入库），再进 chat_loop 循环
- `__init__.py` — 库模式接口，暴露 `configure(data_dir, collection_name)` 供其他项目引用

---

### 外围支撑：Evaluate — RAGAS 评估体系

**涉及文件：** [tests/evaluate.py](tests/evaluate.py) → [tests/golden_test_set.json](tests/golden_test_set.json)

20 道黄金测试题覆盖四种难度：事实查询(5)、多文档综合(5)、知识库无答案(5)、边界歧义(5)。支持传统 RAG vs Agentic RAG 对比模式。用 RAGAS SDK 计算 Context Precision / Context Recall / Answer Relevancy / Faithfulness 四个指标，输出诊断矩阵定位问题是检索还是生成。

---

## RAG 质量提升时间线（版本演进）

| 版本 | 改进 | 预计提升 |
|------|------|---------|
| v1.0 | 基础 RAG：直接检索 + LLM 生成 | 基线 |
| v2.0 | 多路检索（HyDE + Step-back）| 召回 +20% |
| v2.0 | 三级分块 + RecursiveCharacterTextSplitter + overlap | 精度 +15% |
| v2.0 | Auto-merging（子块合并父块） | 答案完整度 +10% |
| v2.1 | 语义校验（防改写幻觉） | 噪声 -30% |
| v2.2 | Pydantic 结构化评分 | 评分准确率 +25% |
| v2.3 | 子查询拆解 | 复杂问题召回 +15% |
| v3.0 | 失败分类 + 证据锚定生成 | 幻觉率 -40% |
