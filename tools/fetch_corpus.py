# -*- coding: utf-8 -*-
"""
从公开渠道抓取测绘语料到 data/。

用法:
    python tools/fetch_corpus.py
    python tools/fetch_corpus.py --only cehuifa,shijing3d-2021
    python tools/fetch_corpus.py --skip-metadata

只下载公开可获取的政府文件、技术大纲和标准元数据。
受版权保护且无公开下载渠道的标准，写入 data/MANUAL_DOWNLOAD.md，不自动抓取。
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CURL = shutil.which("curl.exe") or shutil.which("curl")

SELECTORS = [
    "#UCAP-CONTENT", ".TRS_Editor", "#zoom", ".content",
    "#content", ".article-content", ".pages_content", ".view",
    "article", "main",
]

SOURCES = [
    # ---- 法律法规（政府网站公开全文） ----
    {
        "id": "cehuifa", "kind": "html", "category": "regulations",
        "url": "http://www.npc.gov.cn/npc/c2/c30834/201905/t20190521_277996.html",
        "filename": "cehuifa-2017.md",
        "title": "中华人民共和国测绘法（2017年修订）",
        "source": "中国人大网",
    },
    {
        "id": "ditu-guanli-tiaoli", "kind": "html", "category": "regulations",
        "url": "https://www.gov.cn/zhengce/content/2015-12/14/content_10403.htm",
        "filename": "ditu-guanli-tiaoli.md",
        "title": "地图管理条例（国务院令第664号）",
        "source": "中国政府网",
    },
    {
        "id": "jichu-cehui-tiaoli", "kind": "html", "category": "regulations",
        "url": "https://www.gov.cn/zhengce/content/2009-05/20/content_5657.htm",
        "filename": "jichu-cehui-tiaoli.md",
        "title": "基础测绘条例（国务院令第556号）",
        "source": "中国政府网",
    },
    {
        "id": "cehui-chengguo-tiaoli", "kind": "html", "category": "regulations",
        "url": "https://www.gov.cn/zhengce/content/2008-03/28/content_5765.htm",
        "filename": "cehui-chengguo-guanli-tiaoli.md",
        "title": "中华人民共和国测绘成果管理条例（国务院令第469号）",
        "source": "中国政府网",
    },
    {
        "id": "cehui-zizhi-banfa", "kind": "html", "category": "regulations",
        "url": "https://gi.mnr.gov.cn/202106/t20210608_2648784.html",
        "filename": "cehui-zizhi-guanli-banfa.md",
        "title": "测绘资质管理办法和测绘资质分类分级标准",
        "source": "自然资源部",
    },
    {
        "id": "cehui-zizhi-tongyong", "kind": "file", "category": "regulations",
        "url": "http://gi.mnr.gov.cn/202106/P020210608571811567042.docx",
        "filename": "cehui-zizhi-fenlei-tongyong.docx",
        "title": "测绘资质分类分级标准-通用标准",
        "source": "自然资源部",
    },
    {
        "id": "cehui-zizhi-zhuanye", "kind": "file", "category": "regulations",
        "url": "http://gi.mnr.gov.cn/202106/P020210608528071897272.docx",
        "filename": "cehui-zizhi-fenlei-zhuanye.docx",
        "title": "测绘资质分类分级标准-专业标准",
        "source": "自然资源部",
    },
    {
        "id": "cehui-zhiliang-jiandu", "kind": "html", "category": "regulations",
        "url": "http://zrzyt.hunan.gov.cn/xxgk/zcfg/flfg/chdl/201402/t20140228_4400565.html",
        "filename": "cehui-zhiliang-jiandu-guanli-banfa.md",
        "title": "测绘质量监督管理办法（国测国字〔1997〕28号）",
        "source": "湖南省自然资源厅",
    },
    # ---- 技术大纲 / 实施细则（公开 PDF） ----
    {
        "id": "shijing3d-2021", "kind": "pdf", "category": "guidelines",
        "url": "https://m.mnr.gov.cn/gk/tzgg/202108/P020220608365567427770.pdf",
        "filename": "shijing3d-jishu-dagang-2021.pdf",
        "title": "实景三维中国建设技术大纲（2021版）",
        "source": "自然资源部",
    },
    {
        "id": "xinxing-jichu-cehui-2021", "kind": "pdf", "category": "guidelines",
        "url": "https://casm.ac.cn/upload/zgch/202303/20230313/pdf_202303131030013_21822.pdf",
        "filename": "xinxing-jichu-cehui-jishu-dagang-2021.pdf",
        "title": "新型基础测绘体系建设试点技术大纲",
        "source": "中国测绘科学研究院",
    },
    {
        "id": "fujian-sandiao-xize", "kind": "pdf", "category": "guidelines",
        "url": "https://zrzyt.fj.gov.cn/ztzl/fjsdscqggtdca/bzgf/201904/P020190408568651775289.pdf",
        "filename": "fujian-sandiao-jishu-xize.pdf",
        "title": "福建省第三次全国国土调查（县级）技术实施细则",
        "source": "福建省自然资源厅",
    },
    {
        "id": "yunnan-cunzhuang-jiancha", "kind": "pdf", "category": "guidelines",
        "url": "http://www.rl.gov.cn/gtj/Attach/2303/06S62WY8AQ230324104048001745B0.pdf",
        "filename": "yunnan-cunzhuang-guihua-shujuku-jiancha-xize.pdf",
        "title": "云南省“多规合一”实用性村庄规划数据库标准及数据质量检查细则",
        "source": "云南省自然资源厅",
    },
    {
        "id": "ziran-ziyuan-diaocha", "kind": "doc", "category": "guidelines",
        "url": "https://www.gov.cn/zhengce/zhengceku/2020-01/18/5470398/files/68ebe3064f254d68a1a81ea0c67d9cc2.doc",
        "filename": "ziran-ziyuan-diaocha-jian-ce-zongti-fangan.docx",
        "title": "自然资源调查监测体系构建总体方案",
        "source": "中国政府网",
    },
]

# 在国家标准全文公开系统检索现行版本（只取元数据，不抓全文）
GB_QUERIES = [
    "20257.1", "20257.2", "20257.3", "20257.4",
    "7930-2008", "7931-2008", "18314", "12898",
    "24356", "18316", "17941", "19710",
    "21010", "20258.1", "13989", "17798", "14911",
]

# 无法公开下载全文的标准：只给出官方渠道，由用户自行获取
MANUAL_ITEMS = [
    ("GB/T 20257.1-2017", "国家基本比例尺地图图式 第1部分：1:500 1:1000 1:2000 地形图图式",
     "https://openstd.samr.gov.cn/bzgk/std/index", "国家标准全文公开系统（在线预览）"),
    ("GB/T 20257.2-2017", "国家基本比例尺地图图式 第2部分：1:5000 1:10000 地形图图式",
     "https://openstd.samr.gov.cn/bzgk/std/index", "国家标准全文公开系统（在线预览）"),
    ("GB/T 20257.3-2017", "国家基本比例尺地图图式 第3部分：1:25000 1:50000 1:100000 地形图图式",
     "https://openstd.samr.gov.cn/bzgk/std/index", "国家标准全文公开系统（在线预览）"),
    ("GB/T 20257.4-2017", "国家基本比例尺地图图式 第4部分：1:250000 1:500000 1:1000000 地形图图式",
     "https://openstd.samr.gov.cn/bzgk/std/index", "国家标准全文公开系统（在线预览）"),
    ("GB/T 7930-2008", "1:500 1:1000 1:2000 地形图航空摄影测量内业规范",
     "https://openstd.samr.gov.cn/bzgk/std/index", "国家标准全文公开系统（在线预览）"),
    ("GB/T 7931-2008", "1:500 1:1000 1:2000 地形图航空摄影测量外业规范",
     "https://openstd.samr.gov.cn/bzgk/std/index", "国家标准全文公开系统（在线预览）"),
    ("GB/T 18314-2024", "全球导航卫星系统（GNSS）测量规范",
     "https://openstd.samr.gov.cn/bzgk/std/index", "国家标准全文公开系统（在线预览）"),
    ("GB/T 12898-2009", "国家三、四等水准测量规范",
     "https://openstd.samr.gov.cn/bzgk/std/index", "国家标准全文公开系统（在线预览）"),
    ("GB/T 24356-2023", "测绘成果质量检查与验收",
     "https://openstd.samr.gov.cn/bzgk/std/index", "国家标准全文公开系统（在线预览）"),
    ("GB/T 18316-2008", "数字测绘成果质量检查与验收",
     "https://openstd.samr.gov.cn/bzgk/std/index", "国家标准全文公开系统（在线预览）"),
    ("GB/T 17941-2008", "数字测绘成果质量要求",
     "https://openstd.samr.gov.cn/bzgk/std/index", "国家标准全文公开系统（在线预览）"),
    ("GB/T 19710.1-2023", "地理信息 元数据 第1部分：基础",
     "https://openstd.samr.gov.cn/bzgk/std/index", "国家标准全文公开系统（在线预览）"),
    ("GB/T 21010-2017", "土地利用现状分类",
     "https://openstd.samr.gov.cn/bzgk/std/index", "国家标准全文公开系统（在线预览）"),
    ("GB/T 20258.1-2019", "基础地理信息要素数据字典 第1部分：1:500 1:1000 1:2000",
     "https://openstd.samr.gov.cn/bzgk/std/index", "国家标准全文公开系统（在线预览）"),
    ("GB/T 13989-2012", "国家基本比例尺地形图分幅和编号",
     "https://openstd.samr.gov.cn/bzgk/std/index", "国家标准全文公开系统（在线预览）"),
    ("GB/T 17798-2007", "地理空间数据交换格式",
     "https://openstd.samr.gov.cn/bzgk/std/index", "国家标准全文公开系统（在线预览）"),
    ("GB/T 14911-2008", "测绘基本术语",
     "https://openstd.samr.gov.cn/bzgk/std/index", "国家标准全文公开系统（在线预览）"),
    ("GB 50026-2020", "工程测量标准",
     "https://www.mohurd.gov.cn/gongkai/zc/wjk/art/2021/art_17339_761396.html", "住建部公告（工程建设标准全文公开系统）"),
    ("CJJ/T 8-2011", "城市测量规范",
     "https://www.mohurd.gov.cn/", "住建部（工程建设标准全文公开系统）"),
    ("CH/T 1004-2005", "测绘技术设计规定",
     "http://www.nrsis.org.cn/portal/xxcx/std", "自然资源标准化信息服务平台（在线阅览）"),
    ("CH/T 1001-2005", "测绘技术总结编写规定",
     "http://www.nrsis.org.cn/portal/xxcx/std", "自然资源标准化信息服务平台（在线阅览）"),
    ("CH/T 9008.1-2010", "基础地理信息数字成果 1:500 1:1000 1:2000 数字线划图",
     "http://www.nrsis.org.cn/portal/xxcx/std", "自然资源标准化信息服务平台（在线阅览）"),
    ("CH/T 9008.2-2010", "基础地理信息数字成果 1:500 1:1000 1:2000 数字高程模型",
     "http://www.nrsis.org.cn/portal/xxcx/std", "自然资源标准化信息服务平台（在线阅览）"),
    ("CH/T 9008.3-2010", "基础地理信息数字成果 1:500 1:1000 1:2000 数字正射影像图",
     "http://www.nrsis.org.cn/portal/xxcx/std", "自然资源标准化信息服务平台（在线阅览）"),
    ("CH/T 9008.4-2010", "基础地理信息数字成果 1:500 1:1000 1:2000 数字栅格地图",
     "http://www.nrsis.org.cn/portal/xxcx/std", "自然资源标准化信息服务平台（在线阅览）"),
    ("TD/T 1055-2019", "第三次全国国土调查技术规程",
     "https://gi.mnr.gov.cn/201902/t20190220_2396074.html", "自然资源部（官方发布/购买渠道）"),
    ("自然资发〔2023〕31号", "实景三维中国建设总体实施方案（2023—2025年）",
     "https://www.mnr.gov.cn/", "自然资源部（按文号检索；已收录配套技术大纲）"),
    ("自然资发〔2020〕15号", "自然资源调查监测体系构建总体方案",
     "https://www.gov.cn/zhengce/zhengceku/2020-01/18/content_5470398.htm", "已自动收录附件全文"),
]


def download(url: str, dest: Path, timeout: int = 180) -> int:
    """用 Windows 自带 curl 下载（走系统证书链）。"""
    if not CURL:
        raise RuntimeError("未找到 curl，请确认系统已安装 curl.exe")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    if tmp.exists():
        tmp.unlink()
    cmd = [CURL, "-s", "-L", "--fail", "--retry", "2",
           "--max-time", str(timeout), "-o", str(tmp), url]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        raise RuntimeError(f"curl 失败 (code={result.returncode}): {result.stderr.strip()[:200]}")
    tmp.replace(dest)
    return dest.stat().st_size


def decode_html(raw: bytes) -> str:
    """政府网站编码不统一，按中文字符数量挑最合适的解码方式。"""
    best, best_score = "", -1
    for enc in ("utf-8", "gb18030"):
        try:
            text = raw.decode(enc)
        except UnicodeDecodeError:
            continue
        score = len(re.findall(r"[一-鿿]", text))
        if score > best_score:
            best, best_score = text, score
    return best


def extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    candidates = []
    for sel in SELECTORS:
        candidates.extend(soup.select(sel))
    if candidates:
        node = max(candidates, key=lambda n: len(n.get_text(strip=True)))
    else:
        nodes = soup.find_all(["div", "td", "article", "main"])
        node = max(nodes, key=lambda n: len(n.get_text(strip=True))) if nodes else soup

    for tag in node(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    noise_prefixes = (
        "当前位置", "编辑：", "编 辑：", "责编：", "责 编：",
        "<< 返回首页", "相关文章", "扫一扫", "打印本页", "关闭窗口", "网站地图",
    )
    lines = []
    for line in node.get_text("\n", strip=True).splitlines():
        line = re.sub(r"[ \t　]+", " ", line).strip()
        if not line or line == "首页" or line.startswith(noise_prefixes):
            continue
        lines.append(line)
    return "\n".join(lines)


def convert_doc_to_docx(src: Path, dst: Path) -> None:
    """用本机 Word 把 .doc 转成 .docx（政府网站常提供旧版 .doc 附件）。"""
    import win32com.client as win32

    word = win32.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    try:
        doc = word.Documents.Open(str(src.resolve()), ReadOnly=True)
        try:
            doc.SaveAs2(str(dst.resolve()), FileFormat=16)  # wdFormatXMLDocument
        finally:
            doc.Close(False)
    finally:
        word.Quit()


def inspect_pdf(path: Path):
    """返回 (页数, 前10页可提取字符数)，失败返回 (None, 0)。"""
    try:
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            pages = len(pdf.pages)
            text = "".join((p.extract_text() or "") for p in pdf.pages[:10])
        return pages, len(text)
    except Exception:
        return None, 0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_standard_metadata(queries):
    records = []
    for query in queries:
        url = ("https://openstd.samr.gov.cn/bzgk/std/std_list?"
               "p.p1=0&p.p2=" + urllib.parse.quote(query) +
               "&p.p90=circulation_date&p.p91=desc")
        tmp = DATA_DIR / "metadata" / f".search_{query.replace('.', '_')}.html"
        try:
            download(url, tmp, timeout=40)
            html = decode_html(tmp.read_bytes())
            soup = BeautifulSoup(html, "html.parser")
            for tr in soup.find_all("tr"):
                if "showInfo" not in str(tr):
                    continue
                cells = [c.get_text(" ", strip=True) for c in tr.find_all("td")]
                if len(cells) < 7:
                    continue
                # 同一标准可能有外文版（显示 EN），优先收录中文版
                if "EN" in cells:
                    continue
                m = re.search(r"showInfo\('([0-9A-F]+)'", str(tr))
                no, title, status = cells[1], cells[4], cells[6]
                if "现行" not in status:
                    continue
                if not no.startswith("GB/T"):
                    continue
                base = re.sub(r"-\d{4}$", "", query)
                if not no.startswith("GB/T " + base):
                    continue
                rec = {
                    "query": query,
                    "standard_no": no,
                    "title": title,
                    "status": status,
                    "publish_date": cells[7][:10] if len(cells) > 7 else "",
                    "implement_date": cells[8][:10] if len(cells) > 8 else "",
                    "hcno": m.group(1) if m else "",
                }
                if m:
                    rec["detail_url"] = ("https://openstd.samr.gov.cn/bzgk/std/"
                                         "newGbInfo?hcno=" + m.group(1))
                records.append(rec)
        except Exception as e:
            records.append({"query": query, "error": str(e)[:200]})
        finally:
            if tmp.exists():
                tmp.unlink()
        time.sleep(0.6)
    return records


def write_metadata(records):
    meta_dir = DATA_DIR / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # 去重（同一标准号只留一条）
    seen, unique = set(), []
    for r in records:
        key = r.get("standard_no") or r.get("query") + "|error"
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)

    (meta_dir / "gb_standards.json").write_text(
        json.dumps(unique, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 国家标准元数据（现行版本）",
        "",
        "来源：国家标准全文公开系统（openstd.samr.gov.cn），抓取日期 "
        + datetime.now().strftime("%Y-%m-%d") + "。",
        "",
        "| 标准号 | 标准名称 | 状态 | 实施日期 | 官方详情 |",
        "|---|---|---|---|---|",
    ]
    for r in unique:
        if r.get("error"):
            lines.append(f"| {r['query']} | 查询失败：{r['error']} | - | - | - |")
            continue
        link = r.get("detail_url", "https://openstd.samr.gov.cn/bzgk/std/index")
        lines.append(
            f"| {r['standard_no']} | {r['title']} | {r['status']} | "
            f"{r.get('implement_date', '')} | [详情]({link}) |")
    (meta_dir / "standards_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return unique


def write_manual_list():
    lines = [
        "# 需要手动获取的标准",
        "",
        "以下标准受版权保护，官方只提供在线预览或付费购买，未自动下载。",
        "需要全文时，按“获取渠道”自行下载，放到 `data/standards/` 即可被 RAG 自动收录。",
        "",
        "| 标准号 | 标准名称 | 获取渠道 | 说明 |",
        "|---|---|---|---|",
    ]
    for no, title, url, note in MANUAL_ITEMS:
        lines.append(f"| {no} | {title} | [链接]({url}) | {note} |")
    (DATA_DIR / "MANUAL_DOWNLOAD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(manifest):
    ok = [r for r in manifest["sources"] if r["status"] == "ok"]
    fail = [r for r in manifest["sources"] if r["status"] != "ok"]
    lines = [
        "# data 语料说明",
        "",
        f"生成时间：{manifest['generated_at']}",
        "",
        "## 目录",
        "",
        "- `regulations/` 法律法规、部门规章（政府网站公开全文）",
        "- `guidelines/` 技术大纲、实施细则（公开 PDF）",
        "- `standards/` 标准全文（需手动放入，见 `MANUAL_DOWNLOAD.md`）",
        "- `metadata/` 现行国家标准元数据（来自国家标准全文公开系统）",
        "- `projects/` 预留：你自己的项目文档、技术设计书、质检细则",
        "",
        f"本次成功下载 {len(ok)} 个文件，失败 {len(fail)} 个。",
        "",
        "## 来源说明",
        "",
        "只收录公开可获取的政府文件和公开技术资料。标准正文的版权归发布机构所有；",
        "未自动抓取仅提供在线预览或付费购买的标准，完整清单见 `MANUAL_DOWNLOAD.md`。",
        "",
        "## 重新抓取",
        "",
        "```",
        "python tools/fetch_corpus.py",
        "```",
    ]
    (DATA_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="只抓取指定 id，逗号分隔")
    parser.add_argument("--skip-metadata", action="store_true", help="跳过国家标准元数据")
    args = parser.parse_args()

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    sources = [s for s in SOURCES if not only or s["id"] in only]

    for sub in ("regulations", "guidelines", "standards", "metadata", "projects"):
        (DATA_DIR / sub).mkdir(parents=True, exist_ok=True)

    manifest = {"generated_at": datetime.now().isoformat(timespec="seconds"), "sources": []}

    for src in sources:
        dest = DATA_DIR / src["category"] / src["filename"]
        entry = {
            "id": src["id"], "title": src["title"], "category": src["category"],
            "kind": src["kind"], "file": str(dest.relative_to(DATA_DIR)).replace("\\", "/"),
            "source_url": src["url"], "source": src["source"],
        }
        print(f"[下载] {src['id']} -> {entry['file']}")
        try:
            if src["kind"] == "doc":
                tmp_doc = dest.with_suffix(".doc")
                download(src["url"], tmp_doc)
                convert_doc_to_docx(tmp_doc, dest)
                tmp_doc.unlink(missing_ok=True)
            else:
                download(src["url"], dest)

            if not dest.exists() or dest.stat().st_size == 0:
                raise RuntimeError("下载结果为空")
            entry.update({"status": "ok", "bytes": dest.stat().st_size,
                          "sha256": sha256(dest)})

            if src["kind"] == "html":
                text = extract_main_text(decode_html(dest.read_bytes()))
                if len(text) < 300:
                    raise RuntimeError(f"正文提取过短（{len(text)} 字符），可能是动态页面")
                header = (
                    f"# {src['title']}\n\n"
                    f"- 来源：{src['source']}\n"
                    f"- URL：{src['url']}\n"
                    f"- 抓取日期：{datetime.now().strftime('%Y-%m-%d')}\n\n---\n\n"
                )
                dest.write_text(header + text, encoding="utf-8")
                entry["bytes"] = dest.stat().st_size
                entry["text_chars"] = len(text)
                entry["sha256"] = sha256(dest)
            elif src["kind"] == "pdf":
                head = dest.read_bytes()[:5]
                if head != b"%PDF-":
                    raise RuntimeError("不是有效的 PDF")
                pages, chars = inspect_pdf(dest)
                entry.update({"pages": pages, "text_chars": chars,
                              "needs_ocr": bool(pages and chars < 100)})
            elif dest.suffix.lower() == ".docx" and dest.read_bytes()[:2] != b"PK":
                raise RuntimeError("不是有效的 docx")
            print(f"        OK {entry['bytes']} bytes")
        except Exception as e:
            entry.update({"status": "failed", "error": str(e)[:300]})
            print(f"        失败: {e}")
        manifest["sources"].append(entry)

    metadata = []
    if not args.skip_metadata:
        print("[元数据] 查询国家标准现行版本...")
        metadata = write_metadata(fetch_standard_metadata(GB_QUERIES))
        print(f"        记录 {len(metadata)} 条")

    write_manual_list()
    write_readme(manifest)
    (DATA_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    ok = sum(1 for r in manifest["sources"] if r["status"] == "ok")
    print(f"\n完成：{ok}/{len(manifest['sources'])} 个文件下载成功")
    print(f"清单：{DATA_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
