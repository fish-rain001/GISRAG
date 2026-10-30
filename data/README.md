# data 语料说明

生成时间：2026-10-28T19:35:19

## 目录

- `regulations/` 法律法规、部门规章（政府网站公开全文）
- `guidelines/` 技术大纲、实施细则（公开 PDF）
- `standards/` 标准全文（需手动放入，见 `MANUAL_DOWNLOAD.md`）
- `metadata/` 现行国家标准元数据（来自国家标准全文公开系统）
- `projects/` 预留：你自己的项目文档、技术设计书、质检细则

本次成功下载 13 个文件，失败 0 个。

## 来源说明

只收录公开可获取的政府文件和公开技术资料。标准正文的版权归发布机构所有；
未自动抓取仅提供在线预览或付费购买的标准，完整清单见 `MANUAL_DOWNLOAD.md`。

## 重新抓取

```
python tools/fetch_corpus.py
```
