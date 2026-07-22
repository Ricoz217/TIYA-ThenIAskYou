---
name: websearch
description: 内置 SKILL，提供 WebSearch(网络搜索)、Extract(Crawl/爬虫)、UrlMap(抓取子链接)、 Raw_Fetch(获取原始内容) 的能力
origin: TIYA
version: 0.1.0
metadata:
  category: builtin
  owner: agent-core
---
# Usage

- 当需要 "WebSearch" 相关功能时使用本SKILL。

# 特别提醒  

为避免 WebSearch 阻塞AGENT主控请求，本工具的SKILL提供的工具不会等待结果返回。  
若下次主控请求时 WebSearch 仍无结果，则统一返回任务运行中提示，并在得到结果后通过callback返回给AGENT。  
所有工具返回的结果，无论请求 **成功与否** 均会返回一个 `标准搜索结果` 的列表。结构如下:  

```
{
  "ok": boolean,                               # 标记请求是否成功
  "result_from": enum["CACHE", "SEARCH"],      # 结果来源: 本地缓存 或 在线请求
  "query": string,                             # 请求的目标: 自然语言query 或 目标url
  "answer": string,                            # 针对search: API提供的AI聚合回复
  "results": object,                           # 具体结果: 由具体工具决定
  "error_message": string,                     # 错误提示信息
  "create_time": string,                       # 结果创建时间
}
```

仅在工具执行发生错误时才会抛出异常，因此需要根据结果的 `ok` 项判断请求是否成功  

# 附属工具说明

`websearch` 提供 4 个工具:  

1. `search`
2. `url_map`
3. `extract`
4. `raw_crawl`

其中 1~3 为爬虫平台API实现，4为本地直接fetch

## 1) search  

函数签名
`async def search(self, query: str, include_domains: list[str] = None, purge: bool = False) -> list[dict]`

> 智能搜索 `query` ，返回符合要求的 url 和该 url 的内容概况

参数:  

- `query`: 要查询的内容，用自然语言表示，优先使用英语。
- `include_domains`: 筛选符合指定域名的结果，可包括多个域名。仅当明确要求时使用，否则 **不应** 传入该参数。示例:  
  1. 默认搜索、不限制域名: 不传入 `include_domains`(即`include_domains`=None)  
  2. 对话: "你没看过我视频？我在B站，叫MiraiYukari": 传入参数 `query="MiraiYukari", include_domains=["bilibili.com"]`
  3. 对话: "在github上面找到个工具xxx": 传入参数 `query="xxx", include_domains=["github.com"]`
- `purge`: 是否调用缓存，默认为False。当需要搜索具有时效性、发展中的技术、随时间更新的内容时，显式传入True。解释如下:  
  1. False: 尝试根据 `query` 获取缓存结果，若无则自动进行API请求  
  2. True: 不查询缓存，强制进行API请求

返回：  
- 成功/失败: 一个字典列表，为 `标准搜索结果` 的列表

## 2) url_map  

函数签名:  
`async def url_map(self, url: str, purge: bool = False) -> list[dict]`

> 智能遍历递归分析给定 `url` 包含的子链接。当搜索技术文档、维基词条可选择调用工具分析子链接。若 `search` 返回的信息已满足任务需求则 **不应** 使用  

参数:  

- `url`: 要分析的目标 URL  
- `purge`: 是否调用缓存，默认为False。当链接具有较强时效性(例如开发中项目的技术文档、新闻、商品、期刊等)时，显式传入True。解释如下:  
  1. False: 尝试根据 `url` 获取缓存结果，若无则自动进行API请求  
  2. True: 不查询缓存，强制进行API请求

返回：  
- 成功/失败: 一个字典列表，为 `标准搜索结果` 的列表

## 3) extract  

函数签名:  
`async def extract(self, urls: list[str] | str, query: str, purge: bool = False) -> list[dict]`

> 智能爬取指定urls的内容，支持一次爬取多个  

参数:  

- `urls`: 需要爬取的urls列表，仅爬取一个时可直接输入string  
- `query`: 可选项。需要爬取的具体内容，自然语言表示，优先使用英语:  
  1. 不传入: 爬取网页所有内容，尽可能返回全部内容  
  2. 传入: 返回结果会根据传入的内容重排序，符合 `query` 的靠前  
- `purge`: 是否调用缓存，默认为False。当链接具有较强时效性(例如开发中项目的技术文档、新闻、商品、期刊等)时，显式传入True。解释如下:  
  1. False: 尝试根据 `url` 获取缓存结果，若无则自动进行API请求  
  2. True: 不查询缓存，强制进行API请求

返回：  
- 成功/失败: 一个字典列表，为 `标准搜索结果` 的列表  

## 4) raw_crawl  

函数签名:  
`async def raw_crawl(self, url: str, extra_parameters: dict = None,  purge: bool = False) -> list[dict]`

> 本地直接 `get` 指定url，返回 `headers` 和 `HTML text` ，为后备方案，仅在 `extract` 失败时可选使用。
> 由于会返回全文原始内容，因此会消耗大量token和context窗口

参数:  

- `url`: 目标URL
- `extra_parameters`: httpx支持的参数，例如 `cookies`, `headers` 等，可选项，默认为空。当爬取失败时可根据返回的信息自行构建合适的参数
- `purge`: 是否调用缓存，默认为False。

返回：  
- 成功/失败: 一个字典列表，为 `标准搜索结果` 的列表  

# 使用条件  

## 0) 大前提  

本条目 **高于** 以下任何规则，具体规则如下:  
1. 在执行任何 `websearch` 功能前，应 **至少** 尝试一次直接回复，再根据用户反馈/回复与任务匹配度决定是否请求搜索  
2. 节省API用量很重要，任何搜索都应 **精练、切中要点** ，尽可能用少数搜索请求获取满足任务的信息  
3. 不得虚构搜索结果，若你的答复引用了搜索结果，应附上相应的来源URL  
4. 尽可能利用缓存  
5. 若用户明确要求搜索任务，可直接请求  

## 1) search  

### 应当使用

1. 不能理解的术语、专属名词、人名/角色名、新技术、特殊领域
2. 能理解但了解不深、需要外部知识补充的内容
3. 带有时效性的新闻、时政、世界局势、期刊、实时数据
4. 带有更新性质的持续性内容，例如: 游戏新版本、mod/程序新版本、活跃开发中的技术文档、娱乐话题
5. 用户明确要求进行搜索时

### 不应使用

1. 你已深刻理解的问题、内容，例如: 简单的代码编写、稳定的项目、成熟的技术栈、日常生活的常识
2. 普通对话、不包含明确的任务  
3. 对话中的昵称、用户称谓、玩笑或幽默打趣内容
4. 明显不合理或胡编乱造的内容，例如: 如何摘月亮、如何毁灭世界
5. 需要获取完整或详细信息的任务，不应依赖 `search` ，而应通过 `search` 获取url后 `extract` url

## 2) url_map  

### 应当使用  

1. 在处理搜寻文档、维基能具有明显结构树的任务时，获得主链接后分析子项链接  
2. 产品总览页面、介绍页面等任务需要的链接不在主链接时
3. `search` 返回的内容和回答较为笼统、不能满足任务要求时

### 不应使用

1. `search` 返回的内容和回答已能满足任务需求
2. `youtube`、`bilibili`、`tiktok` 等流媒体或多媒体平台
3. `qq`、`twitter(x)`、`weibo`等社交媒体平台
4. 主链接不具有明显的结构树内容

## 3) extract

### 应当使用

1. 学术期刊、技术文档、复杂问题等需要获取完整详细信息时
2. `search` 返回的内容和回答较为笼统、不能满足任务要求时
3. 用户明确指定链接或明确要求时

### 不应使用
1. `search` 返回的内容和回答已能满足任务需求
2. 仅作为知识面补充，不需要处理深层次问题时
3. 链接的概括明显不符合任务要求

## 4) raw_crawl

### 应当使用
1. `extract` 失败、无返回内容，根据任务需要可尝试获取URL原始信息
2. 执行网络调试任务时，可调用获取更加完整和原始的信息

### 不应使用
1. `extract` 成功，且返回的信息足以完成任务
2. 图片、视频、文件等非网页(HTML)链接

# 推荐使用方法

1. 使用 `search` 搜索需要的信息，一般情况下尽量使用缓存
2. 判断 `search` 返回的聚合回答和内容是否满足任务需求，若满足则结束WebSearch
3. 若不满足，可尝试 `extract` 或 `url_map` 获取更多信息
4. 若 `extract` 可根据情况选择使用 `raw_crawl` 获取url原始信息