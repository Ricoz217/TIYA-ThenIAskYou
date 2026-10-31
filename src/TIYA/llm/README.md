# LLM 通信接口

业务继续从 `TIYA.LLM_connect` 导入 `Chat`、`Context`、Prompt、`ChatConfig`、`parse_llm_setting()`；这些入口重新导出内部的同一组类型。配置文件仅支持下文的 v2 字典结构，旧列表配置会提示人工填写新版，需与代码一起部署。

兼容范围是调用接口、返回契约和业务行为。请求使用 Runtime → Provider → Gateway 链路。`get_llm()` 返回原始配置快照；`parse_llm_setting()` 通过该入口获取预设，再解释为供 Chat 使用的结构化配置。旧私有 `_payload_constructor()` 不再提供。

## 预设数据和 ChatConfig

`TIYA.config.get_llm(name)` 返回简单的数据对象 `LLMPresetConfig`：`configuration` 保存完整原始 LLM 配置快照，`data` 指向选中的预设，`provider_definitions` 是原始 Providers 字典，`providers` 是原始有序候选列表。开放字段完整保留，修改快照不会影响全局配置。

它支持字典读取；`max_context` 和 `extra_parameter` 属性分别读取首个候选的原始 `max_context` 和 `llm_params`，缺失返回 None，不继承连接默认值。`preset_name`、`auto_compress_rate`、`proxy_mode` 也可直接读取。不在这里合成 model、endpoint、凭证等需要业务解释的字段，也不随会话 fallback 改写全局配置。

根 `TIYA/config.py` 负责文件读写、注释、结构校验和上述简单读取，不解析 Provider 引用、不导入 `TIYA.llm`。下游 `llm/config.py` 负责连接查找、协议地址、凭证和参数覆盖；`llm/settings.py` 再转换为 ChatConfig。不存在 LLM 配置迁移函数或命令，旧配置必须人工重新填写。

`parse_llm_setting(name)` 返回独立的 `ChatConfig`。类型定义及字段注释位于 `config_model.py`，由 `model.py` 和 `LLM_connect.py` 重新导出：

- `ChatConfig.candidates`：有序的 `LLMConfig` 列表，第一项是首选。
- `LLMConfig`：协议分类、连接名、完整 endpoint、凭证、模型、原生参数、HTTP 选项、窗口、历史策略及关联 `FilesConfig`。
- `RequestPolicy`：生成尝试总预算、单候选重试上限和 fallback 冷却时间；取消、部分输出和远端执行不明的停止规则不能用这些设置绕过。
- `provider_options`：开放连接节点中交给 Provider 解释的选项，内部只存 JSON 数据；YAML 不需要新增同名包装字段。

`ChatConfig.model/endpoint/token/api_provider/max_context/model_params/client_params` 是首选候选的普通属性。`model_params` getter 返回该候选的实际字典，修改会生效；setter 只替换首选参数。Chat 接收配置时复制可变内容，logger 保持原引用。

```python
from TIYA.LLM_connect import Chat, ChatConfig, LLMConfig, parse_llm_setting

settings = parse_llm_setting("MAIN")
settings.model_params["stream"] = False  # 修改首选，备用仍使用自己的参数
settings.keep_alive = True
chat.setting(settings)

# 显式局部更新，不替换候选列表；Agent 使用这个入口。
chat.update_config(model_params={"stream": False}, keep_alive=False)

# Memory 原有的局部构造仍可用，源码无需改动。
chat.setting(ChatConfig(model_params={"stream": False}, keep_alive=False))

# 直接构造完整单模型配置，使用显式候选类型。
settings = ChatConfig([LLMConfig(
    provider_type="chat_completions", connection_name="direct",
    endpoint="https://example.invalid/v1/chat/completions", token="YOUR_KEY",
    model="YOUR_MODEL_ID", model_params={"stream": False},
)])
```

`candidates=None` 表示局部更新，非空列表替换整套候选配置，空列表报错。完整替换不会继承旧预设的 Files、价格、参数或策略；logger/keep_alive/log_off 等会话选项只在明确提供时更新。没有 `preset: Any` 或 `complete` 标记，也不再拼接第二个内部预设对象。旧 `Chat(...)` 单模型构造参数仍可用；完整 `ChatConfig(endpoint=..., model=...)` 构造改用上述候选列表形式，仓库业务没有这种旧式完整构造调用。

## 配置 v2

```yaml
LLM模型配置:
  schema_version: 2
  Providers:
    responses_api:
      primary:  # 任意连接名，不用于猜测供应商
        endpoint: https://api.openai.com/v1/responses
        key: YOUR_KEY
        responses:
          supports_previous_response_id: true
    chat_completions:
      backup:
        base_url: https://example.invalid/v1
        key: YOUR_KEY
    files_api:
      images:
        endpoint: https://api.openai.com/v1/files
        key: YOUR_KEY
        expires_after_seconds: 86400
  LLM_List:
    MAIN:
      auto_compress_rate: 0.7
      proxy_mode: GlobalProxy
      providers:
        - llm: "responses_api:primary:YOUR_MODEL_ID"
          max_context: 128000
          context_history_mode: auto
          files_api: "files_api:images"
          llm_params:
            stream: true
        - llm: "chat_completions:backup:YOUR_BACKUP_MODEL_ID"
          max_context: 64000
          llm_params:
            stream: true
```

分类由代码注册表选择实现，连接名任意。模型引用只拆前两个冒号，剩余部分原样作为模型 ID。未知连接配置保留且忽略；`llm_params` 是目标接口原生请求参数，未知参数也会转发，系统不把 `max_token` 自动改成其他名称。运行时管理的 input/messages/tools/父 response 字段不能通过此字典覆盖。

连接的 `llm_params` 提供默认值，预设同名参数覆盖它；嵌套值整体替换。完整 `endpoint` 优先于 `base_url`。`base_url` 应包含需要的版本前缀，例如 `/v1`，实现只追加操作路径。

`context_history_mode: full` 始终回放当前完整业务历史；`auto` 只在明确声明支持续接时使用父 ID。未知网关不因返回 ID 就自动启用续接。显式 `store: false` 关闭续接。支持续接时默认请求加密 reasoning，部分兼容接口可通过 `responses.include_encrypted_reasoning: false` 关闭；也可在原生 `llm_params.include` 中显式指定所需字段。

Files 是独立能力，有配置且模型协议支持图片文件引用时优先使用；失败恢复有界，明确不支持时回到内联。模型协议差异使用以下已知字段：

| 场景 | 连接设置 |
| --- | --- |
| DeepSeek Chat 图片 file 内容块 | 模型连接：`chat_completions: {image_file_format: deepseek}` |
| DeepSeek Files 表单和结果 | Files 连接：`dialect: deepseek` |
| Anthropic Files | Files 连接：`dialect: anthropic`；普通 Messages 保持现有版本 |
| 需要旧 Files beta 头的接口 | Files 连接显式设置 `anthropic_beta` |
| 供应商错误码 | 模型连接：`error_dialect: openai/deepseek/anthropic/glm/minimax` |

不靠连接名或域名选择这些设置。官方 OpenAI Chat 图片仍使用 URL/Base64，配置 Files 本身不会把它改为图片 file_id；Responses 可使用图片文件引用。[OpenAI Files](https://developers.openai.com/api/reference/resources/files/methods/create)、[DeepSeek Files](https://api-docs.deepseek.com/guides/files_api/)、[Anthropic Files](https://platform.claude.com/docs/en/build-with-claude/files) 分别定义了各自的用途、过期参数和返回格式。

安全拒绝后才会有限重试并切换已配置的下一组合。人工取消、部分输出、远端结果不明均停止生成重发。备用组合成功提交后，会话进入默认一小时冷却；普通成功不续期，到期由下一次业务请求尝试首选，并记录进入/恢复日志。

## 独立图片生成和编辑

```python
from TIYA.LLM_connect import ImagesClient, ImageGenerateRequest, ImageEditRequest, MediaRef

settings = {
    "endpoint": "https://api.openai.com/v1/images/generations",
    "edit_endpoint": "https://api.openai.com/v1/images/edits",
    "key": "YOUR_KEY",
}

async with ImagesClient(settings) as images:
    response = await images.generate(ImageGenerateRequest(
        model="YOUR_IMAGE_2_5_MODEL_ID", prompt="生成一张图片", output_format="png",
    ))
    cache_id = response.data[0].image.value  # kind == "cache"
    edited = await images.edit(ImageEditRequest(
        model="YOUR_IMAGE_2_5_MODEL_ID", prompt="根据参考图修改目标图的局部",
        images=[MediaRef("path", "target.png"), MediaRef("path", "reference.png")],
        mask=MediaRef("path", "mask.png"),
    ))
```

模型 ID 原样发送。生成和编辑分别使用明确的 endpoint，不从生成 URL 猜编辑 URL；配置了 `base_url` 时可按标准路径追加。默认编辑走 multipart；`images: {edit_format: json}` 启用 JSON 的 images/mask 引用。参考图保持顺序，mask 对应第一张目标图，必须保持 PNG alpha 且尺寸一致。不会把编辑降级成纯文字生成，也不调用 Responses 内置生图工具。

最终 URL/Base64 图片在通信层下载、验证并进入现有 `file_cache` 后才返回。业务获得缓存 ID、MIME 和可用尺寸。没有永久保留或新的清理机制。`MediaError.stage` 区分下载/解码和验证/缓存失败；错误中保留 request_id，可调用 `images.recover_images(error)` 只重试接收落盘，不重复生图。

`FilesClient(settings)` 提供 `upload(MediaRef)`、`retrieve(id)`、`list(params=...)`、`delete(id)`、`download(id)`，可单独使用。下载能力仍以目标服务及具体文件为准。传入共享 httpx client 时，Files 使用自己的显式请求头，不继承模型密钥。

`ImagesClient(settings, provider_ref="image_generation:连接名")` 可沿用配置树中的连接引用；`settings` 仍须显式传入该连接节点，`provider_ref` 选择协议实现并标识定价作用域，不自动读取全局配置。默认使用 `image_generation:direct`。`images.pricing_identity(model, operation="images_generate", preset="")` 返回实际使用的定价绑定。生成与编辑分别使用 images_generate/images_edit。流式预览回调接收的也是已缓存的 `ImageResponseItem`，预览不会计为最终图片。

## 状态与记账

- 文件内容和 TTL：现有 `data/file_cache`。
- 上传映射、能力失败证据、检查点和会话冷却：`data/llm/runtime.sqlite3`。
- 请求账本：`data/token_usage/usage.sqlite3`。
- 独立定价目录：`data/llm/pricing.sqlite3`。
- 业务历史和原生回放项：原有 Context 存档。读取旧存档不要求新增状态存在。

账本每次真实尝试分配独立 request_id，同一逻辑调用共享 call_id；平台 request ID、response ID 分开记录。先持久化初始估算，再发送网络请求。reported 与 estimated 分别保存，effective 逐字段优先采用 reported。缺失真实值不填成真实零值。观察幂等，费用修订不会增加请求数量；确认未执行的拒绝保留初始估算证据，并以新的估算观察注明拒绝状态。

`Ledger` 提供 begin_call/begin_request/observe_usage/finish_request、分页 query_requests/query_observations 和 summarize。旧 `LLM_usage` 的日志和统计入口通过适配器读取 SQLite。JSON 仅首次备份并导入历史聚合，不将历史聚合伪造成逐请求记录，也不重复累计 last_record。

定价通过 `PriceCatalog.publish(identity, rules, source=...)` 维护，不写回连接 YAML。文本请求 identity 使用已转换配置中的实际候选：

```python
settings = parse_llm_setting(name)
candidate = settings.candidates[index]
identity = {"preset": name, "scope": candidate.scope, "model": candidate.model, "operation": candidate.provider_type}
rules = {
    "currency": "CNY", "tier_mode": "whole", "threshold": "input",
    "units": {"input_uncached": 1000000, "cached": 1000000, "output": 1000000},
    "tiers": [
        {"up_to": 128000, "rates": {"input_uncached": "1", "cached": "0", "output": "2"}},
        {"rates": {"input_uncached": "2", "cached": "0", "output": "4"}},
    ],
}
catalog.publish(identity, rules, source="经核对的价格来源")
```

以上数字只是示例。whole 按完整输入所处区间选择整次请求价格；progressive 按同一个计量项逐段计费。输入/输出总量与子项不能重复定价。可使用 images、input_image、output_image 等计量项；没有匹配价格时费用未知，仍然记账。每次请求冻结价格版本，后续调价不改旧请求。异步调用方通过 `llm.workers.run_sync()` 执行同步维护/查询，避免阻塞事件循环。

## 线程资源与 Runtime 阅读入口

LLM 中原先通过 `to_thread` / `run_in_executor` 卸载的同步 SQLite、token 估算和图片处理，现统一走 `llm/workers.py` 的 `run_sync()`，使用 `TIYA/executor.py` 中的进程级 `LLM_EXECUTOR`，默认最多 4 个工作线程，线程名前缀为 `llm`。线程数在现有全局线程池管理文件中集中调整，无需修改连接配置。所有 Chat、FilesClient 和 ImagesClient 共用这个池；HTTP 仍为异步收发，file_cache 使用它已有的独立线程池。

`run_sync()` 保留调用协程的 contextvars、结果和异常。取消等待可取消尚未开始的工作，但不能杀掉已运行的同步函数。单个 Chat/Client 关闭只负责自己的任务和 HTTP 连接；项目现有 `executor.shutdown_all()` 在业务停止后同步等待 LLM 池收尾，包括已开始的账本写入。

```python
from TIYA.llm.workers import run_sync

rows = await run_sync(ledger.query_requests, preset="MAIN", limit=100)
```

Runtime 的 `execute()` 是主调度入口，只制作 Context 快照并按有序候选执行。参数和提交边界直接写在函数文档中，主要步骤如下：

- `_prepare_candidate()`：窗口检查、连接获取、选择匹配的历史检查点。
- `_execute_candidate()`：单候选的有界尝试，准备将实际发送的媒体。
- `_begin_attempt()` / `_send_attempt()`：冻结价格并先落估算，再执行一次生成收发。
- `_complete_attempt()`：保留真实用量、缓存返回图片，构造 `Pending`。
- `_record_failure()` / `_cancel_background()` / `_recover()`：保留失败证据；仅确认未执行才恢复，取消/部分输出/远端未知直接停止。
- `commit_async()`：由业务持有者校验原 Context 未变化后消费结果，随后保存检查点和非滑动冷却。

`Pending`、逻辑调用、单候选和单次发送状态分别有明确的结构定义，内部执行状态通过这些结构传递。Provider 的 `ConversationProvider` / `StreamDecoder` 契约位于 `provider/base.py`；Runtime 不依赖某家解码器的具体实现类。主调用、网络、媒体、账本和定价接口均有参数类型及必要的副作用说明。

## 已修正的旧行为

取消和超时不会提交未完成结果；队列与 no_wait 共用提交逻辑；损坏的工具 JSON 不会变成空参数后执行；分阶段 usage 合并且失败仍记账；不再根据历史长度伪造真实缓存命中。Context/Prompts 副本隔离内容，工具绑定的业务对象保持原引用；完整切换预设清除旧 Files、价格和路由设置。实际 HTTP 请求不再使用旧的首次 SSE 盲目重发逻辑。

旧入口返回 None 时，调用协程可读取 `Chat.last_outcome` 的 status/retry_allowed/call_id/request_id/remote_state。Runtime 已完成有界重试后的终态不允许上层再盲目重发；项目内 Agent 已接入这一规则。自定义调用方应同样检查终态。跨模型 fallback 时 Context 的旧单模型 usage 投影切换到实际模型，长期账本保留各模型全部用量和费用。

普通请求断开连接只代表本地停止，远端是否停止保留 unknown。后台 Responses 必须显式声明 `responses.supports_background`；已有任务 ID 时尝试协议取消并记录确认状态，不把断流当作后台任务已停止。
