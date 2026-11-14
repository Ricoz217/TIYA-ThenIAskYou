# LLM 通信接口

业务从 `TIYA.LLM_connect` 导入 `Chat`、`Context`、各类 Prompt、`ChatConfig`、`ImageGen` 和 `parse_llm_setting()`。配置使用下文的 v2 字典结构。旧列表配置需要人工改写并与代码一起部署。

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

## 复制语义

`Context`、`Prompts`、`TokenUsage`、`ChatConfig` 和兼容统计对象 `UsageStorage` 的 `.copy()` / `copy.copy()` 都是浅复制：创建独立的外层容器，内部 Prompt、候选配置、用量样本等对象仍共享。修改内部对象会影响共享者；只增删副本外层容器不会修改原容器。

需要隔离嵌套数据时明确使用 `copy.deepcopy()`，这些对象通过各自的 `__deepcopy__(memo)` 实现。Context 的深复制保留消息、轮次和系统提示之间原有的引用关系；工具 callable 和配置 logger 是借用的业务资源，仍保留原引用。

递归使用同一个 memo：Context/Prompts 将新对象先登记，再深复制其属性；同一 Prompt 同时出现在消息列表和轮次列表时只产生一个副本。ImagePrompt 使用 Python 默认复制协议，包含 __dict__ 和 slots；分片 ID 列表、native 字典等可变容器独立复制，bytes、base64 字符串、缓存 ID 和指纹等不可变值复用原对象。Path 只复制路径对象，不读文件；MediaRef 当前会创建新的小型元数据对象，不复制文件内容。不会调用图片 to_dict/load_image、文件读取或重新编码。大批嵌套容器仍需遍历和分配，不能将深复制视为零成本。

`Context.copy()` 继续创建新的会话 ID，作为独立会话入口；`copy.deepcopy(context)` 保留会话 ID，适用于同一会话的请求快照。TokenUsage 浅复制沿用旧约定生成新快照 ID，深复制保留原 ID；两者都不重新分配账本请求 ID。

Runtime 每次调用只深复制一次历史 Context，另复制本轮输入；规范化只重建容器和替换需修补的消息。重试复用工作上下文，提交只复制外层容器，不再次深复制历史。Chat 安装配置时隔离调用方数据；请求借用已安装的配置，更新通过安装新对象完成。`Chat.config` 仍返回独立配置快照。

`Chat.context` / `add_context()` 返回浅副本；`get_context()` 返回原对象。`Chat.copy(new_context=...)` 是已有的会话派生接口：配置独立复制，new_context=False 则共享原 Context。单个 Chat 的 ask 队列负责串行；不同 Chat 共用 Context 时的并发由业务控制，通信层没有跨 Chat 的锁。

`Context.message` 返回包含 system 的 `Prompts`；其中的索引返回单条消息，切片返回浅复制的 `Prompts` 容器。Runtime 使用这些接口读取历史，通过 `replace_contents()` 提交已准备的内容，不直接拆改 Context 的私有容器。

## 指纹与修改入口

`BasePrompt.signature` 是消息签名的统一入口，首次读取计算，之后缓存，直到显式调用 `refresh_fingerprints()`。签名包含 ID、角色、时间和消息内容，排除 native 协议附件与运行期缓存。框架不监听属性或嵌套对象写入；直接修改后未刷新时，一致性由调用方承担。

`Prompts.signature` 组合角色和成员签名；`Context.message_fingerprints` 只按历史顺序读取成员签名，`Context.signature/header_signature` 分别组合消息/工具和 system/工具的签名。上层不再展开成员内容重复哈希。容器增删、回滚、清空只清理聚合缓存；显式修改内容时刷新对应消息及所属容器。框架的图片准备等内容修改入口自行刷新，业务正常调用无需额外操作。

`LLMConfig.scope/revision`、HTTP 连接身份与 `ChatConfig.revision` 仍由配置对象缓存并显式刷新。这次消息与聚合签名的格式变化会使旧格式检查点匹配失败，首次完整回放后建立新检查点；不转换或猜测旧签名。

通用 `fingerprint()` 和 `plain()` 位于全局 `TIYA.utils`；LLM 专用的 `stable()` 位于 `llm/model.py`，负责字节摘要和 callable 工具 schema 等内容转换。`llm/config.py` 只解析连接配置，不再提供这些通用工具。原 `snapshot.py` 已删除，消息前缀和上下文签名直接读取 Context 属性。

不监控业务对 Prompt 字段、嵌套 dict 或 `Chat.model["parameter"]` 的直接修改。需要框架识别这些变化时，应通过正式更新入口，或显式刷新相应对象及所属配置/上下文的指纹；不能把业务的任意原地修改当成通信层自动保护范围。

原生回放保存消息的 ID 和当时 `signature` 的快照，校验时直接读取同一属性；不再维护 projection_fingerprint/projection_hashes。记录缺少统一签名快照时使用通用消息编码，不猜测旧格式。Responses/Anthropic 每次编码只构建一次消息 ID 索引，不逐条原生记录重建整份历史索引。`ToolInput.tool_payloads` 缓存同一工具的协议 schema，供编码、估算和签名共用；请求快照借用这份只读缓存，Provider 设置别名或删除协议字段时只复制需要修改的字典层。显式替换 callable/注解后调用 `tool.refresh_schema()`，并刷新所属 Context；普通容器增删不重建 schema。

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

安全拒绝后的重试耗尽，或遇到可安全切换的终止错误时，按顺序进入下一组合并立即记录会话 CD。CD 内沿用当前备用；该备用也失败时继续向后并刷新 CD。末项失败则删除 CD，本次按失败结束，下一次任务才从首选开始；同一任务不回绕已失败的候选。单候选执行既定重试后也按此规则结束。总尝试次数和截止时间仍是上限，普通成功不续 CD，到期后下一次调用直接回到首选。人工取消、部分输出、远端结果不明均立即停止，不触发路由切换。

压缩水位共用 usage.py 的判定规则：优先使用最近一笔有效请求的 reported.total，或完整的 reported.input + reported.output；零是有效值，不回退到估算。工具轮次含多笔请求时不将 repeated-prefix 用量累加作当前水位。Chat 仅在单候选且已有有效真实计量时做同步快检，不启动 tokenizer，也不猜测 CD 对应的实际模型。

真实计量关联已提交历史的现有 Context.signature、实际模型和候选 scope；清理、回滚、压缩或替换后旧签名不再适用，换模型/连接后旧来源不再适用。在途旧请求仍可按原快照完成和记账，但其计量不会成为新模型的有效水位。直接复用已有缓存签名和 scope，不新增哈希算法；未显式刷新的 BasePrompt 原位修改仍由调用方负责。存档没有 scope 时仍必须匹配 model，来源未知不冒充新模型实测值。

Runtime 在队列任务实际执行、选定候选后做权威检查，按该候选 max_context 与配置系数发出 ContextOverflowError；不是估算值直接与裸窗口比较。只有缺少有效真实计量且原 Context 非空时，才复用 tokenizer 缓存或补算本轮工作上下文；同次调用跨候选复用汇总，媒体分片变化才失效。空 Context 的首次新增输入硬上限仍由服务器判断。服务端明确超限继续按既有 Provider 规则上抛；不自动换大窗口、压缩或删除历史。

ask 的空闲检查、任务登记与无界队列 put_nowait 之间没有 await；queue=False 的竞争调用不会跨越估算等待而一起入队。no_wait 的既有优先级规则不变。排队期间历史和 usage 更新后，执行时预检会看到最新状态；未发送的超限任务不创建实际 request 账目。

Runtime.execute 原有五个位置参数不变，可选关键字 context_usage: TokenUsage、usage_signature、usage_scope 提供对应原始历史的最近计量，按只读借用。签名不匹配不采用，模型/连接在选路后校验；省略时可复用本 Runtime 最近一次 commit/commit_async 保存的计量。Runtime 不访问 Chat 对象，也不把 Context.usage 的累计值当水位；更改日志/系数等无关设置不使真实计量失效。

Chat 内存提交时同步保存 task.response、committed 和最终 outcome，再通知 ask 等待者；检查点写入和连接清理仍由当前队列任务收尾，不创建额外后台任务。已提交任务与超时竞争时返回保存的结果，后续收尾失败不撤销成功。Runtime 的统一 deadline 覆盖候选准备、Files、发送和结果物化；失败记账、镜像等收尾仍须完成，线程内已开始的 SQLite 写入不能靠 asyncio 强制中断。

显式协议 refusal（包括 Chat 的内容过滤结束、Responses refusal、Anthropic refusal stop_reason）不转换为普通 TextPrompt。Provider 通过失败通道报告拒绝，日志可记录原因；Chat 返回 None、不提交本轮问答、不重试或 fallback，真实消耗仍记账，状态为 refused。普通 content 中的自然语言拒绝不做关键词推断。Chat Completions 只支持 n=1，不合并多个候选。

## 启用服务端 Prompt 缓存

OpenAI 和 DeepSeek 的常用对话接口默认构建前缀缓存，Chat/Responses Provider 不额外注入缓存开关；调用方显式配置的原生缓存参数仍按原规则透传。[OpenAI 官方说明](https://developers.openai.com/api/docs/guides/prompt-caching)、[DeepSeek 官方说明](https://api-docs.deepseek.com/zh-cn/guides/kv_cache/)。其他兼容服务需按实际连接和模型确认，不按域名、连接名或缓存命中为零猜测能力。

Anthropic Messages Provider 每次请求默认补充顶层 `cache_control: {type: ephemeral}`，让服务端自动选取缓存点，不遍历历史安放断点，也不建立本地 KV 缓存状态。[Anthropic 官方说明](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)。已有 `llm_params.cache_control` 优先，原生 TTL 等字段原样保留，不默认要求更长保存期。兼容连接已自动缓存或不支持此字段时，在该连接配置 `prompt_caching: false`，只关闭默认补充；调用方显式提供的原生字段仍会发送。默认配置模板已标明这一选项。

这只请求使用服务端缓存，不保证每轮命中；前缀一致、模型阈值和服务端有效期仍需满足。真实 cache read/write 继续按响应 usage 记入账本，不根据已发送开关推测命中。此逻辑只影响 Provider payload，业务 Context、rounds、原生历史和 Responses 续接检查点均不变。

## Provider 错误契约

`provider/base.py` 的 `Failure` 明确区分错误事实和处理建议；未填写的权限默认 False，恢复动作默认 `none`。

| 字段 | 含义 |
| --- | --- |
| `category/code/raw` | 诊断分类、原始码和错误内容；category 不决定 Runtime 重试策略 |
| `remote_state` / `safe` | 对远端执行状态的判断；safe 只表示已确认拒绝执行，不表示值得重试 |
| `retryable` | 允许在同一候选原样重试；仍受重试次数、总次数和总时限限制 |
| `fallback_allowed` | 允许放弃当前模型候选，进入下一个；重试耗尽不自动获得此许可 |
| `retry_after` | 建议等待秒数，0 有效；无效、负数或非有限数按未提供处理 |
| `recovery` | 明确的协议恢复动作，与原样重试分开计次 |

恢复动作包括：`context_overflow` 向上抛出窗口错误；`rebuild_parent` 重建一次完整上下文；`refresh_file` 只重传明确失效的文件；`inline_images` 改传内联；`disable_files` 改传内联并暂存不支持文件引用的证据。`none` 不做专项恢复。改变请求内容的恢复完成后才再发，不盲目重复旧 payload。

模型 Provider 的 `failure(status, body, headers)` 解释 HTTP/业务错误，`transport_failure(error)` 解释传输异常；统一入口 `classify_failure(error)` 保留已解码的 ProviderFailure。默认实现复用现有 error_dialect 码表，各 Provider 可覆盖上述方法给出不同权限或恢复动作，不需要 Runtime 识别供应商原始错误码。

供应商码表集中于 `provider/error_codes.py`：`ErrorDialect` 声明错误对象路径、码字段、成功码和明确的 HTTP 覆盖规则，`ErrorRule` 声明分类、远端状态、重试/切换权限、恢复动作及资源字段。依次匹配业务码、方言 HTTP 规则和通用 HTTP 规则；未知方言不借用其他供应商业务码。未明确覆盖时，带未知业务码的 429 不直接当作可重试限流；DeepSeek 官方 429 按其文档解释为请求速率/并发限制，500/503 仍保持执行状态不明、不自动重发。连接配置只选择 `error_dialect`，不需要携带整份码表。特殊协议仍可覆盖 Provider 方法。

普通 `invalid_request`（如 HTTP 400/422 的非法消息或参数）默认不原样重试、不 fallback，也不推进会话 CD。Chat 返回 None 并保留未提交状态，由上游业务决定回滚或修正上下文；底层不擅自删除历史。父 response/file 引用失效等明确的专项错误仍按各自 Provider 规则恢复。HTTP 错误日志包含状态与服务端 message，生成请求账本的 `diagnostic.error` 保留原始错误体，便于定位具体的 role/tool_call_id/参数错误。

失败响应的完整详情写入 Logger 的 `debug.log`：HTTP 状态、服务端请求 ID 和原始响应文本都保留，覆盖 JSON、流式、文件与图片下载；非 JSON 错误页面还保存在 HTTPFailure.body.response_text。ProviderFailure 的原始业务错误对象也会写 DEBUG。即使之后恢复/重试/fallback 成功，先前的失败详情仍保留；不记录正常请求报文、请求正文或认证头。ERROR 保留终止错误的简短说明，月度状态栏格式不变。日志目录仍取决于启动工作目录，手动烟测从 scripts/manual_tests 启动时写入该目录下的 logs。

对话解码器 `feed()` 返回本事件新增的 `StreamText`（或无文本），Runtime 将其写入当前 logger 的 `LLM：` 日志流，遵循 `log_off`，在完成、异常和取消时关闭。推理到正文的切换提示沿用旧行为，不写入图片载荷或工具参数。Chat 流只缓存收到的片段，结束时用 join 拼接文本/推理/工具字段；图文混排仅合并相邻文本，保留图片位置，不深复制收到的内容块。

Runtime 的 `_recover()` 只执行恢复动作，`_retry_candidate()` 只管理被允许的重试次数及等待。取消、部分输出、已观察到真实消耗、已创建但终态不明的后台任务，以及已完成生成后的本地错误，都不会自动重建生成请求；Provider 的允许字段不能覆盖这些事实。账本诊断保留 Provider 的 retryable/fallback_allowed/recovery 和实际 resend_safe，不修改异常中原始 Provider 建议。

Files Provider 使用相同契约，但“改传 base64”和“切换模型候选”分开表达：上传暂时失败可建议 `inline_images`，文件认证/配额错误不自动伪装为内联成功，而按 fallback_allowed 决定是否换已配置的模型连接。独立 Files/Images Client 不因这些字段增加隐藏重试循环。

## 独立图片生成和编辑

无状态入口 `ImageGen` 可以复用[生图预设](../../../docs/configuration.md#生图和修图预设)。

```python
from pathlib import Path
from TIYA.LLM_connect import ImageGen, parse_llm_setting

async with ImageGen(config=parse_llm_setting("demo-images")) as images:
    generated = await images.generate("根据参考图生成日常插画", Path("reference.png"))
    edited = await images.edit(
        "移除红圈标出的挎包和肩带，补齐周围衣料，保留其余画面",
        Path("original.png"),
        markup=Path("markup.png"),
    )
    cache_id = edited[0].hash_name
```

`ImageGen` 只取预设首个候选，不保存上下文、不排队、不重试或 fallback。图片输入接受 `bytes`、`Path` 或 `ImagePrompt`；返回 `Prompts`，成员为已进入 file_cache 的 assistant `ImagePrompt`，`usage` 附带真实用量及账本请求 ID。`generate()` 无参考图走 generations，有参考图走 edits；`edit()` 始终将独立传入的 `base_image` 放在首位，其余参考图顺序不变。

方法中的可选请求参数默认 `None`。非 `None` 的方法参数覆盖 `model_params`，两处都为 `None` 时省略该字段。`False`、`0` 也可以覆盖。直接用 `endpoint`、`token` 和 `model` 构造时，`model_params` 中缺少 `stream` 会补为 `False`。`kwargs` 透传原生扩展参数。`timeout` 以秒为单位，覆盖输入准备、请求和落盘。`keep_alive=False` 每次调用后关闭连接，True 则复用到 `close()`。可用 `async with` 管理生命周期。

`markup` 是覆盖在未标注底图上的透明 PNG。它指示编辑位置，文字 `prompt` 描述要做的修改。用户在原图上直接圈画时，可用 `TIYA.image_diff.extract_markup(original, annotated, tolerance=60, despeckle=True)` 提取图层。异步业务可以用 `await extract_markup_async(...)`。两个接口均接收 `Path` 或图片字节，返回的 PNG 字节可直接传给 `markup`。持有文件缓存 ID 时，可先用 `get_file_path_async()` 取路径。容差需按回传图的压缩程度调整。回传图全不透明时默认只比较 RGB，避免把丢失的透明度识别成标注。

需要访问原生响应或接收流式预览时，可以使用底层 `ImagesClient`。

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
    cache_id = response.data[0].image.hash_name
    prompts = response.to_prompts()  # 按顺序包装为 assistant ImagePrompt，无文件读写。
    edited = await images.edit(ImageEditRequest(
        model="YOUR_IMAGE_2_5_MODEL_ID", prompt="根据参考图修改目标图的局部",
        images=[MediaRef(hash_name=target_id), MediaRef(hash_name=reference_id)],
        mask=MediaRef(hash_name=mask_id),
    ))
    # 使用透明标注图层时改传 markup，不能与 mask 同时使用。
    marked = await images.edit(ImageEditRequest(
        model="YOUR_IMAGE_2_5_MODEL_ID", prompt="修改红圈内的物体，参考第二张的风格",
        images=[MediaRef(hash_name=target_id), MediaRef(hash_name=reference_id)],
        markup=MediaRef(hash_name=markup_id),
    ))
```

模型 ID 原样发送。生成和编辑分别使用明确的 endpoint，不从生成 URL 猜编辑 URL；配置了 `base_url` 时可按标准路径追加。默认编辑走 multipart；`images: {edit_format: json}` 启用 JSON 的 images/mask/markup 引用。参考图保持顺序，mask/markup 均对应第一张目标图，二者互斥。mask 是 PNG alpha 遮罩；markup 是单独的透明 PNG 标注图层，必须含实际透明像素，不能只给不透明原图添加 alpha 通道。两者尺寸均须与首张目标图匹配（按 EXIF 方向读取），框架保留输入字节，不合成、不缩放。

markup 与其他图片一样接受 `MediaRef(hash_name=...)`、本地路径、bytes、base64；JSON 还可直接使用 URL 或远端 file_id。JSON 的本地输入配置了 files_api 时先上传，否则转为 data URL。校验在上传前完成；远端引用不为校验额外下载，可知尺寸在本地比对，其余内容由服务端校验。不会把编辑降级成纯文字生成，也不调用 Responses 内置生图工具。

最终 URL/Base64 图片在通信层下载、验证并进入现有 `file_cache` 后才返回。业务获得缓存 ID、MIME 和可用尺寸。没有永久保留或新的清理机制。`MediaError.stage` 区分下载/解码和验证/缓存失败；错误中保留 request_id，可调用 `images.recover_images(error)` 只重试接收落盘，不重复生图。

`MediaError.pending_response: Params | None` 明确保存待恢复的原生响应，默认为 None。接收失败时按引用保留响应，不深复制 base64 等大载荷；恢复再次失败，新异常继续保留同一响应和 request_id。恢复使用原账本记录，不新增生成请求或重复统计用量；换 ImagesClient 实例恢复时仍须使用同一账本。

本地文件直接填写 `MediaRef(hash_name=...)`，无需判断 `kind/value`；后两者仅用于原始 path/bytes/base64/url 输入和远端 file_id。远端引用也可携带原件的 hash_name。`ImagePrompt("user", hash_name=...)` 支持直接使用缓存 ID；旧的 Path/bytes/base64 构造方式保留，准备后将原件存入 file_cache，并清空 image_input。`image` 仍是旧版有序分片 ID 列表。新图片请构造新的 ImagePrompt，不依赖偷偷替换已准备对象的 image_input。

上传映射只关联 Files 账户作用域和 hash_name，不再读取整张图片计算另一种内容哈希，也不保存 descriptor/prepared_input 或单消息 media 状态。有效映射命中时不读取本地文件；过期后需要原件重传。拿上传租约前后都检查映射，防止并发重复上传。原件/派生件读取失败统一为带 hash_name 和原异常的 MediaError；旧派生件可从可用原件有界重建一次。TTL 沿用文件库现有规则。

`MediaService.upload()` 在同一事件循环内，按状态库路径、Files 账户作用域和 hash_name 共享正在进行的上传 Task，跨 MediaService/Chat 实例复用。后来者直接等待相同的 MediaRef 或失败异常，不轮询本地 SQLite，也不在等待层自行重传。`force=True` 绕过已完成映射，但仍加入当前上传。Task 完成后从内存表移除，成功后的长期复用仍查询状态库；跨进程继续使用原有租约和持久结果。

`upload(timeout=...)` 限制单个调用的全部等待时间；`MediaService(upload_timeout=300)` 限制共享任务自身的执行时间，单位均为秒。单个等待者超时/取消不会取消其他人的上传；最后一个退出时取消任务并释放租约。SQLite claim 在线程中执行，取消时也必须处理其最终结果，避免后台线程稍后取得无人持有的租约。

`upload()` / `_join_upload()` 成功时只返回 MediaRef；等待或共享任务超时抛 TimeoutError，取消抛 CancelledError，其他读取/上传错误继续传播。None 只用于内部 `_cached_upload()` 表示未命中，不是上传失败的返回值。`prepare_prompt()` 捕获超时等上传错误，再由 Files Provider 判断是否降级，不在公开上传入口吞成 None。

Runtime 按本轮 deadline 限制媒体准备及等待，直接调用 execute 时同样有效；本轮截止后直接抛 TimeoutError，不切换候选继续等待，不取消仍被其他调用等待的共享上传。

Runtime 的 `UploadSender` 固定首次发起者的 call_id/连接，并使用独立上传时限和 Gateway 状态。真正发送的 Files 请求只记一笔账，归首次发起的 call_id，等待者不重复生成请求账目。Chat 取消或 keep_alive=False 释放连接时，仍被共享上传使用的客户端延后关闭，最后一个使用它的上传完成后关闭；复用现有连接池，不为每次上传创建新池。直接传入自定义 send 回调时，借用资源和网络时限仍由回调调用方负责。

`ImagesResponse` 的 output_format/background/quality/size/safety_identifier 和 `ImageResponseItem` 的格式、质量、尺寸、事件类型、图片位置、预览序号、created_at/usage 都是固定字段；`metadata` 仅保留供应商未知扩展，不再混入已知字段或 URL/base64 图片载荷。

`FilesClient(settings)` 提供 `upload(MediaRef)`、`retrieve(id)`、`list(params=...)`、`delete(id)`、`download(id)`，可单独使用。下载能力仍以目标服务及具体文件为准。传入共享 httpx client 时，Files 使用自己的显式请求头，不继承模型密钥。

FilesClient/ImagesClient 的 client_params 按连接节点默认值与构造参数逐字段合并，显式构造参数优先；借入现成 HTTP client 时不修改其网络配置。Chat 关联 Files 的网络选项与模型相同时复用连接，不同时按 FilesConfig 获取独立网络池；遇到需要准备的图片才取 Files 连接。Images 编辑复用同一 FilesClient；网络配置不同时持有独立连接池，到 ImagesClient.close() 时关闭，不逐图/逐次编辑重建。借入的模型客户端不关闭。自定义 Runtime GatewayFactory 需同时接受 LLMConfig 与 FilesConfig。

Files 上传默认请求远端保留 86400 秒，可通过 `expires_after_seconds` 覆盖；显式设为 `null` 不发送自动到期参数，适用于不支持 TTL 的中转。OpenAI/DeepSeek 使用 `expires_after[anchor/seconds]`，Anthropic 使用 `expires_in_seconds`。以服务端返回的 `expires_at` 为准；没有返回时，以创建时间和请求 TTL 计算本地引用的保守刷新时间，不表示已确认服务端执行了删除。协议参考：[OpenAI](https://developers.openai.com/api/reference/resources/files/methods/create)、[DeepSeek](https://api-docs.deepseek.com/api/create-file/)、[Anthropic](https://platform.claude.com/docs/en/build-with-claude/files)。

`ImagesClient(settings, provider_ref="image_generation:连接名")` 可沿用配置树中的连接引用；`settings` 仍须显式传入该连接节点，`provider_ref` 选择协议实现并作为价格引用前缀，不自动读取全局配置。默认使用 `image_generation:direct`。`images.pricing_identity(model)` 返回 `image_generation:连接名:模型ID`，生成与编辑按同一模型引用查询价格；账本仍用 images_generate/images_edit 区分操作。流式预览回调接收的也是已缓存的 `ImageResponseItem`，预览不会计为最终图片。

## 状态与记账

- 文件内容和 TTL：现有 `data/file_cache`。
- 上传映射、能力失败证据、检查点和会话冷却：`data/llm/runtime.sqlite3`。
- 请求账本：`data/token_usage/usage.sqlite3`。
- 独立定价目录：`data/llm/pricing.sqlite3`。
- 业务历史和原生回放项：原有 Context 存档。读取旧存档不要求新增状态存在。

`StateStore` 只清理通信元数据：同进程、同一数据库的所有实例共享模块级维护时钟，普通访问默认每小时触发一批清理，不启动后台任务。`cleanup_interval_seconds=86400` 可改为一天；同库实例应使用一致设置。每批最多清理 500 条状态和 500 条租约，满批后也等待下个间隔，避免把积压连续摊给后续请求；大量积压可在维护时显式多次 cleanup。读取单条状态仍立即检查过期，申请租约仍立即检查该 key，不因批量回收延后而使用失效数据。

明确到期的状态和租约会被删除；没有到期信息的状态也按闲置 30 天回收，活跃读取续本地保留期。`retention_seconds` 可在构造状态库时调整。远端 `expires` 不随读取延长，冷却状态在截止时间后可直接回收。旧数据库首次增加访问时间字段时保留原状态并给予完整保留期。

可同步调用 `StateStore.cleanup(now=..., batch_size=500)` 执行一批清理，异步代码通过 `run_sync` 调用；显式清理更新同库共享时钟，不额外叠加自动批次。清理不扫描/删除本地图片，不调用远端删除，不触碰用量账本；SQLite 空闲页供后续写入复用，不在请求链路执行 VACUUM。

账本每次真实尝试分配独立 request_id，同一逻辑调用共享 call_id；平台 request ID、response ID 分开记录。先持久化初始估算，再发送网络请求。reported 与 estimated 分别保存，effective 逐字段优先采用 reported。缺失真实值不填成真实零值。观察幂等，费用修订不会增加请求数量；确认未执行的拒绝保留初始估算证据，并以新的估算观察注明拒绝状态。

Runtime 的账本 estimated 使用 characters-div3-image1440-v1：字符数约除以三，加每条消息固定开销，图片按每张 1440；completeness=heuristic，既不是实测值，也不保证下界，不用于上下文拦截。新增结构化消息的计数缓存在 BasePrompt 的运行期槽中，显式刷新失效，不参与签名/序列化。候选与重试复用同次逻辑调用的输入假设；输出同样采用轻量假设，不为记账运行 tokenizer。重复镜像同一请求时更新可变快照，不新建请求、不修改冻结价格。

`Ledger` 提供 begin_call/begin_request/observe_usage/finish_request、分页 query_requests/query_observations 和 summarize。旧 `LLM_usage` 的日志和统计入口通过适配器读取 SQLite。JSON 仅首次备份并导入历史聚合，不将历史聚合伪造成逐请求记录，也不重复累计 last_record。

状态栏和 `usage_now()` 的月度展示通过 `MonthlyOverview` 汇总：首次或跨月读取当月明细，之后只重读本进程正式接口修改的请求。每次用量观察/请求结束刷新，价格发布也会刷新；不逐 token 更新，不启动周期任务。进程内只缓存当月的汇总贡献及无单价请求所需计量，重启从账本重建。旧聚合导入结果单独缓存，仅在新增导入后重读；多个账本实例操作同一文件时共享失效通知，不承诺感知其他进程或手工 SQL 改动。

当月原本没有价格快照的请求，按当前匹配单价临时估价，按每条请求的用量分别选择阶梯。展示的 `cost` 包含这部分费用，`temporary_cost` 在数据中单列其金额；状态栏保持旧版格式，只显示用量、缓存命中率和金额，不额外展示估算或未知提示。新增单价不回填原始 `price_snapshot/charge`，原来已有价格的请求仍使用冻结费用。临时结果只驻留内存，不修改原始用量、观察证据或费用记录。旧版导入聚合缺少完整连接身份，沿用原统计，不猜测新价格绑定。

`LLMUsage.overview("MONTH")` 使用上述展示口径；`Ledger.summarize()` 以及 DAY/YEAR/ALL 查询继续反映账本原始费用，便于审计。自定义 `LLMUsage(..., prices=catalog)` 可显式绑定展示价库；全局日志默认绑定 `data/llm/pricing.sqlite3`。通过 `PriceCatalog.publish()` 发布后立即刷新已建立的月度视图，未来生效价格在之后首次刷新到达生效时间时启用。

业务返回的 `Prompts.usage` 和累计 `Context.usage` 使用新版 `TokenUsage`，定义在 `usage_model.py`，原导入入口保持不变。它保存已消费响应的用量快照，完整失败尝试与长期费用仍查询账本。

- `input_t/output_t/cache_hit_t/total_t`：原业务读取方式，按每次请求的有效值累计；缺失时兼容返回零。
- `reported/estimated/unknown`：分别汇总真实证据、本地估算和旧数据未注明来源的计量。`samples` 保留每个请求的原始证据及估算说明。
- `effective`：每次请求先按 reported → estimated → unknown 选值，再累计；真实零有效，未知值不改成真实零。`sources` 标明 reported/estimated/unknown/derived/mixed，`incomplete_fields` 标明仍缺少部分计量的字段。
- `request_ids/call_ids`：关联 SQLite 账本；`id` 只是快照自身标识，复制快照可生成新 id，账本 ID 不变。
- `model`：最后合入的模型；`models`：全部贡献模型。跨模型相加保留各次证据，同 request_id 的右侧新快照更新旧快照，不重复累计。

`total` 缺失时可从已有 input/output 推导，并标记 derived；无法追溯到具体请求的普通数值相减只产生未知来源的比较值，不伪装成真实消耗。账本不使用这些比较值计费。

用量存档 v2 保留旧扁平字段并新增 samples。旧 Context 的扁平用量按 unknown 读取，不补造历史请求 ID。Context 追加和协议规范化保留各轮的用量归属，不重复累计；Memory 源码无需修改。

定价通过 `PriceCatalog.publish(ref, rules, source=...)` 维护，不写回连接 YAML。`ref` 就是配置中的完整 `分类:连接名:模型ID`，Runtime 按实际选中的候选自动查价，不需要额外绑定，也不依赖预设名、Key 或账户 scope。例如：

```python
from TIYA.config import DATA_DIR
from TIYA.llm.pricing import PriceCatalog

catalog = PriceCatalog(DATA_DIR / "llm" / "pricing.sqlite3")
ref = "chat_completions:deepseek:deepseek-flash"
rules = {
    "currency": "CNY", "tier_mode": "whole", "threshold": "input",
    "units": {"input_uncached": 1000000, "cached": 1000000, "output": 1000000},
    "tiers": [
        {"rates": {"input_uncached": "2", "cached": "0.04", "output": "8"}},
    ],
}
catalog.publish(ref, rules, source="经核对的价格来源")
```

同一完整引用可被多个预设共用；协议分类或连接名不同就是不同的价格 ID，不猜域名、不跨服务商套用。DS 当前固定采用高峰价，不实现峰谷切换。whole 按完整输入所处区间选择整次请求价格；progressive 按同一个计量项逐段计费。输入/输出总量与子项不能重复定价。可使用 images、input_image、output_image 等计量项；没有匹配价格时费用未知，仍然记账。每次请求冻结价格版本，后续调价不改旧请求。异步调用方通过 `llm.workers.run_sync()` 执行同步维护/查询，避免阻塞事件循环。

## 线程资源与 Runtime 阅读入口

LLM 中原先通过 `to_thread` / `run_in_executor` 卸载的同步 SQLite、token 估算和图片处理，现统一走 `llm/workers.py` 的 `run_sync()`，使用 `TIYA/executor.py` 中的进程级 `LLM_EXECUTOR`，默认最多 4 个工作线程，线程名前缀为 `llm`。线程数在现有全局线程池管理文件中集中调整，无需修改连接配置。所有 Chat、FilesClient 和 ImagesClient 共用这个池；HTTP 仍为异步收发，file_cache 使用它已有的独立线程池。

tokenizer 估算使用 `llm/usage.py` 的 `estimate(context: Context) -> UsageEstimate`；返回明确的估算字段和依据，不进行媒体 IO、不修改业务内容。消息缓存单条估算，Context 缓存汇总；追加、删除、回滚只使汇总失效，未修改的历史文本和工具 schema 不重复编码。`estimate_async()` 在汇总已缓存时直接返回，否则交给 LLM 专用线程池。只在显式 estimate 调用或 Runtime 缺少有效真实计量的预检中启动 tokenizer，特殊 token 字面量作为普通文本计数；Runtime 记账使用另一套轻量假设，不因此调用 tokenizer。

显式修改消息内容后调用 `Context.refresh_fingerprints()`，同时清除内容指纹和估算缓存；仅改变成员关系或媒体传输信息时，框架使用 `content_changed=False` 保留消息级估算。缓存随进程内副本保留，不写入历史归档，也不参与消息指纹、原生回放校验和网络请求体。重新加载归档后首次估算会计算一次。用量模块单向依赖数据对象，`model.py` 不导入估算、配置解析或 Runtime 模块。

`run_sync()` 保留调用协程的 contextvars、结果和异常。取消等待可取消尚未开始的工作，但不能杀掉已运行的同步函数。单个 Chat/Client 关闭只负责自己的任务和 HTTP 连接；项目现有 `executor.shutdown_all()` 在业务停止后同步等待 LLM 池收尾，包括已开始的账本写入。

```python
from TIYA.llm.workers import run_sync

rows = await run_sync(ledger.query_requests, preset="MAIN", limit=100)
```

Runtime 的 `execute()` 是主调度入口，只制作 Context 快照并按有序候选执行。参数和提交边界直接写在函数文档中，主要步骤如下：

- `_prepare_candidate()`：按实际候选检查水位与协议能力、获取连接、选择历史检查点。
- `_advance_route()`：安全失败后更新下一候选的 CD；末项失败删除 CD，本次结束，下次任务从首选开始。
- `_execute_candidate()`：单候选的有界尝试，准备将实际发送的媒体。
- `_begin_attempt()` / `_send_attempt()`：冻结价格并先落估算，再执行一次生成收发。
- `_complete_attempt()`：保留真实用量、缓存返回图片，构造 `Pending`。
- `_record_failure()` / `_cancel_background()` / `_recover()`：保留失败证据；仅确认未执行才恢复，取消/部分输出/远端未知直接停止。
- `_retry_candidate()`：执行 Provider 允许的原样重试，管理次数、等待和总预算。
- `_send_auxiliary_request()`：发送 Files、后台查询/取消等辅助请求并单独记账，不执行生成重试。
- `commit_async()`：由业务持有者校验原 Context 未变化后消费结果，随后保存续接检查点；路由 CD 已在失败切换时记录。

`Pending`、逻辑调用、单候选和单次发送状态分别有明确的结构定义，内部执行状态通过这些结构传递。Provider 的 `ConversationProvider` / `StreamDecoder` 契约位于 `provider/base.py`；Runtime 不依赖某家解码器的具体实现类。主调用、网络、媒体、账本和定价接口均有参数类型及必要的副作用说明。

## 已修正的旧行为

取消和超时不会提交未完成结果；队列与 no_wait 共用提交逻辑；损坏的工具 JSON 不会变成空参数后执行；分阶段 usage 合并且失败仍记账；不再根据历史长度伪造真实缓存命中。内部请求深快照隔离可变内容，公开 copy 保持浅复制，工具绑定的业务对象保持原引用；完整切换预设清除旧 Files、价格和路由设置。实际 HTTP 请求不再使用旧的首次 SSE 盲目重发逻辑。

旧入口返回 None 时，调用协程可读取 `Chat.last_outcome` 的 status/retry_allowed/call_id/request_id/remote_state。Runtime 已完成有界重试后的终态不允许上层再盲目重发；项目内 Agent 已接入这一规则。自定义调用方应同样检查终态。跨模型 fallback 保留 Context 中此前模型的累计用量；Chat 的窗口判断仍使用最近一次响应用量，长期账本保留所有尝试及费用。

普通请求断开连接只代表本地停止，远端是否停止保留 unknown。后台 Responses 必须显式声明 `responses.supports_background`；已有任务 ID 时尝试协议取消并记录确认状态，不把断流当作后台任务已停止。
