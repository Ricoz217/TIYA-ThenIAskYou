# 配置说明

TIYA 使用 `config/config.yaml` 保存本地运行配置。该文件由程序自动生成，并已被 `.gitignore` 排除。

配置中包含模型密钥、NapCat token、Tavily Key 和 Pixiv token 等敏感信息。不要将真实配置改名后提交，也不要把真实密钥复制到文档、测试、Issue 或日志中。

## 配置结构

生成文件分为五个顶层区块：

| 区块 | 内容 |
| --- | --- |
| `BOT基本配置` | Bot、NapCat、代理、模块开关及各系统模型 |
| `LLM模型配置` | Provider 连接及按名称索引的模型预设 |
| `私聊设置` | 私聊默认值和用户级覆盖 |
| `群聊设置` | 群默认值和群号级配置 |
| `其他参数配置，请勿随意更改` | 超时、容量、缓存和内部运行参数 |

最小运行只需要修改前四个区块。最后一个区块提供完整默认值，初次体验时建议保持不变。

以下示例都是生成配置中的局部片段，不应直接替换整个文件。

## NapCat

```yaml
BOT基本配置:
  BotNetWork:
    NapCatWebSocket: ws://localhost:3000
    NapCatToken: "replace-with-your-napcat-token"
```

`NapCatWebSocket` 必须是 NapCat 已启用的 WebSocket 地址。token 需要与 NapCat 服务端一致；即使只监听 localhost，也不要把真实 token 提交到仓库。

`BotInfo.name` 和 `BotInfo.uid` 会在成功获取登录信息后更新，一般不需要手工维护。

## LLM 预设

v2 配置把连接和预设分别放在 `Providers` 与 `LLM_List` 中。下例定义一个 Chat Completions 连接和一个使用该连接的聊天预设。

```yaml
LLM模型配置:
  schema_version: 2
  Providers:
    chat_completions:
      main:
        endpoint: https://example.com/v1/chat/completions
        key: "replace-with-your-api-key"
        llm_params:
          stream: true
  LLM_List:
    demo-model:
      proxy_mode: ""
      auto_compress_rate: 0.7
      providers:
        - llm: "chat_completions:main:your-model-id"
          max_context: 128000
          llm_params:
            temperature: 0.7
```

| 字段 | 说明 |
| --- | --- |
| `Providers.<协议分类>.<连接名>` | 协议分类选择适配器。连接名可自定义，同一连接可供多个模型引用。 |
| `endpoint` / `base_url` | 完整操作地址，或包含版本前缀的 API 基础地址。两者都填时优先用 `endpoint`。 |
| `key` | 连接使用的 API 密钥。 |
| `LLM_List.<预设名>.providers` | 按优先顺序排列的候选连接。`llm` 的格式为 `协议分类:连接名:模型ID`。 |
| `llm_params` | 目标接口的原生请求参数。候选中的同名参数覆盖连接默认值。 |
| `proxy_mode` | 预设引用的 `Proxies` 配置名称。空字符串表示不使用 TIYA 的代理配置。 |
| `max_context` | 候选模型的上下文窗口上限。示例值应按实际模型填写。 |
| `auto_compress_rate` | Agent 自动压缩上下文的窗口比例。 |

`chat_model`、`agent_model`、`MemoryModel` 等业务配置填写预设名，例如 `demo-model`。多个聊天候选按列表顺序配置，是否重试或切换取决于实际错误。Responses、Anthropic Messages 和可选 Files 连接的字段见[通信层配置说明](../src/TIYA/llm/README.md#配置-v2)。

## 从旧版升级

升级前备份现有 `config/config.yaml`，手工将 `LLM模型配置` 改为 `schema_version: 2` 的结构。把旧预设中的连接地址和密钥移到 `Providers`，把 `extra_parameter` 中仍适用的原生参数移到候选的 `llm_params`。保留原有预设名即可沿用业务引用。代码和对应配置应一起部署，旧版列表没有自动转换。

请求用量保存在 `data/token_usage/usage.sqlite3`。首次读取旧 `usage.json` 时会备份并导入历史汇总。单价单独维护在 `data/llm/pricing.sqlite3`，按完整模型引用匹配，配置中无需填写 `price`。部署到另一台机器时，随运行数据带上已有的价库才能沿用本地单价。缺少匹配单价时，系统仍会记录用量。更多细节见[状态与记账](../src/TIYA/llm/README.md#状态与记账)。

## 生图和修图预设

按需把下面的 `image_generation` 连接和 `demo-images` 预设加入对应字典。`endpoint` 用于文字生图，`edit_endpoint` 用于修图和带参考图的生图。

```yaml
LLM模型配置:
  schema_version: 2
  Providers:
    image_generation:
      openai:
        endpoint: https://api.openai.com/v1/images/generations
        edit_endpoint: https://api.openai.com/v1/images/edits
        key: "replace-with-your-api-key"
  LLM_List:
    demo-images:
      providers:
        - llm: "image_generation:openai:your-image-model-id"
```

代码可通过 `ImageGen(config=parse_llm_setting("demo-images"))` 使用这个预设。`ImageGen` 使用首个候选。`Agents.ImageModel` 和 `ImageModelHigh` 配置的是图片理解模型。调用示例见[独立图片生成和编辑](../src/TIYA/llm/README.md#独立图片生成和编辑)。

## Agent 模型

```yaml
BOT基本配置:
  Agents:
    MemoryModel: demo-model
    MemorySummaryModel: demo-model
    ImageModel: demo-model
    ImageModelHigh: demo-model
```

- `MemoryModel`：长期记忆写入、查询和维护使用的主要模型。
- `MemorySummaryModel`：会话记忆总结使用的模型。
- `ImageModel` / `ImageModelHigh`：图片理解能力；最小配置可以先指向同一支持图片的模型。如果当前模型不支持图片，纯文本聊天仍可运行，但图片相关任务会失败。

Agent 主控和 Speaker 的模型在群聊/私聊区块单独配置，因此可以使用速度和能力不同的模型。

## 群聊配置

`Group_Default_Setting` 是新发现群的模板：

```yaml
群聊设置:
  Groups:
    Group_Default_Setting:
      enable: true
      chat: true
      chat_model: demo-model
      agent_model: demo-model
      default_character: 抹布
      speak_rate_min: 0.05
      speak_rate_max: 0.70
      setu: false
      imgsearch: false
      eh: 0
```

常用字段：

| 字段 | 说明 |
| --- | --- |
| `enable` | 是否创建并启动该群对象 |
| `chat` | 是否启用普通 Agent 聊天 |
| `chat_model` | Speaker 使用的预设 |
| `agent_model` | Agent 主控使用的预设 |
| `default_character` | `data/character/` 中的角色名称 |
| `speak_rate_min/max` | 注意力发言概率的下限和上限 |
| `relative_rate_min/max` | 相关性参与发言判断的范围参数 |
| `agent_round_limit` | 单次 Agent 主控最大轮数 |
| `speaker_round_limit` | 单次 Speaker 最大轮数 |

连接 NapCat 后，程序会按实际群列表加入群号配置，例如：

```yaml
群聊设置:
  Groups:
    "123456789":
      name: 示例群
      enable: true
      chat_model: demo-model
      agent_model: demo-model
```

具体群配置从默认模板生成，此后可以单独调整。`SkipGroups` 中的群号不会创建群对象。

## 私聊配置

```yaml
私聊设置:
  Default:
    enable: true
    chat: true
    chat_model: demo-model
    agent_model: demo-model
    default_character: 抹布
    speak_rate_min: 0.80
    speak_rate_max: 1.00
    attention_fade_out_time: 900
    agent_round_limit: 50
    speaker_round_limit: 30
  Users: {}
```

`Default` 应用于所有未单独配置的用户。`Users` 可以按 QQ 号只覆盖需要改变的字段：

```yaml
私聊设置:
  Users:
    "123456789":
      enable: false
```

私聊对象按消息到达创建，长时间无活动时会保存并回收。再次收到消息后会从本地数据恢复。

## 权限列表

本节配置的是 QQ 侧管理员、所有者和通知接收者，不是 Agent 工具权限。当前 Agent 还不能按工具或具体操作分别授权，参见[已知限制](limitations.md)。

```yaml
BOT基本配置:
  AdminList:
    - "123456789"
  OwnerList:
    - "123456789"
  NoticeList:
    - "123456789"
  SkipGroups: []
```

QQ 号建议写成字符串，避免 YAML 或后续序列化对长数字进行不必要的数值处理。示例中的 `114514` 等默认值只是占位符，应替换或删除。

## 代理

```yaml
BOT基本配置:
  Proxies:
    GlobalProxy:
      http: null
      https: null
    NecessaryProxy:
      http: http://127.0.0.1:7890
      https: http://127.0.0.1:7890
```

模型预设和可选模块通过 `proxy_mode` 引用代理名称。无需代理时将对应值设为 `null`；不要保留一个本机并不存在的代理地址，否则相关请求会表现为网络超时。

## 可选模块

```yaml
BOT基本配置:
  Module:
    setu: false
    eh: false
    imgsearch: false
  WebSearch:
    TavilyKey: ""
    ProxyMode: GlobalProxy
```

最小运行建议全部保持 `false`。启用 Web 搜索时填写 `TavilyKey`。Pixiv 和图片模块还包含登录 token、远程登录地址、OCR 和网络代理等配置，本 Demo 文档不展开这些辅助能力。

## 热重载与写回

配置由 `ruamel.yaml` 读取，以尽量保留原文件结构和注释。运行中的配置修改可以触发 reload，新发现的群也会写回配置。

配置写入采用临时文件加原子替换。字段类型错误或缺失时，校验器会生成报告并使用默认值回退；这能减少配置升级造成的直接崩溃，但仍应根据启动日志修正报告的问题。

修改模型、群和私聊默认值后，最简单可靠的验证方式是正常重启 TIYA。内部超时、缓存和容量参数只有在理解对应模块后再调整。
