# 配置说明

TIYA 使用 `config/config.yaml` 保存本地运行配置。该文件由程序自动生成，并已被 `.gitignore` 排除。

配置中包含模型密钥、NapCat token、Tavily Key 和 Pixiv token 等敏感信息。不要将真实配置改名后提交，也不要把真实密钥复制到文档、测试、Issue 或日志中。

## 配置结构

生成文件分为五个顶层区块：

| 区块 | 内容 |
| --- | --- |
| `BOT基本配置` | Bot、NapCat、代理、模块开关及各系统模型 |
| `LLM模型配置` | 可复用的模型预设列表 |
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

模型预设把业务模块使用的名称与具体 API 参数解耦：

```yaml
LLM模型配置:
  LLM_List:
    - preset_name: demo-model
      model: your-model-name
      endpoint: https://example.com/v1/chat/completions
      token: "replace-with-your-api-key"
      api_type: openai
      proxy_mode: GlobalProxy
      max_context: 128000
      auto_compress_rate: 0.7
      price:
        currency: CNY
        input_token: 0
        cache_hit: 0
        output_token: 0
      extra_parameter:
        max_tokens: 8192
        temperature: 0.7
```

关键字段：

| 字段 | 说明 |
| --- | --- |
| `preset_name` | TIYA 内部引用名称，必须唯一 |
| `model` | 发送给接口的模型 ID |
| `endpoint` | 完整 Chat Completions 地址 |
| `token` | API 密钥；本地无鉴权接口可使用非敏感占位值 |
| `api_type` | 最小示例使用 `openai` |
| `proxy_mode` | 引用 `Proxies` 中的代理配置 |
| `max_context` | 模型上下文上限，用于请求前估算和压缩判断 |
| `auto_compress_rate` | 达到窗口比例后自动压缩 Agent Context |
| `price` | 每百万 token 单价，只影响本地用量统计 |
| `extra_parameter` | 透传给模型的生成参数 |

业务配置引用的是 `preset_name`，不是 `model`。如果名称写错，启动或首次请求时会提示找不到模型预设。

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
