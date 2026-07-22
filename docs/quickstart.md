# 快速开始

本页只覆盖最小可运行流程。TIYA 是个人 Demo，目前通过源码运行，不提供 PyPI 安装包。

当前版本没有完整的 Agent 权限管理和代码运行沙箱。请只在受信任的本地环境中运行，并在安装自定义 SKILL 前阅读[已知限制](limitations.md)。

## 环境要求

- Python 3.12
- Git
- 已登录 QQ 的 NapCat 实例
- 一个兼容 OpenAI Chat Completions 的模型接口

建议使用独立测试账号运行 QQ Bot，并确认所使用的 QQ 客户端、NapCat 和自动化方式符合对应平台规则。

## 获取源码

```powershell
git clone <repository-url>
cd TIYA_ThenIAskYou_2026
```

项目链接确定后，将 `<repository-url>` 替换为实际 GitHub 地址。

## 创建环境

Windows PowerShell：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

Linux / macOS：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

依赖版本以 `requirements.txt` 为准。最后一条命令只把本地 `src/TIYA` 注册到当前虚拟环境，`--no-deps` 避免再次解析依赖；它不会从 PyPI 安装 TIYA。

## 生成配置

启动一次程序：

```powershell
python -m TIYA
```

首次运行会生成：

```text
config/config.yaml
```

程序会在提示配置未完成后退出。打开生成的文件，至少填写：

1. NapCat WebSocket 地址和 token。
2. 一个 LLM 预设的 `model`、`endpoint` 和 `token`。
3. `Agents` 使用的模型预设名称。
4. 群聊默认配置和私聊默认配置中的 `chat_model`、`agent_model`。

完整字段参见[配置说明](configuration.md)。

## 配置 NapCat

在 NapCat 中启用 WebSocket 服务，并使地址、端口和 token 与 TIYA 配置一致。默认示例地址是：

```text
ws://localhost:3000
```

TIYA 启动后会通过 OneBot API 获取登录信息和群列表。首次发现群时，会在 `config.yaml` 的群配置中补充对应群号；如果该群仍使用占位模型名称，系统会等待配置完成，不启动该群对象。

## 启动

完成配置并确保 NapCat 正在运行后，再次执行：

```powershell
python -m TIYA
```

正常启动时，日志会显示登录账号、群对象初始化和额外模块加载情况。运行中产生的数据保存在 `data/`、`logs/` 和 `config/` 下，这些目录中的本地内容默认不会进入 Git。

使用 `Ctrl+C` 退出时，程序会尝试保存群聊、私聊、Agent 和长期记忆状态，再关闭事件循环。

## 验证测试

无需连接 QQ 或真实模型即可运行自动化测试：

```powershell
python -m pytest -q tests
```

测试主要覆盖配置、Agent 任务协议、记忆、相关性、群聊/私聊流程和可选模块的核心逻辑。真实 NapCat、模型质量和第三方服务仍需要在本地环境中验证。

## 可选能力

Web 搜索需要 Tavily Key；Pixiv、OCR、图片搜索和浏览器相关能力还需要各自的网络条件或账号配置。它们不属于最小启动路径，可以保持关闭，待主聊天流程稳定后再逐项启用。
