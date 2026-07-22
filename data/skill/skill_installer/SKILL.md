---
name: skill_installer
description: 内置 SKILL，提供 SKILL 的安装、卸载、加载、按标题读取内容、加载 Python 工具与执行脚本、读写文件以及下载文件的基本能力。
origin: TIYA
version: 0.2.0
metadata:
  category: builtin
  owner: agent-core
---
# Usage

- 当需要“管理 SKILL 包”或读写文件时使用本 SKILL。
- 当主控只需要某个 SKILL 的部分说明时，先调用 `load_skill` 获取标题树，再调用 `get_skill_content` 按需读取内容。
- 当需要把某个 SKILL 的 Python 工具接入 Agent 时，使用 `load_tools_from_python_file`。

# Constraints

- 路径参数必须是字符串，支持占位符 `%SKILL%`（表示 Agent 的 SKILL 根目录）。
- `get_skill_content` 的 `title_tree` 必须来自 `load_skill` 返回的标题树结构，不要手写不存在的标题键。
- 执行脚本有安全风险，优先使用只读内容接口；仅在明确需要时调用 `execute_script`。

# SKILL 定义

SKILL 是以“目录”为单位的能力包，最少包含一个 `SKILL.md`。  

典型结构:  
```text
%SKILL%/
  skill_name/
    SKILL.md            # 必需
    scripts/            # 可选
    references/         # 可选
    assets/             # 可选
```  
`SKILL.md` 必须具备 YAML front matter，至少包含:  
- `name`
- `description`  

# SKILL加载机制  

SKILL采用懒加载:
1. AGENT默认仅加载已安装SKILL的基本信息（`name`、`description`）。
2. 可调用 `load_skill(skill_name)` 获取SKILL的SKILL.md数据，仅返回标题树，不返回正文内容。
3. 调用 `get_skill_content(skill_name, title_tree)` 时，按标题树精确返回对应正文。  
这样可以减少主控 context 占用，避免全文注入。  

# 附属工具说明

`skill_installer` 提供 14 个工具:
1. `install_skill`
2. `load_skill`
3. `list_skills`
4. `uninstall_skill`
5. `get_skill_content`
6. `get_content`
7. `set_content`
8. `remove_path`
9. `list_dir`
10. `load_tools_from_python_file`
11. `execute_script`
12. `is_skill`
13. `download_file`
14. `get_hash_name`

## 路径占位符

- **"%SKILL%"** 表示AGENT的SKILL根目录。  
- **"%TEMP%"** 表示临时目录，创建文件时优先选择此路径  

> 示例: `skill_installer`的目录路径为 `"%SKILL%/skill_installer/"`

## 1) install_skill

函数签名:  
`def install_skill(self, path: str) -> list[dict[str, str]]`  

> 安装指定路径下的SKILL。自动检验目录是否合法SKILL，若目录不在`%SKILL%`，则自动在`%SKILL%`创建一份副本并安装。

参数:  
- `path`:SKILL 目录路径（支持 `%SKILL%` 占位符）。

返回:  
- 成功:已安装的SKILL字典列表 `[{skill_name: skill_description}]`。
- 失败:抛出异常（路径不存在、不是 SKILL、重名冲突等）。  

## 2) load_skill

函数签名:  
`def load_skill(self, skill_name: str) -> dict`

> 加载已安装的SKILL并返回SKILL.md的标题树。  

参数:  
- `skill_name`:SKILL 名称。  

返回:  
- 成功:嵌套标题树字典，例如:
  `{skill_name: {"Usage": {}, "Constraints": {}, "附录": {"接口": {}}}}`  

## 3) list_skills

函数签名:  
`def list_skills(self) -> list[dict[str, str]]`

> 列出当前所有已安装的SKILL。  

参数:
- 无。  

返回:
- 成功:已安装的SKILL字典列表 `[{skill_name: skill_description}]`。  

## 4) uninstall_skill

函数签名:  
`def uninstall_skill(self, skill_name: str) -> list[dict[str, str]]`

> 卸载指定SKILL  

参数:  
- `skill_name`:SKILL 名称。

返回:
- 成功:卸载后的已安装的SKILL字典列表 `[{skill_name: skill_description}]`  

## 5) get_skill_content

函数签名:  
`def get_skill_content(self, skill_name: str, title_tree: dict) -> str`

> 根据标题树返回指定的SKILL内容字典  

参数:
- `skill_name`:SKILL 名称。  
- `title_tree`:一个表示标题树的嵌套字典。  

标题树字典约定:
- 字典中所有的键必须来自`load_skill`  
- 输入空字典`{}`表示读取该 SKILL 全部可读内容。
- 值为 `{}` 表示读取该标题节点下全部内容。
- 可只请求部分分支，降低 context 占用。
- 示例:全文:`{}`; 仅两段:`{skill_name: {"Usage": {}, "Tools": {"install_skill": {}}}}`  

返回:  
- 成功:由指定标题内容重建成的 MarkDown 文本字符串。
- 标题树键名错误:error_messages(string)。  

## 6) get_content

函数签名:  
`def get_content(self, path: str, image_mode: bool = False, encoding: str = "utf-8", start_line: int = 0, end_line: int = 0) -> str`  

> 读取指定文本文件内容，也支持读取图像文件并上传。

参数:  
- `path`:文件路径（字符串）。
- `image_mode`:是否读取图像文件，若True，则后续参数无效，同时返回一个读取提示，并在下轮对话消息中上传图片
- `encoding`:编码，默认 `utf-8`。
- `start_line`:从n行开始读取，若为0则从头开始。  
- `end_line`:读取到第n行，若为0则读取到尾;  支持负数，-1表示读到倒数第一行。

返回:  
- 成功:一个{行数:内容}的字典: {'1': line1, ..., '999': line999} 
- 图像模式:{"image": {"path": file_path, "index": 表示在下轮对话消息中第几张图片}}

## 7) set_content

函数签名:  
`set_content(self, path: str, content: str = "", start_line: int = 0, end_line: int = 0)`  

> 使用 utf-8 写入文本文件。 当文件已存在时会直接覆盖/修改，可使用`start/end_line`替换指定行数范围的内容。  

参数:  
- `path`:文件路径（字符串）。  
- `content`:要写入/修改的内容，使用'\n'作为换行符。  
- `start_line`:从n行开始替换，若为0则从头开始。  
- `end_line`:替换到第n行，若为0则替换到尾; 支持负数，-1表示替换到倒数第一行。  

返回:  
- 成功:返回创建/修改的文件路径。

## 8) remove_path  

函数签名:  
`def remove_path(self, path: str) -> str`  

> 删除指定路径的文件/文件夹，仅允许删除 `%TEMP%` 目录内的路径，超出范围会拒绝并抛出异常。 

参数:  
- `path`:文件路径（字符串）。

返回:  
- 成功:返回成功提示;  失败: 抛出异常

## 9) list_dir

函数签名:  
`def list_dir(self, dir_path: str, iteration: bool = False) -> dict`

> 读取指定路径的目录结构  

参数:  
- `dir_path`:目录路径（字符串）。
- `iteration`:`True` 时递归读取子目录。

返回:  
1. 若路径为文件:`{filename: "file"}`
2. 若路径为目录且 `iteration=False`:`{name: "file"|"dir", ...}`
3. 若路径为目录且 `iteration=True`:嵌套字典结构，叶子为 `"file"`，空目录为 `dir`

## 10) load_tools_from_python_file

函数签名:  
`def load_tools_from_python_file(self, skill_name: str, python_file: Path) -> dict`

> 从指定 Python 文件提取可注册函数并加载为TOOL。  

参数:  
- `skill_name`:目标 SKILL 名称。
- `python_file`:Python 文件路径。

返回:  
- 成功:新加载工具的说明字典。

## 11) execute_script

函数签名:  
`execute_script(self, skill_name: str, script_path: str, arguments: list[str] = "") -> dict:`

> 执行外部脚本（高风险操作），捕捉 `stdout` 和 `stderr`。

参数:  
- `skill_name`:SKILL 名称。
- `script_path`:脚本路径。
- `arguments`:一个字符串列表，可选输入参数，严格顺序。

返回:  
- 成功/失败:脚本执行结果的字典。

## 12) is_skill

函数签名:  
`def is_skill(self, skill_path: str) -> bool`

> 判断指定目录是否合法SKILL

参数:  
- `skill_path`:目录路径。

返回:  
- `True`:是 SKILL。
- `False`:不是 SKILL。

## 13) download_file

函数签名:  
`async def download_file(self, url: str, filename: str = "", save_dir: str = "", extra_parameters: dict = None, overwrite: bool = False):`

> 调用 httpx get 方法下载文件，并保存到本地。会直接提取 response.content，因此只在确保ULR为文件/图片/视频等（二进制数据）内容时才调用

参数:  
- `url`:目标URL。
- `filename`:落盘的文件名，为空则自动创建。
- `save_dir`:落盘的目录，为空则使用 `%TEMP%` 。
- `extra_parameters`:httpx支持的参数，如 `headers`, `cookies` 等，可选。可根据错误信息自行传入。
- `overwrite`:是否覆盖现有文件，默认否。若是: 覆盖文件；若否: 文件存在时抛出异常。

返回:  
- 成功:返回结果字典，包含 `path` 和 `hash_name`
- 失败:抛出异常  

## 14) get_hash_name

函数签名:  
`def get_hash_name(self, path: str):`  

> 获取指定文件的 `hash_name`  

参数:  
- `path`: 文件路径字符串  

返回:  
- 成功: 返回该文件的 `hash_name` 字符串\
- 失败: 抛出异常


# 推荐使用方式  

## 管理/使用SKILL  

1. 调用`list_skills`查看已安装SKILL，确保要使用的SKILL存在  
2. 调用`load_skill`查看SKILL标题树  
3. 除非需要详细信息，否则调用`get_skill_content`时仅传递需要的标题树，尽量不要直接获取全文以节省context

## 读写文件  

1. 不确定文件目录情况时，先调用一次`list_dir`查看目录  
2. 读取文件时，可通过`start_line`/`end_line`参数先读取文件头n行/尾n行，来判断文件大致内容，以及总行数  
3. 再根据需要决定是否读取文件全文  
4. 编辑文件时，可优先替换指定行数内容  
5. 创建新文件或全量写文件时，可以不指定`start_line`/`end_line`参数，以此直接替换文件