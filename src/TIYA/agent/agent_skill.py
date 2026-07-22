from __future__ import annotations
"""
agent_skill.py
内置的 SKILL 管理器，存放基础操作逻辑
"""
__version__ = "0.1.1"

import time
import shutil
import importlib.util
import inspect
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from hashlib import blake2b
from typing import Callable, Awaitable, Any, TypedDict
from ruamel.yaml import YAML
from PIL import Image
from TIYA.markdown_heading_tree import parse_markdown_to_nested_dict
from TIYA.LLM_connect import build_tool_payloads
from TIYA.config import BASE_SKILL_DIR, ROOT_DIR, SETTING_CFG
from TIYA.utils import httpx_downloader
from TIYA.file_cache import add_file, add_file_async

_PATH_BLACKLIST = {ROOT_DIR / "config"}

class SkillUninstalledError(KeyError):
    """尝试加载尚未安装的SKILL"""
    ...


class DownloadError(RuntimeError):
    """下载失败"""
    ...


@dataclass(slots=True)
class SKILL:
    name: str
    description: str
    root: Path
    version: str = ""
    metadata: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)
    tools: dict[str, Callable[..., Awaitable | Any]] = field(default_factory=dict)
    tools_name_mapping: dict = field(default_factory=dict)
    md_content: dict = field(default_factory=dict)
    md_title_tree: dict = field(default_factory=dict)
    last_use: float = field(default_factory=time.time)

    def clear_all(self):
        self.metadata.clear()
        self.data.clear()
        self.tools.clear()
        self.tools_name_mapping.clear()
        self.md_content.clear()
        self.md_title_tree.clear()

    def __del__(self):
        self.clear_all()


@dataclass(slots=True)
class SkillMetadata:
    name: str
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)
    body: str = ""


class SkillManager:
    """
    对skill进行存储、管理，并提供一些基础操作接口
    """

    def __init__(self, data_path: Path, register_helper: Callable, uninstall_helper: Callable, image_helper: Callable):
        self.skills_dir = data_path / "skill"
        self.temp_dir = data_path / "temp"
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self._register_helper = register_helper
        self._uninstall_helper = uninstall_helper
        self._image_helper = image_helper
        self._yaml = YAML(typ="safe")
        self.skills: dict[str, SKILL] = {}  # {skill_name: SKILL}
        self.tools_default_setting: dict[str, dict] = {
            "skill_installer": {
                "install_skill": {"wait": True, "callback": "CALL"},
                "load_skill": {"wait": True, "callback": "CALL"},
                "list_skills": {"wait": True, "callback": "CALL"},
                "uninstall_skill": {"wait": True, "callback": "CALL"},
                "get_skill_content": {"wait": True, "callback": "CALL"},
                "get_content": {"wait": True, "callback": "CALL"},
                "set_content": {"wait": True, "callback": "CALL"},
                "remove_path": {"wait": True, "callback": "EXCEPTION"},
                "list_dir": {"wait": True, "callback": "CALL"},
                "load_tools_from_python_file": {"wait": True, "callback": "CALL"},
                "execute_script": {"wait": False, "callback": "CALL"},
                "is_skill": {"wait": True, "callback": "CALL"},
                "download_file": {"wait": False, "callback": "ALWAYS"},
                "get_hash_name": {"wait": True, "callback": "CALL"},
            },
            "websearch": {
                "search": {"wait": False, "callback": "ALWAYS"},
                "url_map": {"wait": False, "callback": "ALWAYS"},
                "extract": {"wait": False, "callback": "ALWAYS"},
                "raw_crawl": {"wait": False, "callback": "ALWAYS"}
            }
        }

    """
    主要操作接口
    """

    def install_skill(self, path: str) -> list[dict[str, str]]:
        """
        安装指定路径的SKILL。
        :param path: SKILL文件夹的路径。
        :return: 成功: 当前所有已安装的SKILL的列表; 失败: 抛出异常
        """
        source = self._resolve_path(path)
        if not source.is_dir():
            raise FileNotFoundError(f"SKILL path is not a folder: {self._decorate_path(source)}")

        if not self.is_skill(str(source)):
            raise ValueError(f"Invalid SKILL folder: {self._decorate_path(source)}")

        skill_metadata = self._read_skill_frontmatter(source / "SKILL.md")
        if skill_metadata.name in self.skills:
            raise FileExistsError(f"SKILL already installed: {skill_metadata.name}")

        target = (self.skills_dir / source.name).resolve()
        if source != target:
            if source.parent == BASE_SKILL_DIR:
                target = source

            else:
                if target.exists():
                    raise FileExistsError(f"Has same SKILL folder name in Agent skill root: {self._decorate_path(target)}")

                shutil.copytree(source, target)

        self._register_skill(skill_metadata, target)
        return self.list_skills()

    def load_skill(self, skill_name: str) -> dict:
        """
        加载已安装的SKILL，返回SKILL.md的标题树。
        :param skill_name: SKILL的名字。
        :return: 成功: 返回嵌套字典表示的SKILL.md标题树，不含内容
        """
        if skill_name not in self.skills:
            raise SkillUninstalledError(f"<{skill_name}> not installed")

        skill = self.skills[skill_name]
        skill.last_use = time.time()
        return skill.md_title_tree.copy()

    def list_skills(self) -> list[dict[str, str]]:
        """
        列出当前所有已安装的SKILL
        :return: 成功: 一个字典列表[{skill_name: skill_description}]
        """
        return [{k: v.description} for k, v in self.skills.items()]

    def uninstall_skill(self, skill_name: str) -> list[dict[str, str]]:
        """
        卸载指定名字的SKILL。
        :param skill_name: SKILL的名字。
        :return: 成功: 当前所有已安装的SKILL的列表
        """
        if skill_name not in self.skills:
            return self.list_skills()

        skill: SKILL = self.skills[skill_name]
        if skill.root.parent == BASE_SKILL_DIR:
            raise PermissionError(f"Base skill <{skill_name}> can't uninstall")

        disable_folder = self.skills_dir / ".disable"
        disable_folder.mkdir(exist_ok=True, parents=True)
        new_path = disable_folder / skill.root.name
        self.skills.pop(skill_name, None)
        self._uninstall_helper(skill_name)
        shutil.move(skill.root, new_path)
        return self.list_skills()

    def get_skill_content(self, skill_name: str, title_tree: dict) -> str:
        """
        读取指定SKILL的指定标题的内容，应先调用`load_skill`获取完整标题树。
        :param skill_name: SKILL的名字。
        :param title_tree: 一个标题嵌套字典，值为空字典时返回该处的内容。
        <示例1: 所有内容> {};
        <示例2: 一级标题A和一级标题B下的二级标题A的所有内容> {skill_name: {"一级标题A": {}, "一级标题B": {"二级标题A": {}}}}。
        :return: 成功: 由指定标题树的内容重建成的markdown文本; 标题树键名错误: error_message(string)
        """
        if skill_name not in self.skills:
            raise SkillUninstalledError(f"<{skill_name}> not installed")

        skill: SKILL = self.skills[skill_name]
        skill.last_use = time.time()
        content_tree = skill.md_content.copy()

        if title_tree:
            issues = self._check_title_tree(title_tree, content_tree)
            if issues:
                return f"标题树有误: \n{issues}"

        def _recursion_get_content(_content: dict, _level: int = 1) -> list[str]:
            _collect = []
            for _k, _v in _content.items():
                if _k in {"_content", "_preamble"}:
                    _collect.append(_v)
                    continue

                _collect.append(f"{'#'*_level} {_k}  \n")
                if isinstance(_v, dict):
                    _g = _recursion_get_content(_v, _level + 1)
                    if _g:
                        _collect.extend(_g)

                else:
                    _collect.append(_v)

            return _collect

        def _recursion_parse_title(_title_tree: dict, _content: dict, _level: int = 1) -> list[str]:
            _collect = []
            if "_preamble" in _content:
                _collect.append(_content["_preamble"])

            if "_content" in _content:
                _collect.append(_content["_content"])

            for _k, _v in _title_tree.items():
                _collect.append(f"{'#'*_level} {_k}  \n")
                if not _v:
                    _g = _recursion_get_content(_content[_k], _level + 1)
                    if _g:
                        _collect.extend(_g)

                else:
                    _g = _recursion_parse_title(_title_tree[_k], _content[_k], _level + 1)
                    if _g:
                        _collect.extend(_g)

            return _collect

        if not title_tree:
            return '\n'.join(_recursion_get_content(content_tree))

        return '\n'.join(_recursion_parse_title(title_tree, content_tree))

    """
    其他辅助接口
    """

    def get_content(
            self,
            path: str,
            image_mode: bool = False,
            encoding: str = "utf-8",
            start_line: int = 0,
            end_line: int = 0
    ) -> dict:
        """
        读取指定文本文件内容，也支持读取图像文件并上传。
        :param path: 文件路径。
        :param image_mode: 是否读取图像文件，若True，则后续参数无效，同时返回一个读取提示，并在下轮对话消息中上传图片
        :param encoding: 文件编码方式，默认utf-8。
        :param start_line: 从n行开始读取，若为0则从头开始;  支持负数，-1表示从倒数第一行开始。
        :param end_line: 读取到第n行，若为0则读取到尾;  支持负数，-1表示读到倒数第一行。
        :return: 成功: 一个{行数:内容}的字典: {'1': line1, ..., '999': line999}
        """
        file_path = self._resolve_path(path)
        if any(file_path.is_relative_to(p) for p in _PATH_BLACKLIST):
            raise PermissionError("Access denied")

        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {self._decorate_path(file_path)}")

        decorate_path = self._decorate_path(file_path)
        if image_mode:
            # 只是通过报错来判断是否图片
            try:
                with Image.open(file_path):
                    pass

            except Image.UnidentifiedImageError:
                raise TypeError(f"<{decorate_path}> is not image")

            return {"image": {
                "path": f"{decorate_path}",
                "index": self._image_helper(file_path)
            }}

        text = file_path.read_text(encoding=encoding)
        text = text.replace(str(self.skills_dir), "%SKILL%").replace(str(self.temp_dir), "%TEMP%")
        lines = text.splitlines(keepends=True)
        if start_line == 0 and end_line == 0:
            return {str(i): c for i, c in enumerate(lines, start=1)}

        s, e = self._normalize_line_range(start_line, end_line, len(lines))
        return {str(i + 1): c for i, c in zip(range(s - 1, e), lines[s - 1:e])}

    def set_content(
            self,
            path: str,
            content: str = "",
            start_line: int = 0,
            end_line: int = 0
    ) -> str:
        """
        使用 utf-8 写入文本文件。 当文件已存在时会直接覆盖/修改;
        可使用start/end_line替换指定行数范围的内容;
        当content实际行数与指定的范围行数不匹配时，会自动拓展/收缩;
        创建文件时优先选用临时目录 <%TEMP%> 。
        :param path: 写入的文件路径。
        :param content: 要写入/修改的内容，使用'\n'作为换行符。
        :param start_line: 从n行开始替换;  支持负数，-1表示从倒数第一行开始。
        :param end_line: 替换到第n行，若为0则替换到尾; 支持负数，-1表示替换到倒数第一行。
        :return: 成功: 返回创建/修改的文件路径
        """
        file_path = self._resolve_path(path)
        if any(file_path.is_relative_to(p) for p in _PATH_BLACKLIST):
            raise PermissionError("Access denied")

        if not (file_path.is_relative_to(self.temp_dir) or file_path.is_relative_to(self.skills_dir)):
            raise PermissionError("Can only write file below %TEMP% or %SKILL%")

        edit_mode = (start_line > 0) or (end_line != 0)
        content = content.replace("%SKILL%", str(self.skills_dir)).replace("%TEMP%", str(self.temp_dir))
        if not edit_mode:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            return self._decorate_path(file_path)

        if not file_path.is_file():
            raise FileNotFoundError(f"File not found for line range update: {self._decorate_path(file_path)}")

        original = file_path.read_text(encoding="utf-8")
        lines = original.splitlines(keepends=True)
        if not lines:
            if start_line not in (0, 1):
                raise ValueError("Empty file only supports start_line=0 when range mode enabled")

            file_path.write_text(content, encoding="utf-8")
            return self._decorate_path(file_path)

        if not content.endswith('\n'):
            content += '\n'

        s, e = self._normalize_line_range(start_line, end_line, len(lines))
        replacement = content.splitlines(keepends=True)
        new_lines = lines[:s - 1] + replacement + lines[e:]
        file_path.write_text("".join(new_lines), encoding="utf-8")
        return self._decorate_path(file_path)

    def remove_path(self, path: str) -> str:
        """
        删除指定路径的文件/文件夹，仅允许删除%TEMP%目录内的路径，超出范围会拒绝。
        :param path: 删除的文件路径。
        :return: 成功: 返回成功提示;  失败: 抛出异常
        """
        path = self._resolve_path(path)
        if not path.exists():
            return f"<{self._decorate_path(path)}> not exist"

        if path.is_relative_to(self.temp_dir) and path != self.temp_dir:
            if path.is_file():
                path.unlink(missing_ok=True)

            else:
                shutil.rmtree(path)

            return f"<{self._decorate_path(path)}> has been removed"

        else:
            raise PermissionError("Can only remove path below %TEMP%")

    def list_dir(self, dir_path: str, iteration: bool = False) -> dict:
        """
        读取指定目录路径下的文件。
        :param dir_path: 文件夹路径
        :param iteration: 若为True，则迭代读取所有子目录
        :return: 成功: 一个字典:  1. 给定的path是文件: {filename: "file"};
        2. 给定的path是目录且iteration为False: {filename/dirname: "file"/"dir", ...};
        3. 给定的path是目录且iteration为True: 嵌套字典，叶子为{dirname: "dir" # 空目录, ..., filename: "file"}
        """
        path = self._resolve_path(dir_path)
        if any(path.is_relative_to(p) for p in _PATH_BLACKLIST):
            raise PermissionError("Access denied")

        if path.is_file():
            return {path.name: "file"}

        if not path.is_dir():
            raise FileNotFoundError(f"Path not found: {self._decorate_path(path)}")

        if not iteration:
            result = {}
            for child in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                result[child.name] = "dir" if child.is_dir() else "file"

            return result

        def _walk(p: Path) -> dict:
            out = {}
            for _child in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                if _child.is_dir():
                    if not any(_child.iterdir()):
                        out[_child.name] = "dir"

                    else:
                        out[_child.name] = _walk(_child)

                else:
                    out[_child.name] = "file"

            return out

        return _walk(path)

    def load_tools_from_python_file(self, skill_name: str, python_file: str) -> dict:
        """
        从指定py文件获取tools并注册到agent。
        :param skill_name: SKILL的名字。
        :param python_file: py文件的路径。
        :return: 成功: 一个字典，为新增tools的详细说明
        """
        if skill_name not in self.skills:
            raise SkillUninstalledError(f"<{skill_name}> not installed")

        path = self._resolve_path(python_file)
        if not path.is_file():
            raise FileNotFoundError(f"<{self._decorate_path(path)}> not found")

        h = blake2b(str(path).encode(), digest_size=8)
        module_name = f"skill_script_{h.hexdigest()}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None:
            raise ImportError(f"{self._decorate_path(path)} can not load")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        tools = {}
        for name, obj in inspect.getmembers(module):
            if not name.startswith('_') and (inspect.isfunction(obj) or inspect.iscoroutinefunction(obj)):
                if obj.__module__ != module.__name__:
                    continue

                tools[name] = obj

        if not tools:
            raise ImportError(f"Doesn't find any tool in <{self._decorate_path(path)}>")

        skill: SKILL = self.skills[skill_name]
        skill.last_use = time.time()
        tools_list = []
        for k, v in tools.items():
            skill.tools[k] = v
            self._register_helper(function=v, source=path, source_name=skill_name)
            instruction = build_tool_payloads(v)["openai"]["function"]
            tool = {"source_name": skill_name, "tool_name": k, **instruction}
            tool.pop("name")
            tools_list.append(tool)

        output = {"new tools from file": {skill_name: tools_list}}
        return output

    def execute_script(self, skill_name: str, script_path: str, arguments: list[str] = None) -> dict:
        """
        调用外部脚本，使用子进程，会自动处理输入参数。
        :param skill_name: SKILL的名字。
        :param script_path:  脚本入口/文件的路径。
        :param arguments:  可选字符串输入参数，用列表表示，严格顺序。
        :return: 脚本执行结果
        """
        import subprocess
        import sys

        if skill_name not in self.skills:
            raise SkillUninstalledError(f"<{skill_name}> not installed")

        if arguments is None:
            arguments = []

        for _ in range(len(arguments)):
            arguments[_] = arguments[_].replace("%SKILL%", str(self.skills_dir)).replace("%TEMP%", str(self.temp_dir))

        skill = self.skills[skill_name]
        skill.last_use = time.time()
        path = self._resolve_script_path(skill, script_path)
        if not path.is_file():
            raise FileNotFoundError(f"Script not found: {self._decorate_path(path)}")

        if path.suffix.lower() == ".py":
            completed = subprocess.run(
                [sys.executable, str(path)] + arguments,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=300,
                check=False,
            )

        else:
            completed = subprocess.run(
                [str(path)] + arguments,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=300,
                check=False,
                cwd=str(skill.root)
            )

        stdout = completed.stdout or ""
        stdout = stdout.replace(str(self.skills_dir), "%SKILL%").replace(str(self.temp_dir), "%TEMP%")
        stderr = completed.stderr or ""
        stderr = stderr.replace(str(self.skills_dir), "%SKILL%").replace(str(self.temp_dir), "%TEMP%")

        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": stdout,
            "stderr": stderr
        }

    def is_skill(self, skill_path: str) -> bool:
        """
        判断一个路径是否为SKILL。
        :param skill_path: 指定文件夹路径。
        :return: 成功: 是 True;  不是 False。
        """
        try:
            root = self._resolve_path(skill_path)

        except Exception:
            return False

        if not root.is_dir():
            return False

        skill_md = root / "SKILL.md"
        if not skill_md.is_file():
            return False

        try:
            self._read_skill_frontmatter(skill_md)

        except Exception:
            return False

        return True

    async def download_file(
            self,
            url: str,
            filename: str = "",
            save_dir: str = "",
            extra_parameters: dict = None,
            overwrite: bool = False
    ) -> dict[str, str]:
        """
        从指定URL下载文件，只有确定URL带有二进制内容时方可调用
        :param url: 目标URL
        :param filename: 保存到本地的文件名.
        :param save_dir: 保存到本地的目录.
        :param extra_parameters: httpx支持的参数，例如 `headers`, `cookies` 等
        :param overwrite: 默认否。若为真，则覆盖已存在文件；为否则抛出异常
        :return: 成功: 返回保存到本地的路径; 失败: 抛出异常
        """
        class _DownloadResult(TypedDict):
            message: str
            path: str
            hash_name: str

        if extra_parameters is None:
            extra_parameters = {}

        extra_parameters.setdefault("follow_redirects", True)
        if not filename:
            filename = url.split("/")[-1].split("?")[0]
            if not filename:
                dt = datetime.now()
                filename = f"downloaded_file_{dt:%y%m%d_%H%M%S}"

        if not save_dir:
            save_dir = self.temp_dir

        else:
            save_dir = self._resolve_path(save_dir)
            if any(save_dir.is_relative_to(p) for p in _PATH_BLACKLIST):
                raise PermissionError("Access denied")

        file_path = save_dir / filename
        if file_path.exists() and not overwrite:
            raise FileExistsError(f"File <{self._decorate_path(file_path)}> already exist")

        response = await httpx_downloader(
            url=url,
            retry=3,
            timeout=SETTING_CFG.Agent.FileDownloadTimeout,
            **extra_parameters
        )
        if not response.ok:
            raise DownloadError(f"File <{filename}> download fail")

        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(response.content)
        return dict(_DownloadResult(
            message="已完成文件下载",
            path=self._decorate_path(file_path),
            hash_name=await add_file_async(file_path)
        ))

    def get_hash_name(self, path: str) -> str:
        """
        获取指定文件的 hash_name。
        :param path: 文件路径。
        :return: 成功: 该文件的 `hash_name` 字符串; 失败: 抛出异常
        """
        file_path = self._resolve_path(path)
        if any(file_path.is_relative_to(p) for p in _PATH_BLACKLIST):
            raise PermissionError("Access denied")

        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {self._decorate_path(file_path)}")

        return add_file(file_path)

    """
    内部辅助函数
    """

    def initiate(self):
        source = BASE_SKILL_DIR / "skill_installer"
        self._install_exist_skills()
        self._register_helper(
            self.install_skill,
            source = source,
            source_name = "skill_installer",
            display_name = "安装SKILL"
        )
        self._register_helper(
            self.load_skill,
            source=source,
            source_name="skill_installer",
            display_name="加载SKILL信息"
        )
        self._register_helper(
            self.list_skills,
            source=source,
            source_name="skill_installer",
            display_name="读取SKILL列表"
        )
        self._register_helper(
            self.uninstall_skill,
            source=source,
            source_name="skill_installer",
            display_name="卸载SKILL"
        )
        self._register_helper(
            self.get_skill_content,
            source=source,
            source_name="skill_installer",
            display_name="获取SKILL内容"
        )
        self._register_helper(
            self.get_content,
            source=source,
            source_name="skill_installer",
            display_name="读取文件"
        )
        self._register_helper(
            self.set_content,
            source=source,
            source_name="skill_installer",
            display_name="编辑文件"
        )
        self._register_helper(
            self.remove_path,
            source=source,
            source_name="skill_installer",
            display_name="删除目录"
        )
        self._register_helper(
            self.list_dir,
            source=source,
            source_name="skill_installer",
            display_name="获取文件目录"
        )
        self._register_helper(
            self.load_tools_from_python_file,
            source=source,
            source_name="skill_installer",
            display_name="加载外部工具"
        )
        self._register_helper(
            self.execute_script,
            source=source,
            source_name="skill_installer",
            display_name="执行外部脚本"
        )
        self._register_helper(
            self.is_skill,
            source=source,
            source_name="skill_installer",
            display_name="判断SKILL"
        )
        self._register_helper(
            self.download_file,
            source=source,
            source_name="skill_installer",
            display_name="下载文件"
        )
        self._register_helper(
            self.get_hash_name,
            source=source,
            source_name="skill_installer",
            display_name="获取文件的哈希名字"
        )

    def _install_exist_skills(self):
        for path in BASE_SKILL_DIR.iterdir():
            try:
                self.install_skill(str(path))

            except ValueError:
                continue

        for path in self.skills_dir.iterdir():
            try:
                self.install_skill(str(path))

            except ValueError:
                continue

    def _resolve_path(self, path: str | Path) -> Path:
        p = str(path)
        p = p.replace("%SKILL%", str(self.skills_dir))
        p = p.replace("%TEMP%", str(self.temp_dir))
        return Path(p).expanduser().resolve()

    def _decorate_path(self, path: str | Path) -> str:
        path = str(path)
        path = path.replace(str(self.skills_dir), "%SKILL%")
        path = path.replace(str(self.temp_dir), "%TEMP%")
        return path

    def _read_skill_frontmatter(self, skill_md: Path) -> SkillMetadata:
        text = skill_md.read_text(encoding="utf-8")
        frontmatter, body = self._split_frontmatter(text, skill_md)
        data = self._yaml.load(frontmatter)
        if not isinstance(data, dict):
            raise TypeError(f"front matter must be mapping: {skill_md}")

        name = str(data.pop("name", "")).strip()
        description = str(data.pop("description", "")).strip()

        if not name:
            raise ValueError(f"missing frontmatter key 'name': {skill_md}")

        if not description:
            raise ValueError(f"missing frontmatter key 'description': {skill_md}")

        return SkillMetadata(name, description, data, body)

    @staticmethod
    def _split_frontmatter(text: str, source: Path) -> tuple[str, str]:
        if text.startswith("\ufeff"):
            text = text[1:]

        lines = text.replace("\r\n", "\n").split("\n")
        if not lines:
            raise ValueError(f"front matter start not found: {source}")

        start = 0
        while start < len(lines) and not lines[start].strip():
            start += 1

        if start >= len(lines) or lines[start].strip() != "---":
            raise ValueError(f"front matter start not found: {source}")

        end = -1
        for i in range(start + 1, len(lines)):
            if lines[i].strip() == "---":
                end = i
                break

        if end < 0:
            raise ValueError(f"front matter end not found: {source}")

        header = "\n".join(lines[start + 1:end]).strip()
        body = "\n".join(lines[end + 1:]).strip()
        return header, body

    def _resolve_script_path(self, skill: SKILL, path: Path | str):
        p = str(path)
        p = p.replace("%SKILL%", str(self.skills_dir))
        p = p.replace("%TEMP%", str(self.temp_dir))
        p = Path(p).expanduser()
        if p.is_absolute():
            return p

        return (skill.root / p).resolve()

    @staticmethod
    def _get_title_tree(content_tree: dict):
        out = {}
        for k, v in content_tree.items():
            if k in {"_preamble", "_content"}:
                continue

            if isinstance(v, dict):
                out[k] = SkillManager._get_title_tree(v)

        return out

    @staticmethod
    def _check_title_tree(title_tree: dict, content_tree: dict) -> dict:
        """对字典树进行校验"""
        def _recursion(_tr: dict, _src: dict):
            _iss = {}
            for k, v in _tr.items():
                if k not in _src:
                    _iss[k] = f"标题 [{k}] 不存在"
                    continue

                if isinstance(v, dict):
                    if not isinstance(_src[k], dict):
                        _iss[k] = f"标题 [{k}] 已无子标题"
                        continue

                    _iss[k] = _recursion(v, _src[k])

                else:
                    _iss[k] = f"标题 [{k}] 的值应为一个字典"

            return _iss

        def _clear(tree: dict):
            for k, v in tree.copy().items():
                if isinstance(v, dict):
                    _clear(tree[k])

                if not v:
                    tree.pop(k)

        if not title_tree:
            return {}

        issues = _recursion(title_tree, content_tree)
        _clear(issues)
        return issues

    def _register_skill(self, data: SkillMetadata, source: Path) -> SKILL:
        version = str(data.metadata.get("version", "")).strip()
        new_skill = SKILL(data.name, data.description, source, version, data.metadata.copy())
        content_tree = parse_markdown_to_nested_dict(data.body)
        title_tree = self._get_title_tree(content_tree)
        new_skill.md_content = {data.name: content_tree}
        new_skill.md_title_tree = {data.name: title_tree}

        self.skills[new_skill.name] = new_skill
        return new_skill

    @staticmethod
    def _normalize_line_range(start: int, end: int, length: int) -> tuple[int, int]:
        if start < 0:
            start = length + start

        end = min(end, length)
        if end < 0:
            end = length + end

        elif end == 0:
            end = length

        start = max(1, start)
        if start > end:
            raise ValueError("Start can't greater than End")

        return start, end
