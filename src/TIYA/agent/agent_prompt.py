import asyncio
import inspect
import importlib.util
import frontmatter
from dataclasses import dataclass, field
from typing import Awaitable, Callable
from pathlib import Path
from hashlib import blake2b


def parse_prompt_from_markdown(filename: str = None, filedir: Path = None, filepath: Path = None) -> str:
    """从md文件读取，暂不做图片和超链接处理"""
    if isinstance(filepath, Path):
        target = filepath

    else:
        if filename is None:
            raise FileNotFoundError("未指定markdown文件")

        if filedir is None:
            frame = inspect.currentframe()
            try:
                caller = frame.f_back
                caller_file = caller.f_globals.get("__file__")
                if caller_file:
                    filedir = Path(caller_file).resolve().parent

                else:
                    filedir = Path.cwd()

            finally:
                del frame

        target = filedir / filename

    if not target.is_file():
        raise FileNotFoundError(f"{target} is not found")

    md = frontmatter.load(str(target))
    content = md.content
    return content

def _parse_prompt_from_py(path: Path) -> tuple[Callable | Awaitable, dict, str]:
    """
    从符合格式的py文件读取动态prompt方法
    :param path: 文件夹路径
    :return: 动态prompt方法, tools函数集, prompt version
    """
    if not path.is_dir():
        raise FileNotFoundError(f"{path} is not a folder")

    prompt_file = path / "prompt.py"
    if not prompt_file.is_file():
        raise FileNotFoundError(f"{prompt_file} is not found")

    h = blake2b(str(path).encode(), digest_size=8)
    module_name = f"prompt_{h.hexdigest()}"
    spec = importlib.util.spec_from_file_location(module_name, prompt_file)
    if spec is None:
        raise ImportError(f"{prompt_file} can not load")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    tools = {}
    prompt_function = None
    version = "#UNDEFINE"
    builtin_metadata = {'__name__', '__doc__', '__file__', '__package__',
                        '__loader__', '__spec__', '__builtins__', '__cached__'}
    for name, obj in inspect.getmembers(module):
        if not name.startswith('_') and (inspect.isfunction(obj) or inspect.iscoroutinefunction(obj)):
            if obj.__module__ != module.__name__:
                continue

            if name == "prompt":
                prompt_function = obj

            else:
                tools[name] = obj

        elif name.startswith("__") and name.endswith("__"):
            if name not in builtin_metadata:
                if name == "__version__":  # 这里先收集version，以后可以补充其他元数据
                    version = obj

    if prompt_function is None:
        raise ImportError(f"{str(path)} hasn't implement prompt() method")

    return prompt_function, tools, version


@dataclass(slots=True)
class AgentPrompt:
    name: str
    data_path: Path  # 文件夹的路径
    description: str = ""
    role: str = ""
    input_function: dict[str, Callable] = field(default_factory=dict)  # 可注册函数自动填充参数
    _prompt_function : Callable = field(init=False)
    tools: dict[str, Callable] = field(init=False)
    version: str = field(init=False)

    def __post_init__(self):
        prompt_function, tools, version = _parse_prompt_from_py(self.data_path)
        self._prompt_function = prompt_function
        self.tools = tools
        self.version = version

    async def get_prompt(self, *args, input_params: dict[str, dict] = None, **kwargs) -> str:
        if input_params is None:
            input_params = {}

        params = {}
        if self.input_function:
            tasks: dict[str, asyncio.Task] = {}
            for key, func in self.input_function.items():
                p = input_params.get(key, {})
                if inspect.iscoroutinefunction(func):
                    tasks[key] = asyncio.create_task(func(**p))

                elif callable(func):
                    params[key] = func(**p)

            if tasks:
                await asyncio.gather(*tasks.values())
                for k, v in tasks.items():
                    params[k] = v.result()

        params.update(kwargs)
        kwargs = params
        if inspect.iscoroutinefunction(self._prompt_function):
            prompt = await self._prompt_function(*args, **kwargs)

        else:
            prompt = self._prompt_function(*args, **kwargs)

        if not isinstance(prompt, str):
            raise TypeError("Prompt must be str")

        return prompt