from __future__ import annotations
import asyncio
import copy
import datetime
import inspect
import hashlib
import time

from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, overload, Literal
from uuid import uuid4

from TIYA.time_id import next_time_id
from TIYA.utils import fingerprint
from TIYA.file_cache import add_file_async, get_file_path_async
from .legacy_media import (
    _IMAGE_NAME_MAPPING,
    _load_image_hashes,
    handle_unload_image,
)
from .tool_schema import ToolPayloads, build_tool_payloads
from .usage_model import TokenUsage as TokenUsage
from .config_model import (
    Params,
)

type MediaKind = Literal["path", "url", "bytes", "base64", "remote"]
type Role = Literal["user", "tool", "assistant", "function", "system"]


def stable(value: object) -> dict | list | object:
    """转换签名输入：消息读取 signature，字节取摘要，callable 展开 schema。

    只构造用于比较的数据，不执行 Prompt.to_dict() 或图片文件操作。
    """
    if isinstance(value, bytes):
        return {"sha256": hashlib.sha256(value).hexdigest()}

    if isinstance(value, dict):
        return {k: stable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [stable(v) for v in value]

    if callable(value):
        return build_tool_payloads(value)

    if isinstance(value, BasePrompt):
        return value.signature

    if hasattr(value, "__dict__"):
        return stable(vars(value))

    return value


class ContextOverflowError(Exception):
    """超出模型context窗口错误"""

    pass


class ContextEmptyError(Exception):
    pass


class PromptType(Enum):
    BASE = "BASE"
    SYSTEM = "SYSTEM"
    TEXT = "TEXT"
    IMAGE = "IMAGE"
    TOOL_IN = "TOOL_IN"
    TOOL_CALL = "TOOLCALL"
    TOOL_RESP = "TOOL_RESP"
    TOKEN = "TOKEN"


class BasePrompt:
    """单条消息；id 表示身份，signature 缓存消息内容签名。

    按约定不原位修改已进入上下文的消息。没有属性监听或自动变更检测，
    框架自己的内容修改入口负责显式刷新；native 是另存的协议附件。
    """

    # 计算缓存独立于 __dict__，不参与消息指纹、原生回放校验和序列化。
    __slots__ = ("__dict__", "__weakref__", "estimated_tokens", "_assumed_tokens", "_signature")

    estimated_tokens: int | None  # 当前进程的 o200k 估算缓存，非供应商 usage。

    def __init__(self, role: Role, contain_type: str, create_time: str = None):
        if create_time is None:
            create_time = str(time.time())

        self.role: Role = role
        self.contain_type: str = contain_type
        self.time: str = create_time
        self.id: str = str(next_time_id())
        self.native: dict = {}
        self.estimated_tokens = None
        self._assumed_tokens: int | None = None  # 账本字符估算缓存，不调用 tokenizer。
        self._signature: str | None = None

    def _signature_data(self) -> dict[str, object]:
        return {key: value for key, value in vars(self).items() if key != "native"}

    @property
    def signature(self) -> str:
        """消息签名的统一入口，首次读取计算，之后缓存到显式刷新。

        包含 ID、角色、时间和消息内容，不含 native 协议附件及运行期缓存。
        不监控原位写入或嵌套对象修改；调用方直接修改后未刷新时，一致性不保证。
        容器及 Provider 读取本属性，不自行展开消息内容计算另一套指纹。
        """
        if self._signature is None:
            self._signature = fingerprint(stable(self._signature_data()))

        return self._signature

    def refresh_fingerprints(self) -> None:
        """显式清除消息签名和 token 估算；已有容器缓存由所属容器刷新。"""
        self._signature = None
        self.estimated_tokens = None
        self._assumed_tokens = None

    def to_dict(self) -> dict:
        return {
            "type": PromptType.BASE.value,
            "role": self.role,
            "contain_type": self.contain_type,
            "time": self.time,
            "id": self.id,
            **({"native": copy.deepcopy(self.native)} if self.native else {}),
        }

    def __eq__(self, other):
        if isinstance(other, BasePrompt):
            return self.role == other.role and self.contain_type == other.contain_type

        else:
            return False

    def __str__(self):
        dt = datetime.datetime.fromtimestamp(float(self.time))
        output = f"[{self.role}] {dt:%Y-%m-%d %H:%M:%S}: "
        return output

    def __add__(self, other):
        return Prompts(self, other)


class SystemPrompt(BasePrompt):
    def __init__(self, text: str, create_time: str = None):
        super().__init__("system", "TEXT", create_time)
        self.text = text

    def to_dict(self) -> dict:
        base_dict = super().to_dict()
        base_dict["type"] = PromptType.SYSTEM.value
        base_dict["data"] = {"text": self.text}
        return base_dict

    def copy(self):
        return self.__copy__()

    @classmethod
    def from_dict(cls, data: dict):
        new_prompt = SystemPrompt(data["data"]["text"], data["time"])
        new_prompt.id = data["id"]
        return new_prompt

    def __copy__(self):
        result = SystemPrompt(self.text, create_time=self.time)
        result.native = self.native.copy()
        result.estimated_tokens = self.estimated_tokens
        result._assumed_tokens = self._assumed_tokens
        return result

    def __eq__(self, other):
        if isinstance(other, SystemPrompt):
            return super().__eq__(other) and self.text == other.text

        else:
            return False

    def __str__(self):
        output = super().__str__()
        output += self.text
        output += "\n"
        return output

    def __bool__(self):
        return bool(self.text)


class TextPrompt(BasePrompt):
    def __init__(
        self, role: Role, text: str, reasoning_content: str = "", create_time: str = None
    ):
        super().__init__(role, "TEXT", create_time)
        self.text: str = text
        self.reasoning_content: str = reasoning_content

    def to_dict(self) -> dict:
        base_dict = super().to_dict()
        base_dict["type"] = PromptType.TEXT.value
        base_dict["data"] = {
            "text": self.text,
            "reasoning_content": self.reasoning_content,
        }
        return base_dict

    @classmethod
    def from_dict(cls, data: dict):
        new_prompt = TextPrompt(
            role=data["role"],
            text=data["data"].get("text", ""),
            reasoning_content=data["data"].get("reasoning_content", ""),
            create_time=data["time"],
        )
        new_prompt.id = data["id"]
        return new_prompt

    def __eq__(self, other):
        if isinstance(other, TextPrompt):
            return (
                super().__eq__(other)
                and self.text == other.text
                and self.reasoning_content == other.reasoning_content
            )

        else:
            return False

    def __str__(self):
        output = super().__str__()
        output += self.text
        output += "\n"
        return output

    def __bool__(self):
        return bool(self.text)


class ImagePrompt(BasePrompt):
    """图片消息；hash_name 是原件 ID，image 是旧版有序派生件 ID 列表。

    image 参数保留 Path/bytes/base64 输入入口；准备后原件入库，image_input 清空。
    已有缓存可直接传 hash_name，不需要解析路径或重新计算内容指纹。
    """

    # 内容身份由 hash_name/image 决定；补齐 MIME/尺寸不应改变远端上下文前缀。
    __slots__ = ("resource",)

    def __init__(
        self,
        role: Role,
        image: str | Path | bytes | None = None,
        name: str = "",
        create_time: str = None,
        *,
        hash_name: str = "",
    ):
        super().__init__(role, "IMAGE", create_time)
        self.name: str = name  # 业务别名，用于旧版分片映射；不作为文件内容标识。
        self.image: list[str] = []  # 旧版压缩/分片的 file_cache ID，按发送顺序保存。
        self.image_input = image
        self.hash_name = hash_name  # 原件的 file_cache ID；准备后以此为准，image_input 清空。
        self.resource: MediaRef | None = None

    def to_dict(self) -> dict:
        hash_name = self.hash_name
        origin_image_path = None
        if not hash_name and isinstance(self.image_input, Path):
            origin_image_path = str(self.image_input)

        elif not hash_name and self.image_input is not None:
            from TIYA.file_cache import add_file

            # 未发送就存档时的保底；已归一的图片存档不再读取本地文件。
            hash_name = add_file(self.image_input, "IMAGE")

        base_dict = super().to_dict()
        base_dict["type"] = PromptType.IMAGE.value
        base_dict["data"] = {
            "name": self.name,
            "image": self.image,
            "image_input": origin_image_path,
            "hash_name": hash_name,
            "resource": asdict(self.resource) if self.resource else None,
        }
        return base_dict

    @classmethod
    def from_dict(cls, data: dict):
        new_prompt = ImagePrompt(
            data["role"],
            Path(data["data"]["image_input"]) if data["data"].get("image_input") else None,
            data["data"]["name"],
            data["time"],
            hash_name=data["data"].get("hash_name", ""),
        )
        new_prompt.id = data["id"]
        new_prompt.image = data["data"]["image"]
        if data["data"].get("resource"):
            new_prompt.resource = MediaRef(**data["data"]["resource"])
        return new_prompt

    async def load_image(self, cache: dict[str, str]) -> bool:
        if self.name:
            hash_name_list = _IMAGE_NAME_MAPPING.get(self.name, [])
            if hash_name_list and await _load_image_hashes(hash_name_list, cache):
                self.image = list(hash_name_list)
                self.refresh_fingerprints()
                return True

            _IMAGE_NAME_MAPPING.pop(self.name, None)

        self.image = []
        self.refresh_fingerprints()
        try:
            if self.hash_name:
                source = await get_file_path_async(self.hash_name)

            else:
                source = self.image_input
                self.hash_name = await add_file_async(source, "IMAGE")

            hash_name_list = await handle_unload_image(source, cache)
            self.image_input = None

        except (FileNotFoundError, OSError, ValueError):
            return False

        if not hash_name_list:
            return False

        self.image = hash_name_list
        self.refresh_fingerprints()
        if self.name:
            _IMAGE_NAME_MAPPING[self.name] = list(self.image)

        return True

    def __eq__(self, other):
        if isinstance(other, ImagePrompt):
            return (
                bool(self.image)
                and super().__eq__(other)
                and self.name == other.name
                and self.image == other.image
            )

        else:
            return False

    def __str__(self):
        output = super().__str__()
        output += f"[图片;hash_id: {self.image}, name: {self.name}]"
        output += "\n"
        return output


class ToolInput(BasePrompt):
    __slots__ = ("_tool_payloads",)

    def __init__(
        self,
        function: Callable,
        function_name: str = None,
        create_time: str = None,
    ):
        if function_name is None:
            function_name = function.__name__

        super().__init__("function", "FUNCTION", create_time)
        self.function: Callable = function
        self.function_name: str = function_name
        self._tool_payloads: ToolPayloads | None = None
        if inspect.iscoroutinefunction(function):
            self.function_type = "ASYNC"

        else:
            self.function_type = "SYNC"

    @property
    def tool_payloads(self) -> ToolPayloads:
        """惰性构建一次 schema；编码、估算和指纹共用，调用方不得修改缓存。"""
        if self._tool_payloads is None:
            self._tool_payloads = build_tool_payloads(self.function)

        return self._tool_payloads

    def refresh_schema(self) -> None:
        """显式替换 callable/注解后重建 schema；普通历史增删无需调用。"""
        self._tool_payloads = None
        self.refresh_fingerprints()

    def _signature_data(self) -> dict[str, object]:
        return {**super()._signature_data(), "function": self.tool_payloads}

    def to_dict(self) -> dict:
        base_dict = super().to_dict()
        base_dict["type"] = PromptType.TOOL_IN.value
        base_dict["data"] = {
            "function_name": self.function_name,
            "function_type": self.function_type,
        }
        return base_dict

    @classmethod
    def from_dict(cls, data: dict):
        new_prompt = ToolInput(
            data["data"]["function"], data["data"]["function_name"], data["time"]
        )
        new_prompt.id = data["id"]
        return new_prompt

    def __eq__(self, other):
        if isinstance(other, ToolInput):
            return (
                super().__eq__(other)
                and self.function is other.function
                and self.function_type == other.function_type
            )

        else:
            return False

    def __str__(self):
        output = super().__str__()
        output += f"[TOOL输入; function_name: {self.function_name}]"
        output += "\n"
        return output

    def __deepcopy__(self, memo):
        # Bound methods belong to business objects, often containing locks/clients.
        result = object.__new__(type(self))
        memo[id(self)] = result
        result.__dict__ = {
            k: v if k == "function" else copy.deepcopy(v, memo)
            for k, v in vars(self).items()
        }
        result.estimated_tokens = self.estimated_tokens
        result._assumed_tokens = self._assumed_tokens
        result._signature = self._signature
        # 入口快照也复用源工具的 schema；首次复制时构建，后续副本不再生成。
        result._tool_payloads = self.tool_payloads
        return result


class ToolCall(BasePrompt):
    def __init__(self, function_name: str, call_id: str, arguments: Any):
        super().__init__("assistant", "FUNCTION")
        self.function_name: str = function_name
        self.call_id: str = call_id
        self.arguments: Any = arguments

    def to_dict(self) -> dict:
        base_dict = super().to_dict()
        base_dict["type"] = PromptType.TOOL_CALL.value
        base_dict["data"] = {
            "function_name": self.function_name,
            "call_id": self.call_id,
            "arguments": self.arguments,
        }
        return base_dict

    @classmethod
    def from_dict(cls, data: dict):
        new_prompt = ToolCall(
            data["data"]["function_name"],
            data["data"]["call_id"],
            data["data"]["arguments"],
        )
        new_prompt.id = data["id"]
        new_prompt.time = data["time"]
        return new_prompt

    def __eq__(self, other):
        if isinstance(other, ToolCall):
            return (
                super().__eq__(other)
                and self.function_name == other.function_name
                and self.call_id == other.call_id
                and self.arguments == other.arguments
            )

        else:
            return False

    def __str__(self):
        output = super().__str__()
        output += f"[TOOL CALL; function_name: {self.function_name}, arguments: {self.arguments}]"
        output += "\n"
        return output


class ToolResponse(BasePrompt):
    def __init__(
        self, function_name: str, call_id: str, response: Any, create_time: str = None
    ):
        super().__init__("tool", "FUNCTION", create_time)
        self.function_name = function_name
        self.call_id: str = call_id
        self.response = response

    def to_dict(self) -> dict:
        base_dict = super().to_dict()
        base_dict["type"] = PromptType.TOOL_RESP.value
        base_dict["data"] = {
            "function_name": self.function_name,
            "call_id": self.call_id,
            "response": self.response,
        }
        return base_dict

    @classmethod
    def from_dict(cls, data: dict):
        new_prompt = ToolResponse(
            data["data"]["function_name"],
            data["data"]["call_id"],
            data["data"]["response"],
            data["time"],
        )
        new_prompt.id = data["id"]
        return new_prompt

    def __eq__(self, other):
        if isinstance(other, ToolResponse):
            return (
                super().__eq__(other)
                and self.function_name == other.function_name
                and self.call_id == other.call_id
                and self.response == other.response
            )

        else:
            return False

    def __str__(self):
        output = super().__str__()
        output += f"[TOOL RESPONSE; function_name: {self.function_name}, response: {self.response}]"
        output += "\n"
        return output


class Prompts:
    def __init__(
        self, *args, role: str = "", create_time: str = None, duration: float = 0.0
    ):
        if create_time is None:
            create_time = time.time()

        self.prompts: list[BasePrompt] = []
        self.role = role
        self.time = create_time  # 最新一条消息的时间
        self._duration: float = duration  # 耗时
        self.usage: TokenUsage | None = None
        self._signature: str | None = None
        if args:
            self._initiate_prompts(args)

    def _add_usage(self, usage: TokenUsage):
        if self.usage is None:
            self.usage = usage

        else:
            self.usage += usage

    def _unpack_prompts(self, prompts, roles: set):
        if isinstance(prompts, Prompts):
            for p in prompts.prompts:
                roles.add(p.role)
                self.prompts.append(p)

            if prompts.usage is not None:
                self._add_usage(prompts.usage)

            self.time = prompts.time

        elif isinstance(prompts, (list, tuple)):
            for p in prompts:
                self._unpack_prompts(p, roles)

        elif isinstance(prompts, BasePrompt):
            roles.add(prompts.role)
            self.prompts.append(prompts)
            self.time = prompts.time

        elif isinstance(prompts, str):  # 仅提供对文本内容的快捷转换
            self._text2prompt(prompts, roles)

        elif isinstance(prompts, TokenUsage):
            self._add_usage(prompts)

        else:
            raise TypeError(f"不支持{type(prompts)}生成Prompts")

    def _initiate_prompts(self, *args):
        roles = set()
        for prompts in args:
            self._unpack_prompts(prompts, roles)

        if len(roles) == 1:
            self.role = next(iter(roles))

        elif len(roles) > 1:
            self.role = "MIX"

    def _text2prompt(self, text: str, roles: set):
        if len(roles) == 1:
            role = next(iter(roles))

        elif self.role:
            role = self.role

        else:
            raise ValueError("需要指定一个role")

        self.prompts.append(TextPrompt(role, text))

    def to_dict(self) -> dict:
        prompts = [prompt.to_dict() for prompt in self.prompts]
        usage = self.usage.to_dict() if self.usage is not None else {}
        return {
            "prompts": prompts,
            "role": self.role,
            "time": self.time,
            "duration": self._duration,
            "usage": usage,
        }

    @property
    def duration(self) -> float:
        if self._duration:
            return self._duration

        else:
            if len(self.prompts) > 1:
                return max(
                    float(self.prompts[-1].time) - float(self.prompts[0].time), 0.0
                )

            else:
                return 0.0

    def append(self, addition: BasePrompt | Prompts | TokenUsage | str | list) -> None:
        self.refresh_fingerprints(content_changed=False)
        roles = set()
        self._unpack_prompts(addition, roles)
        if len(roles) == 1:
            new_role = next(iter(roles))
            if self.role:
                if self.role != new_role:
                    self.role = "MIX"

            else:
                self.role = new_role

        else:
            self.role = "MIX"

    def extend(self, addition: BasePrompt | Prompts | str | list) -> None:
        return self.append(addition)

    def refresh_fingerprints(self, *, content_changed: bool = True) -> None:
        """刷新内容指纹；仅增删容器成员时保留各消息的 token 估算。"""
        self._signature = None
        if content_changed:
            for prompt in self.prompts:
                prompt.refresh_fingerprints()

    @property
    def signature(self) -> str:
        """组合角色和成员 signature；不重新展开各条消息内容。"""
        if self._signature is None:
            messages = [prompt.signature for prompt in self.prompts]
            self._signature = fingerprint([self.role, messages])

        return self._signature

    def discard_prompt_ids(self, prompt_ids: set[str]) -> int:
        if not prompt_ids:
            return 0

        original_count = len(self.prompts)
        self.prompts = [
            prompt for prompt in self.prompts if prompt.id not in prompt_ids
        ]
        self.refresh_fingerprints(content_changed=False)
        return original_count - len(self.prompts)

    def copy(self):
        return self.__copy__()

    def __str__(self):
        output = ""
        for prompt in self.prompts:
            output += str(prompt)

        return output

    def __contains__(self, item):
        if isinstance(item, BasePrompt):
            for p in self.prompts:
                if p == item:
                    return True

            return False

        else:
            return False

    def __iter__(self):
        return iter(self.prompts)

    def __reversed__(self):
        return reversed(self.prompts)

    def __add__(self, other):
        if isinstance(other, (Prompts, BasePrompt, str)):
            return Prompts(self, other)

        else:
            raise TypeError(f"Prompts add doesn't support {type(other)}")

    def __sub__(self, other):
        origin = self.prompts.copy()
        if isinstance(other, Prompts):
            sub = other.prompts.copy()
            for p in origin[:]:
                if p in sub:
                    origin.remove(p)

            if self.usage and other.usage:
                new_usage = self.usage - other.usage
                return Prompts(origin, new_usage)

            return Prompts(origin)

        elif isinstance(other, BasePrompt):
            for p in origin[:]:
                if p == other:
                    origin.remove(p)
                    return Prompts(origin)

            return Prompts(origin)

        elif isinstance(other, TokenUsage):
            if self.usage is not None:
                new_prompts = self.copy()
                new_usage = self.usage - other
                new_prompts.usage = new_usage
                return new_prompts

            raise ValueError("host Prompts doesn't has TokenUsage yet")

        else:
            raise TypeError(f"Prompts sub doesn't support {type(other)}")

    def __bool__(self):
        return bool(self.prompts)

    def __copy__(self):
        result = Prompts()
        result.prompts = self.prompts.copy()
        result.role = self.role
        result.time = self.time
        result._duration = self._duration
        result.usage = self.usage.copy() if self.usage is not None else None
        result._signature = self._signature
        return result

    def __deepcopy__(self, memo: dict[int, object]) -> Prompts:
        result = object.__new__(type(self))
        memo[id(self)] = result
        result.__dict__ = copy.deepcopy(self.__dict__, memo)
        return result

    @overload
    def __getitem__(self, item: int) -> BasePrompt: ...

    @overload
    def __getitem__(self, item: slice) -> Prompts: ...

    def __getitem__(self, item: int | slice) -> BasePrompt | Prompts:
        """索引取单条消息；切片返回新 Prompts 容器，内部消息仍共享。"""
        if isinstance(item, slice):
            return Prompts(self.prompts[item])

        if not isinstance(item, int):
            raise KeyError("index must be int")

        return self.prompts[item]


class Context:
    def __init__(self, *args):
        self.session_id = uuid4().hex
        self._system_prompt: SystemPrompt | None = None
        self._messages: list[BasePrompt] = []
        self._rounds: dict[str, Prompts] = {}
        self._tools: list[ToolInput] = []
        self._usage_accumulation: TokenUsage = TokenUsage()
        self._signature: str | None = None
        self._message_fingerprints: tuple[str, ...] | None = None
        self._header_signature: str | None = None
        self.estimated_tokens: int | None = None  # 当前消息和工具的本地 token 汇总。
        if args:
            self._initiate(args)

    def _initiate(self, args: tuple):
        new_round: dict[str, list] = {"prompts": [], "usage": []}
        self._unpack_prompts(args, new_round)
        if new_round:
            self._submit_round(new_round)

    def refresh_fingerprints(self, *, content_changed: bool = True) -> None:
        """刷新指纹和估算；仅增删成员/准备媒体时传 False，保留消息级估算。

        默认按内容修改处理，清除各消息缓存。不探测外部对消息字段的直接写入。
        """
        self._signature = None
        self._message_fingerprints = None
        self._header_signature = None
        self.estimated_tokens = None
        if content_changed:
            for prompt in self:
                prompt.refresh_fingerprints()

        for prompts in self._rounds.values():
            prompts.refresh_fingerprints(content_changed=False)

    @property
    def message_fingerprints(self) -> tuple[str, ...]:
        """按历史顺序读取消息 signature，供远端检查点匹配前缀。"""
        if self._message_fingerprints is None:
            self._message_fingerprints = tuple(prompt.signature for prompt in self._messages)

        return self._message_fingerprints

    @property
    def signature(self) -> str:
        """组合有序消息签名和工具签名，作为当前上下文的整体签名。"""
        if self._signature is None:
            tools = [tool.signature for tool in self._tools]
            self._signature = fingerprint([list(self.message_fingerprints), tools])

        return self._signature

    @property
    def header_signature(self) -> str:
        """system/tools 的稳定指纹，用于判断能否沿用远端上下文。"""
        if self._header_signature is None:
            system = None
            if self._system_prompt is not None:
                system = self._system_prompt.signature

            tools = [tool.signature for tool in self._tools]
            self._header_signature = fingerprint([system, tools])

        return self._header_signature

    def _add_usage(self, usage: TokenUsage):
        self._usage_accumulation += usage

    def _submit_round(self, round_list: dict):
        save_round = Prompts(round_list["prompts"], round_list["usage"])
        save_round.role = "ROUND"
        self._rounds[save_round.time] = save_round
        round_list.clear()
        round_list.update({"prompts": [], "usage": []})

    def _unpack_prompts(self, prompt, round_list: dict[str, list]):
        if isinstance(prompt, (list, tuple)):
            for p in prompt:
                self._unpack_prompts(p, round_list)

        elif isinstance(prompt, Prompts):
            temp_list: list[BasePrompt | TokenUsage] = prompt.prompts.copy()
            if prompt.usage is not None:
                temp_list.append(prompt.usage)

            self._unpack_prompts(temp_list, round_list)

        elif isinstance(prompt, SystemPrompt):
            self._system_prompt = prompt
            if self._messages and self._messages[0].role == "system":
                self._messages[0] = prompt

            else:
                self._messages.insert(0, prompt)

        elif isinstance(prompt, ToolInput):
            if prompt not in self._tools:
                self._tools.append(prompt)

        elif isinstance(prompt, BasePrompt):
            self._messages.append(prompt)
            if prompt.role != "assistant":
                if (
                    round_list["prompts"]
                    and round_list["prompts"][-1].role == "assistant"
                ):
                    self._submit_round(round_list)

            round_list["prompts"].append(prompt)

        elif isinstance(prompt, TokenUsage):
            self._add_usage(prompt)
            round_list["usage"].append(prompt)

        else:
            raise TypeError(f"不支持{type(prompt)}类型创建Context")

    def append(self, addition: Prompts | list | BasePrompt) -> None:
        if not addition:
            return

        self.refresh_fingerprints(content_changed=False)
        if self._rounds:
            key = next(reversed(self._rounds))
            previous = self._rounds[key]
            # 重建最后一轮时携带旧用量；只交给 _submit_round，不再次累计到总用量。
            last_round = {
                "prompts": previous.prompts.copy(),
                "usage": [previous.usage] if previous.usage is not None else [],
            }
            del self._rounds[key]

        else:
            last_round = {"prompts": [], "usage": []}

        self._unpack_prompts(addition, last_round)
        if last_round.get("prompts", []):
            self._submit_round(last_round)

    def extend(self, addition: Prompts | list | BasePrompt) -> None:
        return self.append(addition)

    def copy(self):
        return self.__copy__()

    def clear(self, system: bool = False, tools: bool = False):
        self._usage_accumulation.clear()
        self._rounds.clear()
        self._messages.clear()

        if system:
            self._system_prompt = None

        else:
            if self._system_prompt is not None:
                self._messages.append(self._system_prompt)

        if tools:
            self._tools.clear()

        self.refresh_fingerprints(content_changed=False)

    def to_prompts(self) -> Prompts:
        return Prompts(self._messages, self._tools)

    def replace_contents(self, other: Context) -> None:
        """替换当前会话的数据；复制外层容器，消息/轮次/工具仍引用 other 的对象。

        用于提交已准备好的工作上下文，不改变当前 session_id。
        """
        self._system_prompt = other._system_prompt
        self._messages = other._messages.copy()
        self._rounds = other._rounds.copy()
        self._tools = other._tools.copy()
        self.replace_usage(other._usage_accumulation)
        self._signature = other._signature
        self._message_fingerprints = other._message_fingerprints
        self._header_signature = other._header_signature
        self.estimated_tokens = other.estimated_tokens

    def replace_usage(self, usage: TokenUsage) -> None:
        """替换累计用量的外层视图，不改动各轮自己的 usage。"""
        self._usage_accumulation = usage.copy()

    def discard_prompt_ids(self, prompt_ids: set[str]) -> int:
        if not prompt_ids:
            return 0

        original_count = len(self._messages)
        self._messages = [
            prompt for prompt in self._messages if prompt.id not in prompt_ids
        ]

        for round_id, prompts in list(self._rounds.items()):
            filtered_prompts = prompts.copy()
            filtered_prompts.prompts = [
                prompt
                for prompt in filtered_prompts.prompts
                if prompt.id not in prompt_ids
            ]
            filtered_prompts.refresh_fingerprints(content_changed=False)
            if filtered_prompts:
                self._rounds[round_id] = filtered_prompts

            else:
                self._rounds.pop(round_id)

        self.refresh_fingerprints(content_changed=False)
        return original_count - len(self._messages)

    def back2last_round(self) -> Prompts:
        """回滚到上一轮对话，POP当前轮"""
        if not self._rounds:
            raise ContextEmptyError("已无上一轮")

        last_round = self._rounds.pop(next(reversed(self._rounds)))
        self._messages.clear()
        if self._system_prompt is not None:
            self._messages.append(self._system_prompt)

        if self._rounds:
            for prompts in self._rounds.values():
                for prompt in prompts:
                    if not isinstance(prompt, SystemPrompt):
                        self._messages.append(prompt)

        self.refresh_fingerprints(content_changed=False)
        return last_round

    def to_dict(self):
        messages = {prompt.id: prompt.to_dict() for prompt in self._messages.copy()}
        tools = {prompt.id: prompt.to_dict() for prompt in self._tools.copy()}
        system = self._system_prompt.id if self._system_prompt is not None else ""
        usage = self._usage_accumulation.to_dict()
        rounds = {}
        for r, data in self._rounds.copy().items():
            round_data = {
                "messages": [p.id for p in data.prompts],
                "usage": data.usage.to_dict() if data.usage is not None else {},
            }
            rounds[r] = round_data

        dump_content = {
            "communication_session": self.session_id,
            "system": system,
            "messages": messages,
            "round": rounds,
            "tools": tools,
            "usage": usage,
        }
        return dump_content

    @classmethod
    def from_dict(cls, data: dict, function_mapping: dict):
        new_context = Context()
        new_context.session_id = data.get("communication_session") or uuid4().hex
        system = data.get("system", "")
        messages = data.get("messages", {})
        rounds = data.get("round", {})
        tools = data.get("tools", {})
        usage = data.get("usage", {})

        parsed_prompts = {}
        parsed_tools = {}
        parsed_rounds = {}
        # 获取完整prompt对象
        for mid, message in messages.items():
            if PromptType(message["type"]) is PromptType.SYSTEM:
                parsed_prompts[mid] = SystemPrompt.from_dict(message)

            elif PromptType(message["type"]) is PromptType.TEXT:
                parsed_prompts[mid] = TextPrompt.from_dict(message)

            elif PromptType(message["type"]) is PromptType.IMAGE:
                parsed_prompts[mid] = ImagePrompt.from_dict(message)

            elif PromptType(message["type"]) is PromptType.TOOL_CALL:
                parsed_prompts[mid] = ToolCall.from_dict(message)

            elif PromptType(message["type"]) is PromptType.TOOL_RESP:
                parsed_prompts[mid] = ToolResponse.from_dict(message)

            if mid in parsed_prompts:
                parsed_prompts[mid].native = copy.deepcopy(message.get("native", {}))

        # 获取tool对象
        for tid, tool in tools.items():
            if PromptType(tool["type"]) is not PromptType.TOOL_IN:
                continue

            function_name = tool["data"]["function_name"]
            function = function_mapping.get(function_name, None)
            if function is None:
                continue

            tool["data"]["function"] = function
            parsed_tools[tid] = ToolInput.from_dict(tool)

        # 重构rounds
        for r, data in rounds.items():
            round_prompts = Prompts()
            for mid in data["messages"]:
                if mid not in parsed_prompts:
                    continue

                round_prompts.append(parsed_prompts[mid])

            if data["usage"]:
                round_prompts.usage = TokenUsage.from_dict(data["usage"])

            if round_prompts:
                parsed_rounds[r] = round_prompts

        new_context._messages.extend(list(parsed_prompts.values()))
        new_context._rounds.update(parsed_rounds)
        new_context._tools.extend(list(parsed_tools.values()))
        if system and system in parsed_prompts:
            new_context._system_prompt = parsed_prompts[system]

        if usage:
            new_context._usage_accumulation = TokenUsage.from_dict(usage)

        return new_context

    @property
    def last_round(self) -> Prompts:
        if not self._rounds:
            return Prompts(role="ROUND")

        last_key = next(reversed(self._rounds))
        last_round = self._rounds[last_key]
        return last_round.copy()

    @property
    def tools(self) -> Prompts:
        """
        返回输入工具列表
        :return:
        """
        return Prompts(self._tools)

    @property
    def rounds(self) -> dict[str, Prompts]:
        return self._rounds.copy()

    @property
    def usage(self) -> TokenUsage:
        """返回累积token消耗"""
        return self._usage_accumulation.copy()

    @property
    def system(self) -> SystemPrompt | None:
        if self._system_prompt is None:
            return None

        if not self._system_prompt.text:
            return None

        return self._system_prompt.copy()

    @property
    def message(self) -> Prompts:
        """
        返回消息列表
        :return:
        """
        return Prompts(self._messages)

    @property
    def has_content(self) -> bool:
        if self._system_prompt is not None:
            return len(self._messages) > 1

        else:
            return bool(self._messages)

    def __str__(self):
        output = ""
        for prompt in self._messages:
            output += str(prompt)

        for prompt in self._tools:
            output += str(prompt)

        return output

    def __copy__(self):
        """沿用旧接口：新会话及外层容器，内部 Prompt/轮次/工具对象共享。"""
        result = Context()
        result._system_prompt = self._system_prompt
        result._messages = self._messages.copy()
        result._rounds = self._rounds.copy()
        result._tools = self._tools.copy()
        result._usage_accumulation = self._usage_accumulation.copy()
        result._signature = self._signature
        result._message_fingerprints = self._message_fingerprints
        result._header_signature = self._header_signature
        result.estimated_tokens = self.estimated_tokens
        return result

    def __deepcopy__(self, memo: dict[int, object]) -> Context:
        """复制数据快照并保留 session_id；memo 保持消息/轮次内的共享关系。"""
        result = object.__new__(type(self))
        memo[id(self)] = result
        result.__dict__ = copy.deepcopy(self.__dict__, memo)
        return result

    def __iter__(self):
        iter_list = self._messages + self._tools
        return iter(iter_list)

    def __reversed__(self):
        iter_list = self._messages + self._tools
        return reversed(iter_list)

    def __bool__(self):
        return bool(self._messages)


@dataclass(slots=True)
class CallOutcome:
    status: str = "idle"
    retry_allowed: bool = False
    call_id: str = ""
    request_id: str = ""
    remote_state: str = "unknown"
    category: str = ""
    committed: bool = False


@dataclass(slots=True)
class AskTask:
    """调度中的一次请求；单调时钟起点用于包含排队时间的总耗时，不随消息存档。"""

    prompts: Prompts
    timeout: float
    id: str = field(default_factory=lambda: str(time.time_ns()))
    event: asyncio.Event = field(default_factory=asyncio.Event)
    response: None | Prompts = None
    handle: asyncio.Task | None = None
    pending: Any = None
    error: BaseException | None = None
    cancelled: bool = False
    committed: bool = False
    outcome: CallOutcome | None = None
    created_monotonic: float = field(default_factory=time.monotonic, init=False, repr=False)


@dataclass(frozen=True, slots=True)
class MediaRef:
    """资源引用；本地文件直接使用 hash_name，无须解释 kind/value。

    kind/value 仅描述尚未归一的输入或远端传输引用。远端引用可同时携带原件
    hash_name，供失效重传使用；content() 优先读取本地原件。scope 绑定远端账户，
    expires_at 为 Unix 秒。尺寸为已知元数据，访问字段不触发 IO。
    """

    kind: MediaKind | None = None
    value: str | Path | bytes | None = None
    mime_type: str | None = None
    scope: str | None = None
    width: int | None = None
    height: int | None = None
    expires_at: float | None = None
    hash_name: str = ""  # file_cache 的内容标识，不再二次哈希文件内容。
    filename: str | None = None  # 上传文件名，不参与内容身份；省略时由传输层命名。


@dataclass
class ImageGenerateRequest:
    """独立 Images API 生成请求；model 原样发送，params 放原生扩展参数。"""

    model: str
    prompt: str
    n: int | None = None
    size: str | None = None
    quality: str | None = None
    background: str | None = None
    output_format: str | None = None
    output_compression: int | None = None
    stream: bool | None = None
    partial_images: int | None = None
    params: Params = field(default_factory=dict)


@dataclass
class ImageEditRequest(ImageGenerateRequest):
    """编辑请求：images 首张为目标，其余为参考，顺序由业务决定。

    mask 是 PNG alpha 遮罩；markup 是带实际透明像素的 PNG 标注图层。
    两者均对应首张目标图、尺寸须匹配，不能同时提供；原图不与标注层合成。
    """

    images: list[MediaRef] = field(default_factory=list)
    mask: MediaRef | None = None
    markup: MediaRef | None = None


@dataclass(slots=True)
class ImageResponseItem:
    """已落盘的输出/预览；image.hash_name 可直接交给业务的文件消费接口。

    image_index 为结果位置，partial_image_index 为预览序号；created_at 为 Unix 秒。
    metadata 只保留未建模的供应商扩展，不含图片载荷和以下已知字段。
    """

    image: MediaRef
    revised_prompt: str | None = None
    metadata: Params = field(default_factory=dict)
    output_format: str | None = None
    background: str | None = None
    quality: str | None = None
    size: str | None = None
    image_index: int | None = None
    partial_image_index: int | None = None
    created_at: int | None = None
    event_type: str | None = None
    safety_identifier: str | None = None
    usage: Params = field(default_factory=dict)  # 单个流事件附带的原生 usage。


@dataclass(slots=True)
class ImagesResponse:
    """已缓存的最终图片集合；request_id 为本地账本 ID，usage 保留真实返回值。

    格式、背景、质量及尺寸来自供应商实际响应，不以请求参数猜测返回值。
    metadata 仅容纳以下字段之外的供应商扩展。
    """

    created: int | None = None
    data: list[ImageResponseItem] = field(default_factory=list)
    usage: Params = field(default_factory=dict)
    request_id: str = ""
    metadata: Params = field(default_factory=dict)
    output_format: str | None = None
    background: str | None = None
    quality: str | None = None
    size: str | None = None
    safety_identifier: str | None = None

    def to_prompts(self, role: Role = "assistant") -> Prompts:
        """按结果顺序将已缓存图片包装为 ImagePrompt，返回新的 Prompts。

        只复用缓存 ID 和不可变 MediaRef，不读取文件、不重新入库或写入 Context。
        usage、request_id 等请求信息保留在本响应中；无图片时返回空容器。
        """
        prompts = Prompts(role=role)
        for item in self.data:
            image = ImagePrompt(role, hash_name=item.image.hash_name)
            image.resource = item.image
            prompts.append(image)

        return prompts
