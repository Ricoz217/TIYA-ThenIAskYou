from __future__ import annotations

import asyncio
import math
import time

from typing import TYPE_CHECKING
from pathlib import Path
from types import TracebackType

from .model import (
    TextPrompt,
    ImagePrompt,
    Prompts,
    MediaRef,
    ImageGenerateRequest,
    ImageEditRequest,
    ImagesResponse,
    TokenUsage,
)
from .clients import ImagesClient
from .config_model import Params, ChatConfig, JsonValue
from .provider.openai_images import Images
from .usage_compat import LLMUsage
from TIYA.logger import get_logger
from TIYA.config import SETTING_CFG

_log = get_logger()
if TYPE_CHECKING:
    from TIYA.logger import BlockHandle, Logger


class ImageGen:
    """无状态生图入口：只使用首个候选，接收图片对象并返回已缓存的 Prompts。

    每次调用独立执行，不积累 Context、不排队、不重试或切换候选。
    keep_alive=True 时复用网络连接；close() 应在业务等待在途调用结束后执行。
    """

    def __init__(
        self,
        *,
        endpoint: str = "",
        edit_endpoint: str = "",
        token: str = "",
        model: str = "",
        client_params: dict[str, object] | None = None,
        model_params: Params | None = None,
        keep_alive: bool = False,
        logger: Logger | BlockHandle | None = None,
        log_off: bool = False,
        usage_record: list[LLMUsage] | None = None,
        config: ChatConfig | None = None,
    ) -> None:
        """config 使用 parse_llm_setting() 的结果，且只读取 candidates[0]。

        传入 config 时，模型与连接参数取自该配置；未声明的会话选项使用构造值。
        无 config 时可直接填写 endpoint/token/model，编辑地址单独填写 edit_endpoint。
        model_params 为对象默认请求参数，None 值不发送；构造时不访问网络或数据库。
        usage_record 保留可选的附加账本镜像，主账本始终正常记录。
        """
        if config is not None:
            if not config.candidates:
                raise ValueError("生图配置需要至少一个候选")

            route = config.primary
            if route.provider_type != "image_generation":
                raise ValueError("首个候选必须使用 image_generation 协议")

            self.model = route.model
            self.model_params = dict(route.model_params)
            self._settings = dict(route.provider_options)
            self._settings.update(endpoint=route.endpoint, key=route.token, headers=dict(route.headers))
            if route.files is not None:
                self._settings["files_api"] = route.files.as_provider_settings()

            self._client_params = dict(route.client_params)
            self._provider_ref = f"{route.provider_type}:{route.connection_name}"
            self._preset_name = config.preset_name or ""
            keep_alive = config.keep_alive if config.keep_alive is not None else keep_alive
            log_off = config.log_off if config.log_off is not None else log_off
            logger = config.logger if config.logger is not None else logger

        else:
            self.model = model
            self.model_params = dict(model_params or {})
            self.model_params.setdefault("stream", False)
            self._settings = {"endpoint": endpoint, "key": token}
            if edit_endpoint:
                self._settings["edit_endpoint"] = edit_endpoint

            self._client_params = dict(client_params or {})
            self._provider_ref = "image_generation:direct"
            self._preset_name = ""

        if not self.model or not self._settings.get("endpoint"):
            raise ValueError("生图需要明确的模型和 endpoint")

        self.keep_alive = keep_alive
        self.log_off = log_off
        self.logger = logger if logger is not None else _log
        self._usage_record = usage_record if usage_record is not None else []
        self._client: ImagesClient | None = None
        self._closed = False

    async def generate(
        self,
        prompt: str | TextPrompt,
        *args: bytes | Path | ImagePrompt,
        background: str | None = None,
        complexity: str | None = None,
        frame_count: int | None = None,
        frame_duration_ms: int | None = None,
        moderation: str | None = None,
        n: int | None = None,
        output_compression: int | None = None,
        output_format: str | None = None,
        partial_images: int | None = None,
        quality: str | None = None,
        size: str | None = None,
        stream: bool | None = None,
        timeout: float | None = None,
        **kwargs: JsonValue,
    ) -> Prompts:
        """文字或参考生图，返回按结果顺序排列的 assistant ImagePrompt。

        args 为有序参考图；无图走 generations，有图使用 edits 承载参考生成。
        ImagePrompt.name 或 Path 文件名保留；字节输入按位置自动命名。
        显式非 None 参数覆盖 model_params；两处都为 None 时不发送字段。
        kwargs 透传额外原生请求参数，timeout 为整个调用的秒数，不进入请求体。
        所有最终图片进入 file_cache 后才返回；失败抛异常，不自动再次生成。
        """
        parameters = self._parameters({
            **kwargs,
            "background": background,
            "complexity": complexity,
            "frame_count": frame_count,
            "frame_duration_ms": frame_duration_ms,
            "moderation": moderation,
            "n": n,
            "output_compression": output_compression,
            "output_format": output_format,
            "partial_images": partial_images,
            "quality": quality,
            "size": size,
            "stream": stream,
        })
        text = self._text(prompt)
        references = [self._image_ref(image, index) for index, image in enumerate(args)]
        if references:
            request = ImageEditRequest(self.model, text, images=references, params=parameters)

        else:
            request = ImageGenerateRequest(self.model, text, params=parameters)

        return await self._run(request, timeout)

    async def edit(
        self,
        prompt: str | TextPrompt,
        base_image: bytes | Path | ImagePrompt,
        *args: bytes | Path | ImagePrompt,
        mask: bytes | Path | ImagePrompt | None = None,
        markup: bytes | Path | ImagePrompt | None = None,
        background: str | None = None,
        complexity: str | None = None,
        frame_count: int | None = None,
        frame_duration_ms: int | None = None,
        input_fidelity: str | None = None,
        moderation: str | None = None,
        n: int | None = None,
        output_compression: int | None = None,
        output_format: str | None = None,
        partial_images: int | None = None,
        quality: str | None = None,
        size: str | None = None,
        stream: bool | None = None,
        timeout: float | None = None,
        **kwargs: JsonValue,
    ) -> Prompts:
        """编辑 base_image；它固定为首图，args 按原顺序作为后续参考图。

        mask 与 markup 互斥，均对应 base_image，须为同尺寸的 PNG 引导图。
        markup 必须有实际透明像素。框架不合成、缩放或重排输入图片。
        参数继承、timeout、返回值及失败处理规则同 generate()。
        """
        if mask is not None and markup is not None:
            raise ValueError("mask 与 markup 不能同时提供")

        parameters = self._parameters({
            **kwargs,
            "background": background,
            "complexity": complexity,
            "frame_count": frame_count,
            "frame_duration_ms": frame_duration_ms,
            "input_fidelity": input_fidelity,
            "moderation": moderation,
            "n": n,
            "output_compression": output_compression,
            "output_format": output_format,
            "partial_images": partial_images,
            "quality": quality,
            "size": size,
            "stream": stream,
        })
        references = [self._image_ref(image, index) for index, image in enumerate((base_image, *args))]
        request = ImageEditRequest(
            self.model,
            self._text(prompt),
            images=references,
            mask=self._image_ref(mask, 0) if mask is not None else None,
            markup=self._image_ref(markup, 0) if markup is not None else None,
            params=parameters,
        )
        return await self._run(request, timeout)

    def _parameters(self, overrides: Params) -> Params:
        parameters = self.model_params.copy()
        parameters.update({key: value for key, value in overrides.items() if value is not None})
        if "model" in parameters:
            raise ValueError("model 请通过预设或构造参数指定")

        return {key: value for key, value in parameters.items() if value is not None}

    @staticmethod
    def _text(prompt: str | TextPrompt) -> str:
        if isinstance(prompt, TextPrompt):
            return prompt.text

        if isinstance(prompt, str):
            return prompt

        raise TypeError("prompt 需要 str 或 TextPrompt")

    @staticmethod
    def _image_ref(image: bytes | Path | ImagePrompt, index: int) -> MediaRef:
        if isinstance(image, ImagePrompt):
            filename = image.name or f"image_{index}"
            if image.hash_name:
                return MediaRef(hash_name=image.hash_name, filename=filename)

            source = image.image_input
            if isinstance(source, Path):
                filename = image.name or source.name
                return MediaRef("path", source, filename=filename)

            elif isinstance(source, bytes):
                return MediaRef("bytes", source, filename=filename)

            elif isinstance(source, str):
                return MediaRef("base64", source, filename=filename)

            else:
                raise ValueError("图片需要 bytes、Path 或包含原件 hash_name/image_input 的 ImagePrompt")

        elif isinstance(image, Path):
            return MediaRef("path", image, filename=image.name)

        elif isinstance(image, bytes):
            return MediaRef("bytes", image, filename=f"image_{index}")

        else:
            raise ValueError("图片需要 bytes、Path 或包含原件 hash_name/image_input 的 ImagePrompt")

    def _client_for_call(self) -> ImagesClient:
        if self._closed:
            raise RuntimeError("ImageGen 已关闭")

        if self.keep_alive and self._client is not None:
            return self._client

        client = ImagesClient(
            self._settings,
            provider_ref=self._provider_ref,
            client_params=self._client_params,
            usage_record=self._usage_record,
        )
        if self.keep_alive:
            self._client = client

        return client

    async def _run(self, request: ImageGenerateRequest, timeout: float | None) -> Prompts:
        timeout = timeout if timeout is not None else float(SETTING_CFG.LLM.ChatRequestTimeout)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout 必须是有限正数，单位为秒")

        started = time.monotonic()
        client = self._client_for_call()
        response: Prompts | None = None
        try:
            # 外层时限覆盖本地图片读取、上传和最终落盘；底层不自动重发生图。
            async with asyncio.timeout(timeout):
                if isinstance(request, ImageEditRequest):
                    result = await client.edit(request, timeout=timeout, preset=self._preset_name)

                else:
                    result = await client.generate(request, timeout=timeout, preset=self._preset_name)

            response = self._to_prompts(result)
            response._duration = time.monotonic() - started
            usage = response.usage
            if not self.log_off and usage is not None:
                self.logger.info(
                    f"生图完成 | 图片: [{len(result.data)}]; "
                    f"输入: [{usage.input_t:,}]; 输出: [{usage.output_t:,}]"
                )

            return response

        except Exception as exc:
            self.logger.error(f"生图请求失败: {type(exc).__name__}: {exc}")
            raise

        finally:
            if not self.log_off:
                elapsed = response._duration if response is not None else time.monotonic() - started
                self.logger.info(f"耗时：[{elapsed:.2f}]秒")

            if not self.keep_alive:
                await client.close()

    def _to_prompts(self, result: ImagesResponse) -> Prompts:
        response = result.to_prompts()
        reported = Images.usage_values(result.usage)
        reported["images"] = len(result.data)
        response.usage = TokenUsage(model=self.model, request_id=result.request_id, reported=reported)
        return response

    async def close(self) -> None:
        """关闭本对象并释放持有的连接；重复调用无操作，不取消业务协程。"""
        self._closed = True
        if self._client is not None:
            try:
                await self._client.close()

            finally:
                self._client = None

    async def __aenter__(self) -> ImageGen:
        if self._closed:
            raise RuntimeError("ImageGen 已关闭")

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()