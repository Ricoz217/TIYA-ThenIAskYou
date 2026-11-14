import asyncio
import json

import httpx
import pytest

from llm_config_fixtures import candidate, configuration
from TIYA.llm.chat import Chat
from TIYA.llm.clients import FilesClient, ImagesClient
from TIYA.llm.model import ImagePrompt
from TIYA.llm.pricing import PriceCatalog
from TIYA.llm.runtime import Runtime
from TIYA.llm.state import StateStore
from TIYA.llm.usage import Ledger


@pytest.mark.parametrize("client_type", [FilesClient, ImagesClient])
def test_service_network_defaults_and_explicit_overrides_are_merged(tmp_path, client_type):
    async def run():
        service = client_type(
            {"endpoint": "https://test/api", "client_params": {"trust_env": False, "http2": False}},
            client_params={"http2": True},
            ledger=Ledger(tmp_path / "usage.db"),
            state=StateStore(tmp_path / "state.db"),
            prices=PriceCatalog(tmp_path / "prices.db"),
        )
        try:
            assert service.client_params == {"trust_env": False, "http2": True}
            assert not (await service.gateway()).client._trust_env
            if isinstance(service, ImagesClient):
                assert service._candidate("image").client_params == service.client_params

        finally:
            await service.close()

    asyncio.run(run())


@pytest.mark.parametrize("with_image", [False, True])
def test_chat_files_use_their_network_pool_and_own_headers(
    tmp_path,
    monkeypatch,
    with_image,
):
    created = []
    sent = []
    real_client = httpx.AsyncClient

    def client_factory(**kwargs):
        trust_env = kwargs.get("trust_env", True)
        created.append(trust_env)

        def handler(request):
            sent.append((request.url.path, trust_env, request.headers.get("authorization")))
            if request.url.path == "/files":
                return httpx.Response(200, json={"id": "file-1"})

            payload = json.loads(request.content)
            if with_image:
                assert payload["input"][0]["content"][0]["file_id"] == "file-1"
            return httpx.Response(200, json={"status": "completed", "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "ok"}]}
            ], "usage": {"input_tokens": 10, "output_tokens": 3}})

        kwargs.pop("mounts", None)
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)

    async def run():
        route = candidate(
            "responses_api:a:m", "responses_api", "a", "m",
            {"endpoint": "https://test/responses", "key": "model-key"}, {"stream": False},
            files_settings={"endpoint": "https://test/files", "key": "files-key", "client_params": {"trust_env": False}},
        )
        chat = Chat(config=configuration("test", [route]), log_off=True, keep_alive=True)
        rt = Runtime(StateStore(tmp_path / "state.db"), Ledger(tmp_path / "usage.db"))
        chat._runtime = rt
        try:
            from io import BytesIO
            from PIL import Image
            from TIYA.llm.model import MediaRef

            image = BytesIO()
            Image.new("RGB", (1, 1)).save(image, "PNG")
            prompt = ImagePrompt("user", hash_name="local")
            prompt.resource = MediaRef("bytes", image.getvalue(), "image/png", hash_name="local")

            async def content(ref, gateway=None):
                return image.getvalue()

            monkeypatch.setattr(rt.media, "content", content)
            assert (await chat.ask(prompt if with_image else "question", timeout=5))[0].text == "ok"
            assert (await chat.ask(prompt if with_image else "next", timeout=5))[0].text == "ok"
            assert sent == ([
                ("/files", False, "Bearer files-key"),
            ] if with_image else []) + [
                ("/responses", True, "Bearer model-key"),
                ("/responses", True, "Bearer model-key"),
            ]
            assert created.count(False) == int(with_image)

        finally:
            await chat.close()

    asyncio.run(run())


def test_image_edit_uploads_use_and_close_the_files_connection(tmp_path, monkeypatch):
    import base64
    from io import BytesIO
    from PIL import Image
    import TIYA.llm.media as media
    from TIYA.llm.model import ImageEditRequest, MediaRef

    image = BytesIO()
    Image.new("RGBA", (2, 2)).save(image, "PNG")
    content = image.getvalue()
    real_client = httpx.AsyncClient
    file_clients = []
    sent = []

    def handler(request):
        sent.append(request.url.path)
        if request.url.path == "/files":
            return httpx.Response(200, json={"id": f"file-{len(sent)}"})

        assert json.loads(request.content)["markup"] == {"file_id": f"file-{len(sent) - 1}"}
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(content).decode()}]})

    def files_client(**kwargs):
        assert kwargs["trust_env"] is False
        client = real_client(transport=httpx.MockTransport(handler), **kwargs)
        file_clients.append(client)
        return client

    async def cache(data, *args):
        return "output"

    monkeypatch.setattr(httpx, "AsyncClient", files_client)
    monkeypatch.setattr(media, "add_file_async", cache)

    async def run():
        async with real_client(transport=httpx.MockTransport(handler)) as model_client:
            images = ImagesClient(
                {
                    "endpoint": "https://test/generate", "edit_endpoint": "https://test/edit",
                    "images": {"edit_format": "json"},
                    "files_api": {"endpoint": "https://test/files", "client_params": {"trust_env": False}},
                },
                client=model_client,
                ledger=Ledger(tmp_path / "usage.db"),
                state=StateStore(tmp_path / "state.db"),
                prices=PriceCatalog(tmp_path / "prices.db"),
            )
            response = await images.edit(ImageEditRequest(
                "image", "edit", images=[MediaRef("bytes", content)], markup=MediaRef("bytes", content),
            ))
            assert response.data[0].image.hash_name == "output"
            assert len(file_clients) == 1
            assert not file_clients[0].is_closed
            await images.edit(ImageEditRequest(
                "image", "edit again", images=[MediaRef("bytes", content)], markup=MediaRef("bytes", content),
            ))
            assert len(file_clients) == 1
            await images.close()
            assert file_clients[0].is_closed
            assert not model_client.is_closed
            assert sent == ["/files", "/files", "/edit"] * 2

    asyncio.run(run())
