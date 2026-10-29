import asyncio
import hashlib
import json
from io import BytesIO

import httpx
from PIL import Image

from TIYA.llm.config import Preset, Route
from TIYA.llm.gateway import Gateway
from TIYA.llm.model import Context, ImagePrompt, Prompts, TextPrompt
from TIYA.llm.runtime import Runtime
from TIYA.llm.state import StateStore
from TIYA.llm.usage import Ledger


def test_thirty_images_then_two_only_prepare_delta_and_full_replay_reuses_uploads(
    tmp_path, monkeypatch
):
    import TIYA.llm.media as module

    cache = {}

    async def add(content, *args):
        key = hashlib.sha256(content).hexdigest()
        path = tmp_path / key
        path.write_bytes(content)
        cache[key] = path
        return key

    async def get(key):
        return cache[key]

    monkeypatch.setattr(module, "add_file_async", add)
    monkeypatch.setattr(module, "get_file_path_async", get)
    calls = []
    uploads = []

    def handler(request):
        if request.url.path == "/files":
            uploads.append(request)
            return httpx.Response(200, json={"id": f"f{len(uploads)}"})
        calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": f"r{len(calls)}",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
            },
        )

    async def run():
        rt = Runtime(
            StateStore(tmp_path / "runtime.sqlite3"), Ledger(tmp_path / "usage.sqlite3")
        )
        route = Route(
            "responses_api:a:m",
            "responses_api",
            "a",
            "m",
            {
                "endpoint": "https://test/responses",
                "responses": {"supports_previous_response_id": True},
            },
            files_ref="files_api:a",
            files_settings={"endpoint": "https://test/files", "key": "k"},
        )
        config = Preset("p", [route])
        pictures = []
        for i in range(32):
            buffer = BytesIO()
            Image.new("RGB", (16, 16), (i, i, i)).save(buffer, "PNG")
            pictures.append(ImagePrompt("user", buffer.getvalue()))
        context = Context()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:

            async def gateway(route):
                return Gateway(client)

            first = await rt.execute(
                context, Prompts(pictures[:30]), config, gateway, 20
            )
            rt.commit(context, first)
            assert all(
                not isinstance(p.image_input, bytes)
                for p in context._messages
                if isinstance(p, ImagePrompt)
            )
            second = await rt.execute(
                context, Prompts(pictures[30:]), config, gateway, 20
            )
            rt.commit(context, second)
            assert len(uploads) == 32
            assert len(calls[1]["input"]) == 2
            assert all("file_id" in item["content"][0] for item in calls[1]["input"])
            # Local originals may expire. A full stateless replay can still use known remote refs.
            for path in cache.values():
                path.unlink()
            route.history_mode = "full"
            await rt.execute(
                context, Prompts(TextPrompt("user", "again")), config, gateway, 20
            )
            assert len(uploads) == 32
            assert "previous_response_id" not in calls[-1]
            assert len(calls[-1]["input"]) == 35

    asyncio.run(run())


def test_only_explicitly_invalid_file_is_reuploaded(tmp_path, monkeypatch):
    import TIYA.llm.media as module

    contents = {}

    async def add(content, *args):
        key = hashlib.sha256(content).hexdigest()
        path = tmp_path / key
        path.write_bytes(content)
        contents[key] = path
        return key

    async def get(key):
        return contents[key]

    monkeypatch.setattr(module, "add_file_async", add)
    monkeypatch.setattr(module, "get_file_path_async", get)
    uploads, calls = [], []

    def handler(request):
        if request.url.path == "/files":
            uploads.append(request)
            return httpx.Response(200, json={"id": f"f{len(uploads)}"})
        body = json.loads(request.content)
        calls.append(body)
        if len(calls) == 1:
            return httpx.Response(
                400, json={"error": {"code": "file_expired", "file_id": "f1"}}
            )
        return httpx.Response(
            200,
            json={
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ]
            },
        )

    async def run():
        rt = Runtime(
            StateStore(tmp_path / "runtime.sqlite3"), Ledger(tmp_path / "usage.sqlite3")
        )
        pictures = []
        for i in range(2):
            buffer = BytesIO()
            Image.new("RGB", (16, 16), (i, i, i)).save(buffer, "PNG")
            pictures.append(ImagePrompt("user", buffer.getvalue()))
        route = Route(
            "responses_api:a:m",
            "responses_api",
            "a",
            "m",
            {"endpoint": "https://test/responses"},
            files_ref="files_api:a",
            files_settings={"endpoint": "https://test/files"},
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:

            async def gateway(route):
                return Gateway(client)

            await rt.execute(
                Context(), Prompts(pictures), Preset("p", [route]), gateway, 10
            )
        assert len(uploads) == 3
        assert calls[1]["input"][0]["content"][0]["file_id"] == "f3"
        assert calls[1]["input"][1]["content"][0]["file_id"] == "f2"

    asyncio.run(run())


def test_split_image_reuploads_only_its_invalid_part(tmp_path, monkeypatch):
    import TIYA.llm.media as module

    paths = {}
    for index in range(2):
        path = tmp_path / f"part{index}.png"
        Image.new("RGB", (16, 16), (index, index, index)).save(path)
        paths[f"part{index}"] = path

    async def get(key):
        return paths[key]

    monkeypatch.setattr(module, "get_file_path_async", get)
    uploads, calls = [], []

    def handler(request):
        if request.url.path == "/files":
            uploads.append(request)
            return httpx.Response(200, json={"id": f"f{len(uploads)}"})
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(
                400, json={"error": {"code": "file_expired", "file_id": "f1"}}
            )
        return httpx.Response(200, json={"output": []})

    async def run():
        rt = Runtime(
            StateStore(tmp_path / "state.sqlite3"), Ledger(tmp_path / "usage.sqlite3")
        )
        picture = ImagePrompt("user", paths["part0"])
        picture.image = ["part0", "part1"]
        route = Route(
            "responses_api:x:m",
            "responses_api",
            "x",
            "m",
            {"endpoint": "https://test/responses"},
            files_ref="files_api:x",
            files_settings={"endpoint": "https://test/files"},
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:

            async def gateway(route):
                return Gateway(client)

            await rt.execute(
                Context(), Prompts(picture), Preset("p", [route]), gateway, 5
            )
        assert len(uploads) == 3

    asyncio.run(run())


def test_file_preflight_failure_can_switch_to_declared_backup(tmp_path, monkeypatch):
    import TIYA.llm.media as module

    path = tmp_path / "source.png"
    buffer = BytesIO()
    Image.new("RGB", (16, 16)).save(buffer, "PNG")
    path.write_bytes(buffer.getvalue())

    async def add(content, *args):
        return "cached"

    async def get(key):
        return path

    monkeypatch.setattr(module, "add_file_async", add)
    monkeypatch.setattr(module, "get_file_path_async", get)
    calls = []

    def handler(request):
        calls.append(request.url.host)
        if request.url.host == "badfiles":
            return httpx.Response(401, json={"error": {"code": "invalid_api_key"}})
        assert request.url.host == "backup"
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    async def run():
        rt = Runtime(
            StateStore(tmp_path / "runtime.sqlite3"),
            Ledger(tmp_path / "ledger.sqlite3"),
        )
        primary = Route(
            "responses_api:a:m",
            "responses_api",
            "a",
            "m",
            {"endpoint": "https://primary"},
            files_ref="files_api:a",
            files_settings={"endpoint": "https://badfiles"},
        )
        backup = Route(
            "chat_completions:b:m",
            "chat_completions",
            "b",
            "m",
            {"endpoint": "https://backup"},
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:

            async def gateway(route):
                return Gateway(client)

            pending = await rt.execute(
                Context(),
                Prompts(ImagePrompt("user", path)),
                Preset("p", [primary, backup]),
                gateway,
                5,
            )
        assert pending.route_index == 1
        assert calls == ["badfiles", "backup"]

    asyncio.run(run())
