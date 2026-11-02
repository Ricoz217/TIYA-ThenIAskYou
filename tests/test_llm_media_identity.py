import asyncio
import base64
import json
from io import BytesIO

import pytest
from PIL import Image

from llm_config_fixtures import candidate
from TIYA.llm.media import MediaError, MediaService
from TIYA.llm.model import ImagePrompt, MediaRef
from TIYA.llm.provider import get_provider
from TIYA.llm.state import StateStore


@pytest.fixture
def local_cache(tmp_path, monkeypatch):
    from TIYA.file_cache import FileCache
    import TIYA.file_cache as files
    import TIYA.llm.media as media
    import TIYA.llm.model as model

    cache = FileCache(tmp_path / "files")
    monkeypatch.setattr(media, "add_file_async", cache.add_file_async)
    monkeypatch.setattr(media, "get_file_path_async", cache.get_file_path_async)
    monkeypatch.setattr(files, "add_file", cache.add_file)
    monkeypatch.setattr(files, "get_file_path", cache.get_file_path)
    monkeypatch.setattr(files, "add_file_async", cache.add_file_async)
    monkeypatch.setattr(files, "get_file_path_async", cache.get_file_path_async)
    monkeypatch.setattr(model, "add_file_async", cache.add_file_async)
    monkeypatch.setattr(model, "get_file_path_async", cache.get_file_path_async)
    yield cache
    asyncio.run(cache.close())


def picture():
    output = BytesIO()
    Image.new("RGB", (3, 4)).save(output, "PNG")
    return output.getvalue()


def responses_route():
    return candidate(
        "responses_api:a:m", "responses_api", "a", "m", {"endpoint": "https://model"},
        files_settings={"endpoint": "https://files"},
    )


def test_expired_cache_and_delete_after_lookup_are_input_errors(tmp_path, local_cache, monkeypatch):
    import TIYA.llm.media as media

    service = MediaService(StateStore(tmp_path / "state.db"))
    missing = "a" * 32

    async def run():
        with pytest.raises(MediaError) as error:
            await service.content(MediaRef(hash_name=missing))

        assert error.value.stage == "input"
        assert error.value.hash_name == missing
        assert isinstance(error.value.__cause__, FileNotFoundError)
        key = await local_cache.add_file_async(picture())

        async def deleted_after_lookup(hash_name):
            path = await local_cache.get_file_path_async(hash_name)
            path.unlink()
            return path

        monkeypatch.setattr(media, "get_file_path_async", deleted_after_lookup)
        with pytest.raises(MediaError) as error:
            await service.content(MediaRef(hash_name=key))

        assert error.value.hash_name == key

    asyncio.run(run())


def test_warm_upload_reuses_cache_id_without_reading_local_bytes(tmp_path, local_cache, monkeypatch):
    import TIYA.llm.media as media

    sends = []

    async def send(wire):
        sends.append(wire)
        return {"id": "remote", "expires_at": 9999999999}

    async def run():
        key = await local_cache.add_file_async(picture())
        service = MediaService(StateStore(tmp_path / "state.db"))
        first = await service.upload(MediaRef(hash_name=key), {"endpoint": "https://files"}, send)
        assert first.hash_name == key
        assert first.mime_type == "image/png"
        assert (first.width, first.height) == (3, 4)
        (await local_cache.get_file_path_async(key)).unlink()

        async def forbidden(*args):
            pytest.fail("a valid remote mapping must not read or re-add local content")

        monkeypatch.setattr(media, "get_file_path_async", forbidden)
        monkeypatch.setattr(media, "add_file_async", forbidden)
        second = await MediaService(service.state).upload(
            MediaRef(hash_name=key), {"endpoint": "https://files"}, send,
        )
        assert second == first

    asyncio.run(run())
    assert len(sends) == 1


def test_preparation_uses_hash_identity_across_prompts_and_restart(tmp_path, local_cache, monkeypatch):
    import TIYA.llm.media as media

    sends = []
    additions = []
    original_add = local_cache.add_file_async

    async def add(data, *args):
        additions.append(data)
        return await original_add(data, *args)

    monkeypatch.setattr(media, "add_file_async", add)

    async def send(wire):
        sends.append(wire)
        return {"id": "remote"}

    async def run():
        state = StateStore(tmp_path / "state.db")
        prompt = ImagePrompt("user", picture())
        refs = await MediaService(state).prepare_prompt(prompt, get_provider(responses_route()), send)
        assert prompt.hash_name == refs[0].hash_name
        assert prompt.image_input is None
        restored = ImagePrompt.from_dict(json.loads(json.dumps(prompt.to_dict())))
        assert restored.hash_name == prompt.hash_name
        different_prompt = ImagePrompt("user", hash_name=prompt.hash_name)
        assert different_prompt.id != prompt.id
        for item in (restored, different_prompt):
            prepared = await MediaService(state).prepare_prompt(item, get_provider(responses_route()), send)
            assert prepared[0].value == "remote"

        assert "prepared_input" not in prompt.to_dict()["data"]

    asyncio.run(run())
    assert len(additions) == len(sends) == 1


def test_invalid_remote_id_uploads_only_the_affected_part(tmp_path, local_cache):
    from TIYA.utils import fingerprint
    from TIYA.llm.provider.files import get_files_provider

    uploads = []

    async def send(wire):
        uploads.append(wire)
        return {"id": f"remote-{len(uploads)}"}

    async def run():
        first = await local_cache.add_file_async(picture())
        different = BytesIO()
        Image.new("RGB", (5, 6)).save(different, "PNG")
        second = await local_cache.add_file_async(different.getvalue())
        prompt = ImagePrompt("user", hash_name=first)
        prompt.image = [first, second]
        route = responses_route()
        service = MediaService(StateStore(tmp_path / "state.db"))
        await service.prepare_prompt(prompt, get_provider(route), send)
        scope = get_files_provider(route.files.as_provider_settings()).scope
        service.state.put("invalid_files", fingerprint([scope, "remote-1"]), {"invalid": True})
        refs = await service.prepare_prompt(prompt, get_provider(route), send)
        assert [ref.value for ref in refs] == ["remote-3", "remote-2"]

    asyncio.run(run())
    assert len(uploads) == 3


def test_expired_remote_and_missing_local_content_never_send_upload(tmp_path, local_cache):
    service = MediaService(StateStore(tmp_path / "state.db"))
    count = 0

    async def send(wire):
        nonlocal count
        count += 1
        return {"id": "expired", "expires_at": 1}

    async def run():
        key = await local_cache.add_file_async(picture())
        ref = MediaRef(hash_name=key)
        await service.upload(ref, {"endpoint": "https://files"}, send)
        (await local_cache.get_file_path_async(key)).unlink()
        with pytest.raises(MediaError, match="input"):
            await service.upload(ref, {"endpoint": "https://files"}, send)

    asyncio.run(run())
    assert count == 1


def test_response_known_fields_are_explicit_and_extensions_remain_open(tmp_path, local_cache):
    from TIYA.llm.clients import ImagesClient

    async def run():
        body = {
            "created": 10, "output_format": "png", "background": "transparent",
            "quality": "high", "size": "3x4", "safety_identifier": "safe",
            "vendor_extension": {"value": 1},
            "data": [{"b64_json": base64.b64encode(picture()).decode(),
                      "revised_prompt": "revised", "type": "image_generation.partial_image",
                      "image_index": 2, "partial_image_index": 0,
                      "created_at": 9, "output_format": "png", "vendor_item": True,
                      "safety_identifier": "safe-item"}],
        }
        client = ImagesClient({"endpoint": "https://images"})
        result = await client._materialize(body, "request", None)
        assert result.created == 10
        assert result.output_format == "png"
        assert result.background == "transparent"
        assert result.quality == "high"
        assert result.size == "3x4"
        assert result.safety_identifier == "safe"
        assert result.metadata == {"vendor_extension": {"value": 1}}
        item = result.data[0]
        assert item.image.hash_name
        assert item.event_type == "image_generation.partial_image"
        assert item.image_index == 2
        assert item.partial_image_index == 0
        assert item.created_at == 9
        assert item.output_format == "png"
        assert item.safety_identifier == "safe-item"
        assert item.metadata == {"vendor_item": True}

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["bytes", "path", "base64", "cache", "remote_original", "url"])
def test_content_ingress_and_local_id_are_unambiguous(tmp_path, local_cache, kind):
    service = MediaService(StateStore(tmp_path / "state.db"))
    data = picture()

    class Gateway:
        async def download(self, url):
            assert url == "https://image"
            return data

    async def run():
        key = await local_cache.add_file_async(data)
        path = await local_cache.get_file_path_async(key)
        refs = {
            "bytes": MediaRef("bytes", data),
            "path": MediaRef("path", path),
            "base64": MediaRef("base64", base64.b64encode(data).decode()),
            "cache": MediaRef(hash_name=key),
            "remote_original": MediaRef("remote", "remote-id", hash_name=key),
            "url": MediaRef("url", "https://image"),
        }
        assert await service.content(refs[kind], Gateway()) == data

    asyncio.run(run())


@pytest.mark.parametrize("ref", [MediaRef(), MediaRef("bytes"), MediaRef("base64", "!"),
                                 MediaRef("path", "missing-file"), MediaRef("remote", "remote-id")])
def test_invalid_input_is_reported_without_losing_stage(tmp_path, ref):
    service = MediaService(StateStore(tmp_path / "state.db"))
    with pytest.raises(MediaError) as error:
        asyncio.run(service.content(ref))

    assert error.value.stage == "input"


def test_mime_does_not_fragment_identity_but_files_accounts_do(tmp_path, local_cache):
    uploads = []

    async def send(wire):
        uploads.append(wire)
        return {"id": f"remote-{len(uploads)}"}

    async def run():
        key = await local_cache.add_file_async(picture())
        service = MediaService(StateStore(tmp_path / "state.db"))
        first = await service.upload(MediaRef(hash_name=key), {"endpoint": "https://files-a"}, send)
        again = await service.upload(MediaRef(hash_name=key, mime_type="image/png"),
                                     {"endpoint": "https://files-a"}, send)
        other = await service.upload(MediaRef(hash_name=key), {"endpoint": "https://files-b"}, send)
        assert first == again
        assert first.scope != other.scope

    asyncio.run(run())
    assert len(uploads) == 2


def test_upload_rechecks_after_claim_and_never_reads_winner_content(tmp_path, monkeypatch):
    from TIYA.llm.provider.files import FilesProvider

    state = StateStore(tmp_path / "state.db")
    service = MediaService(state)
    provider = FilesProvider({"endpoint": "https://files"})
    hash_name = "a" * 32

    def claim(key, owner, seconds):
        assert key == f"{provider.scope}:{hash_name}"
        state.put("uploads", key, {"file_id": "winner", "mime_type": "image/png"})
        return True

    async def forbidden(*args, **kwargs):
        pytest.fail("the upload completed by another caller must be reused")

    monkeypatch.setattr(state, "claim", claim)
    monkeypatch.setattr(service, "content", forbidden)
    result = asyncio.run(service.upload(MediaRef(hash_name=hash_name), provider, forbidden))
    assert result.value == "winner"


def test_raw_input_is_read_once_for_inspection_and_upload(tmp_path, local_cache, monkeypatch):
    from pathlib import Path

    path = tmp_path / "source.png"
    path.write_bytes(picture())
    reads = []
    read_bytes = Path.read_bytes

    def read(self):
        reads.append(self)
        return read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", read)

    async def send(wire):
        return {"id": "remote"}

    service = MediaService(StateStore(tmp_path / "state.db"))
    asyncio.run(service.prepare_prompt(ImagePrompt("user", path), get_provider(responses_route()), send))
    assert reads == [path]


def test_legacy_rebuild_uses_original_hash_without_readding_file(tmp_path, local_cache, monkeypatch):
    from TIYA.llm import model

    async def run():
        key = await local_cache.add_file_async(picture())
        original = await local_cache.get_file_path_async(key)
        prompt = ImagePrompt("user", hash_name=key)

        async def rebuild(source, cache):
            assert source == original
            return [key]

        async def forbidden(*args):
            pytest.fail("an original cache ID must not be rehashed/re-added")

        monkeypatch.setattr(model, "add_file_async", forbidden)
        monkeypatch.setattr(model, "handle_unload_image", rebuild)
        assert await prompt.load_image({}) is True
        assert prompt.image == [key]
        assert prompt.image_input is None

    asyncio.run(run())


def test_runtime_returns_cached_image_and_replays_it_without_readding(tmp_path, local_cache, monkeypatch):
    import httpx
    import TIYA.llm.media as media
    from llm_config_fixtures import configuration
    from TIYA.llm.gateway import Gateway
    from TIYA.llm.model import Context, Prompts, TextPrompt
    from TIYA.llm.runtime import Runtime
    from TIYA.llm.usage import Ledger

    payload = "data:image/png;base64," + base64.b64encode(picture()).decode()
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        content = [{"type": "image_url", "image_url": {"url": payload}}] if len(requests) == 1 else "ok"
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    async def run():
        runtime = Runtime(StateStore(tmp_path / "state.db"), Ledger(tmp_path / "usage.db"))
        context = Context()
        config = configuration("main", [candidate("chat_completions:a:m", "chat_completions", "a", "m",
                                                  {"endpoint": "https://model"}, {"stream": False})])
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            async def gateway(route):
                return Gateway(client)

            pending = await runtime.execute(context, Prompts(TextPrompt("user", "draw")), config, gateway, 5)
            returned = pending.result.prompts[0]
            assert returned.hash_name == returned.resource.hash_name
            assert returned.image_input is None
            assert await runtime.media.content(returned.resource) == picture()
            runtime.commit(context, pending)

            async def forbidden(*args):
                pytest.fail("replaying an output image must not add/re-hash it again")

            monkeypatch.setattr(media, "add_file_async", forbidden)
            await runtime.execute(context, Prompts(TextPrompt("user", "describe")), config, gateway, 5)
            assert requests[1]["messages"][1]["content"][0]["image_url"]["url"] == payload

    asyncio.run(run())


@pytest.mark.parametrize("recovery", ["inline", "file_id"])
def test_runtime_reuses_repaired_media_on_following_rate_limit_retry(tmp_path, monkeypatch, recovery):
    import httpx
    from llm_config_fixtures import configuration
    from TIYA.llm.gateway import Gateway
    from TIYA.llm.model import Context, Prompts
    from TIYA.llm.runtime import Runtime
    from TIYA.llm.usage import Ledger

    runtime = Runtime(StateStore(tmp_path / "state.db"), Ledger(tmp_path / "usage.db"))
    requests = []
    preparations = []

    async def prepare(prompt, provider, send, *, inline=False, **kwargs):
        preparations.append(inline)
        prompt.hash_name = "local-id"
        prompt.image_input = None
        if inline:
            return [MediaRef("base64", "YQ==", "image/png", hash_name="local-id")]

        return [MediaRef("remote", f"file-{len(preparations)}", "image/png", hash_name="local-id")]

    monkeypatch.setattr(runtime.media, "prepare_prompt", prepare)

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            code = "file_reference_not_supported" if recovery == "inline" else "file_expired"
            return httpx.Response(400, json={"error": {"code": code, "file_id": "file-1"}})

        if len(requests) == 2:
            return httpx.Response(429, json={"error": {"code": "rate_limit_exceeded"}},
                                  headers={"retry-after": "0"})

        return httpx.Response(200, json={"output": []})

    async def run():
        route = responses_route()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            async def gateway(route):
                return Gateway(client)

            await runtime.execute(Context(), Prompts(ImagePrompt("user", b"")),
                                  configuration("main", [route]), gateway, 5)

    asyncio.run(run())
    assert len(requests) == 3
    assert preparations == ([False, True] if recovery == "inline" else [False, False])
    assert requests[1] == requests[2]


def test_cached_image_metadata_does_not_change_context_identity():
    import copy
    from TIYA.llm.model import Context
    from TIYA.llm.provider.common import projection_fingerprint

    prompt = ImagePrompt("user", hash_name="original")
    context = Context(prompt)
    identity = context.signature
    projection = projection_fingerprint(prompt)
    prompt.resource = MediaRef(hash_name="original", mime_type="image/png", width=3, height=4)
    context.refresh_fingerprints()
    assert context.signature == identity
    assert projection_fingerprint(prompt) == projection
    assert copy.deepcopy(prompt).resource == prompt.resource
    prompt.hash_name = "replacement"
    context.refresh_fingerprints()
    assert context.signature != identity
