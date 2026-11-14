import asyncio
import base64
from io import BytesIO

from PIL import Image


def png():
    output = BytesIO()
    Image.new("RGBA", (16, 16), (1, 2, 3, 0)).save(output, format="PNG")
    return output.getvalue()


def test_generated_images_are_materialized_before_return(tmp_path, monkeypatch):
    from TIYA.llm.media import MediaService
    from TIYA.llm.state import StateStore
    import TIYA.llm.media as module

    cached = {}

    async def add(data, *args, **kwargs):
        cached["content"] = data
        return "cache-id"

    monkeypatch.setattr(module, "add_file_async", add)

    async def run():
        service = MediaService(StateStore(tmp_path / "runtime.sqlite3"))
        result = await service.materialize(
            {"b64_json": base64.b64encode(png()).decode()}, None
        )
        assert result.image.hash_name == "cache-id"
        assert result.image.mime_type == "image/png"
        assert cached["content"] == png()
        data_url = "data:image/png;base64," + base64.b64encode(png()).decode()
        embedded = await service.materialize({"url": data_url}, None)
        assert embedded.image.hash_name == "cache-id"

    asyncio.run(run())


def test_file_dialects_encode_expiry_and_parse_metadata():
    from TIYA.llm.provider.files import FilesProvider

    provider = FilesProvider(
        {
            "endpoint": "https://files.test/files",
            "key": "key",
            "dialect": "deepseek",
            "expires_after_seconds": 3600,
        }
    )
    request = provider.upload_request(png(), "image.png", "image/png")
    assert request.data["expires_after[seconds]"] == "3600"
    assert request.data["purpose"] == "user_data"
    metadata = provider.decode({"id": "file-1", "created_at": 10, "expires_at": 3610})
    assert metadata.file_id == "file-1"
    assert metadata.expires_at == 3610


def test_images_edit_keeps_mask_and_reference_order():
    from llm_config_fixtures import candidate
    from TIYA.llm.model import ImageEditRequest, MediaRef
    from TIYA.llm.provider.openai_images import Images

    route = candidate(
        "image_generation:x:image-2.5",
        "image_generation",
        "x",
        "image-2.5",
        {
            "endpoint": "https://test/generate",
            "edit_endpoint": "https://test/edit",
            "images": {"edit_format": "json"},
        },
    )
    request = ImageEditRequest(
        "image-2.5",
        "edit",
        images=[MediaRef("remote", "target"), MediaRef("url", "https://ref")],
        mask=MediaRef("remote", "mask"),
        background="transparent",
    )
    encoded = Images(route).encode_operation(request, request.images, request.mask)
    assert encoded.url == "https://test/edit"
    assert encoded.json["images"] == [
        {"file_id": "target"},
        {"image_url": "https://ref"},
    ]
    assert encoded.json["mask"] == {"file_id": "mask"}
    assert encoded.json["background"] == "transparent"


def test_unsupported_files_business_error_falls_back_without_losing_image(
    tmp_path, monkeypatch
):
    from llm_config_fixtures import candidate
    from TIYA.llm.media import MediaService
    from TIYA.llm.state import StateStore
    from TIYA.llm.provider import get_provider
    from TIYA.llm.model import ImagePrompt
    import TIYA.llm.media as module

    path = tmp_path / "image.png"
    path.write_bytes(png())

    async def add(data, *args):
        return "cached"

    async def get(key):
        return path

    monkeypatch.setattr(module, "add_file_async", add)
    monkeypatch.setattr(module, "get_file_path_async", get)
    uploads = []

    async def send(wire):
        uploads.append(wire)
        return {"error": {"code": "file_reference_not_supported"}}

    async def run():
        route = candidate(
            "responses_api:x:m",
            "responses_api",
            "x",
            "m",
            {"endpoint": "https://test"},
            files_ref="files_api:x",
            files_settings={"endpoint": "https://files"},
        )
        service = MediaService(StateStore(tmp_path / "runtime.sqlite3"))
        prompt = ImagePrompt("user", path)
        refs = await service.prepare_prompt(prompt, get_provider(route), send)
        assert refs[0].kind == "base64"
        assert base64.b64decode(refs[0].value) == png()
        refs = await service.prepare_prompt(prompt, get_provider(route), send)
        assert len(uploads) == 1

    asyncio.run(run())


def test_long_concurrent_upload_renews_lease_and_uploads_once(tmp_path):
    from TIYA.llm.media import MediaService
    from TIYA.llm.model import MediaRef
    from TIYA.llm.state import StateStore

    sends = []

    async def send(wire):
        sends.append(wire)
        await asyncio.sleep(0.2)
        return {"id": "shared"}

    async def run():
        path = tmp_path / "runtime.sqlite3"
        first = MediaService(StateStore(path), lease_seconds=0.06)
        second = MediaService(StateStore(path), lease_seconds=0.06)
        settings = {"endpoint": "https://test/files"}
        results = await asyncio.gather(
            first.upload(MediaRef("bytes", png(), "image/png"), settings, send),
            second.upload(MediaRef("bytes", png(), "image/png"), settings, send),
        )
        assert [r.value for r in results] == ["shared", "shared"]
        assert len(sends) == 1

    asyncio.run(run())


def test_state_database_failure_uses_inline_and_keeps_the_image(tmp_path, monkeypatch):
    import sqlite3
    import TIYA.llm.media as module
    from llm_config_fixtures import candidate
    from TIYA.llm.media import MediaService
    from TIYA.llm.model import ImagePrompt
    from TIYA.llm.provider import get_provider

    class BrokenState:
        def get(self, *args):
            raise sqlite3.OperationalError("state unavailable")

        def put(self, *args):
            raise sqlite3.OperationalError("state unavailable")

    path = tmp_path / "image.png"
    path.write_bytes(png())

    async def add(data, *args):
        return "cached"

    async def get(key):
        return path

    async def send(wire):
        raise AssertionError("must not upload without state")

    monkeypatch.setattr(module, "add_file_async", add)
    monkeypatch.setattr(module, "get_file_path_async", get)

    async def run():
        provider = get_provider(
            candidate(
                "responses_api:x:m",
                "responses_api",
                "x",
                "m",
                {"endpoint": "https://test"},
                files_settings={"endpoint": "https://files"},
            )
        )
        result = await MediaService(BrokenState()).prepare_prompt(
            ImagePrompt("user", path), provider, send
        )
        assert result[0].kind == "base64"
        assert base64.b64decode(result[0].value) == png()

    asyncio.run(run())


def test_missing_legacy_variant_is_rebuilt_once_from_available_original(
    tmp_path, monkeypatch
):
    import TIYA.llm.media as module
    from llm_config_fixtures import candidate
    from TIYA.llm.media import MediaService
    from TIYA.llm.model import ImagePrompt
    from TIYA.llm.provider import get_provider
    from TIYA.llm.state import StateStore

    original = tmp_path / "original.png"
    original.write_bytes(png())
    rebuilt = []

    async def get(key):
        if key == "expired":
            raise FileNotFoundError(key)
        return original

    async def rebuild(prompt, cache):
        assert prompt.image_input.is_file()
        rebuilt.append(prompt.id)
        prompt.image = ["rebuilt"]
        return True

    async def send(wire):
        raise AssertionError("inline path must not upload")

    monkeypatch.setattr(module, "get_file_path_async", get)
    monkeypatch.setattr(ImagePrompt, "load_image", rebuild)

    async def run():
        prompt = ImagePrompt("user", original)
        prompt.image = ["expired"]
        provider = get_provider(
            candidate("chat_completions:x:m", "chat_completions", "x", "m", {})
        )
        refs = await MediaService(
            StateStore(tmp_path / "state.sqlite3")
        ).prepare_prompt(prompt, provider, send)
        assert base64.b64decode(refs[0].value) == png()
        assert rebuilt == [prompt.id]

    asyncio.run(run())
