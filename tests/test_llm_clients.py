import asyncio
import base64
from io import BytesIO

import httpx
import pytest
from PIL import Image

from TIYA.llm.model import ImageGenerateRequest, ImageEditRequest, MediaRef


def image():
    data = BytesIO()
    Image.new("RGBA", (16, 16)).save(data, "PNG")
    return data.getvalue()


def test_images_client_caches_results_and_preserves_alpha_mask(tmp_path, monkeypatch):
    from TIYA.llm.clients import ImagesClient
    from TIYA.llm.usage import Ledger
    import TIYA.llm.media as media

    saved = []
    requests = []

    async def cache(data, *args):
        saved.append(data)
        return "cached"

    monkeypatch.setattr(media, "add_file_async", cache)

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "created": 1,
                "data": [{"b64_json": base64.b64encode(image()).decode()}],
                "usage": {"input_tokens": 20, "output_tokens": 30},
            },
        )

    async def run():
        ledger = Ledger(tmp_path / "usage.sqlite3")
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as transport:
            client = ImagesClient(
                {
                    "endpoint": "https://test/gen",
                    "edit_endpoint": "https://test/edit",
                    "key": "k",
                },
                client=transport,
                ledger=ledger,
            )
            response = await client.generate(
                ImageGenerateRequest("image-2.5", "draw", background="transparent")
            )
            assert response.data[0].image.hash_name == "cached"
            await client.edit(
                ImageEditRequest(
                    "image-2.5",
                    "change",
                    images=[MediaRef("bytes", image())],
                    mask=MediaRef("bytes", image()),
                )
            )
            assert requests[1].url.path == "/edit"
            assert b'name="mask"' in requests[1].content
            assert len(saved) == 2
            assert len(ledger.query_requests()) == 2

    asyncio.run(run())


def test_materialization_failure_does_not_regenerate(tmp_path, monkeypatch):
    from TIYA.llm.clients import ImagesClient
    from TIYA.llm.media import MediaError
    from TIYA.llm.usage import Ledger

    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            200, json={"data": [{"b64_json": "bad"}], "usage": {"output_tokens": 30}}
        )

    async def run():
        ledger = Ledger(tmp_path / "usage.sqlite3")
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as transport:
            client = ImagesClient(
                {"endpoint": "https://test/gen"}, client=transport, ledger=ledger
            )
            with pytest.raises(MediaError):
                await client.generate(ImageGenerateRequest("image-2.5", "draw"))
        assert len(calls) == 1
        row = ledger.query_requests()[0]
        assert row["reported"]["output"] == 30
        assert row["status"] == "materialization_failed"

    asyncio.run(run())


def test_media_error_declares_optional_pending_response():
    from TIYA.llm.media import MediaError

    assert MediaError("input", "missing file").pending_response is None


def test_repeated_image_recovery_preserves_response_and_only_generates_once(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from TIYA.llm.clients import ImagesClient
    from TIYA.llm.media import MediaError, MediaService
    from TIYA.llm.usage import Ledger
    from TIYA.llm import usage_compat

    requests = []
    attempts = []
    real_materialize = MediaService.materialize

    async def materialize(self, item, gateway):
        attempts.append(item)
        if len(attempts) < 3:
            raise MediaError("validate_cache", "temporary cache failure")

        return await real_materialize(self, item, gateway)

    monkeypatch.setattr(MediaService, "materialize", materialize)

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={
            "data": [{"b64_json": base64.b64encode(image()).decode()}],
            "usage": {"output_tokens": 30},
        })

    async def run():
        ledger = Ledger(tmp_path / "usage.sqlite3")
        monkeypatch.setattr(usage_compat, "_GLOBAL_USAGE", SimpleNamespace(ledger=ledger))
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
            client = ImagesClient({"endpoint": "https://test/gen"}, client=transport, ledger=ledger)
            with pytest.raises(MediaError) as first:
                await client.generate(ImageGenerateRequest("image-2.5", "draw"))

            response = first.value.pending_response
            assert response is not None
            request_id = first.value.request_id
            with pytest.raises(MediaError) as second:
                await client.recover_images(first.value)

            assert second.value.pending_response is response
            assert second.value.request_id == request_id
            receiver = ImagesClient({"endpoint": "https://test/gen"}, client=transport)
            assert receiver.ledger is None
            result = await receiver.recover_images(second.value)
            assert result.request_id == request_id
            assert result.data[0].image.hash_name
            assert attempts[0] is attempts[1] is attempts[2]
            rows = ledger.query_requests()
            assert len(rows) == len(requests) == 1
            assert rows[0]["status"] == "completed"
            assert rows[0]["reported"]["output"] == 30

    asyncio.run(run())


def test_files_do_not_inherit_model_credentials_from_shared_transport(tmp_path):
    from TIYA.llm.clients import FilesClient
    from TIYA.llm.usage import Ledger

    def handler(request):
        assert "authorization" not in request.headers
        assert request.headers["x-api-key"] == "files-key"
        assert "x-model-secret" not in request.headers
        return httpx.Response(200, json={"id": "f"})

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            headers={"Authorization": "Bearer model-key", "x-model-secret": "private"},
        ) as transport:
            client = FilesClient(
                {
                    "endpoint": "https://files.test/files",
                    "key": "files-key",
                    "dialect": "anthropic",
                },
                client=transport,
                ledger=Ledger(tmp_path / "usage.sqlite3"),
            )
            assert (await client.upload(MediaRef("bytes", image()))).file_id == "f"

    asyncio.run(run())


def test_image_stream_caches_previews_and_deduplicates_final_events(
    tmp_path, monkeypatch
):
    import json
    from TIYA.llm.clients import ImagesClient
    from TIYA.llm.usage import Ledger
    import TIYA.llm.media as media

    payload = base64.b64encode(image()).decode()
    final = {
        "type": "image_generation.completed",
        "b64_json": payload,
        "created_at": 10,
        "output_format": "png",
        "usage": {"input_tokens": 3, "output_tokens": 4},
    }
    events = [
        {
            "type": "image_generation.partial_image",
            "b64_json": payload,
            "partial_image_index": 0,
        },
        final,
        final,
    ]

    def handler(request):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="".join("data: " + json.dumps(e) + "\n\n" for e in events),
        )

    cached = []
    previews = []

    async def cache(data, *args):
        cached.append(data)
        return "cached"

    monkeypatch.setattr(media, "add_file_async", cache)

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as transport:
            client = ImagesClient(
                {"endpoint": "https://test/gen"},
                client=transport,
                ledger=Ledger(tmp_path / "ledger.sqlite3"),
            )
            result = await client.generate(
                ImageGenerateRequest("image-2.5", "draw", params={"stream": True}),
                on_preview=previews.append,
            )
        assert len(result.data) == 1
        assert result.created == 10
        assert result.output_format == "png"
        assert previews[0].image.hash_name == "cached"
        assert len(cached) == 2

    asyncio.run(run())
