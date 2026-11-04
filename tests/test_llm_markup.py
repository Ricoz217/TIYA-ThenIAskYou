import asyncio
import base64
import json
from email import policy
from email.parser import BytesParser
from io import BytesIO

import httpx
import pytest
from PIL import Image

from llm_config_fixtures import candidate
from TIYA.llm.clients import ImagesClient
from TIYA.llm.model import ImageEditRequest, MediaRef
from TIYA.llm.pricing import PriceCatalog
from TIYA.llm.provider.openai_images import Images
from TIYA.llm.state import StateStore
from TIYA.llm.usage import Ledger


def picture(
    size=(32, 16),
    *,
    mode="RGBA",
    color=(0, 0, 0, 0),
    format="PNG",
):
    image = Image.new(mode, size, color)
    image.putpixel((0, 0), (255, 0, 0, 255) if mode == "RGBA" else (255, 0, 0))
    output = BytesIO()
    image.save(output, format)
    return output.getvalue()


@pytest.fixture
def cache(tmp_path, monkeypatch):
    from TIYA.file_cache import FileCache
    import TIYA.llm.media as media

    cache = FileCache(tmp_path / "files")
    monkeypatch.setattr(media, "add_file_async", cache.add_file_async)
    monkeypatch.setattr(media, "get_file_path_async", cache.get_file_path_async)
    yield cache
    asyncio.run(cache.close())


def make_client(
    tmp_path,
    transport,
    edit_format,
    *,
    files=False,
):
    settings = {
        "endpoint": "https://test/generate",
        "edit_endpoint": "https://test/edit",
        "images": {"edit_format": edit_format},
    }
    if files:
        settings["files_api"] = {"endpoint": "https://test/files"}

    return ImagesClient(
        settings,
        client=transport,
        ledger=Ledger(tmp_path / "usage.db"),
        state=StateStore(tmp_path / "state.db"),
        prices=PriceCatalog(tmp_path / "prices.db"),
    )


def multipart_parts(request):
    message = BytesParser(policy=policy.default).parsebytes(
        b"Content-Type: " + request.headers["content-type"].encode()
        + b"\r\n\r\n" + request.content
    )
    return [
        (part.get_param("name", header="content-disposition"), part.get_payload(decode=True))
        for part in message.iter_parts()
    ]


@pytest.mark.parametrize("edit_format,files", [("multipart", False), ("json", False), ("json", True)])
def test_markup_cache_ids_preserve_bytes_order_and_cache_output(
    tmp_path,
    cache,
    edit_format,
    files,
):
    requests = []
    target = picture(mode="RGB", color=(10, 20, 30))
    reference = picture((16, 32))
    markup = picture()

    def handler(request):
        requests.append(request)
        if request.url.path == "/files":
            return httpx.Response(200, json={"id": f"file-{len(requests)}"})

        return httpx.Response(200, json={
            "data": [{"b64_json": base64.b64encode(target).decode()}],
            "usage": {"input_tokens": 20, "output_tokens": 30},
        })

    async def run():
        keys = [await cache.add_file_async(data) for data in (target, reference, markup)]
        request = ImageEditRequest(
            "gpt-image-2.5-sunburst", "change the circled area",
            images=[MediaRef(hash_name=key) for key in keys[:2]],
            markup=MediaRef(hash_name=keys[2]),
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
            client = make_client(tmp_path, transport, edit_format, files=files)
            result = await client.edit(request)

        assert (await cache.get_file_path_async(result.data[0].image.hash_name)).read_bytes() == target
        assert result.usage["input_tokens"] == 20
        assert request.markup.hash_name == keys[2]
        assert [ref.hash_name for ref in request.images] == keys[:2]
        rows = client.ledger.query_requests()
        assert len(rows) == (4 if files else 1)
        edit = requests[-1]
        assert edit.url.path == "/edit"
        if edit_format == "multipart":
            parts = multipart_parts(edit)
            assert [(name, data) for name, data in parts if name == "image[]"] == [
                ("image[]", target), ("image[]", reference),
            ]
            assert [(name, data) for name, data in parts if name in ("mask", "markup")] == [("markup", markup)]

        else:
            body = json.loads(edit.content)
            assert "mask" not in body
            if files:
                assert body["images"] == [{"file_id": "file-1"}, {"file_id": "file-2"}]
                assert body["markup"] == {"file_id": "file-3"}
                assert [multipart_parts(req)[-1][1] for req in requests[:3]] == [target, reference, markup]

            else:
                refs = [*body["images"], body["markup"]]
                assert [base64.b64decode(ref["image_url"].split(",", 1)[1]) for ref in refs] == [target, reference, markup]

    asyncio.run(run())


@pytest.mark.parametrize("guide", [
    picture(mode="RGB", color=(0, 0, 0)),
    picture(color=(0, 0, 0, 255)),
    picture((16, 32)),
    picture(mode="RGB", color=(0, 0, 0), format="JPEG"),
], ids=["no-alpha", "opaque", "wrong-size", "jpeg"])
def test_invalid_markup_rejected_before_upload_or_edit(tmp_path, guide):
    def handler(request):
        pytest.fail(f"invalid markup must not send {request.url}")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
            client = make_client(tmp_path, transport, "json", files=True)
            with pytest.raises(ValueError, match="markup"):
                await client.edit(ImageEditRequest(
                    "image", "change", images=[MediaRef("bytes", picture())],
                    markup=MediaRef("bytes", guide),
                ))

            assert client.ledger.query_requests() == []

    asyncio.run(run())


@pytest.mark.parametrize("typed_mask", [True, False])
def test_mask_and_markup_rejected_before_media_access(tmp_path, typed_mask):
    async def run():
        async with httpx.AsyncClient() as transport:
            client = make_client(tmp_path, transport, "json")
            with pytest.raises(ValueError, match="mask.*markup|markup.*mask"):
                await client.edit(ImageEditRequest(
                    "image", "change", images=[MediaRef(hash_name="missing")],
                    mask=MediaRef(hash_name="missing") if typed_mask else None,
                    markup=MediaRef(hash_name="missing"),
                    params={} if typed_mask else {"mask": {"file_id": "remote"}},
                ))

    asyncio.run(run())


@pytest.mark.parametrize("markup,expected", [
    (MediaRef("remote", "annotation"), {"file_id": "annotation"}),
    (MediaRef("url", "https://source/markup.png"), {"image_url": "https://source/markup.png"}),
])
def test_json_remote_markup_does_not_download(tmp_path, cache, markup, expected):
    def handler(request):
        assert request.method == "POST"
        assert json.loads(request.content)["markup"] == expected
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(picture()).decode()}]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
            client = make_client(tmp_path, transport, "json")
            await client.edit(ImageEditRequest(
                "image", "change", images=[MediaRef("url", "https://source/image.png")],
                markup=markup,
            ))

    asyncio.run(run())


def test_palette_markup_and_oriented_target_keep_original_bytes(tmp_path, cache):
    original = Image.new("RGB", (16, 32))
    exif = Image.Exif()
    exif[274] = 6
    target = BytesIO()
    original.save(target, "JPEG", exif=exif)
    overlay = Image.new("P", (32, 16))
    overlay.putpixel((0, 0), 1)
    markup = BytesIO()
    overlay.save(markup, "PNG", transparency=0)

    def handler(request):
        body = json.loads(request.content)
        assert base64.b64decode(body["images"][0]["image_url"].split(",", 1)[1]) == target.getvalue()
        assert base64.b64decode(body["markup"]["image_url"].split(",", 1)[1]) == markup.getvalue()
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(picture()).decode()}]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
            client = make_client(tmp_path, transport, "json")
            await client.edit(ImageEditRequest(
                "image", "change", images=[MediaRef("bytes", target.getvalue())],
                markup=MediaRef("base64", base64.b64encode(markup.getvalue()).decode()),
            ))

    asyncio.run(run())


@pytest.mark.parametrize("edit_format", ["json", "multipart"])
def test_provider_checks_effective_guides_and_typed_markup_wins(edit_format):
    route = candidate("images_api:x:image", "images_api", "x", "image", {
        "endpoint": "https://test/generate", "edit_endpoint": "https://test/edit",
        "images": {"edit_format": edit_format},
    }, params={"mask": {"file_id": "default-mask"}})
    provider = Images(route)
    target = MediaRef("bytes", picture(), "image/png")
    if edit_format == "json":
        target = MediaRef("remote", "target")

    markup = MediaRef("bytes", picture(), "image/png")
    request = ImageEditRequest("image", "change", images=[target])
    with pytest.raises(ValueError, match="mask.*markup"):
        provider.encode_operation(request, request.images, markup=markup)

    route.model_params.clear()
    request.params["markup"] = {"file_id": "overridden"}
    if edit_format == "json":
        markup = MediaRef("remote", "prepared")

    wire = provider.encode_operation(request, request.images, markup=markup)
    if edit_format == "json":
        assert wire.json["markup"] == {"file_id": "prepared"}

    else:
        assert "markup" not in wire.data
        assert [part[0] for part in wire.files] == ["image[]", "markup"]
