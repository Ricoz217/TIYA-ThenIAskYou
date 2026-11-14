import asyncio
from io import BytesIO

import pytest
from PIL import Image

from TIYA.image_diff import extract_markup, extract_markup_async


def encode(image, *, format="PNG", **kwargs):
    output = BytesIO()
    image.save(output, format=format, **kwargs)
    return output.getvalue()


@pytest.mark.parametrize("color", [(0, 0, 0), (0, 120, 0), (50, 50, 250)])
def test_extracts_any_color_and_clears_unchanged_background(color):
    original = Image.new("RGB", (8, 6), "white")
    marked = original.copy()
    marked.putpixel((2, 3), color)

    output = extract_markup(encode(original), encode(marked), tolerance=10)
    with Image.open(BytesIO(output)) as layer:
        assert layer.format == "PNG"
        assert layer.mode == "RGBA"
        assert layer.size == original.size
        assert layer.getpixel((2, 3)) == (*color, 255)
        assert layer.getpixel((0, 0)) == (0, 0, 0, 0)
        assert layer.getchannel("A").getbbox() == (2, 3, 3, 4)

    assert original.getpixel((2, 3)) == (255, 255, 255)


def test_tolerance_uses_max_channel_difference_and_excludes_equal_threshold():
    original = Image.new("RGB", (3, 1), (100, 100, 100))
    marked = original.copy()
    marked.putpixel((0, 0), (130, 100, 100))
    marked.putpixel((1, 0), (100, 131, 100))
    marked.putpixel((2, 0), (100, 100, 131))

    with Image.open(BytesIO(extract_markup(encode(original), encode(marked), tolerance=30))) as layer:
        assert list(layer.getchannel("A").getdata()) == [0, 255, 255]


def test_suppresses_small_background_variations_without_color_filter():
    original = Image.new("RGB", (16, 16), (130, 130, 130))
    marked = Image.new("RGB", original.size, (135, 125, 133))
    marked.putpixel((7, 7), (70, 90, 140))

    with Image.open(BytesIO(extract_markup(encode(original), encode(marked), tolerance=10))) as layer:
        assert layer.getchannel("A").getbbox() == (7, 7, 8, 8)


def test_preserves_selected_alpha_and_detects_alpha_only_changes():
    original = Image.new("RGBA", (2, 1), (0, 0, 0, 0))
    marked = original.copy()
    marked.putpixel((1, 0), (0, 0, 0, 120))

    with Image.open(BytesIO(extract_markup(encode(original), encode(marked), tolerance=10))) as layer:
        assert layer.getpixel((1, 0)) == (0, 0, 0, 120)
        assert layer.getpixel((0, 0)) == (0, 0, 0, 0)


def test_opaque_return_does_not_treat_lost_alpha_as_markup():
    original = Image.new("RGBA", (4, 3), (138, 181, 251, 83))
    marked = Image.new("RGBA", original.size, (136, 178, 252, 255))
    marked.putpixel((2, 1), (220, 40, 35, 255))
    original_data = encode(original)
    marked_data = encode(marked)

    with Image.open(BytesIO(extract_markup(original_data, marked_data, tolerance=128))) as layer:
        assert layer.getchannel("A").getbbox() == (2, 1, 3, 2)
        assert layer.getpixel((2, 1)) == (220, 40, 35, 255)

    explicit = extract_markup(original_data, marked_data, tolerance=128, compare_alpha=True)
    with Image.open(BytesIO(explicit)) as layer:
        assert layer.getchannel("A").getextrema() == (255, 255)


def test_despeckle_removes_small_fragments_and_preserves_thin_stroke_endpoints():
    original = Image.new("RGB", (12, 10), "white")
    marked = original.copy()
    stroke = [(0, y) for y in range(2, 7)]
    speckles = [(11, 0), (6, 5), (7, 6)]
    for point in stroke + speckles:
        marked.putpixel(point, (0, 0, 180))

    output = extract_markup(encode(original), encode(marked), despeckle=True)
    with Image.open(BytesIO(output)) as layer:
        assert layer.getchannel("A").getbbox() == (0, 2, 1, 7)
        assert all(layer.getpixel(point) == (0, 0, 180, 255) for point in stroke)
        assert all(layer.getpixel(point) == (0, 0, 0, 0) for point in speckles)


def test_jpeg_recompression_preserves_clear_marks_with_tolerance():
    original = Image.new("RGB", (32, 32), (130, 130, 130))
    marked = original.copy()
    marked.paste((40, 60, 20), (8, 8, 16, 16))

    output = extract_markup(encode(original), encode(marked, format="JPEG", quality=75), tolerance=30)
    with Image.open(BytesIO(output)) as layer:
        assert layer.getpixel((0, 0)) == (0, 0, 0, 0)
        assert layer.getpixel((12, 12))[3] == 255


@pytest.mark.parametrize("tolerance", [0, 30, 255])
def test_identical_inputs_produce_fully_transparent_layer(tolerance):
    content = encode(Image.new("RGB", (8, 6), "blue"))
    with Image.open(BytesIO(extract_markup(content, content, tolerance=tolerance))) as layer:
        assert layer.getchannel("A").getbbox() is None


def test_dimension_restoration_and_strict_mode():
    original = encode(Image.new("RGB", (16, 8), "white"))
    marked = encode(Image.new("RGB", (8, 4), "black"))
    with Image.open(BytesIO(extract_markup(original, marked))) as layer:
        assert layer.size == (16, 8)
        assert layer.getpixel((0, 0)) == (0, 0, 0, 255)

    with pytest.raises(ValueError, match="尺寸"):
        extract_markup(original, marked, resize=False)


def test_path_input_exif_orientation_and_unchanged_source_files(tmp_path):
    original = Image.new("RGB", (3, 2), "white")
    original.putpixel((0, 0), (255, 0, 0))
    exif = Image.Exif()
    exif[274] = 6
    original_path = tmp_path / "original.png"
    original_path.write_bytes(encode(original, exif=exif))
    marked = original.transpose(Image.Transpose.ROTATE_270)
    marked.putpixel((0, 0), (0, 0, 0))
    marked_path = tmp_path / "marked.png"
    marked_path.write_bytes(encode(marked))
    before = original_path.read_bytes(), marked_path.read_bytes()

    with Image.open(BytesIO(extract_markup(original_path, marked_path, resize=False))) as layer:
        assert layer.size == (2, 3)
        assert layer.getchannel("A").getbbox() == (0, 0, 1, 1)

    assert (original_path.read_bytes(), marked_path.read_bytes()) == before


@pytest.mark.parametrize("tolerance", [-1, 256, 1.5, True])
def test_invalid_tolerance_is_rejected_before_reading_inputs(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        extract_markup(b"not an image", b"not an image", tolerance=tolerance)


def test_invalid_source_type_and_corrupt_images_are_not_guessed():
    with pytest.raises(TypeError, match="Path.*bytes|bytes.*Path"):
        extract_markup("cache-id", b"")

    with pytest.raises(OSError):
        extract_markup(b"not an image", b"not an image")


def test_async_entry_returns_png_bytes_and_propagates_missing_files(tmp_path):
    original = Image.new("RGB", (8, 6), "white")
    marked = original.copy()
    marked.putpixel((3, 2), (80, 90, 0))
    original_path = tmp_path / "original.png"
    marked_path = tmp_path / "marked.png"
    original.save(original_path)
    marked.save(marked_path)

    async def run():
        markup = await extract_markup_async(original_path, marked_path, tolerance=20)
        with Image.open(BytesIO(markup)) as layer:
            assert layer.format == "PNG"
            assert layer.getchannel("A").getbbox() == (3, 2, 4, 3)

        cleaned = await extract_markup_async(original_path, marked_path, despeckle=True)
        with Image.open(BytesIO(cleaned)) as layer:
            assert layer.getchannel("A").getbbox() is None

        with pytest.raises(FileNotFoundError):
            await extract_markup_async(tmp_path / "missing.png", marked_path)

    asyncio.run(run())
