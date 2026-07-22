from io import BytesIO
from unittest.mock import patch

import pytest
from PIL import Image, ImageSequence

from TIYA.utils import postprocess_image


def _make_gif() -> bytes:
    frames = [
        Image.new("RGB", (24, 20), (index * 60, 20, 40))
        for index in range(3)
    ]
    output = BytesIO()
    frames[0].save(
        output,
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=[40, 70, 110],
        loop=3,
        optimize=False,
    )
    return output.getvalue()


def test_postprocess_image_crops_all_gif_frames_with_the_same_offsets() -> None:
    original = _make_gif()

    with patch("TIYA.utils.random.randint", side_effect=[1, 2, 3, 4]):
        result = postprocess_image(original)

    with Image.open(result) as image:
        assert image.format == "GIF"
        assert image.size == (21, 13)
        assert image.n_frames == 3
        assert image.info["loop"] == 3
        assert [
            int(frame.info.get("duration", 0))
            for frame in ImageSequence.Iterator(image)
        ] == [40, 70, 110]


def test_postprocess_image_adds_exact_number_of_static_noise_pixels() -> None:
    source = Image.new("RGBA", (24, 24), (12, 34, 56, 255))
    original = BytesIO()
    source.save(original, format="PNG")

    with (
        patch("TIYA.utils.random.randint", side_effect=[2, 2, 2, 2]),
        patch("TIYA.utils.random.sample", return_value=[0, 20, 40, 60, 80]),
        patch("TIYA.utils.random.choice", side_effect=[True, False, True, False, True]),
    ):
        result = postprocess_image(original.getvalue(), noise_pixels=5)

    with Image.open(result) as image:
        pixels = list(image.convert("RGBA").getdata())

    changed = [pixel for pixel in pixels if pixel != (12, 34, 56, 255)]
    assert image.size == (20, 20)
    assert len(changed) == 5
    assert set(changed) == {(255, 255, 255, 255), (255, 255, 255, 0)}


def test_postprocess_image_rejects_noise_larger_than_cropped_image() -> None:
    source = Image.new("RGB", (20, 20), "black")
    original = BytesIO()
    source.save(original, format="PNG")

    with patch("TIYA.utils.random.randint", side_effect=[4, 4, 4, 4]):
        with pytest.raises(ValueError, match="noise_pixels"):
            postprocess_image(original.getvalue(), noise_pixels=145)


def test_postprocess_image_uses_white_noise_for_jpeg_paths(tmp_path) -> None:
    source = Image.new("RGB", (16, 16), (20, 30, 40))
    source_path = tmp_path / "source.jpg"
    source.save(source_path, format="JPEG", quality=100, subsampling=0)

    with (
        patch("TIYA.utils.random.randint", side_effect=[1, 1, 1, 1]),
        patch("TIYA.utils.random.sample", return_value=[0]),
    ):
        result = postprocess_image(source_path, noise_pixels=1)

    with Image.open(result) as image:
        assert image.format == "JPEG"
        assert image.mode == "RGB"
        assert image.size == (14, 14)
        assert min(image.getpixel((0, 0))) >= 240


@pytest.mark.parametrize(
    ("image", "noise_pixels", "error"),
    [
        (b"not an image", 0, ValueError),
        (b"not an image", -1, ValueError),
        (b"not an image", 1.5, TypeError),
    ],
)
def test_postprocess_image_validates_input_and_noise_count(
        image: bytes,
        noise_pixels: int,
        error: type[Exception],
) -> None:
    with pytest.raises(error):
        postprocess_image(image, noise_pixels=noise_pixels)
