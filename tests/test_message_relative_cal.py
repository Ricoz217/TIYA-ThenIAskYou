from TIYA.model.message import (
    AtMsg,
    FileMsg,
    GroupMsg,
    ImgMsg,
    RawMsg,
    ReplyMsg,
    SetuMsg,
    TextMsg,
)


def test_to_relative_cal_returns_structured_multimodal_evidence() -> None:
    message = GroupMsg(
        msg_id="message",
        group_id="group",
        user_id="user",
        username="",
        nickname="",
        content=[
            ReplyMsg(msg_id="origin"),
            AtMsg(user_id="target"),
            TextMsg(text="看看这个"),
            ImgMsg(
                name="meme.png",
                hash_name="hash",
                description="一张表达无语的猫猫表情包",
            ),
            FileMsg(filename="设计方案.pdf", hash_name="file"),
            SetuMsg(
                illust_id="123",
                page=1,
                hash_name="setu",
                artist_id=9,
                artist_name="测试画师",
                type="illust",
                title="海边",
                caption="夏日插画",
                tags=["蓝天", "海浪"],
            ),
            RawMsg(type="face", content={"id": "14"}),
        ],
    )

    result = message.to_relative_cal()

    assert result.text == "看看这个"
    assert "无语" in result.semantic_text
    assert "设计方案.pdf" in result.semantic_text
    assert "海边" in result.semantic_text
    assert result.media_ids == (
        "image:hash",
        "setu:123:1",
        "face:14",
    )
    assert result.reply_to == "origin"
    assert result.mention_ids == frozenset({"target"})
    assert result.pending_media is False


def test_to_relative_cal_marks_unparsed_images_as_pending() -> None:
    message = GroupMsg(
        msg_id="message",
        group_id="group",
        user_id="user",
        username="",
        nickname="",
        content=[ImgMsg(name="pending.png")],
    )

    result = message.to_relative_cal()

    assert result.text == ""
    assert result.semantic_text == ""
    assert result.media_ids == ()
    assert result.pending_media is True
