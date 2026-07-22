import asyncio

import pytest

from TIYA.model.message import GroupMsg, ReplyMsg, SetuMsg, TextMsg


def _setu_message() -> SetuMsg:
    return SetuMsg(
        illust_id="123456",
        page=0,
        hash_name="hash-name",
        artist_id=789,
        artist_name="artist",
        type="IMAGE",
        title="test title",
        caption="",
        tags=["tag"],
    )


def _format_reply(text: str, origin_content: list) -> dict:
    origin = GroupMsg(
        msg_id="origin-message",
        user_id="bot-user",
        username="Bot",
        nickname="",
        group_id="test-group",
        content=origin_content,
    )
    reply = ReplyMsg(msg_id=origin.msg_id)
    asyncio.run(reply.process(origin))
    message = GroupMsg(
        msg_id="reply-message",
        user_id="test-user",
        username="User",
        nickname="Tester",
        group_id="test-group",
        content=[reply, TextMsg(text=text)],
    )

    return message._format_text_llm()


@pytest.mark.parametrize(
    ("raw_score", "expected_score"),
    [
        ("114514", "10.00"),
        ("0.005", "0.01"),
        ("-1919810", "0.00"),
    ],
)
def test_setu_reply_tip_clamps_and_quantizes_score(
        raw_score: str,
        expected_score: str,
) -> None:
    formatted = _format_reply(raw_score, [_setu_message()])

    assert formatted["message_tips"][-1] == (
        f"对色图[123456_p0]《test title》打分 {expected_score} 分"
    )


def test_setu_reply_tip_marks_interaction_command() -> None:
    formatted = _format_reply("--ban-id", [_setu_message()])

    assert formatted["message_tips"][-1] == (
        "对色图[123456_p0]《test title》发送互动指令：--ban-id"
    )


def test_setu_reply_tip_marks_plain_comment() -> None:
    formatted = _format_reply("好色，我喜欢", [_setu_message()])

    assert formatted["message_tips"][-1] == (
        "对色图[123456_p0]《test title》进行回复或评价"
    )


def test_normal_reply_keeps_generic_tip() -> None:
    formatted = _format_reply("8", [TextMsg(text="ordinary message")])

    assert all("色图[" not in tip for tip in formatted["message_tips"])
