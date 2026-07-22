from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
INLINE_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
HTML_MEDIA_TAG_RE = re.compile(r"<\s*(img|video|audio|source|picture)\b[^>]*>", re.IGNORECASE)
MEDIA_EXT_RE = re.compile(
    r"\.(png|jpe?g|gif|webp|svg|bmp|ico|mp4|m4v|mov|avi|mkv|webm|mp3|wav|ogg|flac|aac|m4a)(\?.*)?$",
    re.IGNORECASE,
)


def _clean_heading_text(raw: str) -> str:
    """清理标题文本，避免把装饰符带入 key。"""
    text = raw.strip()
    text = re.sub(r"\s+#+$", "", text)  # 去掉尾部 ####
    text = re.sub(r"`([^`]+)`", r"\1", text)  # 行内代码
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # 粗体
    text = re.sub(r"\*([^*]+)\*", r"\1", text)  # 斜体
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)  # 链接文本
    text = re.sub(r"\s+", " ", text).strip()
    return text or "UNTITLED"


def _ensure_unique_key(parent_node: dict[str, Any], heading: str) -> str:
    """同级标题重名时自动追加序号。"""
    if heading not in parent_node:
        return heading

    i = 2
    while True:
        candidate = f"{heading} ({i})"
        if candidate not in parent_node:
            return candidate
        i += 1


def _strip_inline_media(line: str) -> str:
    """移除行内图片标记，不影响其他文本。"""
    line = INLINE_IMAGE_RE.sub("", line)
    line = HTML_MEDIA_TAG_RE.sub("", line)
    return line


def _is_media_only_line(line: str) -> bool:
    """判断是否为仅包含多媒体内容的行。"""
    stripped = line.strip()
    if not stripped:
        return False

    # 纯 Markdown 图片
    if INLINE_IMAGE_RE.fullmatch(stripped):
        return True

    # HTML 多媒体标签
    if HTML_MEDIA_TAG_RE.search(stripped):
        non_tag = HTML_MEDIA_TAG_RE.sub("", stripped).strip()
        if not non_tag:
            return True

    # 纯链接且为媒体扩展
    link_match = re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", stripped)
    if link_match:
        url = link_match.group(2).strip()
        return bool(MEDIA_EXT_RE.search(url))

    # 裸 URL 且为媒体扩展
    if stripped.startswith("http://") or stripped.startswith("https://"):
        return bool(MEDIA_EXT_RE.search(stripped))

    return False


def parse_markdown_to_nested_dict(markdown_text: str) -> dict[str, Any]:
    """
    根据 Markdown 标题构建嵌套字典。

    返回结构示例：
    {
      "_preamble": "首个标题前正文",
      "一级标题": {
        "_content": "一级标题下正文",
        "二级标题": {
          "_content": "..."
        }
      }
    }
    """
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(0, root)]
    current_node: dict[str, Any] = root

    in_fence = False
    fence_token = ""

    for raw_line in markdown_text.replace("\r\n", "\n").split("\n"):
        fence_match = FENCE_RE.match(raw_line)
        if fence_match:
            token = fence_match.group(1)
            if not in_fence:
                in_fence = True
                fence_token = token
            elif token == fence_token:
                in_fence = False
                fence_token = ""

            # 代码块边界行也作为正文保留（非媒体）
            target_key = "_content" if current_node is not root else "_preamble"
            current_node[target_key] = (current_node.get(target_key, "") + raw_line + "\n")
            continue

        if not in_fence:
            m = HEADING_RE.match(raw_line)
            if m:
                level = len(m.group(1))
                heading_raw = m.group(2)
                heading = _clean_heading_text(heading_raw)

                # 找到父级
                while stack and stack[-1][0] >= level:
                    stack.pop()

                if not stack:
                    stack = [(0, root)]

                parent = stack[-1][1]
                unique_key = _ensure_unique_key(parent, heading)
                new_node: dict[str, Any] = {}
                parent[unique_key] = new_node
                current_node = new_node
                stack.append((level, new_node))
                continue

        # 正文处理：过滤多媒体行，清理行内多媒体
        if _is_media_only_line(raw_line):
            continue

        clean_line = _strip_inline_media(raw_line)
        target_key = "_content" if current_node is not root else "_preamble"
        current_node[target_key] = current_node.get(target_key, "") + clean_line + "\n"

    # 统一清理每个节点里的末尾空白
    def _trim(node: dict[str, Any]) -> None:
        for k, v in list(node.items()):
            if isinstance(v, dict):
                _trim(v)
            elif isinstance(v, str):
                node[k] = v.strip()

    _trim(root)
    return root


def parse_markdown_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Markdown 文件不存在: {path}")

    text = path.read_text(encoding="utf-8")
    return parse_markdown_to_nested_dict(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="将 Markdown 按标题解析为嵌套字典(JSON)")
    parser.add_argument("input", type=Path, help="输入 Markdown 文件路径")
    parser.add_argument("-o", "--output", type=Path, default=None, help="输出 JSON 文件路径")
    args = parser.parse_args()

    tree = parse_markdown_file(args.input)
    json_text = json.dumps(tree, ensure_ascii=False, indent=2)

    if args.output is None:
        print(json_text)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json_text, encoding="utf-8")
    print(f"已输出: {args.output}")


if __name__ == "__main__":
    main()
