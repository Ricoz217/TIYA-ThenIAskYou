__version__ = "0.1.0"

def prompt(query: str = "") -> str:
    if query:
        output = f"根据下方问题或要求分析图片，**不超过** 1000 字符  \n\n# 问题/要求\n\n{query}"

    else:
        output = "描述图片内容，**不超过** 400 字符。  \n若图片为文字内容，在尽可能保留原文的前提下将内容概括到 400 字符以内。"

    return output