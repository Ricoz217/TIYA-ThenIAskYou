__version__ = "0.1.0"


async def prompt(first: str, second: str) -> str:
    return first + second


async def megumin(boom: int) -> str:
    """
    EXPLOSION!!!
    :param boom: 今天要炸几次
    :return: 爆！！
    """
    return "BOOM " * boom