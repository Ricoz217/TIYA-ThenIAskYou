import asyncio
import traceback
from PicImageSearch import Network, Ascii2D, SauceNAO, AnimeTrace, Bing, BaiDu
from ncatbot.utils.logger import get_log


_log = get_log()
_ASCII2D_URL = "https://ascii2d.obfs.dev/search/file"
_SAUCE_API_KEYS = {
    "1": {"short": 4, "long": 100},
    "2": {"short": 4, "long": 100},
    "3": {"short": 4, "long": 100}
}
_SAUCE_API_KEY = ""
_PROXY = "http://127.0.0.1:7890"

A2D_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7,ja;q=0.6",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Origin": "https://ascii2d.net",
    "Content-Type": "multipart/form-data; boundary=----WebKitFormBoundaryABxqCt56BZBjmDI7",  # 必须与实际 boundary 一致
    "Priority": "u=0, i",  # 真实浏览器默认携带
    "Referer": "https://ascii2d.net/",  # 关键：模拟从官网跳转
    "DNT": "1",
    "Connection": "keep-alive",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-User": "?1",
    "Sec-Ch-Ua": '"Chromium";v="136", "Google Chrome";v="136", "Not.A/Brand";v="99"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"'
}

# "https://ascii2d.obfs.dev"
A2D_COLOR = Ascii2D("https://ascii2d.net", verify_ssl=True, headers=A2D_HEADERS, proxies=_PROXY if _PROXY else None)
A2D_FEATURE = Ascii2D("https://ascii2d.net", verify_ssl=True, headers=A2D_HEADERS, bovw=True, proxies=_PROXY if _PROXY else None)
SAUCE = SauceNAO(api_key=_SAUCE_API_KEY, proxies=_PROXY if _PROXY else None)


def switch_sauce_api(used_api: list):
    global _SAUCE_API_KEY
    index_now = list(_SAUCE_API_KEYS.keys()).index(_SAUCE_API_KEY)
    if index_now == len(_SAUCE_API_KEYS) - 1:
        index_new = 0

    else:
        index_new = index_now + 1

    if index_new == index_now:
        return []

    while index_new in used_api:
        index_new += 1
        if index_new == index_now:
            return []

    used_api.append(index_new)
    _SAUCE_API_KEY = list(_SAUCE_API_KEYS.keys())[index_new]
    return used_api


async def a2d_search(img_file: str):
    async with Network(proxies=_PROXY if _PROXY else None) as client:
        a2d_color = Ascii2D("https://ascii2d.obfs.dev", verify_ssl=True, client=client)
        a2d_feature = Ascii2D("https://ascii2d.obfs.dev", verify_ssl=True, bovw=True, client=client)

        try:
            color_resp = await a2d_color.search(file=img_file)
            feature_resp = await a2d_feature.search(file=img_file)

        except asyncio.CancelledError:
            _log.info("搜图已超时")
            return [[], []]

        except Exception as E:
            _log.error(f"搜图出错：{E}\n完整信息：{traceback.format_exc()}")
            return [[], []]

        result = []
        if color_resp.raw:
            data = [i for i in color_resp.raw if i.title or i.url_list][0]
            result.append({
                "thumbnail": data.thumbnail,
                "title": data.title,
                "author": data.author,
                "url": data.url,
                "author_url": data.author_url
            })

        else:
            result.append([])

        if feature_resp.raw:
            data = [i for i in feature_resp.raw if i.title or i.url_list][0]
            result.append({
                "thumbnail": data.thumbnail,
                "title": data.title,
                "author": data.author,
                "url": data.url,
                "author_url": data.author_url
            })

        else:
            result.append([])

        return result


async def sauce_search(img_file: str, used_api: list):
    async with Network(proxies=_PROXY if _PROXY else None) as client:
        sauce = SauceNAO(api_key=_SAUCE_API_KEY, client=client)

        try:
            resp = await sauce.search(file=img_file)

        except asyncio.CancelledError:
            _log.error("SauceNao搜图已超时")
            return {"status": "#error"}

        except Exception as E:
            _log.error(f"搜图出错：{E}\n完整信息：{traceback.format_exc()}")
            return {"status": "#error"}

        if resp.status_code != 200:
            if resp.status_code == 403:
                used_api = switch_sauce_api(used_api)
                if not used_api:
                    return {
                        "status": "#bad_request",
                        "code": resp.status_code,
                        "message": resp.origin["header"]["message"]
                    }

                result = await sauce_search(img_file, used_api)
                return result

            else:
                return {
                    "status": "#bad_request",
                    "code": resp.status_code,
                    "message": resp.origin["header"]["message"]
                }

        _SAUCE_API_KEYS[_SAUCE_API_KEY]["short"] = int(resp.short_remaining)
        _SAUCE_API_KEYS[_SAUCE_API_KEY]["long"] = int(resp.long_remaining)

        if not int(resp.short_remaining) or not int(resp.short_remaining):
            switch_sauce_api([])

        results = [info for info in resp.raw[:3] if float(info.similarity) >= 60]
        if not results:
            return {
                "status": "#low_relative"
            }

        result = {
            "status": "#sucess",
            "data": []
        }

        for item in results:
            result["data"].append({
                "thumbnail": item.thumbnail,
                "similarity": float(item.similarity),
                "title": item.title,
                "author": item.author,
                "url": item.url,
                "author_url": item.author_url
            })

        return result


async def ac_search(img_file: str):
    async with Network(proxies=_PROXY) as client:
        ac = AnimeTrace(client=client)

        try:
            resp = await ac.search(file=img_file)

        except asyncio.CancelledError:
            _log.info("搜图已超时")
            return {}

        except Exception as E:
            _log.error(f"搜图出错：{E}\n完整信息：{traceback.format_exc()}")
            return {}

    if resp.code:
        _log.error(f"搜图异常：\n{resp.origin}")
        return {}

    result = {"ai": resp.ai, "data": []}
    if not resp.raw:
        return result

    target = [item.characters for item in resp.raw if item.characters][0]
    for index, char in enumerate(target):
        result["data"].append({
            "character": char.name,
            "work": char.work
        })
        if index >= 2:
            break

    return result


async def baidu_search(img_file: str):
    async with Network(proxies=None) as client:
        baidu = BaiDu(client=client)

        try:
            resp = await baidu.search(file=img_file)

        except asyncio.CancelledError:
            _log.info("搜图已超时")
            return []

        except Exception as E:
            _log.error(f"搜图出错：{E}\n完整信息：{traceback.format_exc()}")
            return []

    if resp.origin.get("status", None):
        _log.error(f"搜图异常：\n{resp.origin}")
        return []

    results = []
    for index, item in enumerate(resp.exact_matches, start=1):
        result = {
            "url": item.url,
            "thumbnail": item.thumbnail,
            "title": item.title
        }
        results.append(result)
        if index >= 3:
            break

    return results


async def bing_search(img_file: str):
    async with Network(proxies=_PROXY) as client:
        bing = Bing(client=client)

        try:
            resp = await bing.search(file=img_file)

        except asyncio.CancelledError:
            _log.info("搜图已超时")
            return []

        except Exception as E:
            _log.error(f"搜图出错：{E}\n完整信息：{traceback.format_exc()}")
            return []

    results = []
    for index, item in enumerate(resp.pages_including, start=1):
        result = {
            "url": item.url,
            "thumbnail": item.thumbnail,
            "title": item.name
        }
        results.append(result)
        if index >= 3:
            break

    return results


if __name__ == "__main__":
    async def main():
        # file = r"D:\OneDrive\Pictures\素材\102439408_p0.jpg"
        # sauce_result = await sauce_search(file, [])
        img_url = "https://public-share.obs.cn-south-1.myhuaweicloud.com/motd.png"
        a2d_result = await a2d_search(img_url)

        # print("sauce:", sauce_result)
        # print('=' * 100)
        print("a2d:", a2d_result)


    asyncio.run(main())