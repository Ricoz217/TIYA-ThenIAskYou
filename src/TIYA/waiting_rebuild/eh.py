import copy
import datetime
import os
import random
import re
import time
import json
import math
import base64
import traceback
import zstandard as zstd
import py7zr
import asyncio
import requests
from requests import Response
from io import BytesIO
from json import JSONDecodeError
from threading import Thread
from threading import Lock
from bs4 import BeautifulSoup
from PIL import Image, UnidentifiedImageError, ImageSequence
from ncatbot.utils.logger import get_log
from ncatbot.core.message import MessageChain
from ncatbot.core.message import Image as Im
from TIYA.utils import debug, DecoratedDict

_illegal_filter = re.compile(r'[<>:"/\\|?*\x00-\x1F]')
_log = get_log()
_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-encoding': 'gzip, deflate, br, zstd',
    'Accept-language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7,ja;q=0.6',
    "Connection": "keep-alive",
    "Referer": "https://exhentai.org/",
    'Sec-ch-ua': '"Google Chrome";v="137", "Chromium";v="137", "Not/A)Brand";v="24"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-User": "?1",
    'Upgrade-Insecure-Requests': '1',
}
_EH_URL = "https://exhentai.org/g/"
_API_URL = "https://api.e-hentai.org/api.php"
_COOKIES = {}


class Ehentai:
    def __init__(self, host):
        self.host = host
        self.cache = {}
        self.get_cache()
        self.lock = Lock()
        self.task_finished = 0
        self.killer = self.killer = asyncio.create_task(self.alive())

        # 参数
        self.imgsize_limit = 1
        self.EXECUTE = False
        self.BUSY = False
        self.worker_limit = 4
        self.timeout = 300

    async def ehviewer(self, msg_id: str, gallery: str):
        if self.BUSY:
            await self.host.say("已经有人在发色图了喵~晚点再来试试吧！", msg_id)
            return

        if not gallery:
            await self.host.say("忘记写链接了喵？", msg_id)
            return

        purge = False
        only_display = False
        splited = gallery.split(' ')
        if "--purge" in splited:
            purge = True
            splited.remove("--purge")

        if "--d" in splited:
            only_display = True
            splited.remove("--d")

        gallery_link = splited[0].strip()
        if len(splited) > 1:
            gallery_page = ''.join(splited[1:]).strip()

        else:
            gallery_page = None

        self.EXECUTE = False
        self.keep_alive()
        status = {"status_code": 0}
        t_eh = Thread(target=self.run, args=(gallery_link, gallery_page, purge, only_display, status), daemon=True)
        t_eh.start()

        got_info = False
        got_img = False
        got_url = False
        got_7z = False
        got_long = False
        report_download_time = time.time()
        report_upload_time = time.time()
        download_change = 0
        sending_min = 0
        while True:
            if self.EXECUTE:
                return

            if status["status_code"] == -1:
                if self.EXECUTE:
                    return

                if not got_info:
                    if status.get("info") == "out_of_range":
                        await self.host.say(f"选定的范围出错了喵！重新再来一次吧！", msg_id)

                    elif status.get("info") == "network_error":
                        await self.host.say(f"网络连接异常！看看是不是梯子挂了喵...", msg_id)

                    else:
                        await self.host.say(f"找本子出错了喵！确定链接是对的吗？", msg_id)

                elif not got_img:
                    if status.get("info") == "network_error":
                        await self.host.say(f"网络连接异常！看看是不是梯子挂了喵...", msg_id)

                    elif status.get("info") == "technical_error":
                        await self.host.say(f"发生了技术错误！请联系开发者", msg_id)

                    else:
                        await self.host.say(f"图片全部下载失败了...", msg_id)

                self.execute()
                self.BUSY = False
                return

            if status["status_code"] >= 1 and not got_info:
                self.keep_alive()
                page_text = ""
                if gallery_page:
                    page_text = f"的第{gallery_page}页"

                await self.host.say(f"来点色图！正在下载{status['gallery']}{page_text}喵~", msg_id)
                got_info = True

            if status["status_code"] >= 2 and not got_url:
                self.keep_alive()
                report_download_time = time.time()
                await self.host.say(f"总共有[{status['total_page']}]张图喵~")
                got_url = True

            if status["status_code"] >= 2 and not got_img:
                time_now = time.time()
                if time_now - report_download_time > 30:
                    if self.task_finished - download_change > 10:
                        self.keep_alive()

                    download_change = self.task_finished
                    await self.host.say(
                        f"已经下载了[{self.task_finished}]张图，还有[{status['download_page'] - self.task_finished}]张图喵~")
                    report_download_time = time.time()

            if status["status_code"] >= 3 and not got_img:
                self.keep_alive()
                fail_text = f"失败了[{status['fail']}]张..."
                send_text = f"成功下载[{status['done']}]张图片喵~"
                if status['fail']:
                    send_text += fail_text
                await self.host.say(send_text)
                got_img = True

            if status["status_code"] >= 4 and not got_7z:
                self.keep_alive()
                if status["info"] == "complete":
                    await self.host.say(f"已经打包完毕，正在上传！", msg_id)
                    try:
                        await asyncio.wait_for(self.host.bot.api.upload_group_file(
                            group_id=self.host.target,
                            file=status["7z"],
                            name=str(status["gid"]) + ".dll",
                            folder_id='/'
                        ), timeout=180)

                    except Exception as E:
                        _log.error(f"上传压缩包发生错误: {E}")

                elif status["info"] == "selection":
                    pass

                elif status["info"] == "display":
                    pass

                else:
                    await self.host.say(f"因为有图片下载失败所以先不上传文件了喵...可以重新下载试试~", msg_id)

                await self.host.say(f"正在压缩预览图喵~")
                got_7z = True

            if status["status_code"] >= 5 and not got_long:
                got_long = True
                self.keep_alive()
                if status["long_error"]:
                    await self.host.say(f"有[{status["long_error"]}]张图片压缩失败了喵...")

                if self.EXECUTE:
                    return

                msg_set = []
                for img in status["long"]:
                    if self.EXECUTE:
                        return

                    msg = MessageChain([Im(f"base64://{img}")])
                    msg_set.append(msg)

                total_img = len(msg_set)
                sended_img = 0
                self.BUSY = False

                # 尝试用聊天记录去发送
                await self.host.say(f"开始发色图！")
                report_upload_time = time.time()
                got_long = True
                await self.host.say("正在尝试用聊天记录发送喵")
                temp = []
                for index, msg in enumerate(msg_set, start=1):
                    if self.EXECUTE:
                        return

                    if time.time() - report_upload_time > 60:
                        self.keep_alive()
                        await self.host.say(f"等一下！还有[{total_img - sended_img}]张色图喵~")
                        report_upload_time = time.time()

                    temp.append(msg)
                    if index % 4 == 0 or index == len(msg_set):
                        sended = None
                        try:
                            sended = await self.host.bot.send_group_message_chain_forward(
                                self.host.target,
                                temp,
                                self.host.host.bot_name,
                                self.host.host.bot_uid
                            )

                        except Exception as E:
                            _log.error(f"发送发生错误: {E}")

                        if not sended or sended["status"] != "ok":
                            await self.host.say("聊天记录发送失败了...尝试直接发送！")
                            for single_msg in temp:
                                if self.EXECUTE:
                                    return

                                if time.time() - report_upload_time > 60:
                                    self.keep_alive()
                                    await self.host.say(f"等一下！还有[{total_img - sended_img}]张色图喵~")
                                    report_upload_time = time.time()

                                try:
                                    single_sended = await self.host.say(rtf=single_msg)
                                    if single_sended is not None:
                                        sended_img += 1

                                    else:
                                        continue

                                except Exception as E:
                                    _log.error(f"发送发生错误: {E}")
                                    continue

                        else:
                            sended_img += len(temp)
                            await asyncio.sleep(0.5)

                        temp = []

                if total_img - sended_img > 0:
                    await asyncio.sleep(5)
                    await self.host.say(f"有[{total_img - sended_img}]张图片发送失败了...", msg_id)

                else:
                    await asyncio.sleep(5)
                    await self.host.say(f"色图发完啦~", msg_id)

                self.execute()
                return

            await asyncio.sleep(1)

    def execute(self):
        self.EXECUTE = True
        self.killer.cancel()

    @staticmethod
    def select(page: str, filecount: int):
        page = page.replace('，', ',')
        new_selection = []
        choices = page.split(',')
        for choice in choices:
            choice = choice.strip()
            if choice.isdigit():
                if 0 < int(choice) <= filecount:
                    new_selection.append(int(choice))

                else:
                    return []

            elif any(symbol in choice for symbol in ['~', '-']):
                choice = choice.replace('~', '-')
                temp = choice.split('-')
                if len(temp) != 2:
                    continue

                if any(not c.isdigit() for c in temp):
                    continue

                if 0 < int(temp[0]) <= filecount and 0 < int(temp[1]) <= filecount and (temp[0] != temp[1]):
                    lower_num = min([int(i) for i in temp])
                    upper_num = max([int(i) for i in temp]) + 1
                    new_selection += [i for i in range(lower_num, upper_num)]

                else:
                    return []

        # 去重
        new_list = []
        for i in new_selection:
            if i not in new_list:
                new_list.append(i)

        return new_list

    # 读取缓存
    def get_cache(self):
        base_dir = os.path.join(os.getcwd(), "eh")
        cache_file = os.path.join(base_dir, "cache.json")
        if not os.path.exists(cache_file):
            return

        with open(cache_file, 'r', encoding="utf-8") as f:
            load_str = f.read()

        self.cache = json.loads(load_str)
        return

    def check_cache(self, total_file: int, gid: str):
        exist_gallery = []
        base_dir = os.path.join(os.getcwd(), "eh")
        cache_file = os.path.join(base_dir, "cache.json")
        need_update = False
        for gallery_dir in os.listdir(base_dir):
            if os.path.isdir(os.path.join(base_dir, gallery_dir)):
                info_file = os.path.join(base_dir, gallery_dir, "info.txt")
                if not os.path.exists(info_file):
                    continue

                with open(info_file, 'r', encoding="utf-8") as f:
                    load_str = f.read()

                try:
                    load_dict: dict = json.loads(load_str)

                except JSONDecodeError:
                    continue

                exist_gallery.append(str(load_dict["gid"]))
                if not load_dict["failure"] and load_dict["gid"] != gid:
                    self.cache[str(load_dict["gid"])] = load_dict["name"]

                elif not load_dict["failure"] and load_dict["gid"] == gid:
                    if total_file <= load_dict["total_file"]:
                        self.cache[str(load_dict["gid"])] = load_dict["name"]

                    else:
                        need_update = True
                        exist_gallery.remove(gid)

                else:
                    exist_gallery.remove(str(load_dict["gid"]))

        del_gallery = []
        for gid in self.cache.keys():
            if gid not in exist_gallery:
                del_gallery.append(gid)

        for gid in del_gallery:
            del self.cache[gid]

        with open(cache_file, 'w', encoding="utf-8") as f:
            f.write(json.dumps(self.cache, ensure_ascii=False, indent=4))

        return need_update

    # 根据最新的画廊信息更新本地文件信息
    @staticmethod
    def update_storage_info(gallery_name: str, filecount: int):
        gallery_dir = os.path.join(os.getcwd(), "eh", gallery_name)
        info_file = os.path.join(gallery_dir, "info.txt")
        with open(info_file, 'r', encoding="utf-8") as f:
            load_str = f.read()

        load_dict: dict = json.loads(load_str)
        old_files = os.listdir(gallery_dir)
        new_image_count = filecount - load_dict["total_file"]

        # 根据新增的图片数修改编号
        failures = [fail + new_image_count for fail in load_dict["failure"]]
        for file in old_files:
            if file == "info.txt":
                continue

            os.rename(
                os.path.join(gallery_dir, file),
                os.path.join(gallery_dir, f"{(int(file.split('.')[0]) + new_image_count):03d}", file.split('.')[1])
            )

        load_dict["total_file"] = filecount
        load_dict["failure"] = failures
        with open(info_file, 'w', encoding="utf-8") as f:
            f.write(json.dumps(load_dict, ensure_ascii=False, indent=4))

        return load_dict

    # 检查本地文件
    @staticmethod
    def check_exist_file(gallery_name: str, gid: str, filecount: int):
        gallery_dir = os.path.join(os.getcwd(), "eh", gallery_name)
        info_file = os.path.join(gallery_dir, "info.txt")
        exists_files = os.listdir(gallery_dir)
        if "info.txt" in exists_files:
            exists_files.remove("info.txt")

        def save_gallery_info():
            info_dict = {"name": gallery_name, "gid": gid, "total_file": filecount}

            if len(exists_files) == filecount:
                info_dict["failure"] = []

            else:
                info_dict["failure"] = [index + 1 for index in range(filecount) if
                                     index + 1 not in [int(file.split('.')[0]) for file in exists_files]]

            with open(info_file, 'w', encoding="utf-8") as _f:
                _f.write(json.dumps(info_dict, ensure_ascii=False, indent=4))

            return info_dict

        if os.path.exists(info_file):
            with open(info_file, 'r', encoding="utf-8") as f:
                load_str = f.read()

            try:
                gallery_info: dict = json.loads(load_str)

            except JSONDecodeError:
                return save_gallery_info()

            if gallery_name != gallery_info["name"]:
                return save_gallery_info()

            if len(exists_files) != gallery_info["total_file"]:
                gallery_info["failure"] = [index + 1 for index in range(filecount) if
                                         index + 1 not in [int(file.split('.')[0]) for file in exists_files]]

            with open(info_file, 'w', encoding="utf-8") as f:
                f.write(json.dumps(gallery_info, ensure_ascii=False, indent=4))

            return gallery_info

        elif len(exists_files) == filecount:
            return save_gallery_info()

    # 刷新全部已完成下载的缓存
    def update_cache(self, name: str, gid: str):
        base_dir = os.getcwd() + r"\eh"
        cache_file = base_dir + r"\cache.json"
        os.makedirs(base_dir, exist_ok=True)
        self.cache[gid] = name
        save_str = json.dumps(self.cache, ensure_ascii=False, indent=4)
        with open(cache_file, 'w', encoding="utf-8") as f:
            f.write(save_str)

    # 解码zstd
    @staticmethod
    def zstd_decoder(zstd_content: bytes) -> str:
        dctx = zstd.ZstdDecompressor()
        reader = dctx.stream_reader(zstd_content)
        return reader.read().decode()


    # 获取画册信息
    @staticmethod
    def get_gallery_info(gallery: str):
        if any(char in gallery for char in ["http://", "https://"]):
            if "exhentai.org/" in gallery:
                str_slice = ''.join(gallery.split("exhentai.org/")[-1]).split('/')
                g = str_slice[1:]

            elif "e-hentai.org/" in gallery:
                str_slice = ''.join(gallery.split("e-hentai.org/")[-1]).split('/')
                g = str_slice[1:]

            else:
                return "#error",

        else:
            g = gallery.strip().split('/')

        if not g[0].strip().isdigit():
            return "#error",

        gid, gt = int(g[0].strip()), g[1].strip()
        payload = {
            "method": "gdata",
            "gidlist": [
                [gid, gt]
            ],
            "namespace": 1
        }

        status = 0
        retry = 0
        response = None
        while status != 200 and retry <= 3:
            retry += 1
            try:
                response = requests.post(_API_URL, headers=_HEADERS, data=json.dumps(payload), cookies=_COOKIES,
                                         timeout=15)

                status = response.status_code

            except Exception as E:
                _log.error(f"{E}\n{traceback.format_exc()}\n\nEH爬虫发生错误")
                continue

            finally:
                time.sleep(random.uniform(0.5, 1.5))


        if status != 200:
            _log.error("EH爬虫获取本子信息出错，已超过重试次数")
            return "#network",

        try:
            result: dict = json.loads(response.content)

        except JSONDecodeError:
            _log.error(f"{response.content}\n\nEH爬虫解析本子信息发生错误")
            return "#error",

        if result.get("gmetadata"):
            if "error" in result["gmetadata"][0]:
                return "#error",

            title = result["gmetadata"][0]["title"]
            title = _illegal_filter.sub('', title[:min(len(title), (250 - len(os.getcwd())))]).strip()

            title_jpn = result["gmetadata"][0]["title_jpn"]
            title_jpn = _illegal_filter.sub('', title_jpn[:min(len(title), (250 - len(os.getcwd())))]).strip()

            file_count = result["gmetadata"][0]["filecount"]
            return title_jpn if title_jpn else title, f"{gid}/{gt}/", str(gid), file_count

    # 爬取画册图片链接
    def get_gallery_site(self, gallery: str, filecount: str, selection: list, status: dict):
        base_url = _EH_URL + gallery

        # 获取一页的图片数量
        response = None
        retry = 0
        response_status = 0
        while response_status != 200 and retry <= 3:
            retry += 1
            try:
                response = requests.get(base_url, headers=_HEADERS, cookies=_COOKIES, timeout=15)
                response_status = response.status_code

            except Exception as E:
                _log.error(f"EH爬虫获取画廊图片失败: {E}")
                continue

            finally:
                time.sleep(random.uniform(0.5, 1.5))

        page_capacity = 20
        if response_status == 200:
            HTML_TEXT = response.text
            soup = BeautifulSoup(HTML_TEXT, 'lxml')
            div = soup.find('div', id='gdt')
            if div:
                class_name = div.get('class')
                if class_name:
                    if class_name[0] == "gt100":
                        page_capacity = 40

                    elif class_name[0] == "gt200":
                        page_capacity = 20

                    elif class_name[0] == "gt400":
                        page_capacity = 10

                    else:
                        temp = class_name[0][-3:]
                        if temp.isdigit():
                            page_capacity = 4000 / int(temp)

                        if not page_capacity is int:
                            page_capacity = 20

        else:
            _log.error("EH爬虫获取画廊图片失败")

        url_list = {}
        img_url_dict = {}
        if selection:
            for index in selection:
                page = (index - 1) // page_capacity
                if page == 0:
                    url = base_url

                else:
                    url = base_url + f"?p={page}"

                if url in url_list:
                    url_list[url].append(index)

                else:
                    url_list[url] = [index]

            for url, index_list in url_list.items():
                response = None
                retry = 0
                response_status = 0
                while response_status != 200 and retry <= 3:
                    retry += 1
                    try:
                        response = requests.get(url, headers=_HEADERS, cookies=_COOKIES, timeout=15)
                        response_status = response.status_code

                    except Exception as E:
                        _log.error(f"EH爬虫网络错误: {E}")
                        continue

                    finally:
                        time.sleep(random.uniform(0.5, 1.5))

                if response_status != 200:
                    _log.error("EH爬虫获取图片token出错，已超过重试次数")
                    status["info"] = "network_error"
                    return {}

                if response.headers.get("Content-Encoding") == "zstd":
                    HTML_TEXT = self.zstd_decoder(response.content)

                else:
                    HTML_TEXT = response.text

                soup = BeautifulSoup(HTML_TEXT, 'lxml')
                div = soup.find('div', id='gdt')
                if div:
                    image_list = div.find_all('a', href=True)
                    for index in index_list:
                        img_url_dict[index] = image_list[(index % page_capacity - 1)].get("href")

                else:
                    _log.error("EH爬虫爬取图片token出错")
                    debug(HTML_TEXT)
                    status["info"] = "technical_error"
                    return {}

        else:
            url_list = []
            page = (int(filecount) - 1) // page_capacity + 1
            for i in range(page):
                if i == 0:
                    url_list.append(base_url)

                else:
                    url_list.append(base_url + f"?p={i}")

            index = 1
            for url in url_list:
                response = None
                retry = 0
                response_status = 0
                while response_status != 200 and retry <= 3:
                    retry += 1
                    try:
                        response = requests.get(url, headers=_HEADERS, cookies=_COOKIES, timeout=15)
                        response_status = response.status_code

                    except Exception as E:
                        _log.error(f"EH爬虫网络错误: {E}")
                        continue

                    finally:
                        time.sleep(random.uniform(0.5, 1.5))

                if response_status != 200:
                    _log.error("EH爬虫获取图片token出错，已超过重试次数")
                    status["info"] = "network_error"
                    return {}

                if response.headers.get("Content-Encoding") == "zstd":
                    HTML_TEXT = self.zstd_decoder(response.content)

                else:
                    HTML_TEXT = response.text
                soup = BeautifulSoup(HTML_TEXT, 'lxml')
                div = soup.find('div', id='gdt')

                if div:
                    for a in div.find_all('a', href=True):
                        img_url_dict[index] = a.get("href")
                        index += 1

                else:
                    _log.error("EH爬虫爬取图片token出错")
                    debug(HTML_TEXT)
                    status["info"] = "technical_error"
                    return {}

        status["total_page"] = len(img_url_dict)
        return img_url_dict

    # 多线程下载图片
    def task_pool(self, gallery_name: str, url_dict: dict, gid: str, filecount: int, status: dict, force=False):
        """
        多线程下载图片
        :param gallery_name: gallery's title
        :param url_dict: {index: url}
        :param gid: gallery's id
        :param filecount: total files from api report
        :param status: information container through Threads
        :param force: force download ignore cache
        :return: [done, fail]
        """
        print(f"正在下载[{gallery_name}]")
        self.task_finished = 0
        save_dir = os.path.join(os.getcwd(), "eh", gallery_name)
        os.makedirs(save_dir, exist_ok=True)
        gallery_info = self.check_exist_file(gallery_name, gid, filecount)

        if gallery_info and not force:
            # 检查本子是否有更新
            need_update = self.check_cache(filecount, gid)
            if need_update:
                gallery_info = self.update_storage_info(gallery_name, filecount)

            # 检查已存在文件
            if gallery_info["failure"]:
                download_dict = {index: url for index, url in url_dict.items() if index in gallery_info["failure"]}

            else:
                download_dict = url_dict

        else:
            download_dict = url_dict

        status["download_page"] = len(download_dict)

        # 检查是否有缓存
        if not force and gid in self.cache:
            return len(url_dict), 0

        if not download_dict:
            return len(url_dict), 0

        # debug
        # print(gallery_info)
        # print(download_dict)

        print(f"总共[{len(download_dict)}]张图片")

        # 重试列表
        fail_list = []

        # 任务列表
        task_list = []

        for index, url in download_dict.items():
            # [url: str, index: int, retry: int]
            new_list = [url, index, 0]
            task_list.append(new_list)

        # 线程池
        worker_list = []
        for i in range(self.worker_limit):
            worker_list.append(Thread(target=self.img_downloader, args=(task_list, fail_list, save_dir), daemon=True))

        for w in worker_list:
            w.start()
            if self.EXECUTE:
                return 0, 0

        for w in worker_list:
            w.join()
            if self.EXECUTE:
                return 0, 0

        # 写入信息文件
        save_file = os.path.join(save_dir, "info.txt")
        save_dict = {"name": gallery_name, "gid": gid, "total_file": filecount}
        if len(url_dict) == filecount:
            failures = [index[1] for index in fail_list]

        else:
            selection_done_index = [index for index in url_dict.keys() if index not in [i[1] for i in fail_list]]
            if gallery_info:
                failures = [index for index in gallery_info["failure"] if index not in selection_done_index]

            else:
                failures = [index + 1 for index in range(filecount) if index + 1 not in selection_done_index]

        if not failures:
            self.update_cache(gallery_name, str(gid))

        save_dict["failure"] = failures
        with open(save_file, 'w', encoding="utf-8") as f:
            f.write(json.dumps(save_dict, ensure_ascii=False, indent=4))

        fail_text = f"，有{len(fail_list)}张失败了喵..."
        _log.info(f"eh成功下载{self.task_finished}张图片{fail_text if fail_list else ''}")
        return self.task_finished, len(fail_list)

    # 图片爬虫
    def img_downloader(self, task_list: list, fail_list: list, save_dir: str):
        def download_img_sub(_url: str) -> tuple[str, Response | None]:
            try:
                response = requests.get(_url, headers=_HEADERS, cookies=_COOKIES, timeout=15)

            except Exception as E:
                _log.error(f"EH下载错误：{E}")
                return "#error", None

            if response.status_code != 200:
                _log.error("EH下载请求被拒绝")
                return "#error", None

            if response.headers.get("Content-Encoding") == "zstd":
                HTML_TEXT = self.zstd_decoder(response.content)

            else:
                HTML_TEXT = response.text

            soup = BeautifulSoup(HTML_TEXT, 'lxml')
            img = soup.find('img', id='img')
            if not img:
                _log.error("EH下载链接错误")
                return "#error", None

            download_url = img.get("src")
            try:
                with requests.get(download_url, headers=_HEADERS, cookies=_COOKIES, timeout=30,
                                  stream=True) as downloaded_img:
                    if downloaded_img.status_code != 200:
                        _log.error("EH下载请求被拒绝")
                        return "#error", None

                    expect_size = downloaded_img.headers.get('Content-Length', 0)
                    chunks = []
                    total_received_size = 0
                    for chunk in downloaded_img.iter_content(chunk_size=8192):
                        if chunk:
                            chunks.append(chunk)
                            total_received_size += len(chunk)

                    downloaded_content = b''.join(chunks)
                    fake_response = DecoratedDict(
                        {
                            "status_code": downloaded_img.status_code,
                            "headers": copy.deepcopy(downloaded_img.headers),
                            "content": downloaded_content
                        }
                    )

                    if int(expect_size) > 0:
                        if int(expect_size) != total_received_size:
                            self.ex_debug(fake_response)
                            _log.error(f"eh爬虫下载图片不完整。期望 {expect_size} 字节, 实际收到 {total_received_size} 字节")
                            return "#error", None

                    temp_buffer = BytesIO(downloaded_content)
                    try:
                        with Image.open(temp_buffer):
                            pass

                    except UnidentifiedImageError:
                        self.ex_debug(fake_response)
                        _log.error(f"eh爬虫下载图片不完整。")
                        return "#error", None

                    except Exception as E:
                        _log.error(f"eh图片打开发生未知错误:\n{E}")
                        self.ex_debug(fake_response, E)
                        return "#error", None

                    # 创建模拟的Response对象返回
                    return "#pass", fake_response

            except Exception as E:
                _log.error(f"EH下载错误：{E}\n{traceback.format_exc()}")
                return "#error", None

            # try:
            #     downloaded_img = requests.get(download_url, headers=_HEADERS, cookies=_COOKIES, timeout=30)
            #
            # except Exception as E:
            #     _log.error(f"EH下载错误：{E}")
            #     return "#error", None
            #
            # if downloaded_img.status_code != 200:
            #     _log.error("EH下载请求被拒绝")
            #     return "#error", None
            #
            # temp_buffer = BytesIO(downloaded_img.content)
            # expect_size = downloaded_img.headers.get('Content-Length', 0)
            # if int(expect_size) > 0:
            #     if temp_buffer.getbuffer().nbytes != int(expect_size):
            #         self.ex_debug(downloaded_img)
            #         _log.error(f"eh爬虫下载图片不完整。")
            #         return "#error", None
            #
            # try:
            #     with Image.open(temp_buffer):
            #         pass
            #
            # except UnidentifiedImageError:
            #     self.ex_debug(downloaded_img)
            #     _log.error(f"eh爬虫下载图片不完整。")
            #     return "#error", None
            #
            # except Exception as E:
            #     _log.error(f"eh图片打开发生未知错误:\n{E}")
            #     self.ex_debug(downloaded_img, E)
            #     return "#error", None
            #
            # return "#pass", downloaded_img

        while task_list:
            time.sleep(random.uniform(2, 5))
            with self.lock:
                if not task_list:
                    break

                task = task_list.pop(0)

            url = task[0]
            index = task[1]
            if self.EXECUTE:
                return

            print(f"正在下载第{index}张图片...")
            result = download_img_sub(url)
            if result[0] != "#pass":
                with self.lock:
                    fail_list.append(task)

                _log.error(f"eh下载第{index}张出错")
                continue

            file_type = result[1].headers.get('content-type', '').split('/')[-1]
            file_name = f"{index:03d}.{file_type if file_type else "jpg"}"
            save_file = os.path.join(save_dir, file_name)
            with open(save_file, 'wb') as f:
                f.write(result[1].content)

            with self.lock:
                self.task_finished += 1

        while fail_list:
            time.sleep(random.uniform(2, 5))
            with self.lock:
                if not fail_list:
                    break

                task = fail_list.pop(0)

            task[2] += 1
            # 重试5次以后，结束
            if task[2] > 5:
                fail_list.append(task)
                return

            url = task[0]
            index = task[1]

            if task[2] == 4:
                url += r"?nl=1"

            if self.EXECUTE:
                return

            print(f"正在下载第{index}张图片...")
            result = download_img_sub(url)
            if result[0] != "#pass":
                with self.lock:
                    fail_list.append(task)

                _log.error(f"eh下载第{index}张出错")
                continue

            file_type = result[1].headers.get('content-type', '').split('/')[-1]
            file_name = f"{index:03d}.{file_type if file_type else "jpg"}"
            save_file = os.path.join(save_dir, file_name)
            with open(save_file, 'wb') as f:
                f.write(result[1].content)

            with self.lock:
                self.task_finished += 1

    @staticmethod
    def fold27zip(gallery_name: str):
        save_dir = os.getcwd() + r"\eh"
        target_dir = save_dir + f"\\{gallery_name}"
        save_file = target_dir + ".7z"
        with py7zr.SevenZipFile(save_file, 'w', password="114514", header_encryption=True, ) as archive:
            archive.writeall(target_dir, arcname='')

        return save_file

    @staticmethod
    def compress_gif(gif_buffer: BytesIO, max_size: float = 6):
        def save_gif(_frames: list, buffer: BytesIO, optimize=False):
            _frames[0].save(
                buffer,
                format="GIF",
                append_images=_frames[1:],
                save_all=True,
                duration=_durations,
                loop=0,
                optimize=optimize
            )

        max_size = max_size * 1024 * 1024
        sample_buffer = BytesIO()
        frames = []
        _durations = []
        with Image.open(gif_buffer) as GIF:
            for frame in ImageSequence.Iterator(GIF):
                frames.append(frame.copy())
                _durations.append(frame.info.get("duration", 100))

        frames_count = len(frames)
        save_gif(frames, sample_buffer)
        estimated_size = sample_buffer.getbuffer().nbytes
        # print(estimated_size)
        # print(max_size)
        if estimated_size <= max_size:
            return gif_buffer

        compressed_buffer = BytesIO()

        # 先缩放分辨率
        width_now, height_now = frames[0].size
        if width_now > height_now:
            min_height = math.floor(max(600.0, 0.5 * height_now))
            min_width = math.floor(width_now * min_height / height_now)

        else:
            min_width = math.floor(max(600.0, 0.5 * width_now))
            min_height = math.floor(height_now * min_width / width_now)

        if not min_width < width_now or not min_height < height_now:
            min_width, min_height = width_now, height_now

        scale_rate = (min_width * min_height) / (width_now * height_now)
        # print(f"正在重采样，共有{len(frames)}帧")
        if estimated_size * scale_rate <= max_size:
            scale_rate = max_size / estimated_size - 0.05
            new_frames = []
            new_width = math.floor(width_now * scale_rate)
            new_height = math.floor(height_now * scale_rate)
            for frame in frames:
                new_frame = frame.resize((new_width, new_height))
                new_frames.append(new_frame)
                frame.close()

            save_gif(new_frames, compressed_buffer)
            while compressed_buffer.getbuffer().nbytes > max_size:
                compressed_buffer.seek(0)
                compressed_buffer.truncate(0)
                final_frames = []
                width_now, height_now = new_frames[0].size
                new_width = math.floor(width_now * 0.95)
                new_height = math.floor(height_now * 0.95)
                for frame in new_frames:
                    new_frame = frame.resize((new_width, new_height))
                    final_frames.append(new_frame)
                    frame.close()

                new_frames = final_frames
                save_gif(new_frames, compressed_buffer, optimize=True)

            return compressed_buffer

        # 再丢弃帧
        estimated_size = estimated_size * scale_rate
        min_frames_count = math.floor(max(60.0, 0.5 * frames_count))
        if min_frames_count > frames_count:
            min_frames_count = frames_count

        skip_rate = min_frames_count / frames_count
        # print(f"正在丢帧，共有{len(frames)}帧")
        if estimated_size * skip_rate <= max_size:
            skip = max(1, int(frames_count / min_frames_count)) - 1
            skip_frames = frames[::skip + 1]
            _durations = _durations[::skip + 1]
            compensate_delay = frames_count / len(skip_frames)
            _durations = [i * compensate_delay for i in _durations]

            new_frames = []
            for frame in skip_frames:
                new_frame = frame.resize((min_width, min_height))
                new_frames.append(new_frame)
                frame.close()

            # print(f"还剩{len(new_frames)}帧")
            save_gif(new_frames, compressed_buffer)
            while compressed_buffer.getbuffer().nbytes > max_size:
                final_frames = []
                compressed_buffer.seek(0)
                compressed_buffer.truncate(0)
                width_now, height_now = new_frames[0].size
                new_width = math.floor(width_now * 0.95)
                new_height = math.floor(height_now * 0.95)
                for frame in new_frames:
                    new_frame = frame.resize((new_width, new_height))
                    final_frames.append(new_frame)
                    frame.close()

                new_frames = final_frames
                # print(f"还剩{len(final_frames)}帧")
                save_gif(new_frames, compressed_buffer, optimize=True)

            return compressed_buffer

        # 最后压缩颜色
        # print(f"正在压缩颜色，共有{len(frames)}帧")
        colors_now = 256
        min_colors = 128
        estimated_size = estimated_size * skip_rate
        skip = max(1, int(frames_count / min_frames_count)) - 1
        frames = frames[::skip + 1]
        _durations = _durations[::skip + 1]
        compensate_delay = frames_count / len(frames)
        _durations = [i * compensate_delay for i in _durations]
        target_colors = max(
            min_colors,
            int(colors_now * (max_size / estimated_size)))

        final_frames = []
        for frame in frames:
            frame = frame.resize((min_width, min_height))
            frame = frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=target_colors)
            final_frames.append(frame)

        # print(f"还剩{len(final_frames)}帧")
        save_gif(final_frames, compressed_buffer, optimize=True)
        if compressed_buffer.getbuffer().nbytes > max_size:
            return gif_buffer

        return compressed_buffer

    # 暂不使用
    # def fold2pdf(self, gallery_name: str):
    #     save_dir = os.getcwd() + r"\eh"
    #     target_dir = save_dir + f"\\{gallery_name}"
    #     save_file = target_dir + ".pdf"
    #
    #     file_list = []
    #     for filename in os.listdir(target_dir):
    #         if filename == "info.txt":
    #             continue
    #
    #         file_list.append(os.path.join(target_dir, filename))
    #
    #     img_list = [Image.open()]

    # 后处理
    def resize(self, img_file: str):
        max_size = 1024 * 1024 * self.imgsize_limit
        try:
            with Image.open(img_file) as img:
                buffer = BytesIO()

                # 判断是否是有多帧的动画
                if getattr(img, 'n_frames', 1) > 1:
                    with open(img_file, 'rb') as f:
                        data = f.read()

                    gif_buffer = BytesIO(data)
                    compress_buffer = self.compress_gif(gif_buffer)

                    base64_str = base64.b64encode(compress_buffer.getvalue()).decode()
                    return [base64_str, "#animation"]

                # 转换成JPG时要处理透明度（像摘下猫咪的隐身斗篷）
                if img.mode in ('RGBA', 'LA'):
                    img = img.convert('RGB')

                new_img = img.copy()

                new_img.save(buffer, format='JPEG', quality=90)

        except Exception as E:
            _log.error(f"爬虫转换图片出错: {E}")
            return ["", "#error"]

        if buffer.tell() < max_size:
            return [base64.b64encode(buffer.getvalue()).decode(), "#img"]

        new_quality = 80
        while new_quality > 20:
            save_args = {'format': 'JPEG', 'quality': new_quality, 'optimize': True}
            buffer.truncate(0)
            buffer.seek(0)
            new_img.save(buffer, **save_args)
            if buffer.tell() < max_size:
                return [base64.b64encode(buffer.getvalue()).decode(), "#img"]

            new_quality -= 5

        orig_width, orig_height = new_img.size
        for attempt in range(10):
            new_width = math.floor(orig_width * 0.9)
            new_height = math.floor(orig_height * 0.9)
            resized_img = new_img.resize((new_width, new_height))
            buffer.truncate(0)
            buffer.seek(0)
            resized_img.save(buffer, format='JPEG', quality=20)  # 开启最低质量喵
            if buffer.tell() <= max_size:
                return [base64.b64encode(buffer.getvalue()).decode(), "#img"]

        # 图片过大，缩放超时
        return [base64.b64encode(buffer.getvalue()).decode(), "#img"]

    @staticmethod
    def img2long(base64_list: list):
        formated_img = []
        for base64_str in base64_list:
            data = base64.b64decode(base64_str)
            buffer = BytesIO(data)
            img = Image.open(buffer)
            w_percent = 1200 / float(img.size[0])
            h_size = int(float(img.size[1]) * float(w_percent))
            f_img = img.resize((1200, h_size))
            formated_img.append(f_img)

        # (✪ω✪) 计算总高度（悄悄把小鱼叠高高）
        total_height = sum(img.height for img in formated_img)

        # (=ↀωↀ=) 准备好画布开始作画啦~
        composite = Image.new('RGB', (1200, total_height))
        y_offset = 0

        for idx, img in enumerate(formated_img):
            composite.paste(img, (0, y_offset))
            y_offset += img.height

        buffer = BytesIO()
        composite.save(buffer, format='JPEG', quality=100)
        new_quality = 80
        while buffer.tell() > 1024 * 1024 * 12 and new_quality >= 20:
            buffer.truncate(0)
            buffer.seek(0)
            composite.save(buffer, format='JPEG', quality=new_quality)
            new_quality -= 5

        return base64.b64encode(buffer.getvalue()).decode()

    def run(self, gallery: str, page: str, force: bool, display: bool, status: dict):
        # [title, id/token/, id, filecount]
        gallery_info = self.get_gallery_info(gallery)
        if gallery_info[0] == "#error":
            status["status_code"] = -1
            return

        elif gallery_info[0] == "#network":
            status["status_code"] = -1
            status["info"] = "network_error"
            return

        status["gallery"] = gallery_info[0]
        status["gid"] = gallery_info[2]
        status["status_code"] = 1
        if page:
            selection = self.select(page, int(gallery_info[3]))
            if not selection:
                status["status_code"] = -1
                status["info"] = "out_of_range"
                return

        else:
            selection = []

        url_dict = self.get_gallery_site(gallery_info[1], gallery_info[3], selection, status)
        if not url_dict:
            status["status_code"] = -1
            return

        status["status_code"] = 2
        done, fail = self.task_pool(gallery_info[0], url_dict, gallery_info[2], int(gallery_info[3]), status, force)
        if done or status["total_page"] != status["download_page"]:
            status["done"] = done
            status["fail"] = fail

        else:
            status["status_code"] = -1
            return

        if self.EXECUTE:
            return

        status["status_code"] = 3
        if fail:
            status["info"] = "incomplete"

        elif selection:
            status["info"] = "selection"

        elif display:
            status["info"] = "display"

        else:
            _7z_file = self.fold27zip(gallery_info[0])
            status["7z"] = _7z_file
            status["info"] = "complete"

        if self.EXECUTE:
            return

        status["status_code"] = 4
        img_list = []
        for filename in os.listdir(os.path.join(os.getcwd(), "eh", gallery_info[0])):
            if filename == "info.txt":
                continue

            if selection:
                if not int(filename.split('.')[0]) in selection:
                    continue

            img_list.append(filename)

        img_list_sorted = sorted(img_list)
        base64_list = [[]]
        animation_list = []
        fail_count = 0
        for img in img_list_sorted:
            if self.EXECUTE:
                return

            img_file = os.path.join(os.getcwd(), "eh", gallery_info[0], img)
            base64_str = self.resize(img_file)
            if base64_str[1] == "#error":
                _log.error("eh重采样图像出错")
                fail_count += 1
                continue

            if base64_str[1] == "#animation":
                animation_list.append(base64_str[0])
                continue

            if len(base64_list[-1]) > 5:
                base64_list.append([])

            base64_list[-1].append(base64_str[0])

        upload_list = []
        for img_set in base64_list:
            if self.EXECUTE:
                return

            if not img_set:
                continue

            base64_str = self.img2long(img_set)
            upload_list.append(base64_str)

        status["long"] = upload_list + animation_list
        status["long_error"] = fail_count
        status["status_code"] = 5

    async def alive(self):
        await asyncio.sleep(self.timeout)
        self.execute()
        await self.host.say("色图已超时！")

    def keep_alive(self):
        self.killer.cancel()
        self.killer = asyncio.create_task(self.alive())

    @staticmethod
    def ex_debug(task_response: Response | DecoratedDict, message = None):
        error_path = os.path.join(os.getcwd(), "logs", "errorfiles")
        os.makedirs(error_path, exist_ok=True)

        error_logs = os.path.join(error_path, "errors.txt")
        dt = datetime.datetime.now()
        output_message = f"[{dt:%Y-%m-%d %H:%M:%S}]\n"
        output_message += f"Response: \nheader: {task_response.headers}\n"
        output_message += f"Message: {message}\n"
        output_message += f"{'=' * 80}\n"

        with open(error_logs, 'a', encoding="utf-8") as f:
            f.write(output_message)

        with open(os.path.join(error_path, f"{dt:%Y-%m-%d %H-%M-%S}_error_file"), 'wb') as f:
            try:
                f.write(task_response.content)

            except Exception:
                return


if __name__ == "__main__":
    async def main():
        eh = Ehentai(None)
        # print(eh.get_gallery_info("https://exhentai.org/g/2781293/53741b4d56"))
        # test_urls = eh.get_gallery_site("2781293/53741b4d56", "202", [], {})
        # print(len(test_urls))
        # print([test_urls[idx] for idx in range(32, 57)])

    # asyncio.run(main())

    response = requests.get('https://exhentai.org/g/1585661/cff388ba07', headers=_HEADERS, cookies=_COOKIES, timeout=15)
    print(response.headers)
    dctxa = zstd.ZstdDecompressor()
    readera = dctxa.stream_reader(response.content)
    print(readera.read().decode())
    print(response.text)
    # print(Ehentai.zstd_decoder(response.content))

    # downloaded_img = requests.get("https://uxtmwym.gavcvusbvdie.hath.network/h/9eb510d62f548613538c8f38bdd31ee6418bad96-647885-1269-1800-jpg/keystamp=1751168700-2e3469686f;fileindex=140382584;xres=org/a_027.jpg", headers=_HEADERS, cookies=_COOKIES, timeout=90)
    # print(downloaded_img.status_code)
    # print(downloaded_img.headers)
