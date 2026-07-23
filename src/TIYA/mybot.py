import asyncio
import traceback
import websockets
from websockets.exceptions import ConnectionClosed
import datetime
import json as j
from threading import Event, Thread
from TIYA.config import BASE_CFG, config_update
from TIYA.logger import get_logger

from ncatbot.core.message import MessageChain, Reply
from ncatbot.core.client import BotClient
from ncatbot.conn.http import WsRoute
from ncatbot.conn.gateway import Websocket


_log = get_logger()


class CustomWsRoute(WsRoute):
    async def post(self, path, params=None, json=None, dumped = None):
        async with websockets.connect(
            self.url,
            extra_headers=self.headers,
            ping_interval=300,   # 显式设置心跳
            ping_timeout=330,
        ) as websocket:
            if params:
                await websocket.send(
                    j.dumps(
                        {
                            "action": path.replace("/", ""),
                            "params": params,
                            "echo": int(datetime.datetime.now().timestamp()),
                        }
                    )
                )

            elif json:
                await websocket.send(
                    j.dumps(
                        {
                            "action": path.replace("/", ""),
                            "params": json,
                            "echo": int(datetime.datetime.now().timestamp()),
                        }
                    )
                )

            elif dumped:
                print("正在发送")
                await websocket.send(dumped)

            else:
                await websocket.send(
                    j.dumps(
                        {
                            "action": path.replace("/", ""),
                            "params": {},
                            "echo": int(datetime.datetime.now().timestamp()),
                        }
                    )
                )

            response = await websocket.recv()
            return j.loads(response)


class CustomWebsocket(Websocket):
    def __init__(self, client, config=None):
        super().__init__(client, config)

    async def ws_connect(self):
        while True:
            try:
                await super().ws_connect()

            except asyncio.CancelledError:
                raise

            except (
                ConnectionClosed,
                OSError,
                asyncio.TimeoutError,
            ) as error:
                _log.warning(f"WebSocket 已断开，3 秒后重新连接: {error}")
                await asyncio.sleep(3)

    async def receive(self, message):
        msg = j.loads(message)
        if msg["post_type"] == "message" or msg["post_type"] == "message_sent":
            if msg["message_type"] == "group":
                asyncio.create_task(self.client.handle_group_event(msg))

            elif msg["message_type"] == "private":
                asyncio.create_task(self.client.handle_private_event(msg))

            else:
                _log.error("这个报错说明message_type不属于group,private\n" + str(msg))

        elif msg["post_type"] == "notice":
            asyncio.create_task(self.client.handle_notice_event(msg))

        elif msg["post_type"] == "request":
            asyncio.create_task(self.client.handle_request_event(msg))

        elif msg["post_type"] == "meta_event":
            if msg["meta_event_type"] == "lifecycle":
                _log.info(f"机器人 {msg.get('self_id')} 成功启动")
                # _config.set_bot_uid(msg.get('self_id'))
                # asyncio.create_task(self.client.get_bot_name_by_login())

            else:
                pass

        else:
            _log.error("这是一个错误，请反馈给开发者\n" + str(msg))


class MyBot(BotClient):
    def __init__(self, use_ws=True, plugins_path="plugins"):
        super().__init__(use_ws=True, plugins_path="plugins")
        self._additional_handlers = []
        self.event_loop = None
        self._run_task: asyncio.Task | None = None
        self.startup_done = Event()
        self.startup_error: BaseException | None = None
        if type(self.api._http) is WsRoute:
            self.api._http = CustomWsRoute()

    async def run_async(self):
        self.event_loop = asyncio.get_running_loop()
        self._run_task = asyncio.current_task()
        websocket_server = CustomWebsocket(self)
        # await self.plugin_sys.load_plugin(self.api)
        asyncio.create_task(self.get_bot_name_by_login())
        try:
            await websocket_server.ws_connect()
        except asyncio.CancelledError:
            return
        finally:
            self._run_task = None

    async def get_bot_name_by_login(self):
        """
        检查登录状态
        :return:
        """
        await asyncio.sleep(3)
        login_info: dict = await self.api.get_login_info()
        if login_info and login_info["status"] == "ok" and "data" in login_info:
            BASE_CFG.BotInfo.name = str(login_info["data"]["nickname"])
            BASE_CFG.BotInfo.uid = str(login_info["data"]["user_id"])
            config_update()
            _log.info(f"已登录账号：\n昵称：{BASE_CFG.BotInfo.name}\nQ号：{BASE_CFG.BotInfo.uid}")

        else:
            _log.error("获取登录信息失败，请检查配置与网络连接")
            raise RuntimeError("获取登录信息失败")

    def run_thread_sub(self):
        if self.event_loop is None:
            raise RuntimeError("event loop 错误")

        # asyncio.set_event_loop(loop)
        try:
            future = asyncio.run_coroutine_threadsafe(self.handle_additional_func(), self.event_loop)
            future.result()
            _log.info("全部任务执行完毕！")

            # 维持event_loop运行
            # self.event_loop.run_until_complete(asyncio.Future())

        except KeyboardInterrupt:
            exit(0)

        except Exception as E:
            self.startup_error = E.__cause__ or E
            _log.error(f"处理消息时发生错误: {E}")

        finally:
            self.startup_done.set()

    def request_stop(self):
        if self.event_loop is None or self._run_task is None:
            return

        def _cancel():
            if self._run_task is not None and not self._run_task.done():
                self._run_task.cancel()

        self.event_loop.call_soon_threadsafe(_cancel)

    def load_tiya_modules(self, func):
        self._additional_handlers.append(func)
        return func

    async def handle_additional_func(self):
        _log.info(f"目前有{len(self._additional_handlers)}个额外任务")
        _log.info("正在执行额外任务...")
        task_list = []
        errors: list[BaseException] = []
        for func in self._additional_handlers:
            task_list.append(asyncio.create_task(func()))

        for task in task_list:
            try:
                await asyncio.wait_for(task, timeout=900)

            except asyncio.TimeoutError as E:
                errors.append(E)
                _log.error(f"已超时: {E}")

            except Exception as E:
                errors.append(E)
                _log.error(f"处理消息时发生错误: {E}，完整错误信息：\n")
                traceback.print_exc()

        if errors:
            raise RuntimeError(f"TIYA 模块初始化失败，共 {len(errors)} 个错误") from errors[0]

    async def send_group_message_chain_forward(
            self,
            group_id: str,
            msg: list[MessageChain],
            bot_name: str,
            bot_uid: str,
            title: str = ""
    ):
        def dump_large_data(container: list, path, data):
            content = j.dumps(
                {
                    "action": path.replace("/", ""),
                    "params": data,
                    "echo": int(datetime.datetime.now().timestamp()),
                }
            )
            container.append(content)

        async def wait_dump(container: list):
            while not container:
                await asyncio.sleep(1)

            return container[0]

        if not msg or not msg[0].elements:
            return

        if not title:
            title = f"{bot_name}的聊天记录"

        # 从消息链构造转发信息
        payload = {"group_id": group_id}
        nodes = []
        news = []
        for chain in msg:
            message = []

            # 首先检查是否有 reply，只取第一个
            reply_elem = None
            for elem in chain.elements:
                if elem["type"] == "reply":
                    reply_elem = Reply(elem["data"]["id"])
                    break

            # 如果有 reply，插入到消息开头
            if reply_elem:
                message.insert(0, reply_elem)

            # 检查是否包含基本元素(at/图片/文本/表情/猜拳/骰子)
            basic_types = {"at", "image", "text", "face", "dice", "rps"}
            basic_elems = [elem for elem in chain.elements if elem["type"] in basic_types]

            # 如果存在基本元素只添加基本元素
            if basic_elems:
                message.extend(basic_elems)

            # 如果没有基本元素，才使用所有非reply元素
            else:
                message.extend(
                    [elem for elem in chain.elements if elem["type"] != "reply"]
                )

            if not message:
                continue

            news.append({"text": f"{bot_name}: {chain.display()}"})

            new_node = {
                "type": "node",
                "data": {
                    "nickname": bot_name,
                    "user_id": bot_uid,
                    "content": message
                }
            }
            nodes.append(new_node)

        if not nodes:
            return

        payload["messages"] = nodes  # type: ignore
        payload["news"] = news  # type: ignore
        payload["prompt"] = "聊天记录"
        payload["summary"] = f"查看{len(nodes)}条转发消息"
        payload["source"] = title

        # if type(self.api._http) is WsRoute:
        #     self.api._http = CustomWsRoute()

        json_pass = []
        t_dump = Thread(target=dump_large_data, args=(json_pass, "/send_forward_msg", payload))
        t_dump.start()

        try:
            json_str = await asyncio.wait_for(wait_dump(json_pass), timeout=300)
            # print(f"json编码完成：{json_str[0][:50]}，正在发送")

        except asyncio.CancelledError:
            return

        except asyncio.TimeoutError:
            return

        try:
            response = await self.api._http.post("/send_forward_msg", dumped=json_str)

        except Exception as E:
            _log.error(f"{traceback.format_exc()}\n{E}\n\n聊天记录发送失败")
            return

        return response


if __name__ == '__main__':
    async def main():
        async with websockets.connect(
                "ws://127.0.0.1:5050/api",
                extra_headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer 114514",
                },
                ping_interval=300,  # 显式设置心跳
                ping_timeout=330,
        ) as websocket:
            await websocket.send(
                j.dumps(
                    {
                        "action": "get_login_info",
                        "params": {},
                        "echo": int(datetime.datetime.now().timestamp()),
                    }
                )
            )

            response = await websocket.recv()
            print(j.loads(response))

    asyncio.run(main())
