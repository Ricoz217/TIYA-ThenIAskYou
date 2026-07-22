from __future__ import annotations
"""
QQ_bot.py
TIYA_BOT 的主入口
"""
__version__ = "0.3.0"

import time
import asyncio
import sys
import traceback
from typing import TYPE_CHECKING, Any, Mapping
from threading import Thread

from TIYA.mybot import MyBot
from TIYA.api.common_api import get_group_list
from TIYA.memory import ContextMemoryConfig, get_context_memory_engine
from TIYA.input_manager import read_command_async
from TIYA.global_vars import QQ_GROUPS, PRIVATE_CHATS
from TIYA.model.character import initiate_groups_character_manager
from TIYA.private_chat import PrivateChat
from TIYA.qq_group import QQGroup
from TIYA.setu.setu import AsyncPixivApi
from TIYA.relatedness import close_relatedness_runtime
from TIYA.storage_cleanup import run_auto_storage_cleanup
from TIYA.executor import shutdown_all as shutdown_executors
from TIYA.event_loop_monitor import EventLoopLagMonitor
from TIYA.runtime_control import (
    clear_hot_reload_handler,
    hot_reload,
    pause_before_exit,
    save_groups,
    set_hot_reload_handler,
    shutdown_groups,
    spawn_restart_process,
    start_new_groups,
)
from TIYA.config import (
    initiate_config_1,
    initiate_config_2,
    reload_config,
    error_printer,
    BASE_CFG,
    GROUPS_CFG,
    get_bot_uid,
    set_bot,
    DATA_DIR,
    ROOT_DIR,
)
from TIYA.logger import get_logger

# ncatbot SDK
from ncatbot.utils.config import config as ncat_config

if TYPE_CHECKING:
    from ncatbot.core.message import GroupMessage, PrivateMessage


_log = get_logger()
_NEW_GROUP_REFRESH_DELAY = 10.0


async def _refresh_group_members_for_ban_notice(msg: Mapping[str, Any]) -> None:
    """收到群禁言事件后，强制刷新对应群的成员与禁言状态。"""
    group_id = str(msg.get("group_id", ""))
    group = QQ_GROUPS.get(group_id)
    if group is None:
        return

    await group.update_member(force=True)


async def _refresh_groups_for_self_increase_notice(msg: Mapping[str, Any]) -> bool:
    """BOT 自身加入新群后，等待 NapCat 同步群列表并触发热更新。"""
    if msg.get("notice_type") != "group_increase":
        return False

    self_id = str(msg.get("self_id", ""))
    user_id = str(msg.get("user_id", ""))
    group_id = str(msg.get("group_id", ""))
    if not self_id or user_id != self_id or not group_id:
        return False

    await asyncio.sleep(_NEW_GROUP_REFRESH_DELAY)
    try:
        await hot_reload()

    except Exception as error:
        _log.error(f"BOT 加入群[{group_id}]后自动刷新群配置失败: {error}")
        _log.debug(traceback.format_exc())
        return False

    return True


def main():
    """程序主入口"""

    # =========================================================
    # 加载并检查配置
    # =========================================================

    ncat_config.set_bot_uin(get_bot_uid())
    try:
        initiate_config_1()

    except RuntimeError as exc:
        pause_before_exit(f"配置加载失败，请检查或删除配置重试。\n{exc}")
        return

    ncat_config.set_ws_uri(BASE_CFG.BotNetWork.NapCatWebSocket)  # 设置 napcat websocket server 地址
    ncat_config.set_token(BASE_CFG.BotNetWork.NapCatToken)  # 设置 token (napcat 服务器的 token)
    mybot = MyBot()
    set_bot(mybot)  # 设置全局实例
    # 更新记忆配置
    _memory_config = ContextMemoryConfig(
        base_dir=DATA_DIR / "memory",
        max_bucket_depth=6,
        llm_preset=BASE_CFG.Agents.MemoryModel,
        image_llm_preset=BASE_CFG.Agents.ImageModelHigh
    )
    _memory = get_context_memory_engine(config=_memory_config)
    auto_cleanup_task: asyncio.Task[None] | None = None
    event_loop_monitor: EventLoopLagMonitor | None = None

    async def save_all():
        errors = await save_groups(QQ_GROUPS.values())
        for error in errors:
            _log.error(f"群数据保存失败: {error}")

    def save():
        save_future = asyncio.run_coroutine_threadsafe(save_all(), mybot.event_loop)
        save_future.result()

    # =========================================================
    # 注册基础回调函数
    # =========================================================

    @mybot.group_event()
    async def on_group_message(msg: GroupMessage):
        group_id = str(msg.group_id)
        if not msg.message:
            return

        if group_id in QQ_GROUPS:
            await QQ_GROUPS[group_id].accept_message(msg)

    @mybot.private_event()
    async def on_private_message(msg: PrivateMessage):
        if not msg.message:
            return

        user_id = str(msg.user_id)
        username = str(msg.sender.nickname or "")
        private_chat = PrivateChat.get_or_create(user_id, username)
        await private_chat.accept_message(msg)

    @mybot.notice_event
    async def on_notice_message(msg):
        notice_type = msg.get("notice_type", "")

        # 实现一个全局持久化命令
        if notice_type == "bot_offline":
            await save_all()

        elif notice_type == "group_ban":
            await _refresh_group_members_for_ban_notice(msg)

        elif notice_type == "group_increase":
            await _refresh_groups_for_self_increase_notice(msg)

    # TODO
    @mybot.request_event
    async def on_request_message(msg):  # 绑定请求消息回调函数
        """不知道这个是什么"""
        ...

    @mybot.load_tiya_modules
    async def load_groups(*, strict: bool = True):
        """加载TIYA模块"""
        group_data_list: list[dict] | None = None
        for _ in range(3):
            group_list = await get_group_list()
            if not group_list or group_list.get("status", "") != "ok":
                continue

            data: list[dict] = group_list.get("data", [])
            if not data:
                raise RuntimeError("当前 BOT 账号没有添加群，无法初始化群对象")

            group_data_list = data
            break

        if group_data_list is None:
            raise RuntimeError("获取群列表失败，请检查网络或 napcat")

        def initialize_group_config(new_groups: dict[str, str]):
            initiate_config_2(group_list=new_groups)
            initiate_groups_character_manager()

        # noinspection PyTypeChecker
        result = await start_new_groups(
            group_data_list,
            groups=QQ_GROUPS,
            skipped_groups=BASE_CFG.SkipGroups,
            group_configs=GROUPS_CFG.Groups,
            initialize_config=initialize_group_config,
            make_group=QQGroup.make_group,
            strict=strict,
        )

        _log.info(f"已加载{len(QQ_GROUPS)}个群")
        print(error_printer())
        return result

    @mybot.load_tiya_modules
    async def initiate_setu():
        """初始化色图单例，登录"""
        if not BASE_CFG.Module.setu:
            return

        SETU = AsyncPixivApi.get_api()
        asyncio.create_task(SETU.pixiv_get_new_illusts())

    @mybot.load_tiya_modules
    async def initiate_storage_cleanup():
        """启动由主入口托管的低频存储清理任务。"""
        nonlocal auto_cleanup_task
        if auto_cleanup_task is not None and not auto_cleanup_task.done():
            return

        auto_cleanup_task = asyncio.create_task(
            run_auto_storage_cleanup(logger=_log),
            name="tiya-storage-cleanup",
        )

    # =========================================================
    # 执行网络连接
    # =========================================================

    # 连接 napcat
    t_main = Thread(target=mybot.run, args=(True,), daemon=True)
    t_main.start()

    # 等待获取事件循环
    start_time = time.time()
    while mybot.event_loop is None and time.time() - start_time < 15:
        time.sleep(0.1)

    if mybot.event_loop is None:
        _log.error("获取事件循环失败")
        pause_before_exit("获取事件循环失败")
        return

    async def start_event_loop_monitor() -> EventLoopLagMonitor:
        monitor = EventLoopLagMonitor(
            logger=_log,
            threshold=3.0,
            probe_interval=0.25,
        )
        await monitor.start()
        return monitor

    monitor_future = None
    try:
        monitor_future = asyncio.run_coroutine_threadsafe(
            start_event_loop_monitor(),
            mybot.event_loop,
        )
        event_loop_monitor = monitor_future.result(timeout=5)

    except Exception as error:
        if monitor_future is not None:
            monitor_future.cancel()

        _log.warning(f"事件循环阻塞监控启动失败，将继续运行 BOT: {error}")

    # 启动辅助线程
    t_sub = Thread(target=mybot.run_thread_sub, daemon=True)
    t_sub.start()
    mybot.startup_done.wait()
    if mybot.startup_error is not None:
        if event_loop_monitor is not None:
            event_loop_monitor.stop()
        mybot.request_stop()
        pause_before_exit(f"程序初始化失败:\n{mybot.startup_error}")
        return

    # =========================================================
    # 主线程直接命令函数
    # =========================================================

    shutdown_requested = False
    restart_requested = False

    async def reload_all():
        config_result = reload_config()
        if not config_result.ok:
            raise RuntimeError(config_result.error or "配置重新载入失败")

        result = await load_groups(strict=False)
        if result.pending_config:
            pending = ", ".join(result.pending_config)
            _log.warning(f"新增群配置尚未完成，已暂缓启动: {pending}")

        _log.info(
            f"配置热更新完成，新增群[{len(result.started)}]个，"
            f"待配置群[{len(result.pending_config)}]个"
        )
        return result

    set_hot_reload_handler(reload_all)

    def reload():
        """重新加载配置，并发现、启动新增群。"""
        reload_future = asyncio.run_coroutine_threadsafe(
            hot_reload(),
            mybot.event_loop
        )
        reload_future.result()

    async def shutdown_all():
        nonlocal auto_cleanup_task, event_loop_monitor
        cleanup_task = auto_cleanup_task
        auto_cleanup_task = None
        if cleanup_task is not None:
            cleanup_task.cancel()
            await asyncio.gather(cleanup_task, return_exceptions=True)

        errors = await shutdown_groups(QQ_GROUPS.values(), timeout=30)
        for _error in errors:
            _log.error(f"群对象关闭失败: {_error}")

        errors = await shutdown_groups(PRIVATE_CHATS.values(), timeout=30)
        for _error in errors:
            _log.error(f"私聊对象关闭失败: {_error}")

        QQ_GROUPS.clear()
        PRIVATE_CHATS.clear()
        await close_relatedness_runtime()

        try:
            await _memory.close(wait=False)

        except Exception as _error:
            _log.error(f"记忆系统关闭失败: {_error}")

        monitor = event_loop_monitor
        event_loop_monitor = None
        if monitor is not None:
            await monitor.close()

    def shutdown():
        """正常关闭所有群对象和后台模块，然后退出程序。"""
        nonlocal shutdown_requested
        if shutdown_requested:
            return

        shutdown_requested = True
        _log.info("正在正常关闭 TIYA...")
        shutdown_future = asyncio.run_coroutine_threadsafe(shutdown_all(), mybot.event_loop)
        try:
            shutdown_future.result(timeout=35)

        except TimeoutError:
            _log.error("正常关闭超时，将停止 BOT 事件循环")

        finally:
            mybot.request_stop()

    def restart():
        """正常关闭当前进程，并在新的终端窗口中重启 BOT。"""
        nonlocal restart_requested
        if shutdown_requested:
            return

        restart_requested = True
        _log.info("正在重启 TIYA...")
        shutdown()

    try:
        while not shutdown_requested:
            future = asyncio.run_coroutine_threadsafe(read_command_async(prompt='>', logger=_log), mybot.event_loop)
            command = future.result()
            try:
                exec(command)

            except Exception as E:
                _log.error(f"无效指令:\n {E}")
                _log.debug(traceback.format_exc())

    except KeyboardInterrupt:
        try:
            save()
        finally:
            mybot.request_stop()

    except asyncio.CancelledError:
        try:
            save()
        finally:
            mybot.request_stop()

    finally:
        if event_loop_monitor is not None:
            event_loop_monitor.stop()
            event_loop_monitor = None
        clear_hot_reload_handler()
        t_main.join(timeout=5)
        shutdown_executors()
        if t_main.is_alive():
            _log.error("BOT 事件循环未在 5 秒内退出")
        else:
            _log.shutdown()
            if restart_requested:
                try:
                    spawn_restart_process(
                        executable=sys.executable,
                        cwd=ROOT_DIR
                    )
                except OSError as error:
                    pause_before_exit(f"重启新进程失败:\n{error}")


if __name__ == "__main__":
    main()
