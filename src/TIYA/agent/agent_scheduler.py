from __future__ import annotations
__version__ = "0.1.0"

from typing import TYPE_CHECKING
from datetime import datetime
from threading import RLock
from pathlib import Path
from tzlocal import get_localzone
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.base import BaseTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.cron import CronTrigger
from apscheduler.job import Job
from zoneinfo import ZoneInfo

from TIYA.logger import get_logger

if TYPE_CHECKING:
    from .agent_structure import AgentScheTask, Trigger, AgentScheTaskManage
    from .agent import BaseAgent


_log = get_logger()
_AGENTS: dict[str, BaseAgent] = {}

async def run_task(task_id: str, agent_id: str):
    if agent_id not in _AGENTS:
        _log.warning(f"定时任务宿主 Agent 已丢失，id: {agent_id}")
        return

    try:
        await _AGENTS[agent_id].run_schedule_task(task_id)

    except ReferenceError:
        _AGENTS.pop(agent_id, None)

class ScheduleTaskExistError(Exception):
    """已存在相同的定时任务"""
    pass


class AsyncSchedulerManager:
    """
    异步定时任务管理器
    采用APScheduler(v3.11)
    """
    def __init__(self, agent_id: str, tasks_manager: AgentScheTaskManage, persist_path: Path, timezone: str = None):
        try:
            if timezone is None:
                timezone = get_localzone()

            else:
                timezone = ZoneInfo(timezone)

        except:
            timezone = ZoneInfo("Asia/Shanghai")

        self.tz = timezone
        self.persist_path = persist_path
        self.agent_id = agent_id
        self.scheduler = AsyncIOScheduler(timezone=timezone)
        self.tasks_manager = tasks_manager
        self._file_lock = RLock()

    async def start(self):
        self.scheduler.start(paused=True)
        self.load_task()
        self.scheduler.resume()

    async def shutdown(self):
        self.save_task()
        self.scheduler.shutdown(wait=False)

    def save_task(self):
        with self._file_lock:
            temp_file = self.persist_path.with_name(f"{self.persist_path.name}.tmp")
            self.scheduler.export_jobs(str(temp_file))
            if temp_file.is_file():
                temp_file.replace(self.persist_path)

            try:
                temp_file.unlink(missing_ok=True)

            except:
                pass

    def load_task(self):
        if not self.persist_path.is_file():
            return

        self.scheduler.import_jobs(str(self.persist_path))
        for job in self.scheduler.get_jobs():
            job._scheduler = self.scheduler
            job._jobstore_alias = "default"
            job.modify(args=[job.id, self.agent_id])

    def _normalize_dt(self, dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=self.tz)

        return dt.astimezone(self.tz)

    def _parse_trigger(self, data: Trigger) -> BaseTrigger:
        if data.trigger_once:
            dt = datetime.fromisoformat(data.trigger_once)
            return DateTrigger(self._normalize_dt(dt))

        else:
            data = data.to_dict()
            data.pop("trigger_once", None)
            if "start_date" in data:
                data["start_date"] = self._normalize_dt(datetime.fromisoformat(data["start_date"]))

            if "end_date" in data:
                data["end_date"] = self._normalize_dt(datetime.fromisoformat(data["end_date"]))

            return CronTrigger(**data, timezone=self.tz)

    def fire_job_immediately(self, task: str | AgentScheTask):
        if not isinstance(task, str):
            task = task.id

        self.scheduler.modify_job(task, next_run_time=datetime.now(tz=self.tz))

    # ==========
    # 增删改查
    # ==========

    # 增
    def add_job(self, task: AgentScheTask) -> Job:
        trigger = self._parse_trigger(task.trigger_spec)
        with self._file_lock:
            if self.tasks_manager.is_exist(task):
                raise ScheduleTaskExistError(f"id: [{task.id}], name: [{task.name}]")

            self.tasks_manager.tasks[task.id] = task
            try:
                job = self.scheduler.add_job(
                    func=run_task,
                    trigger=trigger,
                    id=task.id,
                    name=task.name if task.name else None,
                    args=[task.id, self.agent_id],
                    misfire_grace_time=30,
                    coalesce=True,
                    max_instances=1,
                    replace_existing=True
                )

            except Exception:
                self.tasks_manager.remove(task)
                raise

            task.next_run_at = str(job.next_run_time)
            self.save_task()
            return job

    # 删
    def remove_job(self, task: AgentScheTask | str):
        with self._file_lock:
            job = self.check_job(task)
            if job is not None:
                self.scheduler.remove_job(job.id)

            self.tasks_manager.remove(task)
            self.save_task()

    # 改
    def edit_job(self, task: AgentScheTask):
        job = self.scheduler.get_job(task.id)
        if job is None:
            return

        trigger = self._parse_trigger(task.trigger_spec)
        changes = {
            "trigger": trigger,
            "name": task.name,
        }
        self.scheduler.modify_job(job.id, **changes)
        self.save_task()

    # 查
    def list_job(self) -> list[Job]:
        return self.scheduler.get_jobs()

    # 查
    def check_job(self, task: AgentScheTask | str) -> Job | None:
        if not isinstance(task, str):
            task = task.id

        return self.scheduler.get_job(task)

    def pause_job(self, task: AgentScheTask | str):
        job = self.check_job(task)
        if job is None:
            return

        self.scheduler.pause_job(job.id)
        self.save_task()

    def resume_job(self, task: AgentScheTask | str):
        job = self.check_job(task)
        if job is None:
            return

        self.scheduler.resume_job(job.id)
        self.save_task()
