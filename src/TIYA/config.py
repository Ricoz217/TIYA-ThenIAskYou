from __future__ import annotations

import asyncio
import copy
import os
import re
import tempfile
import traceback
import weakref
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Literal,
    Mapping,
    Sequence,
    SupportsIndex,
    overload,
)

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from TIYA.global_vars import QQ_GROUPS
from TIYA.logger import get_logger
from TIYA.utils import plain as _plain

if TYPE_CHECKING:
    from TIYA.mybot import MyBot


ROOT_DIR = Path(__file__).parents[2]
SRC_DIR = Path(__file__).parent
LOGS_DIR = ROOT_DIR / "logs"
DATA_DIR = ROOT_DIR / "data"
BUILTIN_RESOURCES_DIR = DATA_DIR / "builtin_resources"
TIME_ID_STATE_FILE = DATA_DIR / "time_id_state.json"
PROMPTS_DIR = DATA_DIR / "prompt"
GROUPS_DIR = DATA_DIR / "groups_data"
PRIVATE_CHATS_DIR = DATA_DIR / "private_chats_data"
CONFIG_DIR = ROOT_DIR / "config"
MEMORY_DIR = DATA_DIR / "memory"
CHARACTER_DIR = DATA_DIR / "character"
CONFIG_PATH = CONFIG_DIR / "config.yaml"
BASE_SKILL_DIR = DATA_DIR / "skill"
PIXIV_CHUNKS_DIR = DATA_DIR / "pixiv" / "chunks"
FILE_CACHE_DIR = DATA_DIR / "file_cache"

for _directory in (
    LOGS_DIR,
    PROMPTS_DIR,
    GROUPS_DIR,
    PRIVATE_CHATS_DIR,
    CONFIG_DIR,
    MEMORY_DIR,
    BASE_SKILL_DIR,
    PIXIV_CHUNKS_DIR,
    FILE_CACHE_DIR
):
    _directory.mkdir(exist_ok=True, parents=True)

_log = get_logger()

TITLE_BASE = "BOT基本配置"
TITLE_LLM = "LLM模型配置"
TITLE_GROUPS = "群聊设置"
TITLE_PRIVATE = "私聊设置"
TITLE_SETTING = "其他参数配置，请勿随意更改"
_PUBLIC_CONFIG_NAMES = {
    TITLE_BASE: "BASE_CFG",
    TITLE_LLM: "LLM_CFG",
    TITLE_GROUPS: "GROUPS_CFG",
    TITLE_PRIVATE: "PRIVATE_CFG",
    TITLE_SETTING: "SETTING_CFG",
}


DEFAULT_CONFIG: dict[str, Any] = {
    TITLE_BASE: {
        "BotInfo": {"name": "", "uid": ""},
        "AdminList": ["114514", "1919810"],
        "OwnerList": ["114514"],
        "NoticeList": ["114514"],
        "SkipGroups": ["114514"],
        "BotNetWork": {
            "NapCatWebSocket": "ws://localhost:3000",
            "NapCatToken": "114514",
        },
        "Proxies": {
            "GlobalProxy": {"http": None, "https": None},
            "NecessaryProxy": {
                "http": "http://127.0.0.1:7890",
                "https": "http://127.0.0.1:7890",
            },
        },
        "Module": {"setu": False, "eh": False, "imgsearch": False},
        "Agents": {
            "MemoryModel": "YourModelPresetName",
            "MemorySummaryModel": "YourModelPresetName",
            "ImageModel": "YourModelPresetName",
            "ImageModelHigh": "YourModelPresetName",
        },
        "WebSearch": {
            "TavilyKey": "",
            "ProxyMode": "GlobalProxy",
        },
        "SETU": {
            "TagsLLM": "YourModelPresetName",
            "PixivToken": "",
            "ProxyMode": "NecessaryProxy",
            "PixivLoginPublicUrl": "",
            "PixivLoginAutoDiscoverIP": True,
            "PixivLoginPublicPort": 0,
            "PixivLoginIPv4Endpoints": [
                "https://api.ipify.org",
                "https://ddns.oray.com/checkip",
                "https://ip.3322.net",
                "https://4.ipw.cn",
                "https://v4.yinghualuo.cn/bejson",
            ],
            "PixivLoginIPv6Endpoints": ["https://api6.ipify.org", "https://6.ipw.cn"],
            "PixivLoginBindHost": "0.0.0.0",
            "PixivLoginBindPort": 8765,
            "PixivLoginLocale": "zh-HK",
            "PixivLoginTLSCert": "",
            "PixivLoginTLSKey": "",
        },
    },
    TITLE_LLM: {
        "schema_version": 2,
        "Providers": {
            "chat_completions": {
                "deepseek": {
                    "endpoint": "https://api.deepseek.com/chat/completions",
                    "key": "YOUR_DEEPSEEK_API_KEY",
                    "error_dialect": "deepseek",
                    "chat_completions": {"image_file_format": "deepseek"},
                    "llm_params": {"stream": True},
                },
                "openai": {
                    "base_url": "https://api.openai.com/v1",
                    "key": "YOUR_OPENAI_API_KEY",
                    "llm_params": {"stream": True},
                },
                "local": {
                    "endpoint": "http://127.0.0.1:8080/v1/chat/completions",
                    "key": "sk-no-key-required",
                    "llm_params": {"stream": True},
                },
            },
            "responses_api": {
                "openai": {
                    "endpoint": "https://api.openai.com/v1/responses",
                    "key": "YOUR_OPENAI_API_KEY",
                    "responses": {"supports_previous_response_id": True},
                    "llm_params": {"stream": True},
                },
            },
            "anthropic_api": {
                "anthropic": {
                    "endpoint": "https://api.anthropic.com/v1/messages",
                    "key": "YOUR_ANTHROPIC_API_KEY",
                    "error_dialect": "anthropic",
                    "prompt_caching": True,
                    "llm_params": {"stream": True},
                },
            },
            "files_api": {
                "deepseek": {
                    "endpoint": "https://api.deepseek.com/files",
                    "key": "YOUR_DEEPSEEK_API_KEY",
                    "dialect": "deepseek",
                    "error_dialect": "deepseek",
                    "expires_after_seconds": 86400,
                },
                "openai": {
                    "endpoint": "https://api.openai.com/v1/files",
                    "key": "YOUR_OPENAI_API_KEY",
                    "dialect": "openai",
                    "expires_after_seconds": 86400,
                },
                "anthropic": {
                    "endpoint": "https://api.anthropic.com/v1/files",
                    "key": "YOUR_ANTHROPIC_API_KEY",
                    "dialect": "anthropic",
                    "error_dialect": "anthropic",
                    "anthropic_beta": "files-api-2025-04-14",
                    "expires_after_seconds": None,
                },
            },
            "image_generation": {
                "openai": {
                    "endpoint": "https://api.openai.com/v1/images/generations",
                    "edit_endpoint": "https://api.openai.com/v1/images/edits",
                    "key": "YOUR_OPENAI_API_KEY",
                },
            },
        },
        "LLM_List": {
            "YourModelPresetName": {
                "auto_compress_rate": 0.7,
                "proxy_mode": "",
                "providers": [
                    {
                        "llm": "chat_completions:deepseek:YOUR_DEEPSEEK_MODEL_ID",
                        "llm_params": {"max_tokens": 8192, "temperature": 0.7},
                        "max_context": 128_000,
                        "context_history_mode": "full",
                        "files_api": "files_api:deepseek",
                    },
                ],
            },
            "Responses示例": {
                "auto_compress_rate": 0.7,
                "proxy_mode": "GlobalProxy",
                "providers": [
                    {
                        "llm": "responses_api:openai:YOUR_OPENAI_MODEL_ID",
                        "llm_params": {"max_output_tokens": 8192},
                        "max_context": 128_000,
                        "context_history_mode": "auto",
                        "files_api": "files_api:openai",
                    },
                    {
                        "llm": "chat_completions:openai:YOUR_OPENAI_MODEL_ID",
                        "llm_params": {"max_completion_tokens": 8192},
                        "max_context": 128_000,
                        "context_history_mode": "full",
                    },
                ],
            },
            "Anthropic示例": {
                "auto_compress_rate": 0.7,
                "proxy_mode": "GlobalProxy",
                "providers": [
                    {
                        "llm": "anthropic_api:anthropic:YOUR_ANTHROPIC_MODEL_ID",
                        "llm_params": {"max_tokens": 8192},
                        "max_context": 128_000,
                        "context_history_mode": "full",
                        "files_api": "files_api:anthropic",
                    },
                ],
            },
            "本地模型": {
                "auto_compress_rate": 0.7,
                "proxy_mode": "",
                "providers": [
                    {
                        "llm": "chat_completions:local:YOUR_LOCAL_MODEL_ID",
                        "llm_params": {"max_tokens": 8192, "temperature": 0.7},
                        "max_context": 128_000,
                        "context_history_mode": "full",
                    },
                ],
            },
        },
    },
    TITLE_PRIVATE: {
        "Default": {
            "enable": True,
            "chat": True,
            "speak_rate_min": 0.80,
            "speak_rate_max": 1.00,
            "attention_fade_out_time": 900,
            "agent_round_limit": 50,
            "speaker_round_limit": 30,
            "chat_model": "YourModelPresetName",
            "agent_model": "YourModelPresetName",
            "default_character": "抹布",
        },
        "Users": {},
    },
    TITLE_GROUPS: {
        "Groups": {
            "114514": {
                "name": "群名",
                "enable": True,
                "debug": False,
                "chat": True,
                "speak_limit": False,
                "energy_per_hours": 10,
                "energy_limit": 0,
                "speak_rate_min": 0.05,
                "speak_rate_max": 0.70,
                "attention_fade_out_time": 300,
                "relative_rate_min": 0.60,
                "relative_rate_max": 0.80,
                "agent_round_limit": 50,
                "speaker_round_limit": 20,
                "ban_personality": [],
                "ban_topic": [],
                "get_shit": False,
                "post_shit": False,
                "setu": False,
                "setu_nsfw": False,
                "imgsearch": False,
                "eh": 0,
                "chat_model": "YourModelPresetName",
                "agent_model": "YourModelPresetName",
                "default_character": "抹布",
            },
            "Group_Default_Setting": {
                "name": "",
                "enable": True,
                "debug": False,
                "chat": True,
                "speak_limit": False,
                "energy_per_hours": 10,
                "energy_limit": 0,
                "speak_rate_min": 0.05,
                "speak_rate_max": 0.70,
                "attention_fade_out_time": 300,
                "relative_rate_min": 0.60,
                "relative_rate_max": 0.80,
                "agent_round_limit": 50,
                "speaker_round_limit": 20,
                "ban_personality": [],
                "ban_topic": [],
                "get_shit": False,
                "post_shit": False,
                "setu": False,
                "setu_nsfw": False,
                "imgsearch": False,
                "eh": 0,
                "chat_model": "本地模型",
                "agent_model": "本地模型",
                "default_character": "抹布",
            },
        }
    },
}

DEFAULT_SETTING: dict[str, Any] = {
    "Agent": {
        "AgentControlTimeout": 900,  # 主控默认请求超时
        "AgentWorkerTimeout": 900,  # 默认的任务超时
        "AgentCompressContextTimeout": 900,  # 自动压缩上下文超时
        "AgentTaskRetryLimit": 3,  # 默认的任务最大重试次数
        "AgentControlRetryLimit": 3,  # 默认主控最大请求重试次数
        "AgentInitInputMessages": 100,  # 默认的主控输入聊天消息数
        "DefaultHookObjectTTL": 3600,  # 默认的对象工具托管对象TTL，秒
        "ExpiredDays": 7,  # 历史文件保存天数
        "FileDownloadTimeout": 1800,  # 下载文件超时
        "ForceSaveMemoryRetry": 3,  # 强制保存记忆重试限制
        "NoteReactTimes": 15,  # React提醒次数，超过后会注入停止提示词
        "MaxAgentHistory": 5000,  # 最大Agent历史长度
        "MaxAgentDoneTask": 100,  # 最大已完成任务记录
        "MaxAgentContext": 10,  # 最大历史Context数量
        "MaxReactTimes": 20,  # React最大自请求次数，超过后会强制挂起
        "MaxWorker": 16,  # AGENT最大并行任务数量
        "SpeakerInitInputMessages": 50,  # 发言 LLM 默认获取的消息数量
        "SaidReviewPeriod": 30,  # 每发言多少次进行一次审阅
        "WebSearchCacheExpire": 15,  # websearch缓存过期时间
        "WebSearchCoMeGate": 0.8,  # CoMe的置信度门控
    },
    "Aqueue": {
        "DefaultTaskTimeout": 60,  # 每个任务的默认超时
        "MaxWorkers": 16,
        "MinWorkers": 4,
        "StopTimeout": 15,
    },
    "AutoFav": {
        "ImageRepeatGate": 3,  # 一个周期内重复n次视为表情，不包含
        "ImageToTitleTimeout": 120,  # 识图请求超时
        "SaveTime": 900,
        "StatisticDelay": 30,
        "StatisticTimeout": 180,
        "TimePeriod": 3600,  # 自动表情时间周期
        "DescriptionWordsLimit": 15,
    },
    "Common": {
        "CommonRequestTimeout": 15,
        "CommonDialogLiveTime": 900,
        "CompressImageSizeAttempt": 5,
        "CompressImageSizeFactor": 0.7,
        "DialogUploadTimeout": 30,
        "EnableConfigCheck": True,
        "FileCacheExpire": 30,  # 缓存文件保留时间/日
        "LoopTasksGuardPeriod": 900,  # 守护循环任务的检查周期
        "MinImageQuality": 20,
        "MaxMessageHistory": 5000,  # 消息管理器默认最大容量
    },
    "Groups": {
        "ApiTimeout": 60,  # 各种napcatAPI请求的超时
        "AgentPostQueueTimeout": 180,  # Agent请求的队列超时，不是请求超时
        "AgentRequestTime": 300,  # Agent请求超时
        "AutoUnmuteMessages": 300,
        "AutoClearHistoryTime": 3600 * 6,
        "AutoClearMemoryCycle": 300,
        "AutoFavExpire": 30,  # 自动新增表情的过期天数
        "AutoResetAttitude": 3600 * 3,
        "AutoSaveTime": 900,  # 自动持久化的间隔
        "ActiveMemberPersonaLimit": 5,  # 最多只获取多少名活跃群员的画像
        "BanCommandAttempt": 3,
        "BanCommandColdDown": 1800,
        "BanNoticeLiveTime": 1800,
        "CallOrAtProbabilityDecreasing": 0.016,  # 回复和@有多少概率无视
        "CommonDialogEnergyRecover": 10,
        "CommonRepeatRate": 0.8,  # 一般发言复读概率(连续三次以上才会)
        "ClearCacheInterval": 259_200,  # 清理缓存的周期
        "DefaultBanTime": 600,
        "DefaultMuteTime": 10_800,  # 默认的拉黑时间
        "DynamicRelativeDetectTime": 3600,
        "ForceMemoryTipSaid": 25,  # 自己发言多少次生成记忆保存提示
        "ForceMemoryTipMessages": 100,  # 群聊多少条消息生成记忆保存提示
        "ForceMemorySaveSaid": 30,  # 自己发言多少次强制保存一次记忆
        "ForceMemorySaveMessages": 150,  # 群聊多少条消息强制保存一次记忆
        "GroupPersonaCacheTime": 10_800,  # 群画像缓存有效时间
        "GotBanDetectTime": 3600 * 3,
        "GotBanLiveTime": 3600,
        "IgnoreSelfCount": 5,
        "IgnoreMessageCount": 50,
        "ImageRepeatMessageWindow": 15,  # 复读图片的统计窗口
        "ImageRepeatRate": 0.9,  # 复读图片的概率(连续三次以上才会)
        "MaxAskHistory": 300,
        "MaxMessageHistory": 5000,
        "MaxRelativeNews": 5,  # 最多发送多少条新闻
        "MaxShortImageCache": 100,
        "MaxLongImageCache": 5000,
        "MaxConsecutiveRobotCall": 3,
        "MaxHistorySample": 3,
        "MemberUpdatePeriod": 300,  # 成员信息更新周期
        "MemberPersonaCacheTime": 10_800,  # 成员画像缓存有效时间
        "MessageParseTimeout": 15,  # 等待消息解析时间
        "MuteNoticeLiveTime": 1800,
        "MaxNewMemoryDisplay": 3,
        "MemoryExpireTime": 14 * 24 * 3600,
        "MemorySummaryMessageCount": 1000,  # 每多少条消息触发一次自动记忆总结。确保小于消息队列储存上限
        "PendingDeleteMemoryExpire": 300,  # 拟删除记忆列表过期时间
        "ResetAttitudeSelfCount": 5,
        "ResetAttitudeLiveTime": 600,
        "RobotDetectHistories": 50,
        "SETUNoticeSelfCount": 15,
        "SETUNoticeMessageCount": 150,
        "SleepPerWord": 0.6,  # 多句的逐字模拟延迟
    },
    "ImageSearch": {
        "AnimeSearchRequestTimeout": 90,
        "DownloadSearchRequestTimeout": 60,
        "DownloadImageTimeout": 10,
        "ImageSearchDialogLiveTime": 60,
    },
    "ImageRecognize": {
        "CacheExpire": 14,  # 缓存过期天数
        "GifMaxFrames": 7,
        "MaxTasks": 5,
        "MaxFileSizePerFrame": 1,
        "MaxImageFileSize": 5,
        "MaxImageToken": 2048,
        "MaxFrameToken": 1024,
        "Proxy": "GlobalProxy",
        "RequestTimeout": 60,
        "ResponseTimeout": 120,
        "RepeatLimit": 10,
        "VLMRetryLimit": 5,
        "VLMTimeout": 180,
        "ThreadLiveTIme": 60,
    },
    "LLM": {
        "AtDialogLiveTime": 1800,
        "AskQueueTimeout": 120,
        "AskRequestTimeout": 900,
        "ChatQueueTimeout": 90,
        "CommonRequestTimeout": 300,
        "ChatRequestTimeout": 300,
        "ChatResponseMaxToken": 80,
        "MaxResponses": 20,
        "MemorySummaryTimeout": 300,
        "MemoryCreateTimeout": 300,
        "PersonalDialogLiveTime": 1800,
        "PromptMemoryFlowCapacity": 10,
        "PromptMemoryDetectHistories": 10,
        "PromptMemoryMaxMembers": 3,
        "SpeakerRequestTimeout": 180,  # 发言LLM请求超时
    },
    "Message": {
        "ActiveUserTimeLimit": 86_400,  # 活跃用户统计限制
        "ActiveUserSpeakGate": 3,  # 至少说多少句话才算
        "ProcessRetry": 1,  # 消息初始化重试次数
        "MessageParseTimeout": 15,  # 等待消息解析时间
    },
    "PrivateChat": {
        "ApiTimeout": 60,  # 私聊 Napcat Api 的通用超时
        "AgentPostQueueTimeout": 180,  # 私聊 Agent 请求队列超时
        "AgentRequestTime": 300,  # 私聊 Agent 单次请求超时
        "AutoSaveTime": 900,  # 私聊数据自动保存间隔
        "ChatObjectTimeout": 10_800,  # 私聊对象空闲回收时间，默认三小时
        "ClearCacheInterval": 259_200,  # 私聊缓存清理周期
        "ForceMemoryTipSaid": 15,
        "ForceMemoryTipMessages": 50,
        "ForceMemorySaveSaid": 20,
        "ForceMemorySaveMessages": 75,
        "ManagementDialogTimeout": 1800,  # 私聊管理界面的空闲退出时间
        "MaxMessageHistory": 5000,
        "MaxResponses": 100,
        "MaxRelativeNews": 5,
        "MemorySummaryMessageCount": 1000,
        "NoticeResponseTimeout": 300,  # 默认的通知超时时间
        "NoticeSendRetry": 3,
        "PersonaCacheTime": 86_400,
        "ReplyFallbackTimeout": 60,  # 用户请求后未调用 speak 的自动发言保底
        "SleepPerWord": 0.4,  # 模拟发言延迟，每个字延迟
    },
    "Persona": {
        "GetPersonaRetry": 1,  # 获取画像重试次数
        "MemoryConfidenceGate": 0.6,  # 获取记忆的置信度门控
        "SearchMemoryConfidenceGate": 0.5,  # 广义搜索记忆的置信度门控
    },
    "Relatedness": {
        "MessageWindow": 1000,
        "TextCandidateLimit": 64,
        "TextEdgeLimit": 12,
        "ContextEdgeLimit": 8,
        "LexicalWeight": 0.65,
        "SubwordWeight": 0.35,
        "ReplyWeight": 1.0,
        "MentionWeight": 0.65,
        "TextWeight": 0.55,
        "ContextWeight": 0.25,
        "SameUserWeight": 0.08,
        "PropagationDepth": 3,
        "PropagationNodeLimit": 128,
        "TopicRounds": 6,
        "TopicMembershipLimit": 3,
        "TopicMembershipMin": 0.15,
        "TopicMinEdge": 0.35,
        "MaintenanceMessageInterval": 50,
        "MaintenanceSeconds": 900,
        "RuntimeQueueLimit": 128,
    },
    "SETU": {
        "AutoFetchPeriod": 86_400,  # 自动获取新图的周期
        "CrawlPages": 3,  # API 默认自动多爬取多少页
        "CrawlForSearchPagesLimit": 10,  # 通过广度爬取算法时，页数的上限
        "ConcurrentDownloadLimit": 5,  # 每个画廊的图片下载并发限制
        "DownloaderProxy": "NecessaryProxy",
        "DownloaderTimeout": 180,
        "DownloaderRetry": 3,
        "FilterSizeRate": 4,
        "IdsExpireDay": 30,  # 色图查重ID表过期时间
        "MaxGifFileSize": 6,  # GIF 大小限制
        "MaxImageFileSize": 12,  # 静态图片大小限制
        "MaxLoadedChunks": 50,  # 最多常驻多少个 chunks
        "MaxGroupLoadedChunks": 20,  # 群适配器最多常驻多少个 chunks
        "ResultHistoryLimit": 1000,  # 群色图结果互动历史与已使用画廊上限
        "MinBookmarksGate": 2000,  # 入库时默认的最小收藏要求
        "NoisePixels": 5,  # 随机噪点数
        "OcrEndPoint": "https://status.ocr.space/",
        "OcrStatusPage": "https://status.ocr.space/",
        "OcrProxy": "GlobalProxy",
        "OnlineSearchCrawlPages": 5,  # 原生搜索时每个tag爬取的页面数量
        "RelatedSearchCrawlPages": 10,  # 相关性搜索时爬取页面数量
        "SendSETUTaskTimeout": 600,  # 自动解冻的超时
        "SendImageTimeout": 30,
        "SubstituteMultiplier": 4,  # 替补图乘数
        "TransferChunksToGroup": 3,  # 每次获取新图时，传递多少个chunk
        "TokenRefreshTime": 3600,
        "TagsRequestTimeout": 30,
        "TokenResponseTimeout": 900,  # 获取私聊返回的 token 超时
        "UpdatePeriod": 3600 * 24,
        "UnfreezeTimesLimit": 5,  # 解冻次数限制
    },
    "Relative": {
        "InterestTimeout": 90,
        "CommonTimeWeight": 0.1,
        "CommonTimeNowWeight": 0.1,
        "CommonTimeNerfWeight": 0.2,
        "CommonContextWeight": 0.1,
        "CommonSameUserWeight": 0.02,
        "CommonAtWeight": 0.2,
        "CommonReplyWeight": 0.8,
        "CommonBotNerfWeight": 0.1,
        "CommonRepeatNerfWeight": 0.4,
        "DynamicSimilarityTimeWindow": 600,
        "DynamicSimilarityMessageWindow": 50,
        "DynamicSimilarityInheritFactor": 0.3,
        "FilterGroupDocRate": 0.11,
        "FilterGroupDocTFIDF": 0.05,
        "FilterGroupUserRate": 0.6,
        "FilterGroupUserTFIDF": 0.05,
        "FilterGroupTimeRateLow": 0.15,
        "FilterGroupTimeRateUp": 0.3,
        "FilterGroupTimeTFIDFLow": 0.01,
        "FilterGroupTimeTFIDFUp": 0.05,
        "LouvainMinResolution": 0.2,
        "LouvainMaxResolution": 1.6,
        "LouvainMinModularity": 0.8,
        "LouvainMaxCommunity": 0.15,
        "HotWordVertexWeight": 0.1,
        "MinWordAlphaLimit": 3,
        "MinWordNumLimit": 2,
        "MinTFIDFDocuments": 100,
        "MinGroupCommunityVertex": 100,
        "MaxWordCharLimit": 12,
        "MaxWordNumLimit": 8,
        "MaxTFIDFDocuments": 5000,
        "MaxPreviousMessage": 10,
        "MaxNextMessage": 10,
        "MaxCommunityVertex": 5000,
        "RefreshSimilarityWindow": 50,
        "RefreshSimilarityPeriod": 600,
        "TFIDFVectorDimension": 262_144,
    },
}

COMMENT: dict[str, Any] = {
    TITLE_BASE: {
        "BotInfo": {
            "_comment": {"name": "BOT的昵称，自动获取", "uid": "BOT的QQ号，自动获取"}
        },
        "BotNetWork": {
            "_comment": {
                "NapCatWebSocket": "websocket的地址，注意端口。type: str",
                "NapCatToken": "websocket服务器的token。type: str",
            }
        },
        "Proxies": {
            "_comment": {
                "GlobalProxy": "全局代理，默认不启用，可在LLM单独覆盖设置。type: dict",
                "NecessaryProxy": "\n类似色图，eh爬虫等国内使用必须配置。type: dict",
            }
        },
        "Module": {
            "_comment": {
                "setu": "P站智能爬虫，需要配置P站账号，赋予bot发色图的能力",
                "eh": "e站爬虫，需要配置内站(EX)账号，赋予bot直接上传本子的能力",
                "imgsearch": "聚合多个搜图引擎的爬虫，需要配置token，赋予bot搜图的能力",
            }
        },
        "Agents": {
            "_comment": {
                "MemoryModel": "用于处理记忆的语言模型，必须填，唯一建议 DSV4-flash。type: str",
                "MemorySummaryModel": "自动总结记忆所用的模型，必须填，调用不会太多，不建议太弱。type: str",
                "ImageModel": "通用的图片识别模型，调用较多，不要求精度，建议使用价格低廉的。type: str",
                "ImageModelHigh": "高质量的图片识别模型，选旗舰VLM就好。type: str",
            }
        },
        "WebSearch": {
            "_comment": {
                "TavilyKey": "websearch采用tavily。 前往 [https://www.tavily.com] 创建一个APIKey",
                "ProxyMode": "websearch用的代理",
            }
        },
        "SETU": {
            "_comment": {
                "TagsLLM": "用于搜索生成合适的TAGS的LLM，请用非思维模型，且能力不能太弱",
                "PixivToken": "请按照提示输入token，在管理后台输入也可",
                "ProxyMode": "色图模块所使用的代理",
                "PixivLoginPublicUrl": "远程一键登录的 HTTPS 公网地址；留空时自动获取公网 IP",
                "PixivLoginAutoDiscoverIP": "公网地址为空时是否自动获取公网 IP",
                "PixivLoginPublicPort": "公网映射端口；为 0 时复用监听端口",
                "PixivLoginIPv4Endpoints": "公网 IPv4 探测 API 列表，并发请求并采用首个有效结果",
                "PixivLoginIPv6Endpoints": "IPv4 探测全部失败后使用的公网 IPv6 探测 API 列表",
                "PixivLoginBindHost": "远程一键登录临时服务的监听地址",
                "PixivLoginBindPort": "远程一键登录临时服务的监听端口",
                "PixivLoginLocale": "Pixiv 登录浏览器语言，默认 zh-HK",
                "PixivLoginTLSCert": "直连 HTTPS 使用的 PEM 证书；由反向代理终止 TLS 时留空",
                "PixivLoginTLSKey": "直连 HTTPS 使用的 PEM 私钥；由反向代理终止 TLS 时留空",
            }
        },
        "_comment": {
            "AdminList": "BOT管理员列表，填写管理员的QQ号。type: list[str]",
            "OwnerList": "BOT拥有者列表，填写拥有者的QQ号。type: list[str]",
            "NoticeList": "接受BOT通知（上线、报错、信息等）的列表，填写QQ号。type: list[str]",
            "SkipGroups": "不需要加载的群，填写群号。type: list[str]",
            "BotNetWork": "填写BOT框架的ws地址，推荐使用napcat",
            "Proxies": "代理预设设置，请勿更改预设名字，程序内会默认调用",
            "Module": "附加模块总开关，开启需要进行相应配置，关闭则不加载模块。若关闭，即使群配置中开启也无法使用。type: bool",
            "Agents": "处理其他任务所用的模型，与主对话模型分离",
            "WebSearch": "Websearch配置",
            "SETU": "色图模块的配置",
        },
    },
    TITLE_LLM: {
        "_comment": {
            "schema_version": "LLM 配置格式版本，仅支持 2；旧配置须人工填写新版，不自动迁移。",
            "Providers": "\n连接配置树。分类选择协议实现，连接名可自定；仅保留需要的示例，填写真实 Key 和模型 ID。\nendpoint 优先于 base_url；同一服务在不同分类下分别填写连接，不按连接名猜测供应商。",
            "LLM_List": "\n字典键为预设名，供 chat_model/agent_model/MemoryModel 等引用。YourModelPresetName 对应上方默认引用，可改名后同步修改。\nmax_context 的 128000 仅为示例，按实际模型/网关填写；定价在独立数据库维护，不在这里填写 price。",
        },
        "Providers": {
            "_comment": {
                "chat_completions": "标准 Chat Completions；可用于现有主对话和 Memory。",
                "responses_api": "Responses 协议；只对已确认有状态续接的连接启用父 response ID。",
                "anthropic_api": "普通 Anthropic Messages，不使用托管 Agent/Session。",
                "files_api": "独立上传连接，认证与模型连接分别填写；是否引用由预设决定。",
                "image_generation": "独立生图/编辑连接，由 ImagesClient 使用，不作为 Chat 候选，也不启用 Responses 内置工具。",
            },
            "chat_completions": {
                "deepseek": {
                    "_comment": {
                        "error_dialect": "选择供应商错误码表；连接名不参与判断。",
                        "llm_params": "连接级原生请求参数默认值；预设同名字段覆盖，嵌套值整体替换。",
                    },
                    "chat_completions": {
                        "_comment": {"image_file_format": "DeepSeek 的图片 file_id 内容块格式；不适用于官方 OpenAI Chat。"}
                    },
                },
                "openai": {
                    "_comment": {"base_url": "包含版本前缀，系统只追加 chat/completions；也可改用完整 endpoint。"}
                },
                "local": {
                    "_comment": {"endpoint": "本地 OpenAI 兼容服务示例，按实际监听地址和路径修改。"}
                },
            },
            "responses_api": {
                "openai": {
                    "responses": {
                        "_comment": {"supports_previous_response_id": "仅服务明确支持时设 true；兼容 Responses 格式不代表支持续接。"}
                    }
                }
            },
            "anthropic_api": {
                "anthropic": {
                    "_comment": {
                        "prompt_caching": "每次请求默认补充 cache_control，让服务端自动选择缓存点；已隐式缓存或不支持此参数的兼容服务可设 false。llm_params.cache_control 的显式值仍优先。"
                    }
                }
            },
            "files_api": {
                "deepseek": {
                    "_comment": {
                        "dialect": "Files 请求/响应格式，与模型协议分类分开选择。",
                        "expires_after_seconds": "远端 TTL，秒；示例保留一天，不影响本地 file_cache 的现有 TTL。",
                    }
                },
                "openai": {
                    "_comment": {"expires_after_seconds": "请求远端保留一天；不支持此参数的服务可显式填 null。"}
                },
                "anthropic": {
                    "_comment": {
                        "anthropic_beta": "Files beta 头，按使用的服务版本确认。",
                        "expires_after_seconds": "该示例不发送远端过期参数；本地文件仍按现有 TTL 清理。",
                    }
                },
            },
            "image_generation": {
                "openai": {
                    "_comment": {
                        "edit_endpoint": "编辑接口独立地址；模型 ID（如所需的 image 2.5）和生成参数由 ImageGenerateRequest/ImageEditRequest 传入。"
                    }
                }
            },
        },
        "LLM_List": {
            "YourModelPresetName": {
                "_comment": {
                    "auto_compress_rate": "交给上游业务的压缩水位比例；Runtime 不自动压缩或截断。",
                    "proxy_mode": "空字符串表示不使用代理；也可填 BOT基本配置.Proxies 中的名字，如 GlobalProxy。",
                    "providers": "有序候选列表，首项首选，其后按顺序 fallback；只用一个服务就保留一项。",
                },
                "providers": [
                    {
                        "_comment": {
                            "llm": "分类:连接名:模型ID；前两段定位 Providers，剩余部分是完整模型 ID。",
                            "llm_params": "目标接口的原生请求参数，直接发送；不把 max_token 自动改名。",
                            "max_context": "本候选的上下文水位，token 数；按实际模型/服务填写。",
                            "context_history_mode": "full 发送完整历史；auto 仅在支持时续接；previous_response_id 显式请求续接。",
                            "files_api": "分类:连接名；可删除此项以直接内联图片，上传失败按 Provider 策略回退。",
                        }
                    }
                ],
            },
            "Responses示例": {
                "_comment": {"providers": "演示 Responses → Chat Completions 的有序 fallback，按实际需求删除或调整。"},
                "providers": [
                    {
                        "_comment": {
                            "llm_params": "Responses 使用 max_output_tokens；不要照搬 Chat 的输出限制字段。",
                            "context_history_mode": "auto 配合连接声明使用父 response，失效后按恢复策略回放。",
                        }
                    },
                    {"_comment": {"llm": "备用 Chat 使用 URL/Base64 图片；官方 OpenAI Chat 不配置图片 file_id。"}},
                ],
            },
        },
    },
    TITLE_PRIVATE: {
        "Default": {
            "_comment": {
                "enable": "是否允许创建该用户的私聊会话。type: bool",
                "chat": "是否启用私聊 Agent。type: bool",
                "speak_rate_min": "私聊注意力曲线最低请求概率。type: float",
                "speak_rate_max": "私聊注意力曲线最高请求概率。type: float",
                "attention_fade_out_time": "私聊注意力时间窗口，单位秒。type: int",
                "agent_round_limit": "私聊 Agent 每个 Context 的轮次上限。type: int",
                "speaker_round_limit": "私聊 Speaker 每个 Context 的轮次上限。type: int",
                "chat_model": "私聊 Speaker 使用的模型预设。type: str",
                "agent_model": "私聊主控 Agent 使用的模型预设。type: str",
                "default_character": "私聊默认角色。type: str",
            }
        },
        "_comment": {
            "Default": "全部私聊用户共享的默认配置",
            "Users": "按 QQ 号填写的稀疏覆盖配置；不会自动生成",
        },
    },
    TITLE_GROUPS: {
        "Groups": {
            "114514": {
                "_comment": {
                    "name": "群名，自动获取。type: str",
                    "enable": "是否开启该群对象，关闭则连消息都不接收。type: bool",
                    "debug": "是否开启DEBUG模式。type: bool",
                    "chat": "是否开启聊天BOT，可以使用命令随时开关。type: bool",
                    "speak_limit": "若开启，则体力为0时不会进行任何发言；若关闭，则体力仅限制强制发言。type: bool",
                    "energy_per_hours": "每小时发言次数回复量，体力制。type: int",
                    "energy_limit": "体力上限，最小为10，若设置为0则为回复量的两倍。type: int",
                    "ban_personality": "本群不允许使用的人格。type: list[str]",
                    "ban_topic": "本群禁止讨论的话题，并非严格限制，只是告知LLM减少讨论。应少而精的填写关键词，可能会导致反作用。type: list[str]",
                    "get_shit": "自动搬屎功能，是否获取本群的聊天记录，可能会泄露隐私。type: bool",
                    "post_shit": "自动搬屎功能，是否发送屎到本群，可能会发出奇奇怪怪的东西。type: bool",
                    "setu": "色图功能，需要先配置色图模块，有封号风险。type: bool",
                    "setu_nsfw": "是否发黄图，有封号、封群风险。type: bool",
                    "imgsearch": "是否开启搜图功能，部分引擎可能会返回R18结果。type:bool",
                    "eh": "是否开放e站爬虫权限。0:不开放；1:开放；2:仅管理员。type: int",
                    "speak_rate_min": "随机发言的最小值，不建议修改。type: float",
                    "speak_rate_max": "随机发言的最大值，不建议修改。type: float",
                    "attention_fade_out_time": "随机发言注意力时间窗口(秒)，超过该时间随机发言概率回到最低值。 type: int",
                    "relative_rate_min": "相关性发言的最小值，控制发言频率优先修改此值。type: float",
                    "relative_rate_max": "相关性发言的最大值，控制发言频率优先修改此值。type: float",
                    "agent_round_limit": "Agent每个Context窗口的请求轮次上线",
                    "speaker_round_limit": "发言LLM每个Context窗口的发言轮次上线",
                    "chat_model": "发言 LLM 的预设。type: str",
                    "agent_model": "群聊 Agent 的预设。type: str",
                    "default_character": "群聊和提问的默认人格，若不存在则会使用自带的'抹布'。type: str",
                }
            },
            "_comment": {
                "114514": "示例群，根据说明修改你的群配置",
                "Group_Default_Setting": "\n群默认设置，当群配置无效或新增群时，会根据该设置默认生成配置",
            },
        }
    },
}


PathPart = str | int
ConfigPath = tuple[PathPart, ...]
ReportStyle = Literal["combined", "tree", "list"]


def _path_text(path: Sequence[PathPart]) -> str:
    output = ""
    for part in path:
        if isinstance(part, int):
            output += f"[{part}]"

        else:
            output += ("." if output else "") + part

    return output


def _runtime_path_text(path: Sequence[PathPart]) -> str:
    if path and path[0] in _PUBLIC_CONFIG_NAMES:
        return _path_text((_PUBLIC_CONFIG_NAMES[path[0]], *path[1:]))

    return _path_text(path)


@dataclass(frozen=True, slots=True)
class ConfigIssue:
    path: ConfigPath
    code: str
    expected_type: str = ""
    actual_type: str = ""
    actual_value: Any = None
    fallback_used: bool = False
    message: str = ""

    @property
    def path_text(self) -> str:
        return _path_text(self.path)

    def describe(self) -> str:
        if self.code == "missing_key":
            detail = "缺失配置项"

        elif self.code == "type_error":
            sensitive_names = ("token", "secret", "password", "apikey", "api_key")
            leaf_name = str(self.path[-1]).lower() if self.path else ""
            if any(name in leaf_name for name in sensitive_names):
                actual_value = "<已隐藏>"

            else:
                actual_value = repr(self.actual_value)
                if len(actual_value) > 120:
                    actual_value = actual_value[:117] + "..."

            detail = (
                f"类型错误：期望 {self.expected_type}，实际 {self.actual_type}"
                f"，值 {actual_value}"
            )
        elif self.code == "load_error":
            detail = self.message or "配置读取失败"

        else:
            detail = self.message or self.code

        if self.fallback_used:
            detail += "；运行态已使用默认值"

        return detail


@dataclass(frozen=True, slots=True)
class ValidationReport:
    errors: tuple[ConfigIssue, ...] = ()
    repairs: tuple[ConfigIssue, ...] = ()

    @property
    def issues(self) -> tuple[ConfigIssue, ...]:
        return self.errors + self.repairs

    def render(self, style: ReportStyle = "combined") -> str:
        if not self.issues:
            return ""
        tree = self._render_tree()
        path_list = self._render_list()
        if style == "tree":
            return tree

        if style == "list":
            return path_list

        return f"{tree}\n\n{path_list}"

    def _render_list(self) -> str:
        lines = [
            f"配置检查：{len(self.errors)} 个错误，{len(self.repairs)} 个自动修复"
        ]
        for issue in self.errors:
            lines.append(f"- [错误] {issue.path_text}：{issue.describe()}")

        for issue in self.repairs:
            lines.append(f"- [修复] {issue.path_text}：已补入默认值并写回")

        return "\n".join(lines)

    def _render_tree(self) -> str:
        root: dict[PathPart, Any] = {}
        for issue in self.issues:
            node = root
            for part in issue.path:
                node = node.setdefault(part, {})

            node.setdefault("__issues__", []).append(issue)

        lines = ["配置错误树"]

        def walk(node: dict[PathPart, Any], prefix: str) -> None:
            entries = [(key, value) for key, value in node.items() if key != "__issues__"]
            for index, (key, value) in enumerate(entries):
                is_last = index == len(entries) - 1
                branch = "└─" if is_last else "├─"
                label = f"[{key}]" if isinstance(key, int) else str(key)
                leaf_issues = value.get("__issues__", [])
                suffix = ""
                if leaf_issues:
                    suffix = "：" + "；".join(issue.describe() for issue in leaf_issues)

                lines.append(f"{prefix}{branch} {label}{suffix}")
                child_prefix = prefix + ("   " if is_last else "│  ")
                walk(value, child_prefix)

        walk(root, "")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ConfigWriteResult:
    ok: bool
    path: Path
    error: str = ""


@dataclass(frozen=True, slots=True)
class ConfigLoadResult:
    ok: bool
    report: ValidationReport = field(default_factory=ValidationReport)
    wrote_repairs: bool = False
    error: str = ""


CheckedError = ConfigIssue


class ConfigObserver:
    def __init__(self) -> None:
        self._observers: weakref.WeakSet[Any] = weakref.WeakSet()

    def register(self, observer: Any) -> None:
        if not callable(getattr(observer, "update_config", None)):
            raise TypeError("observer must implement update_config()")

        self._observers.add(observer)

    def remove(self, observer: Any) -> None:
        self._observers.discard(observer)

    def update(self) -> None:
        for observer in tuple(self._observers):
            try:
                observer.update_config()

            except Exception:
                _log.error(f"配置观察者更新失败\n{traceback.format_exc()}")


MutationCallback = Callable[[ConfigPath], None]


def _wrap_runtime(value: Any, callback: MutationCallback, path: ConfigPath) -> Any:
    if isinstance(value, Mapping):
        return ConfigMap(value, callback=callback, path=path)

    if isinstance(value, list):
        return ConfigList(value, callback=callback, path=path)

    return copy.deepcopy(value)


class ConfigMap(dict[str, Any]):
    _INTERNAL_NAMES = {"_callback", "_path", "_muted"}

    def __init__(
        self,
        value: Mapping[str, Any] | None = None,
        *,
        callback: MutationCallback | None = None,
        path: ConfigPath = (),
    ) -> None:
        dict.__init__(self)
        object.__setattr__(self, "_callback", callback or (lambda _path: None))
        object.__setattr__(self, "_path", path)
        object.__setattr__(self, "_muted", True)
        for key, item in (value or {}).items():
            dict.__setitem__(self, key, _wrap_runtime(item, self._callback, path + (key,)))

        object.__setattr__(self, "_muted", False)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]

        except KeyError:
            raise AttributeError(
                f"configuration attribute '{_runtime_path_text(self._path + (key,))}' does not exist"
            ) from None

    def __setattr__(self, key: str, value: Any) -> None:
        if key in self._INTERNAL_NAMES:
            object.__setattr__(self, key, value)
            return

        self[key] = value

    def __setitem__(self, key: str, value: Any) -> None:
        dict.__setitem__(self, key, _wrap_runtime(value, self._callback, self._path + (key,)))
        if not self._muted:
            self._callback(self._path + (key,))

    def __delitem__(self, key: str) -> None:
        dict.__delitem__(self, key)
        if not self._muted:
            self._callback(self._path + (key,))

    def clear(self) -> None:
        if self:
            super().clear()
            self._callback(self._path)

    def pop(self, key: str, *default: Any) -> Any:
        existed = key in self
        result = dict.pop(self, key, *default)
        if existed:
            self._callback(self._path + (key,))

        return result

    def popitem(self) -> tuple[str, Any]:
        key, value = super().popitem()
        self._callback(self._path + (key,))
        return key, value

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key in self:
            return self[key]

        self[key] = default
        return self[key]

    def update(self, *args: Any, **kwargs: Any) -> None:
        incoming = dict(*args, **kwargs)
        for key, value in incoming.items():
            self[key] = value

    def replace(self, value: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_muted", True)
        try:
            super().clear()
            for key, item in value.items():
                dict.__setitem__(
                    self,
                    key,
                    _wrap_runtime(item, self._callback, self._path + (key,)),
                )

        finally:
            object.__setattr__(self, "_muted", False)


class ConfigList(list[Any]):
    def __init__(
        self,
        value: Iterable[Any] = (),
        *,
        callback: MutationCallback | None = None,
        path: ConfigPath = (),
    ) -> None:
        self._callback = callback or (lambda _path: None)
        self._path = path
        list.__init__(
            self,
            (_wrap_runtime(item, self._callback, path + (index,)) for index, item in enumerate(value)),
        )

    def _changed(self) -> None:
        self._callback(self._path)
        self._rewrap()

    def _rewrap(self) -> None:
        for index, item in enumerate(list(self)):
            list.__setitem__(self, index, _wrap_runtime(_plain(item), self._callback, self._path + (index,)))

    @overload
    def __setitem__(self, key: SupportsIndex, value: Any) -> None: ...

    @overload
    def __setitem__(self, key: slice, value: Iterable[Any]) -> None: ...

    def __setitem__(self, key: SupportsIndex | slice, value: Any) -> None:
        list.__setitem__(self, key, value)
        self._changed()

    def __delitem__(self, key: SupportsIndex | slice) -> None:
        list.__delitem__(self, key)
        self._changed()

    def append(self, value: Any) -> None:
        list.append(self, value)
        self._changed()

    def extend(self, values: Iterable[Any]) -> None:
        list.extend(self, values)
        self._changed()

    def insert(self, index: SupportsIndex, value: Any) -> None:
        list.insert(self, index, value)
        self._changed()

    def pop(self, index: SupportsIndex = -1) -> Any:
        result = list.pop(self, index)
        self._changed()
        return result

    def remove(self, value: Any) -> None:
        list.remove(self, value)
        self._changed()

    def clear(self) -> None:
        if self:
            list.clear(self)
            self._changed()

    def reverse(self) -> None:
        list.reverse(self)
        self._changed()

    def sort(self, *args: Any, **kwargs: Any) -> None:
        list.sort(self, *args, **kwargs)
        self._changed()


@dataclass(slots=True)
class LLMPresetConfig(Mapping[str, object]):
    """完整原始 LLM 配置快照及选中的预设名，不解释任何连接或协议字段。

    data/provider_definitions 保留配置原貌；旧字典入口只投影首个候选中
    直接存在的值。没有 Provider 默认值继承、引用展开或凭据处理。
    """

    preset_name: str
    configuration: ConfigMap

    def __post_init__(self):
        if not isinstance(self.preset_name, str):
            raise TypeError("LLM 预设名必须是字符串")

        issues = _llm_format_issues(self.configuration, (TITLE_LLM,))
        if issues:
            raise ValueError(issues[0].message)

        self.configuration = ConfigMap(_plain(self.configuration))
        if self.preset_name not in self.configuration["LLM_List"]:
            raise KeyError("LLM 预设不存在")

    @property
    def data(self) -> ConfigMap:
        return self.configuration["LLM_List"][self.preset_name]

    @property
    def provider_definitions(self) -> ConfigMap:
        return self.configuration["Providers"]

    @property
    def providers(self) -> ConfigList:
        return self.data["providers"]

    @property
    def max_context(self) -> int | None:
        # 只读取首项原始值，缺失就保持 None，绝不去连接配置中猜测。
        return self.providers[0].get("max_context")

    @property
    def extra_parameter(self) -> ConfigMap | None:
        return self.providers[0].get("llm_params")

    @property
    def auto_compress_rate(self) -> float | None:
        return self.data.get("auto_compress_rate")

    @property
    def proxy_mode(self) -> str | ConfigMap | None:
        return self.data.get("proxy_mode")

    def __getitem__(self, key: str) -> object:
        if key == "preset_name":
            return self.preset_name

        if key == "max_context":
            return self.max_context

        if key == "extra_parameter":
            return self.extra_parameter

        return self.data[key]

    def __iter__(self):
        return iter(
            dict.fromkeys((*self.data, "preset_name", "max_context", "extra_parameter"))
        )

    def __len__(self) -> int:
        return sum(1 for _ in self)


def _llm_format_issues(raw: object, path: ConfigPath) -> list[ConfigIssue]:
    """只检查文件结构和字段类型；连接是否存在/协议含义由 llm 层检查。"""

    def issue(at, message):
        return ConfigIssue(at, "invalid_llm_format", message=message)

    if not isinstance(raw, Mapping):
        return [issue(path, "LLM模型配置必须是字典")]
    if isinstance(raw.get("schema_version"), bool) or raw.get("schema_version") != 2:
        return [
            issue(
                path + ("schema_version",),
                "仅支持新版 LLM 配置（schema_version: 2），请人工填写新配置；不自动转换旧版",
            )
        ]
    errors = []
    for key in ("Providers", "LLM_List"):
        if not isinstance(raw.get(key), Mapping):
            errors.append(issue(path + (key,), f"{key} 必须是字典"))
    if errors:
        return errors
    for name, spec in raw["LLM_List"].items():
        location = path + ("LLM_List", str(name))
        if not isinstance(spec, Mapping):
            errors.append(issue(location, "LLM 预设必须是字典"))
            continue
        candidates = spec.get("providers")
        if not isinstance(candidates, list) or not candidates:
            errors.append(issue(location + ("providers",), "providers 必须是非空列表"))
            continue
        for index, candidate in enumerate(candidates):
            item = location + ("providers", index)
            if not isinstance(candidate, Mapping):
                errors.append(issue(item, "候选配置必须是字典"))
                continue
            if not isinstance(candidate.get("llm"), str):
                errors.append(issue(item + ("llm",), "llm 必须是字符串"))
            if candidate.get("max_context") is not None and (
                isinstance(candidate["max_context"], bool)
                or not isinstance(candidate["max_context"], int)
            ):
                errors.append(issue(item + ("max_context",), "max_context 必须是整数"))
            if candidate.get("llm_params") is not None and not isinstance(
                candidate["llm_params"], Mapping
            ):
                errors.append(issue(item + ("llm_params",), "llm_params 必须是字典"))
    return errors


def _merge_mapping(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return a recursive copy of *base* with a sparse override applied."""
    result = {str(key): _plain(value) for key, value in base.items()}
    for key, value in override.items():
        key_text = str(key)
        if isinstance(result.get(key_text), Mapping) and isinstance(value, Mapping):
            result[key_text] = _merge_mapping(result[key_text], value)
        else:
            result[key_text] = _plain(value)

    return result


def _commented(value: Any) -> Any:
    if isinstance(value, Mapping):
        result = CommentedMap()
        for key, item in value.items():
            result[key] = _commented(item)

        return result
    if isinstance(value, list):
        result = CommentedSeq()
        result.extend(_commented(item) for item in value)
        return result

    return copy.deepcopy(value)


def _mapping_identity(value: Any) -> tuple[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    for key in ("preset_name", "id", "name"):
        if key in value and isinstance(value[key], (str, int)):
            return key, value[key]
    return None


def _merge_commented(existing: Any, value: Any) -> Any:
    """Merge a runtime value into a ruamel node without detaching comments."""
    if isinstance(existing, CommentedMap) and isinstance(value, Mapping):
        result = copy.deepcopy(existing)
        for key in tuple(result):
            if key not in value:
                del result[key]
        for key, item in value.items():
            if key in result:
                result[key] = _merge_commented(result[key], item)
            else:
                result[key] = _commented(item)
        return result

    if isinstance(existing, CommentedSeq) and isinstance(value, list):
        old_values = [_plain(item) for item in existing]
        unused_indexes = set(range(len(existing)))
        matched_indexes: dict[int, int] = {}

        # Exact values retain their comments through insertions, removals and
        # reordering. Mapping entries may also match by a stable identity so an
        # edited LLM preset keeps comments attached to its nested fields.
        for new_index, item in enumerate(value):
            plain_item = _plain(item)
            match = next(
                (
                    old_index
                    for old_index in sorted(unused_indexes)
                    if old_values[old_index] == plain_item
                ),
                None,
            )
            if match is None:
                identity = _mapping_identity(plain_item)
                if identity is not None:
                    match = next(
                        (
                            old_index
                            for old_index in sorted(unused_indexes)
                            if _mapping_identity(old_values[old_index]) == identity
                        ),
                        None,
                    )
            if match is not None:
                matched_indexes[new_index] = match
                unused_indexes.remove(match)

        result = CommentedSeq()
        result.ca.comment = copy.deepcopy(existing.ca.comment)
        result.ca.end = copy.deepcopy(existing.ca.end)
        flow_style = existing.fa.flow_style()
        if flow_style is True:
            result.fa.set_flow_style()
        elif flow_style is False:
            result.fa.set_block_style()

        for new_index, item in enumerate(value):
            old_index = matched_indexes.get(new_index)
            if old_index is None:
                result.append(_commented(item))
                continue
            result.append(_merge_commented(existing[old_index], item))
            if old_index in existing.ca.items:
                result.ca.items[new_index] = copy.deepcopy(existing.ca.items[old_index])
        return result

    return _commented(value)


def _matches_type(value: Any, sample: Any) -> bool:
    if isinstance(sample, bool):
        return isinstance(value, bool)

    if isinstance(sample, (int, float)):
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    return isinstance(value, type(sample))


class Configer:
    def __init__(
        self,
        config_path: str | os.PathLike[str] = CONFIG_PATH,
        *,
        default_config: Mapping[str, Any] | None = None,
        default_setting: Mapping[str, Any] | None = None,
        comments: Mapping[str, Any] | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self._yaml = YAML()
        self._yaml.indent(mapping=2, sequence=4, offset=2)
        self._yaml.width = 100
        self._default_config = copy.deepcopy(dict(default_config or DEFAULT_CONFIG))
        self._default_setting = copy.deepcopy(dict(default_setting or DEFAULT_SETTING))
        self._comments = copy.deepcopy(dict(COMMENT if comments is None else comments))
        self._document: CommentedMap = CommentedMap()
        self._dirty_paths: set[ConfigPath] = set()
        self._last_report = ValidationReport()
        self._loaded = False
        self._base_config = ConfigMap(callback=self._mark_dirty, path=(TITLE_BASE,))
        self._llm_config = ConfigMap(callback=self._mark_dirty, path=(TITLE_LLM,))
        self._group_config = ConfigMap(callback=self._mark_dirty, path=(TITLE_GROUPS,))
        self._private_config = ConfigMap(callback=self._mark_dirty, path=(TITLE_PRIVATE,))
        self._setting_config = ConfigMap(callback=self._mark_dirty, path=(TITLE_SETTING,))
        self._sections = {
            TITLE_BASE: self._base_config,
            TITLE_LLM: self._llm_config,
            TITLE_GROUPS: self._group_config,
        }
        if TITLE_PRIVATE in self._default_config:
            self._sections[TITLE_PRIVATE] = self._private_config
        self._sections[TITLE_SETTING] = self._setting_config

    @property
    def last_report(self) -> ValidationReport:
        return self._last_report

    def get_base_config(self) -> ConfigMap:
        return self._base_config

    def get_llm_config(self) -> ConfigMap:
        return self._llm_config

    def get_group_config(self) -> ConfigMap:
        return self._group_config

    def get_private_config(self) -> ConfigMap:
        return self._private_config

    def get_private_user_config(self, user_id: str) -> ConfigMap:
        default = _plain(self._private_config.get("Default", {}))
        if not default:
            default = _plain(self._default_config[TITLE_PRIVATE]["Default"])
        users = self._private_config.get("Users", {})
        override = users.get(str(user_id), {}) if isinstance(users, Mapping) else {}
        return ConfigMap(_merge_mapping(default, override))

    def get_setting_config(self) -> ConfigMap:
        return self._setting_config

    def get_all_config(self) -> list[ConfigMap]:
        return list(self._sections.values())

    def get_container_by_title(self, title: str) -> ConfigMap | None:
        return self._sections.get(title)

    def _mark_dirty(self, path: ConfigPath) -> None:
        if self._loaded:
            self._dirty_paths.add(path)

    def load_config(self, config: Mapping[str, Any] | None = None) -> ConfigLoadResult:
        if config is None:
            try:
                with self.config_path.open("r", encoding="utf-8") as stream:
                    loaded = self._yaml.load(stream)

            except Exception as exc:
                # ruamel has several parser exception classes; returning a
                # result keeps this API stable without coupling callers to them.
                return self._load_failure(exc)

        else:
            loaded = _commented(config)

        if loaded is None:
            return self._load_failure(ValueError("配置文件为空"))

        if not isinstance(loaded, Mapping):
            return self._load_failure(TypeError("配置文件顶层必须是映射"))

        document = copy.deepcopy(loaded)
        if not isinstance(document, CommentedMap):
            document = _commented(document)

        errors: list[ConfigIssue] = []
        repairs: list[ConfigIssue] = []
        effective = self._resolve_document(document, errors, repairs)
        report = ValidationReport(tuple(errors), tuple(repairs))

        wrote_repairs = False
        if repairs and config is None:
            try:
                self._atomic_write(self.config_path, document)

            except Exception as exc:
                return ConfigLoadResult(
                    ok=False,
                    report=report,
                    error=f"设置参数自动补全写回失败：{exc}",
                )
            wrote_repairs = True

        self._document = document
        for title, root in self._sections.items():
            root.replace(effective[title])

        self._dirty_paths.clear()
        self._last_report = report
        self._loaded = True
        return ConfigLoadResult(True, report, wrote_repairs)

    def _load_failure(self, exc: Exception) -> ConfigLoadResult:
        issue = ConfigIssue(
            path=(str(self.config_path),),
            code="load_error",
            actual_type=type(exc).__name__,
            message=str(exc),
        )
        return ConfigLoadResult(False, ValidationReport((issue,), ()), error=str(exc))

    def _resolve_document(
        self,
        document: CommentedMap,
        errors: list[ConfigIssue],
        repairs: list[ConfigIssue],
    ) -> dict[str, dict[str, Any]]:
        effective: dict[str, dict[str, Any]] = {}
        for title, defaults in self._default_config.items():
            raw = document.get(title, CommentedMap())
            effective[title] = self._resolve_value(
                raw,
                defaults,
                (title,),
                errors,
                repairs,
                materialize=False,
            )

        raw_setting = document.get(TITLE_SETTING)
        if raw_setting is None:
            raw_setting = CommentedMap()
            document[TITLE_SETTING] = raw_setting

        effective[TITLE_SETTING] = self._resolve_value(
            raw_setting,
            self._default_setting,
            (TITLE_SETTING,),
            errors,
            repairs,
            materialize=True,
        )
        return effective

    def _resolve_value(
        self,
        raw: Any,
        default: Any,
        path: ConfigPath,
        errors: list[ConfigIssue],
        repairs: list[ConfigIssue],
        *,
        materialize: bool,
        report_missing: bool = True,
    ) -> Any:
        if path == (TITLE_LLM,):
            errors.extend(_llm_format_issues(raw, path))
            return _plain(raw) if isinstance(raw, Mapping) else _plain(default)
        if not _matches_type(raw, default):
            errors.append(
                ConfigIssue(
                    path,
                    "type_error",
                    type(default).__name__,
                    type(raw).__name__,
                    _plain(raw),
                    fallback_used=True,
                )
            )
            return copy.deepcopy(default)

        if isinstance(default, Mapping):
            result = {key: copy.deepcopy(value) for key, value in raw.items()}
            for key, default_value in default.items():
                child_path = path + (str(key),)
                if key not in raw:
                    result[key] = copy.deepcopy(default_value)
                    if materialize:
                        raw[key] = _commented(default_value)
                        repairs.append(
                            ConfigIssue(child_path, "missing_key", type(default_value).__name__)
                        )

                    elif report_missing:
                        errors.append(
                            ConfigIssue(
                                child_path,
                                "missing_key",
                                type(default_value).__name__,
                                fallback_used=True,
                            )
                        )
                    continue

                if path == (TITLE_GROUPS,) and key == "Groups":
                    result[key] = self._resolve_groups(
                        raw[key], default_value, child_path, errors
                    )

                elif path == (TITLE_PRIVATE,) and key == "Users":
                    result[key] = self._resolve_private_users(
                        raw[key], default.get("Default", {}), child_path, errors
                    )

                else:
                    result[key] = self._resolve_value(
                        raw[key],
                        default_value,
                        child_path,
                        errors,
                        repairs,
                        materialize=materialize,
                        report_missing=report_missing,
                    )

            return result

        if isinstance(default, list):
            if not default:
                return copy.deepcopy(list(raw))

            template = default[0]
            list_result: list[Any] = []
            for index, item in enumerate(raw):
                list_result.append(
                    self._resolve_value(
                        item,
                        template,
                        path + (index,),
                        errors,
                        repairs,
                        materialize=False,
                        report_missing=report_missing,
                    )
                )

            return list_result

        return copy.deepcopy(raw)

    def _resolve_groups(
        self,
        raw: Any,
        defaults: Mapping[str, Any],
        path: ConfigPath,
        errors: list[ConfigIssue],
    ) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            errors.append(
                ConfigIssue(path, "type_error", "dict", type(raw).__name__, _plain(raw), True)
            )
            return copy.deepcopy(dict(defaults))

        result: dict[str, Any] = {}
        group_template = defaults.get("Group_Default_Setting", {})
        for group_id, value in raw.items():
            template = defaults.get(group_id, group_template)
            result[group_id] = self._resolve_value(
                value,
                template,
                path + (str(group_id),),
                errors,
                [],
                materialize=False,
            )

        for reserved in ("114514", "Group_Default_Setting"):
            if reserved not in result and reserved in defaults:
                result[reserved] = copy.deepcopy(defaults[reserved])
                errors.append(
                    ConfigIssue(
                        path + (reserved,),
                        "missing_key",
                        "dict",
                        fallback_used=True,
                    )
                )

        return result

    def _resolve_private_users(
        self,
        raw: Any,
        template: Mapping[str, Any],
        path: ConfigPath,
        errors: list[ConfigIssue],
    ) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            errors.append(
                ConfigIssue(path, "type_error", "dict", type(raw).__name__, _plain(raw), True)
            )
            return {}

        result: dict[str, Any] = {}
        for raw_user_id, raw_override in raw.items():
            user_id = str(raw_user_id)
            user_path = path + (user_id,)
            if not isinstance(raw_override, Mapping):
                errors.append(
                    ConfigIssue(
                        user_path,
                        "type_error",
                        "dict",
                        type(raw_override).__name__,
                        _plain(raw_override),
                        True,
                    )
                )
                result[user_id] = {}
                continue

            override: dict[str, Any] = {}
            for key, value in raw_override.items():
                key_text = str(key)
                if key_text not in template:
                    override[key_text] = copy.deepcopy(value)
                    continue

                expected = template[key_text]
                if not _matches_type(value, expected):
                    errors.append(
                        ConfigIssue(
                            user_path + (key_text,),
                            "type_error",
                            type(expected).__name__,
                            type(value).__name__,
                            _plain(value),
                            True,
                        )
                    )
                    continue

                override[key_text] = self._resolve_value(
                    value,
                    expected,
                    user_path + (key_text,),
                    errors,
                    [],
                    materialize=False,
                    report_missing=False,
                )

            result[user_id] = override

        return result

    def _candidate_document(self) -> CommentedMap:
        candidate = copy.deepcopy(self._document)
        for path in sorted(self._dirty_paths, key=len):
            self._apply_dirty_path(candidate, path)

        return candidate

    def _runtime_value(self, path: ConfigPath) -> Any:
        current: Any = self._sections[str(path[0])]
        for part in path[1:]:
            current = current[part]

        return _plain(current)

    def _apply_dirty_path(self, document: CommentedMap, path: ConfigPath) -> None:
        try:
            value = self._runtime_value(path)
            deleted = False

        except (KeyError, IndexError):
            value = None
            deleted = True

        current: Any = document
        for part in path[:-1]:
            if isinstance(current, MutableMapping):
                if part not in current:
                    current[part] = CommentedMap()

                current = current[part]

            else:
                current = current[part]

        leaf = path[-1]
        if deleted:
            if isinstance(current, MutableMapping):
                current.pop(leaf, None)

            elif isinstance(leaf, int) and -len(current) <= leaf < len(current):
                current.pop(leaf)

            return

        if isinstance(current, MutableMapping) and leaf in current:
            current[leaf] = _merge_commented(current[leaf], value)
        elif isinstance(current, list) and isinstance(leaf, int) and leaf < len(current):
            current[leaf] = _merge_commented(current[leaf], value)
        else:
            current[leaf] = _commented(value)

    def commit_and_write_config(self) -> ConfigWriteResult:
        if not self._loaded:
            return ConfigWriteResult(False, self.config_path, "尚未加载配置")

        if not self._dirty_paths:
            return ConfigWriteResult(True, self.config_path)

        candidate = self._candidate_document()
        try:
            self._atomic_write(self.config_path, candidate)

        except Exception as exc:
            return ConfigWriteResult(False, self.config_path, str(exc))

        self._document = candidate
        self._dirty_paths.clear()
        return ConfigWriteResult(True, self.config_path)

    def _atomic_write(self, target: Path, document: Mapping[str, Any]) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=target.parent,
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                stream.write(self._render_document(document))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, target)
            temporary_path = None

        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _render_document(self, document: Mapping[str, Any]) -> str:
        """Keep one readable heading per known root, including legacy YAML files."""
        stream = StringIO()
        self._yaml.dump(copy.deepcopy(document), stream)
        lines = stream.getvalue().splitlines(keepends=True)
        titles = tuple(dict.fromkeys((*self._default_config, TITLE_SETTING)))
        heading_pattern = re.compile(
            r"^#\s*=+\s*(?:" + "|".join(re.escape(title) for title in titles) + r")\s*=+\s*$"
        )
        output: list[str] = []

        for line in lines:
            if heading_pattern.fullmatch(line.rstrip("\r\n")):
                continue

            title = next((title for title in titles if line.startswith(f"{title}:")), None)
            if title is not None:
                while output and not output[-1].strip():
                    output.pop()
                if output:
                    output.append("\n")
                output.append(f"# {f' {title} ':=^100}\n")

            output.append(line)

        return "".join(output)

    def show_config(self) -> str:
        candidate = self._candidate_document() if self._loaded else self._document
        return self._render_document(candidate)

    def build_default_document(self) -> CommentedMap:
        document = _commented(
            {
                **copy.deepcopy(self._default_config),
                TITLE_SETTING: copy.deepcopy(self._default_setting),
            }
        )
        self._apply_builtin_comments(document, self._comments)
        return document

    def _apply_builtin_comments(self, document: Any, comments: Any) -> None:
        if isinstance(document, CommentedSeq) and isinstance(comments, list):
            nested_comments = comments
            item_comments: list[str] = []
            if comments and isinstance(comments[-1], list):
                nested_comments = comments[:-1]
                item_comments = [str(item) for item in comments[-1]]

            for index, text in enumerate(item_comments[: len(document)]):
                if text and document.ca.items.get(index) is None:
                    document.yaml_set_comment_before_after_key(index, before=text.lstrip("\n"))

            for index, sub_comments in enumerate(nested_comments[: len(document)]):
                self._apply_builtin_comments(document[index], sub_comments)

            return

        if not isinstance(document, CommentedMap) or not isinstance(comments, Mapping):
            return

        inline = comments.get("_comment", {})
        if isinstance(inline, Mapping):
            for key, text in inline.items():
                if key in document and text and document.ca.items.get(key) is None:
                    comment_text = str(text)
                    if "\n" in comment_text:
                        document.yaml_set_comment_before_after_key(
                            key, before=comment_text.strip()
                        )

                    else:
                        document.yaml_add_eol_comment(comment_text, key)

        for key, sub_comments in comments.items():
            if key == "_comment" or key not in document:
                continue

            self._apply_builtin_comments(document[key], sub_comments)


CONFIG_OBSERVER = ConfigObserver()
EXCEPTION: dict[str, Any] = {}


def set_bot(bot: MyBot) -> None:
    global BOT
    BOT = bot


def get_bot() -> MyBot | None:
    return BOT


def get_bot_uid() -> str:
    try:
        return str(BASE_CFG.BotInfo.uid or "")

    except AttributeError:
        return ""


def get_bot_name() -> str:
    try:
        return str(BASE_CFG.BotInfo.name or "")

    except AttributeError:
        return ""


def get_proxy(proxy_name: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(proxy_name, Mapping):
        return dict(proxy_name)

    if isinstance(proxy_name, str):
        try:
            proxy = BASE_CFG.Proxies[proxy_name]

        except (AttributeError, KeyError):
            proxy = None

        if isinstance(proxy, Mapping):
            return dict(proxy)

    return {"http": None, "https": None}


def get_llm(preset_name: str) -> LLMPresetConfig:
    """读取完整原始预设快照；兼容属性仅映射首项，不解释通信配置。"""
    return LLMPresetConfig(preset_name, LLM_CFG)


def get_llm_list() -> list[str]:
    issues = _llm_format_issues(LLM_CFG, (TITLE_LLM,))
    if issues:
        raise ValueError(issues[0].message)

    return list(LLM_CFG["LLM_List"])


def get_private_user_config(user_id: str) -> ConfigMap:
    return CONFIGER.get_private_user_config(str(user_id))


def get_character_dict() -> dict[str, Any]:
    return CHARACTER_DICT


def set_character_dict(new_dict: Mapping[str, Any]) -> None:
    CHARACTER_DICT.clear()
    CHARACTER_DICT.update(new_dict)


def update_character_dict(new_dict: Mapping[str, Any]) -> None:
    CHARACTER_DICT.update(new_dict)


def write_default_config() -> ConfigWriteResult:
    document = CONFIGER.build_default_document()
    groups = document[TITLE_GROUPS]["Groups"]
    template = groups["Group_Default_Setting"]
    for group_id, group_obj in QQ_GROUPS.items():
        groups[group_id] = copy.deepcopy(template)
        groups[group_id]["name"] = group_obj.group_name

    try:
        CONFIGER._atomic_write(CONFIGER.config_path, document)

    except Exception as exc:
        return ConfigWriteResult(False, CONFIGER.config_path, str(exc))

    return ConfigWriteResult(True, CONFIGER.config_path)


def check_key_and_type(*_args: Any, **_kwargs: Any) -> ValidationReport:
    return CONFIGER.last_report


def restore_setting() -> ConfigLoadResult:
    return CONFIGER.load_config()


def check_config_1() -> ValidationReport:
    return CONFIGER.last_report


def check_config_2() -> ValidationReport:
    return CONFIGER.last_report


def initiate_config_1() -> ConfigLoadResult:
    if not CONFIGER.config_path.is_file():
        write_result = write_default_config()
        if not write_result.ok:
            raise RuntimeError(f"默认配置生成失败：{write_result.error}")

        raise RuntimeError("未进行配置")

    load_result = CONFIGER.load_config()
    if not load_result.ok:
        raise RuntimeError(f"配置读取失败：{load_result.error}")

    try:
        websocket = BASE_CFG.BotNetWork.NapCatWebSocket

    except AttributeError as exc:
        raise RuntimeError("未配置websocket服务器") from exc

    if not websocket:
        raise RuntimeError("未配置websocket服务器")

    return load_result


def initiate_config_2(*, group_list: Mapping[str, str]) -> ConfigWriteResult:
    changed = False
    groups = GROUPS_CFG.Groups
    template = groups.Group_Default_Setting
    for group_id, group_name in group_list.items():
        if group_id not in groups:
            groups[group_id] = _plain(template)
            changed = True

        if groups[group_id].name != group_name:
            groups[group_id].name = group_name
            changed = True

    if changed:
        return CONFIGER.commit_and_write_config()

    return ConfigWriteResult(True, CONFIGER.config_path)


def config_update() -> ConfigWriteResult:
    return CONFIGER.commit_and_write_config()


def reload_config() -> ConfigLoadResult:
    result = CONFIGER.load_config()
    if not result.ok:
        return result

    for group_id, group_obj in QQ_GROUPS.items():
        group_config = GROUPS_CFG.Groups.get(group_id)
        if group_config is not None:
            group_obj.group_config = group_config

    CONFIG_OBSERVER.update()
    return result


def error_printer(
    *,
    style: ReportStyle = "combined",
) -> str:
    try:
        if SETTING_CFG.Common.EnableConfigCheck is False:
            return ""

    except AttributeError:
        pass

    return CONFIGER.last_report.render(style)


def debug_mode(debug_config_path: str | os.PathLike[str] = CONFIG_DIR / "test.yaml") -> ConfigLoadResult:
    CONFIGER.config_path = Path(debug_config_path)
    result = initiate_config_1()
    initiate_config_2(group_list={})
    return result


def set_event(event: asyncio.Event) -> None:
    if BOT is None or BOT.event_loop is None:
        raise RuntimeError("No Running EventLoop")

    async def _set() -> None:
        event.set()

    asyncio.run_coroutine_threadsafe(_set(), BOT.event_loop)


BOT: MyBot | None = None
CONFIGER = Configer()
BASE_CFG = CONFIGER.get_base_config()
LLM_CFG = CONFIGER.get_llm_config()
GROUPS_CFG = CONFIGER.get_group_config()
PRIVATE_CFG = CONFIGER.get_private_config()
SETTING_CFG = CONFIGER.get_setting_config()
ALL_CFG = CONFIGER.get_all_config()

BOT_NAME = get_bot_name
BOT_UID = get_bot_uid
PRIVATES_LIST: list[Any] = []
PRIVATES_DICT: dict[str, Any] = {}
NOTICE_LIST: list[Any] = []
NOTICE_DICT: dict[str, Any] = {}
CHARACTER_DICT: dict[str, Any] = {}
