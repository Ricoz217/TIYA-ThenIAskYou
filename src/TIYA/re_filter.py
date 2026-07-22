import re


_at_badformat = re.compile(r"\[@.*?]")
_at_filter = re.compile(r'\[at=(.*?)]')
_setu_filter = re.compile(r'\[setu(?:=(.*))?]')
_ban_filter = re.compile(r'\[ban=(.*?)]')
_mute_filter = re.compile(r'\[mute=(.*?)]')
_think_filter = re.compile(r'<think>.*?</think>', flags=re.DOTALL)
_img_filter = re.compile(r'\[image=(.*?)]')
_img_badformat_filter = re.compile(r'\[image:(.*?)]')
_reply_filter = re.compile(r'\[reply=(\d+)]')
_memory_filter = re.compile(r'\[memory=*(.*?)]')
_fav_badformat_filter = re.compile(r'\[(.+?)]')
_split_filter = re.compile(r'\[split=(.*?)(?:(\[at=\d+])(.*))?]')
_all_filter = re.compile(r'\[.*?]', re.DOTALL)
_robot_cmd_filter = re.compile(r'\[@.*?\(QQ号:(\w+?)\)] *(.+)')
_pixiv_artwork_filter = re.compile(r'pixiv\.net/artworks/(\d+)', flags=re.DOTALL)
_pixiv_user_filter = re.compile(r'pixiv\.net/users/(\d+)', flags=re.DOTALL)
_json_filter = re.compile(r'^(?:```)?(?:json)?\s*(.*?)\s*(?:```)?$', flags=re.DOTALL)
_cq_data_filter = re.compile(r'\[CQ:(.+?),(.+?)]', re.DOTALL)
_cq_data2dict = re.compile(r'(\w+?)=(.*?)(?=,\w+=|$)', re.DOTALL)
_cq_pure_filter = re.compile(r'\[[@|reply=].*?]', re.DOTALL)
_pure_filter = re.compile(r'\[.*?]', re.DOTALL)
_custom_punctuation = re.compile(r'[，。、！？“‘”’,!?\"\'. ]')
_illegal_filter = re.compile(r'[<>:"/\\|?*\x00-\x1F]')
_chunk_index_filter = re.compile(r'chunk_(\w+?).json')
_jieba_dot_filter = re.compile(r'[^\w\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]', flags=re.DOTALL)
_checkpoint_filter = re.compile(r'^checkpoint_(\w+?).json$')
_date_dir_filter = re.compile(r'^(\d{2}_\d{2}_\d{2})$')
_pixiv_link_filter = re.compile(
    r"(?<![/\w.-])(?:https?://)?(?:[a-z0-9-]+\.)*pixiv\.net/"
    r"(?P<kind>artworks|users)/(?P<target_id>\d+)",
    flags=re.IGNORECASE,
)