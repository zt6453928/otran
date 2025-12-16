import os

# API配置 - 优先使用环境变量

# MinerU文档解析API
MINERU_API_URL = os.environ.get("MINERU_API_URL", "https://mineru.net/api/v4/extract/task")
MINERU_API_TOKEN = os.environ.get("MINERU_API_TOKEN", "")

# DeepLX翻译API
DEEPLX_API_URL = os.environ.get("DEEPLX_API_URL", "")

# 默认翻译设置
DEFAULT_SOURCE_LANG = os.environ.get("DEFAULT_SOURCE_LANG", "EN")
DEFAULT_TARGET_LANG = os.environ.get("DEFAULT_TARGET_LANG", "ZH")
