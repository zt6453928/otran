import os

# API配置 - 优先使用环境变量

# MinerU文档解析API
MINERU_API_URL = os.environ.get("MINERU_API_URL", "https://mineru.net/api/v4/extract/task")
MINERU_API_TOKEN = os.environ.get("MINERU_API_TOKEN", "eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI1NDgwMDQ3NSIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc2NTcyOTg5MiwiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiIiwib3BlbklkIjpudWxsLCJ1dWlkIjoiNzM1ODU1ZTYtZDY5ZC00ZDFlLWJjNTMtYTQ0ZjdhZDg1YzI4IiwiZW1haWwiOiIiLCJleHAiOjE3NjY5Mzk0OTJ9.QpvpIG6lwF_hz58sVzVaPpCffixArXoT2EXKHD4bsevsOn6fwkEqOHhSvty2VkTebzBtPEwRQp73uWMRxic1lw")

# DeepLX翻译API
DEEPLX_API_URL = os.environ.get("DEEPLX_API_URL", "")

# 默认翻译设置
DEFAULT_SOURCE_LANG = os.environ.get("DEFAULT_SOURCE_LANG", "EN")
DEFAULT_TARGET_LANG = os.environ.get("DEFAULT_TARGET_LANG", "ZH")
