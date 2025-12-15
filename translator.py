"""
翻译模块 - 调用DeepLX翻译API
"""
import requests
import re
import time
from typing import List, Tuple

from config import DEEPLX_API_URL, DEFAULT_SOURCE_LANG, DEFAULT_TARGET_LANG


class Translator:
    def __init__(self, source_lang: str = None, target_lang: str = None):
        self.api_url = DEEPLX_API_URL
        self.source_lang = source_lang or DEFAULT_SOURCE_LANG
        self.target_lang = target_lang or DEFAULT_TARGET_LANG

    def translate(self, text: str) -> str:
        """
        翻译单段文本

        Args:
            text: 要翻译的文本

        Returns:
            翻译后的文本
        """
        if not text or not text.strip():
            return text

        payload = {
            "text": text,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=30)
            result = response.json()

            if result.get("code") == 200:
                return result.get("data", text)
            else:
                print(f"⚠️ 翻译失败: {result.get('message', 'Unknown error')}")
                return text
        except Exception as e:
            print(f"⚠️ 翻译请求异常: {e}")
            return text

    def translate_markdown(self, md_content: str) -> str:
        """
        翻译Markdown内容，保留格式

        Args:
            md_content: Markdown格式的内容

        Returns:
            翻译后的Markdown内容
        """
        # 分割成段落
        paragraphs = self._split_paragraphs(md_content)
        translated_parts = []

        total = len(paragraphs)
        for i, para in enumerate(paragraphs, 1):
            if self._should_translate(para):
                print(f"🔄 翻译进度: {i}/{total}", end="\r")
                translated = self._translate_paragraph(para)
                translated_parts.append(translated)
                # 避免请求过快
                time.sleep(0.1)
            else:
                translated_parts.append(para)

        print(f"✅ 翻译完成: {total}/{total}")
        return "\n\n".join(translated_parts)

    def _split_paragraphs(self, content: str) -> List[str]:
        """分割段落"""
        # 按空行分割
        paragraphs = re.split(r'\n\s*\n', content)
        return [p.strip() for p in paragraphs if p.strip()]

    def _should_translate(self, text: str) -> bool:
        """判断是否需要翻译"""
        # 跳过纯图片、公式、代码块等
        text = text.strip()

        # 跳过图片
        if re.match(r'^!\[.*\]\(.*\)$', text):
            return False

        # 跳过代码块
        if text.startswith('```') or text.startswith('~~~'):
            return False

        # 跳过纯数学公式
        if re.match(r'^\$\$.*\$\$$', text, re.DOTALL):
            return False

        # 跳过纯数字/符号
        if re.match(r'^[\d\s\.\,\-\+\=\*\/\(\)\[\]\{\}]+$', text):
            return False

        # 跳过空白
        if not text or len(text.strip()) < 2:
            return False

        return True

    def _translate_paragraph(self, para: str) -> str:
        """翻译单个段落，保留Markdown格式"""
        # 保存并替换特殊格式
        placeholders = {}
        counter = [0]

        def save_placeholder(match):
            key = f"__PH{counter[0]}__"
            placeholders[key] = match.group(0)
            counter[0] += 1
            return key

        # 保护行内代码
        protected = re.sub(r'`[^`]+`', save_placeholder, para)

        # 保护链接
        protected = re.sub(r'\[([^\]]+)\]\([^\)]+\)', save_placeholder, protected)

        # 保护图片
        protected = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', save_placeholder, protected)

        # 保护行内公式
        protected = re.sub(r'\$[^\$]+\$', save_placeholder, protected)

        # 保护粗体/斜体标记（但翻译内容）
        # 这里我们保留标记，只翻译文本

        # 翻译
        translated = self.translate(protected)

        # 恢复占位符
        for key, value in placeholders.items():
            translated = translated.replace(key, value)

        return translated

    def translate_batch(self, texts: List[str], batch_size: int = 10) -> List[str]:
        """
        批量翻译文本

        Args:
            texts: 文本列表
            batch_size: 批次大小

        Returns:
            翻译后的文本列表
        """
        results = []
        total = len(texts)

        for i, text in enumerate(texts, 1):
            print(f"🔄 翻译进度: {i}/{total}", end="\r")
            if self._should_translate(text):
                results.append(self.translate(text))
                time.sleep(0.1)
            else:
                results.append(text)

        print(f"✅ 翻译完成: {total}/{total}")
        return results


if __name__ == "__main__":
    # 测试
    translator = Translator()

    # 测试单句翻译
    test_text = "The quick brown fox jumps over the lazy dog."
    result = translator.translate(test_text)
    print(f"原文: {test_text}")
    print(f"译文: {result}")

    # 测试Markdown翻译
    test_md = """
# Introduction

This is a test paragraph with **bold** and *italic* text.

The formula $E=mc^2$ is famous.

```python
print("Hello World")
```

Another paragraph here.
"""
    result_md = translator.translate_markdown(test_md)
    print("\n--- Markdown翻译结果 ---")
    print(result_md)
