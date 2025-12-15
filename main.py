#!/usr/bin/env python3
"""
PDF翻译工具 - 主程序入口

使用Gitee AI文档解析API + DeepLX翻译API实现PDF翻译
"""
import os
import sys
import argparse
import zipfile
import tempfile
import json
from pathlib import Path

from document_parser import DocumentParser
from translator import Translator
from pdf_builder import PDFBuilder


class PDFTranslator:
    def __init__(self, source_lang: str = "EN", target_lang: str = "ZH"):
        """
        初始化PDF翻译器

        Args:
            source_lang: 源语言
            target_lang: 目标语言
        """
        self.parser = DocumentParser()
        self.translator = Translator(source_lang, target_lang)

    def translate(self, input_path: str, output_path: str = None) -> str:
        """
        翻译PDF文件

        Args:
            input_path: 输入PDF路径
            output_path: 输出PDF路径（可选）

        Returns:
            输出文件路径
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"文件不存在: {input_path}")

        # 生成输出路径
        if not output_path:
            base_name = Path(input_path).stem
            output_dir = Path(input_path).parent
            output_path = str(output_dir / f"{base_name}_translated.pdf")

        print("=" * 50)
        print("📚 PDF翻译工具")
        print("=" * 50)
        print(f"📄 输入文件: {input_path}")
        print(f"📝 输出文件: {output_path}")
        print("=" * 50)

        # 步骤1: 解析文档
        print("\n🔍 步骤1: 解析文档...")
        parse_result = self.parser.parse(input_path, output_format="md")

        # 处理解析结果
        md_content, images = self._process_parse_result(parse_result)

        if not md_content:
            raise ValueError("文档解析失败，未获取到内容")

        print(f"✅ 解析完成，内容长度: {len(md_content)} 字符")

        # 步骤2: 翻译内容
        print("\n🌐 步骤2: 翻译内容...")
        translated_content = self.translator.translate_markdown(md_content)
        print(f"✅ 翻译完成")

        # 保存翻译后的Markdown（用于调试）
        md_output_path = output_path.replace('.pdf', '.md')
        with open(md_output_path, 'w', encoding='utf-8') as f:
            f.write(translated_content)
        print(f"📄 Markdown已保存: {md_output_path}")

        # 步骤3: 重建PDF
        print("\n📄 步骤3: 重建PDF...")
        builder = PDFBuilder(input_path)
        builder.build_from_markdown(translated_content, output_path, images)
        builder.close()

        print("\n" + "=" * 50)
        print("🎉 翻译完成!")
        print(f"📄 输出文件: {output_path}")
        print("=" * 50)

        return output_path

    def _process_parse_result(self, result: dict) -> tuple:
        """
        处理解析结果

        Args:
            result: 解析API返回的结果

        Returns:
            (markdown内容, 图片字典)
        """
        images = {}

        if result["type"] == "text":
            return result["content"], images

        elif result["type"] == "zip":
            # 解压zip文件
            with tempfile.TemporaryDirectory() as temp_dir:
                zip_path = os.path.join(temp_dir, "result.zip")
                with open(zip_path, "wb") as f:
                    f.write(result["content"])

                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)

                # 查找markdown文件
                md_content = ""
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)

                        if file.endswith('.md'):
                            with open(file_path, 'r', encoding='utf-8') as f:
                                md_content = f.read()

                        elif file.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                            with open(file_path, 'rb') as f:
                                images[file] = f.read()

                        elif file.endswith('.json'):
                            # 可能包含结构化数据
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                    # 如果JSON中有markdown内容
                                    if isinstance(data, dict) and 'markdown' in data:
                                        md_content = data['markdown']
                            except:
                                pass

                return md_content, images

        return "", images


def main():
    parser = argparse.ArgumentParser(
        description="PDF翻译工具 - 保持原文档格式的翻译",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py document.pdf
  python main.py document.pdf -o translated.pdf
  python main.py document.pdf --source EN --target ZH
        """
    )

    parser.add_argument("input", help="输入PDF文件路径")
    parser.add_argument("-o", "--output", help="输出PDF文件路径")
    parser.add_argument("--source", default="EN", help="源语言 (默认: EN)")
    parser.add_argument("--target", default="ZH", help="目标语言 (默认: ZH)")

    args = parser.parse_args()

    try:
        translator = PDFTranslator(args.source, args.target)
        output_path = translator.translate(args.input, args.output)
        print(f"\n✅ 成功! 输出文件: {output_path}")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
