"""
PDF重建模块 - 基于翻译后的内容重建PDF
"""
import fitz  # PyMuPDF
import re
import os
from typing import List, Dict, Tuple, Optional
import base64
import io
from PIL import Image


class PDFBuilder:
    def __init__(self, original_pdf_path: str):
        """
        初始化PDF构建器

        Args:
            original_pdf_path: 原始PDF文件路径
        """
        self.original_pdf_path = original_pdf_path
        self.original_doc = fitz.open(original_pdf_path)

        # 页面尺寸
        self.page_width = self.original_doc[0].rect.width
        self.page_height = self.original_doc[0].rect.height

        # 字体设置
        self.font_path = self._find_chinese_font()

    def _find_chinese_font(self) -> Optional[str]:
        """查找系统中的中文字体"""
        # macOS常见中文字体路径
        font_paths = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
            "/System/Library/Fonts/Supplemental/Songti.ttc",
        ]

        for path in font_paths:
            if os.path.exists(path):
                return path

        return None

    def build_from_markdown(self, md_content: str, output_path: str,
                           images: Dict[str, bytes] = None) -> str:
        """
        从Markdown内容构建PDF

        Args:
            md_content: 翻译后的Markdown内容
            output_path: 输出PDF路径
            images: 图片字典 {图片名: 图片数据}

        Returns:
            输出文件路径
        """
        # 创建新PDF
        doc = fitz.open()

        # 解析Markdown
        elements = self._parse_markdown(md_content)

        # 当前页面和位置
        page = doc.new_page(width=self.page_width, height=self.page_height)
        y_position = 50  # 顶部边距
        x_margin = 50
        max_width = self.page_width - 2 * x_margin

        for element in elements:
            elem_type = element["type"]
            content = element["content"]

            # 计算需要的高度
            height_needed = self._estimate_height(element, max_width)

            # 检查是否需要新页面
            if y_position + height_needed > self.page_height - 50:
                page = doc.new_page(width=self.page_width, height=self.page_height)
                y_position = 50

            # 渲染元素
            if elem_type == "heading":
                y_position = self._render_heading(page, element, x_margin, y_position, max_width)
            elif elem_type == "paragraph":
                y_position = self._render_paragraph(page, content, x_margin, y_position, max_width)
            elif elem_type == "image":
                y_position = self._render_image(page, element, x_margin, y_position, max_width, images)
            elif elem_type == "code":
                y_position = self._render_code(page, content, x_margin, y_position, max_width)
            elif elem_type == "formula":
                y_position = self._render_formula(page, content, x_margin, y_position, max_width)
            elif elem_type == "table":
                y_position = self._render_table(page, content, x_margin, y_position, max_width)

            y_position += 10  # 元素间距

        # 保存
        doc.save(output_path)
        doc.close()

        print(f"✅ PDF已保存: {output_path}")
        return output_path

    def overlay_translation(self, translated_blocks: List[Dict], output_path: str) -> str:
        """
        在原PDF上覆盖翻译文本（保持原有布局）

        Args:
            translated_blocks: 翻译后的文本块列表，每个包含 {page, rect, text}
            output_path: 输出路径

        Returns:
            输出文件路径
        """
        doc = fitz.open(self.original_pdf_path)

        for block in translated_blocks:
            page_num = block.get("page", 0)
            rect = block.get("rect")
            text = block.get("text", "")

            if page_num >= len(doc):
                continue

            page = doc[page_num]

            # 用白色矩形覆盖原文
            if rect:
                rect_obj = fitz.Rect(rect)
                page.draw_rect(rect_obj, color=(1, 1, 1), fill=(1, 1, 1))

                # 插入翻译文本
                fontsize = block.get("fontsize", 10)
                self._insert_text_in_rect(page, rect_obj, text, fontsize)

        doc.save(output_path)
        doc.close()

        print(f"✅ PDF已保存: {output_path}")
        return output_path

    def _parse_markdown(self, md_content: str) -> List[Dict]:
        """解析Markdown为元素列表"""
        elements = []
        lines = md_content.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i]

            # 标题
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if heading_match:
                level = len(heading_match.group(1))
                elements.append({
                    "type": "heading",
                    "level": level,
                    "content": heading_match.group(2)
                })
                i += 1
                continue

            # 代码块
            if line.startswith('```'):
                code_lines = []
                lang = line[3:].strip()
                i += 1
                while i < len(lines) and not lines[i].startswith('```'):
                    code_lines.append(lines[i])
                    i += 1
                elements.append({
                    "type": "code",
                    "language": lang,
                    "content": '\n'.join(code_lines)
                })
                i += 1
                continue

            # 图片
            img_match = re.match(r'!\[([^\]]*)\]\(([^\)]+)\)', line)
            if img_match:
                elements.append({
                    "type": "image",
                    "alt": img_match.group(1),
                    "src": img_match.group(2),
                    "content": line
                })
                i += 1
                continue

            # 公式块
            if line.strip().startswith('$$'):
                formula_lines = [line]
                if not line.strip().endswith('$$') or line.strip() == '$$':
                    i += 1
                    while i < len(lines) and '$$' not in lines[i]:
                        formula_lines.append(lines[i])
                        i += 1
                    if i < len(lines):
                        formula_lines.append(lines[i])
                elements.append({
                    "type": "formula",
                    "content": '\n'.join(formula_lines)
                })
                i += 1
                continue

            # 表格
            if '|' in line and i + 1 < len(lines) and re.match(r'^[\|\s\-:]+$', lines[i + 1]):
                table_lines = [line]
                i += 1
                while i < len(lines) and '|' in lines[i]:
                    table_lines.append(lines[i])
                    i += 1
                elements.append({
                    "type": "table",
                    "content": '\n'.join(table_lines)
                })
                continue

            # 普通段落
            if line.strip():
                para_lines = [line]
                i += 1
                while i < len(lines) and lines[i].strip() and not lines[i].startswith('#'):
                    if lines[i].startswith('```') or lines[i].startswith('$$'):
                        break
                    if re.match(r'!\[.*\]\(.*\)', lines[i]):
                        break
                    para_lines.append(lines[i])
                    i += 1
                elements.append({
                    "type": "paragraph",
                    "content": ' '.join(para_lines)
                })
                continue

            i += 1

        return elements

    def _estimate_height(self, element: Dict, max_width: float) -> float:
        """估算元素高度"""
        elem_type = element["type"]
        content = element.get("content", "")

        if elem_type == "heading":
            return 30 + element.get("level", 1) * 5
        elif elem_type == "paragraph":
            # 估算行数
            chars_per_line = max_width / 8
            lines = len(content) / chars_per_line + 1
            return lines * 14
        elif elem_type == "image":
            return 200
        elif elem_type == "code":
            return len(content.split('\n')) * 12 + 20
        elif elem_type == "formula":
            return 50
        elif elem_type == "table":
            return len(content.split('\n')) * 20 + 10

        return 20

    def _render_heading(self, page, element: Dict, x: float, y: float, max_width: float) -> float:
        """渲染标题"""
        level = element.get("level", 1)
        content = element["content"]

        # 根据级别设置字体大小
        font_sizes = {1: 24, 2: 20, 3: 16, 4: 14, 5: 12, 6: 11}
        fontsize = font_sizes.get(level, 12)

        # 插入文本
        text_point = fitz.Point(x, y + fontsize)

        if self.font_path:
            page.insert_text(text_point, content, fontsize=fontsize,
                           fontfile=self.font_path, fontname="chinese")
        else:
            page.insert_text(text_point, content, fontsize=fontsize)

        return y + fontsize + 10

    def _render_paragraph(self, page, content: str, x: float, y: float, max_width: float) -> float:
        """渲染段落"""
        fontsize = 11
        rect = fitz.Rect(x, y, x + max_width, y + 500)

        # 清理Markdown格式
        clean_text = self._clean_markdown(content)

        if self.font_path:
            rc = page.insert_textbox(rect, clean_text, fontsize=fontsize,
                                    fontfile=self.font_path, fontname="chinese",
                                    align=fitz.TEXT_ALIGN_LEFT)
        else:
            rc = page.insert_textbox(rect, clean_text, fontsize=fontsize,
                                    align=fitz.TEXT_ALIGN_LEFT)

        # 计算实际使用的高度
        lines = len(clean_text) / (max_width / 7) + 1
        return y + lines * 14

    def _render_image(self, page, element: Dict, x: float, y: float,
                     max_width: float, images: Dict = None) -> float:
        """渲染图片"""
        src = element.get("src", "")

        # 尝试从base64或文件加载图片
        img_data = None

        if images and src in images:
            img_data = images[src]
        elif src.startswith("data:image"):
            # base64图片
            try:
                base64_data = src.split(",")[1]
                img_data = base64.b64decode(base64_data)
            except:
                pass
        elif os.path.exists(src):
            with open(src, "rb") as f:
                img_data = f.read()

        if img_data:
            try:
                # 插入图片
                img_rect = fitz.Rect(x, y, x + min(max_width, 400), y + 200)
                page.insert_image(img_rect, stream=img_data)
                return y + 210
            except Exception as e:
                print(f"⚠️ 图片插入失败: {e}")

        # 如果无法加载图片，显示占位符
        page.insert_text(fitz.Point(x, y + 15), f"[图片: {element.get('alt', src)}]", fontsize=10)
        return y + 30

    def _render_code(self, page, content: str, x: float, y: float, max_width: float) -> float:
        """渲染代码块"""
        fontsize = 9

        # 绘制背景
        lines = content.split('\n')
        height = len(lines) * 12 + 20
        rect = fitz.Rect(x, y, x + max_width, y + height)
        page.draw_rect(rect, color=(0.9, 0.9, 0.9), fill=(0.95, 0.95, 0.95))

        # 插入代码文本
        text_y = y + 15
        for line in lines:
            page.insert_text(fitz.Point(x + 10, text_y), line, fontsize=fontsize,
                           fontname="courier")
            text_y += 12

        return y + height + 5

    def _render_formula(self, page, content: str, x: float, y: float, max_width: float) -> float:
        """渲染公式（简单显示LaTeX源码）"""
        fontsize = 10

        # 清理$$符号
        clean_formula = content.replace('$$', '').strip()

        page.insert_text(fitz.Point(x, y + fontsize), f"[公式] {clean_formula}", fontsize=fontsize)
        return y + 30

    def _render_table(self, page, content: str, x: float, y: float, max_width: float) -> float:
        """渲染表格"""
        lines = content.split('\n')
        fontsize = 9
        row_height = 18
        current_y = y

        for i, line in enumerate(lines):
            # 跳过分隔行
            if re.match(r'^[\|\s\-:]+$', line):
                continue

            cells = [c.strip() for c in line.split('|') if c.strip()]
            if not cells:
                continue

            cell_width = max_width / len(cells)

            for j, cell in enumerate(cells):
                cell_x = x + j * cell_width

                # 绘制单元格边框
                cell_rect = fitz.Rect(cell_x, current_y, cell_x + cell_width, current_y + row_height)
                page.draw_rect(cell_rect, color=(0.7, 0.7, 0.7))

                # 插入文本
                if self.font_path:
                    page.insert_text(fitz.Point(cell_x + 5, current_y + 13), cell[:20],
                                   fontsize=fontsize, fontfile=self.font_path, fontname="chinese")
                else:
                    page.insert_text(fitz.Point(cell_x + 5, current_y + 13), cell[:20],
                                   fontsize=fontsize)

            current_y += row_height

        return current_y + 5

    def _insert_text_in_rect(self, page, rect: fitz.Rect, text: str, fontsize: float):
        """在矩形区域内插入文本"""
        if self.font_path:
            page.insert_textbox(rect, text, fontsize=fontsize,
                              fontfile=self.font_path, fontname="chinese",
                              align=fitz.TEXT_ALIGN_LEFT)
        else:
            page.insert_textbox(rect, text, fontsize=fontsize,
                              align=fitz.TEXT_ALIGN_LEFT)

    def _clean_markdown(self, text: str) -> str:
        """清理Markdown格式符号"""
        # 移除粗体
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        # 移除斜体
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        # 移除行内代码
        text = re.sub(r'`([^`]+)`', r'\1', text)
        # 移除链接，保留文本
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        return text

    def close(self):
        """关闭原始文档"""
        self.original_doc.close()


if __name__ == "__main__":
    # 测试
    test_md = """
# 测试文档

这是一个测试段落，包含一些中文内容。

## 第二节

这里有一些**粗体**和*斜体*文本。

```python
print("Hello World")
```

| 列1 | 列2 | 列3 |
|-----|-----|-----|
| A   | B   | C   |
| D   | E   | F   |
"""

    builder = PDFBuilder("/Users/enithz/Downloads/NC-2023_Sulfate_triple-oxygen-isotope_evidence_confirming_oceanic_oxygenation_570_million_years_ago_2.pdf")
    builder.build_from_markdown(test_md, "/Users/enithz/Desktop/otran/test_output.pdf")
    builder.close()
