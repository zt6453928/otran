"""
PDF生成模块 - 使用ReportLab将Markdown转换为PDF
参考pdfnew项目实现，支持公式Unicode转换和图片嵌入
"""
import os
import re
import base64
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase.cidfonts import UnicodeCIDFont


def clean_unicode_characters(text, debug=False):
    """清理文本中无法显示的Unicode字符"""
    import unicodedata

    if not text:
        return text

    original_text = text

    # 字符替换规则
    unicode_replacements = {
        '\ufffd': '',  # REPLACEMENT CHARACTER
        '\u0000': '',  # NULL
        '\ufeff': '',  # BOM
    }

    for old_char, new_char in unicode_replacements.items():
        text = text.replace(old_char, new_char)

    # 查找并处理剩余的问题字符
    cleaned_text = []
    removed_chars = set()

    for char in text:
        code_point = ord(char)

        # REPLACEMENT CHARACTER
        if code_point == 0xFFFD:
            removed_chars.add((char, code_point, 'REPLACEMENT CHARACTER'))
            continue

        # 私有使用区字符
        if (0xE000 <= code_point <= 0xF8FF or
            0xF0000 <= code_point <= 0xFFFFD or
            0x100000 <= code_point <= 0x10FFFD):
            try:
                char_name = unicodedata.name(char, f'PRIVATE_USE_U+{code_point:04X}')
            except:
                char_name = f'PRIVATE_USE_U+{code_point:04X}'
            removed_chars.add((char, code_point, char_name))
            continue

        # 控制字符（除了常见的换行、制表符等）
        if (0x00 <= code_point <= 0x1F and
            code_point not in [0x09, 0x0A, 0x0D]):
            removed_chars.add((char, code_point, 'CONTROL_CHARACTER'))
            continue

        cleaned_text.append(char)

    result = ''.join(cleaned_text)

    if debug and removed_chars:
        print(f"清理了 {len(removed_chars)} 种问题字符", flush=True)

    return result


def convert_latex_to_unicode(text):
    """将常见的 LaTeX 数学符号转换为 Unicode"""
    if not text:
        return text

    original_text = text

    # 上标数字映射
    superscripts = {
        '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
        '+': '⁺', '-': '⁻', '=': '⁼', '(': '⁽', ')': '⁾'
    }

    # 下标数字映射
    subscripts = {
        '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
        '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
        '+': '₊', '-': '₋', '=': '₌', '(': '₍', ')': '₎'
    }

    # 希腊字母映射
    greek_letters = {
        r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\delta': 'δ',
        r'\Delta': 'Δ', r'\epsilon': 'ε', r'\zeta': 'ζ', r'\eta': 'η',
        r'\theta': 'θ', r'\Theta': 'Θ', r'\lambda': 'λ', r'\Lambda': 'Λ',
        r'\mu': 'μ', r'\nu': 'ν', r'\xi': 'ξ', r'\pi': 'π', r'\Pi': 'Π',
        r'\rho': 'ρ', r'\sigma': 'σ', r'\Sigma': 'Σ', r'\tau': 'τ',
        r'\phi': 'φ', r'\Phi': 'Φ', r'\chi': 'χ', r'\psi': 'ψ', r'\Psi': 'Ψ',
        r'\omega': 'ω', r'\Omega': 'Ω'
    }

    # 其他常用符号
    symbols = {
        r'\sim': '∼', r'\approx': '≈', r'\pm': '±', r'\times': '×',
        r'\div': '÷', r'\leq': '≤', r'\geq': '≥', r'\neq': '≠',
        r'\infty': '∞', r'\sum': '∑', r'\prod': '∏', r'\int': '∫',
        r'\partial': '∂', r'\nabla': '∇', r'\cdot': '·'
    }

    # 首先处理希腊字母和符号
    for latex, unicode_char in greek_letters.items():
        text = text.replace(latex, unicode_char)
    for latex, unicode_char in symbols.items():
        text = text.replace(latex, unicode_char)

    # 处理上标 ^{...}
    def replace_superscript(match):
        content = match.group(1)
        result = ''
        for char in content:
            result += superscripts.get(char, char)
        return result

    text = re.sub(r'\^{([^}]+)}', replace_superscript, text)

    # 处理简单上标 ^x
    def replace_simple_superscript(match):
        char = match.group(1)
        return superscripts.get(char, '^' + char)

    text = re.sub(r'\^([0-9+-])', replace_simple_superscript, text)

    # 处理下标 _{...}
    def replace_subscript(match):
        content = match.group(1)
        result = ''
        for char in content:
            result += subscripts.get(char, char)
        return result

    text = re.sub(r'_{([^}]+)}', replace_subscript, text)

    # 处理简单下标 _x
    def replace_simple_subscript(match):
        char = match.group(1)
        return subscripts.get(char, '_' + char)

    text = re.sub(r'_([0-9+-])', replace_simple_subscript, text)

    # 处理 \mathrm{...}
    text = re.sub(r'\\mathrm{([^}]+)}', r'\1', text)

    # 移除剩余的反斜杠命令
    text = re.sub(r'\\[a-zA-Z]+', '', text)

    # 移除多余的花括号
    text = text.replace('{', '').replace('}', '')

    # 清理可能残留的问题Unicode字符
    text = clean_unicode_characters(text, debug=False)

    return text


def markdown_to_pdf(markdown_text, output_path, images=None, task_id=None, get_image_func=None):
    """
    将Markdown转换为PDF

    Args:
        markdown_text: Markdown文本内容
        output_path: 输出PDF路径
        images: 图片字典 {图片名: 图片数据bytes}
        task_id: 任务ID（用于获取API图片）
        get_image_func: 获取图片的函数 func(task_id, image_name) -> bytes
    """
    print("\n" + "="*50, flush=True)
    print("开始生成PDF...", flush=True)
    print(f"原始文本长度: {len(markdown_text)}", flush=True)
    print("="*50, flush=True)

    # 清理文本
    markdown_text = clean_unicode_characters(markdown_text, debug=True)

    # 创建PDF文档
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)

    story = []
    styles = getSampleStyleSheet()

    # 注册字体
    font_registered = False
    font_name = 'Helvetica'
    fallback_font_name = None

    # 字体目录
    local_font_dir = os.path.join(os.path.dirname(__file__), 'fonts')
    font_paths = []

    # 项目内置字体
    if os.path.isdir(local_font_dir):
        for candidate in [
            'NotoSansSC-Regular.ttf',
            'DejaVuSans.ttf',
        ]:
            font_paths.append(os.path.join(local_font_dir, candidate))

    # 系统字体
    font_paths.extend([
        '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
        '/Library/Fonts/Arial Unicode.ttf',
        '/System/Library/Fonts/PingFang.ttc',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    ])

    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                pdfmetrics.registerFont(TTFont('UnicodeFont', font_path))
                font_registered = True
                print(f"✓ 成功注册字体: {font_path}")
                font_name = 'UnicodeFont'
                break
        except Exception as e:
            print(f"⚠️ 无法注册字体 {font_path}: {e}")
            continue

    if not font_registered:
        try:
            pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
            font_name = 'STSong-Light'
            print("✓ 使用内置CID字体: STSong-Light")
        except:
            font_name = 'Helvetica'
            print("⚠️ 无法加载Unicode字体，使用默认字体")

    # 注册备用字体（用于数学符号）
    fallback_candidates = []
    if os.path.isdir(local_font_dir):
        fallback_candidates.extend([
            os.path.join(local_font_dir, 'DejaVuSans.ttf'),
            os.path.join(local_font_dir, 'NotoSansMath-Regular.ttf'),
        ])
    fallback_candidates.extend([
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ])

    for fp in fallback_candidates:
        try:
            if os.path.exists(fp):
                pdfmetrics.registerFont(TTFont('UnicodeFallback', fp))
                fallback_font_name = 'UnicodeFallback'
                print(f"✓ 成功注册备用字体: {fp}")
                break
        except Exception as e:
            print(f"⚠️ 无法注册备用字体 {fp}: {e}")
            continue

    # HTML转义和备用字体应用函数
    def _escape_html(s):
        return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    def _apply_supsub_fallback(s):
        t = _escape_html(s)
        if not fallback_font_name:
            return t

        def need_fallback(cp):
            # 上/下标
            if cp in (0x00B2, 0x00B3, 0x00B9) or 0x2070 <= cp <= 0x209F or 0x2080 <= cp <= 0x208E:
                return True
            # 希腊字母
            if 0x0370 <= cp <= 0x03FF:
                return True
            # 数学符号
            if 0x2100 <= cp <= 0x214F or 0x2190 <= cp <= 0x21FF or 0x2200 <= cp <= 0x22FF:
                return True
            return False

        out = []
        open_tag = False
        for ch in t:
            cp = ord(ch)
            if need_fallback(cp):
                if not open_tag:
                    out.append(f'<font face="{fallback_font_name}">')
                    open_tag = True
                out.append(ch)
            else:
                if open_tag:
                    out.append('</font>')
                    open_tag = False
                out.append(ch)
        if open_tag:
            out.append('</font>')
        return ''.join(out)

    # 创建样式
    chinese_style = ParagraphStyle(
        'Chinese',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=11,
        leading=16,
        alignment=TA_JUSTIFY,
        wordWrap='CJK',
    )
    chinese_title = ParagraphStyle(
        'ChineseTitle',
        parent=styles['Title'],
        fontName=font_name,
        fontSize=20,
        leading=28,
        alignment=TA_CENTER,
        spaceAfter=20,
    )
    chinese_heading1 = ParagraphStyle(
        'ChineseHeading1',
        parent=styles['Heading1'],
        fontName=font_name,
        fontSize=16,
        leading=22,
        spaceAfter=12,
        spaceBefore=12,
    )
    chinese_heading2 = ParagraphStyle(
        'ChineseHeading2',
        parent=styles['Heading2'],
        fontName=font_name,
        fontSize=14,
        leading=20,
        spaceAfter=10,
        spaceBefore=10,
    )

    # 处理数学公式的函数
    def replace_math(match):
        latex = match.group(1)
        return convert_latex_to_unicode(latex)

    # 处理图片的函数
    def process_image(line):
        """处理图片行，返回Image对象或None"""
        match = re.match(r'!\[(.*?)\]\((.*?)\)', line)
        if not match:
            return None, None

        alt_text = match.group(1)
        image_src = match.group(2)

        img_data = None

        # 1. 检查是否是base64图片
        if image_src.startswith('data:image'):
            base64_match = re.search(r'base64,(.+)', image_src)
            if base64_match:
                try:
                    img_data = base64.b64decode(base64_match.group(1))
                    print(f"✓ 解码base64图片: {alt_text}")
                except Exception as e:
                    print(f"⚠️ base64解码失败: {e}")

        # 2. 检查是否是API图片路径 (如 /api/image/xxx/yyy.png)
        elif '/api/image/' in image_src and task_id and get_image_func:
            # 提取图片名
            parts = image_src.split('/')
            if len(parts) >= 2:
                image_name = parts[-1]
                try:
                    img_data = get_image_func(task_id, image_name)
                    if img_data:
                        print(f"✓ 从API获取图片: {image_name}")
                except Exception as e:
                    print(f"⚠️ 获取API图片失败: {e}")

        # 3. 检查images字典
        elif images:
            # 尝试从images字典获取
            image_name = image_src.split('/')[-1] if '/' in image_src else image_src
            if image_name in images:
                img_data = images[image_name]
                print(f"✓ 从字典获取图片: {image_name}")
            # 也尝试完整路径
            elif image_src in images:
                img_data = images[image_src]
                print(f"✓ 从字典获取图片: {image_src}")

        # 4. 检查是否是本地文件
        elif os.path.exists(image_src):
            try:
                with open(image_src, 'rb') as f:
                    img_data = f.read()
                print(f"✓ 读取本地图片: {image_src}")
            except Exception as e:
                print(f"⚠️ 读取本地图片失败: {e}")

        if img_data:
            try:
                img_buffer = BytesIO(img_data)
                img = Image(img_buffer)

                # 调整图片大小
                max_width = 6 * inch
                max_height = 8 * inch
                if img.drawWidth > max_width:
                    ratio = max_width / img.drawWidth
                    img.drawWidth = max_width
                    img.drawHeight = img.drawHeight * ratio
                if img.drawHeight > max_height:
                    ratio = max_height / img.drawHeight
                    img.drawHeight = max_height
                    img.drawWidth = img.drawWidth * ratio

                return img, alt_text
            except Exception as e:
                print(f"⚠️ 创建图片对象失败: {e}")

        return None, alt_text

    # 处理Markdown文本
    lines = markdown_text.split('\n')
    image_count = 0

    for line in lines:
        line = line.strip()

        if not line:
            story.append(Spacer(1, 0.15 * inch))
            continue

        # 处理标题
        if line.startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            text = line.lstrip('#').strip()

            # 处理数学公式
            text = re.sub(r'\$([^\$]+)\$', replace_math, text)
            text = re.sub(r'\$\$([^\$]+)\$\$', replace_math, text)

            # 应用备用字体
            text = _apply_supsub_fallback(text)

            if level == 1:
                p = Paragraph(text, chinese_title)
                story.append(p)
                story.append(Spacer(1, 0.3 * inch))
            elif level == 2:
                p = Paragraph(text, chinese_heading1)
                story.append(p)
                story.append(Spacer(1, 0.2 * inch))
            else:
                p = Paragraph(text, chinese_heading2)
                story.append(p)
                story.append(Spacer(1, 0.15 * inch))

        # 处理图片
        elif line.startswith('!['):
            img, alt_text = process_image(line)
            if img:
                story.append(img)
                story.append(Spacer(1, 0.1 * inch))
                image_count += 1
            elif alt_text:
                p = Paragraph(f"[图片: {_escape_html(alt_text)}]", chinese_style)
                story.append(p)
                story.append(Spacer(1, 0.08 * inch))

        # 跳过HTML img标签
        elif line.startswith('<img'):
            print(f"⚠️ 跳过HTML img标签")
            continue

        # 处理普通文本
        else:
            text = line

            # 处理数学公式
            text = re.sub(r'\$([^\$]+)\$', replace_math, text)
            text = re.sub(r'\$\$([^\$]+)\$\$', replace_math, text)

            # 应用备用字体
            text = _apply_supsub_fallback(text)

            # 处理Markdown格式
            text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
            text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
            text = re.sub(r'`(.+?)`', r'<font face="Courier">\1</font>', text)

            if text.strip():
                try:
                    p = Paragraph(text, chinese_style)
                    story.append(p)
                    story.append(Spacer(1, 0.08 * inch))
                except Exception as e:
                    print(f"⚠️ 处理段落出错: {e}")
                    continue

    # 生成PDF
    try:
        doc.build(story)
        print(f"✓ PDF生成成功: {output_path}")
        print(f"  共处理 {image_count} 张图片")
    except Exception as e:
        print(f"❌ PDF生成失败: {e}")
        raise

    return output_path


def content_list_to_markdown(content_list, images=None):
    """
    将content_list转换为Markdown文本

    Args:
        content_list: 内容列表
        images: 图片字典

    Returns:
        Markdown文本
    """
    md_parts = []

    for item in content_list:
        item_type = item.get('type', '')

        if item_type == 'image':
            # 处理图片
            img_path = item.get('img_path', '')
            alt_text = ' '.join(item.get('image_caption', [])) or '图片'
            if img_path:
                md_parts.append(f"![{alt_text}]({img_path})")
        else:
            # 处理文本（优先使用翻译后的文本）
            text = item.get('translated_text') or item.get('text') or ''
            if text:
                # 根据text_level添加标题标记
                text_level = item.get('text_level')
                if text_level and isinstance(text_level, int) and 1 <= text_level <= 6:
                    text = '#' * text_level + ' ' + text
                md_parts.append(text)

    return '\n\n'.join(md_parts)
