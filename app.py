#!/usr/bin/env python3
"""
PDF文档解析查看器 - Flask Web应用
"""
import os
import uuid
import threading
import json
import base64
import requests
from datetime import datetime
from pathlib import Path
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

from flask import Flask, request, jsonify, send_file, send_from_directory, render_template, after_this_request
from flask_cors import CORS

from document_parser import DocumentParser
from config import DEEPLX_API_URL, DEFAULT_SOURCE_LANG, DEFAULT_TARGET_LANG

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'outputs')
FRONTEND_DIST = BASE_DIR / 'frontend' / 'dist'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

tasks = {}
DEEPLX_MAX_CONCURRENCY = int(os.environ.get("DEEPLX_MAX_CONCURRENCY", "4"))

# 文件清理配置（秒）
FILE_MAX_AGE = 24 * 60 * 60  # 24小时


def cleanup_old_files():
    """清理超过24小时的上传文件和输出文件"""
    import time
    current_time = time.time()
    cleaned_count = 0

    # 清理uploads目录
    if os.path.exists(UPLOAD_FOLDER):
        for filename in os.listdir(UPLOAD_FOLDER):
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(filepath):
                file_age = current_time - os.path.getmtime(filepath)
                if file_age > FILE_MAX_AGE:
                    try:
                        os.remove(filepath)
                        cleaned_count += 1
                        print(f"✓ 清理旧文件: {filename}")
                    except Exception as e:
                        print(f"⚠️ 清理文件失败 {filename}: {e}")

    # 清理outputs目录
    if os.path.exists(OUTPUT_FOLDER):
        for filename in os.listdir(OUTPUT_FOLDER):
            filepath = os.path.join(OUTPUT_FOLDER, filename)
            if os.path.isfile(filepath):
                file_age = current_time - os.path.getmtime(filepath)
                if file_age > FILE_MAX_AGE:
                    try:
                        os.remove(filepath)
                        cleaned_count += 1
                        print(f"✓ 清理旧文件: {filename}")
                    except Exception as e:
                        print(f"⚠️ 清理文件失败 {filename}: {e}")

    if cleaned_count > 0:
        print(f"✅ 共清理 {cleaned_count} 个过期文件")


def start_cleanup_scheduler():
    """启动定时清理任务"""
    import time

    def cleanup_loop():
        while True:
            time.sleep(3600)  # 每小时检查一次
            cleanup_old_files()

    cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
    cleanup_thread.start()
    print("🧹 文件清理服务已启动（每小时检查，清理超过24小时的文件）")


class ParseTask:
    def __init__(self, task_id: str, filename: str):
        self.task_id = task_id
        self.filename = filename
        self.status = "pending"
        self.progress = 0
        self.message = "等待处理..."
        self.result = None
        self.error = None
        self.pdf_path = None
        self.created_at = datetime.now()

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "filename": self.filename,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
            "created_at": self.created_at.isoformat()
        }


def process_parse(task_id: str, input_path: str):
    task = tasks.get(task_id)
    if not task:
        return
    try:
        task.status = "parsing"
        task.progress = 10
        task.message = "正在上传文件..."
        parser = DocumentParser()
        task.progress = 30
        task.message = "正在解析文档..."
        result = parser.parse(file_path=input_path)
        task.progress = 90
        task.message = "处理完成"
        task.result = result
        task.pdf_path = input_path  # 保留PDF路径用于预览
        task.status = "completed"
        task.progress = 100
        task.message = "解析完成!"
    except Exception as e:
        task.status = "failed"
        task.error = str(e)
        task.message = f"处理失败: {str(e)}"
        import traceback
        traceback.print_exc()
        # 解析失败时删除上传的文件
        if os.path.exists(input_path):
            try:
                os.remove(input_path)
                print(f"✓ 已删除上传文件: {input_path}")
            except Exception as e2:
                print(f"⚠️ 删除文件失败: {e2}")


def translate_via_deeplx(text: str, source_lang: str, target_lang: str) -> str:
    payload = {
        "text": text,
        "source_lang": source_lang or DEFAULT_SOURCE_LANG,
        "target_lang": target_lang or DEFAULT_TARGET_LANG
    }
    try:
        response = requests.post(
            DEEPLX_API_URL,
            json=payload,
            timeout=60
        )
        if response.status_code == 429:
            raise ValueError("DeepLX接口触发限流，请稍后重试或切换自定义翻译服务。")
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response else None
        if status == 429:
            raise ValueError("DeepLX接口触发限流，请稍后重试或配置自定义OpenAI翻译。")
        raise ValueError(f"DeepLX翻译失败: {exc}")
    except Exception as exc:
        raise ValueError(f"DeepLX翻译调用异常: {exc}")

    if data.get("code") == 200:
        return data.get("data", text)
    raise ValueError(data.get("message", "DeepLX翻译失败"))


def translate_via_openai(text: str, config: dict, source_lang: str, target_lang: str) -> str:
    base_url = (config or {}).get("base_url")
    api_key = (config or {}).get("api_key")
    model = (config or {}).get("model")

    if not base_url or not api_key or not model:
        raise ValueError("缺少OpenAI翻译配置")

    url = base_url.rstrip('/') + "/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    prompt = f"Translate the following text from {source_lang or 'Source Language'} to {target_lang or 'Target Language'}:\n\n{text}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a professional translation engine."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "stream": False
    }

    response = requests.post(url, headers=headers, json=payload, timeout=90)
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices", [])
    if not choices:
        raise ValueError("OpenAI翻译返回为空")
    return choices[0]["message"]["content"].strip()


def batch_translate_via_deeplx(texts, source_lang, target_lang):
    if not texts:
        return []
    results = ["" for _ in texts]
    worker_count = max(1, min(DEEPLX_MAX_CONCURRENCY, len(texts)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {}
        for idx, text in enumerate(texts):
            future = executor.submit(translate_via_deeplx, text, source_lang, target_lang)
            future_map[future] = idx
        for future in as_completed(future_map):
            idx = future_map[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = texts[idx]
    return results


def batch_translate_via_openai(texts, config, source_lang, target_lang):
    results = []
    for text in texts:
        if not text:
            results.append(text)
            continue
        translated = translate_via_openai(text, config, source_lang, target_lang)
        results.append(translated)
    return results


@app.route('/')
def serve_frontend_index():
    if FRONTEND_DIST.joinpath('index.html').exists():
        return send_from_directory(FRONTEND_DIST, 'index.html')
    return render_template('index.html')


@app.route('/viewer')
def render_viewer():
    return render_template('index.html')


@app.route('/assets/<path:filename>')
def serve_frontend_assets(filename):
    assets_dir = FRONTEND_DIST / 'assets'
    if assets_dir.exists():
        return send_from_directory(assets_dir, filename)
    return jsonify({"error": "前端静态资源尚未构建"}), 404


@app.route('/<path:path>')
def serve_frontend_static(path):
    target = FRONTEND_DIST / path
    if target.exists() and target.is_file():
        return send_from_directory(FRONTEND_DIST, path)
    return serve_frontend_index()


@app.route('/api/parse', methods=['POST'])
def parse():
    if 'file' not in request.files:
        return jsonify({"error": "没有上传文件"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "文件名为空"}), 400
    task_id = str(uuid.uuid4())[:8]
    filename = f"{task_id}_{file.filename}"
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_path)
    task = ParseTask(task_id, file.filename)
    tasks[task_id] = task
    thread = threading.Thread(target=process_parse, args=(task_id, input_path))
    thread.daemon = True
    thread.start()
    return jsonify({"task_id": task_id, "message": "任务已创建"})


@app.route('/api/task/<task_id>')
def get_task(task_id):
    task = tasks.get(task_id)
    if not task:
        return jsonify({"error": "任务不存在"}), 404
    return jsonify(task.to_dict())


@app.route('/api/result/<task_id>')
def get_result(task_id):
    task = tasks.get(task_id)
    if not task:
        return jsonify({"error": "任务不存在"}), 404
    if task.status != 'completed' or not task.result:
        return jsonify({"error": "结果未准备好"}), 400
    return jsonify({
        "markdown": task.result.get("markdown", ""),
        "content_list": task.result.get("content_list", []),
        "page_mappings": task.result.get("page_mappings", {})
    })


@app.route('/api/pdf/<task_id>')
def get_pdf(task_id):
    task = tasks.get(task_id)
    if not task:
        return jsonify({"error": "任务不存在"}), 404
    if not task.pdf_path or not os.path.exists(task.pdf_path):
        return jsonify({"error": "PDF文件不存在"}), 404
    return send_file(task.pdf_path, mimetype='application/pdf')


@app.route('/api/image/<task_id>/<image_name>')
def get_image(task_id, image_name):
    task = tasks.get(task_id)
    if not task or not task.result:
        return jsonify({"error": "任务不存在"}), 404
    images = task.result.get("images", {})
    if image_name not in images:
        return jsonify({"error": "图片不存在"}), 404
    return send_file(BytesIO(images[image_name]), mimetype='image/png')


@app.route('/api/translate', methods=['POST'])
def translate_text():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "缺少文本内容"}), 400

    provider = data.get("provider", "deeplx")
    source_lang = data.get("source_lang") or DEFAULT_SOURCE_LANG
    target_lang = data.get("target_lang") or DEFAULT_TARGET_LANG

    try:
        if provider == "openai":
            translated = translate_via_openai(text, data.get("config"), source_lang, target_lang)
        else:
            translated = translate_via_deeplx(text, source_lang, target_lang)
        return jsonify({"translated_text": translated})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/translate_batch', methods=['POST'])
def translate_batch():
    data = request.get_json(force=True)
    chunks = data.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        return jsonify({"error": "缺少翻译内容"}), 400

    texts = [(chunk.get("text") or "") for chunk in chunks]
    provider = data.get("provider", "deeplx")
    source_lang = data.get("source_lang") or DEFAULT_SOURCE_LANG
    target_lang = data.get("target_lang") or DEFAULT_TARGET_LANG

    try:
        if provider == "openai":
            translations = batch_translate_via_openai(texts, data.get("config"), source_lang, target_lang)
        else:
            translations = batch_translate_via_deeplx(texts, source_lang, target_lang)
        return jsonify({"translations": translations})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/download_pdf/<task_id>', methods=['POST'])
def download_translated_pdf(task_id):
    """下载翻译后的PDF文件"""
    task = tasks.get(task_id)
    if not task:
        return jsonify({"error": "任务不存在"}), 404
    if task.status != 'completed' or not task.result:
        return jsonify({"error": "任务未完成"}), 400

    data = request.get_json(force=True) if request.is_json else {}
    content_list = data.get("content_list", [])

    try:
        from pdf_generator import markdown_to_pdf, content_list_to_markdown

        # 获取图片的辅助函数
        def get_image_data(tid, image_name):
            t = tasks.get(tid)
            if t and t.result:
                imgs = t.result.get("images", {})
                return imgs.get(image_name)
            return None

        # 构建Markdown内容
        if content_list:
            # 使用前端传来的翻译后内容
            markdown_content = content_list_to_markdown(content_list, task.result.get("images", {}))
        else:
            # 使用原始Markdown
            markdown_content = task.result.get("markdown", "")

        if not markdown_content:
            return jsonify({"error": "没有可导出的内容"}), 400

        # 生成PDF
        output_filename = f"translated_{task_id}.pdf"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        markdown_to_pdf(
            markdown_content,
            output_path,
            images=task.result.get("images", {}),
            task_id=task_id,
            get_image_func=get_image_data
        )

        # 在响应完成后删除生成的PDF文件
        @after_this_request
        def cleanup(response):
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
                    print(f"✓ 已删除临时PDF: {output_path}")
            except Exception as e:
                print(f"⚠️ 删除临时PDF失败: {e}")
            return response

        return send_file(
            output_path,
            as_attachment=True,
            download_name=f"translated_{task.filename}",
            mimetype='application/pdf'
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"PDF生成失败: {str(e)}"}), 500


# 启动时执行一次清理并启动定时清理服务
cleanup_old_files()
start_cleanup_scheduler()


if __name__ == '__main__':
    print("=" * 50)
    print("📄 PDF文档解析查看器")
    print("=" * 50)
    print("🌐 访问地址: http://localhost:8080")
    print("=" * 50)
    app.run(host='0.0.0.0', port=8080, debug=True)
