#!/usr/bin/env python3
"""
PDF文档解析查看器 - Flask Web应用
"""
import os
import uuid
import threading
import json
import base64
import shutil
import time
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
DEEPLX_TIMEOUT = float(os.environ.get("DEEPLX_TIMEOUT", "30"))
DEEPLX_MAX_RETRIES = int(os.environ.get("DEEPLX_MAX_RETRIES", "2"))
DEEPLX_RATE_LIMIT = float(os.environ.get("DEEPLX_RATE_LIMIT", "0"))

# 文件清理配置（秒）
FILE_MAX_AGE = 24 * 60 * 60  # 24小时


def cleanup_old_files():
    """清理超过24小时的上传文件和输出文件"""
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
        # 将PDF复制到outputs目录，使用固定命名便于查找
        task_output_dir = os.path.join(OUTPUT_FOLDER, task_id)
        os.makedirs(task_output_dir, exist_ok=True)
        pdf_dest = os.path.join(task_output_dir, "original.pdf")
        shutil.copy2(input_path, pdf_dest)
        task.pdf_path = pdf_dest
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
    if not DEEPLX_API_URL:
        raise ValueError("DEEPLX_API_URL 未配置")

    payload = {
        "text": text,
        "source_lang": source_lang or DEFAULT_SOURCE_LANG,
        "target_lang": target_lang or DEFAULT_TARGET_LANG
    }

    last_error = None
    for attempt in range(1, DEEPLX_MAX_RETRIES + 1):
        try:
            if attempt > 1:
                wait_time = min(2 ** (attempt - 1), 5)
                print(f"⏳ DeepLX重试等待 {wait_time} 秒（第 {attempt}/{DEEPLX_MAX_RETRIES} 次）...", flush=True)
                time.sleep(wait_time)

            start_ts = time.time()
            response = requests.post(
                DEEPLX_API_URL,
                json=payload,
                timeout=(5, DEEPLX_TIMEOUT)
            )
            elapsed = time.time() - start_ts
            print(f"🔁 DeepLX响应: status={response.status_code} time={elapsed:.2f}s", flush=True)

            if response.status_code == 429:
                raise ValueError("DeepLX接口触发限流(429)，请稍后重试或降低批量大小/频率。")

            response.raise_for_status()
            data = response.json()

            if data.get("code") == 200:
                return data.get("data", text)

            raise ValueError(data.get("message") or data.get("msg") or "DeepLX翻译失败")
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_error = exc
            print(f"⚠️ DeepLX请求异常: {exc}", flush=True)
            continue
        except Exception as exc:
            last_error = exc
            print(f"⚠️ DeepLX翻译错误: {exc}", flush=True)
            continue

    raise ValueError(f"DeepLX翻译失败: {last_error}")


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

    # 快速连通性检测：避免翻译服务不可用时请求长时间卡住
    sample = next((t for t in texts if (t or "").strip()), "")
    if sample:
        try:
            requests.post(
                DEEPLX_API_URL,
                json={
                    "text": (sample[:50] if sample else "ping"),
                    "source_lang": source_lang or DEFAULT_SOURCE_LANG,
                    "target_lang": target_lang or DEFAULT_TARGET_LANG,
                },
                # 该检测只用于快速提示，不应成为硬失败条件；某些实例首包较慢
                timeout=(3, min(15, DEEPLX_TIMEOUT))
            )
        except Exception as exc:
            print(f"⚠️ DeepLX连通性检测失败（将继续尝试翻译）: {exc}", flush=True)

    results = []
    success_count = 0
    last_error = None
    for idx, text in enumerate(texts, start=1):
        if not text:
            results.append(text)
            continue
        try:
            translated = translate_via_deeplx(text, source_lang, target_lang)
            results.append(translated)
            success_count += 1
        except Exception as exc:
            last_error = exc
            results.append(text)
        if DEEPLX_RATE_LIMIT > 0 and idx < len(texts):
            time.sleep(DEEPLX_RATE_LIMIT)

    if success_count == 0 and last_error is not None:
        raise ValueError(str(last_error))

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
    # 优先从任务对象获取
    task = tasks.get(task_id)
    if task and task.pdf_path and os.path.exists(task.pdf_path):
        print(f"✓ 返回PDF文件(从任务): {task.pdf_path}")
        return send_file(task.pdf_path, mimetype='application/pdf')
    # 任务对象不存在时，尝试从outputs目录查找
    pdf_path = os.path.join(OUTPUT_FOLDER, task_id, "original.pdf")
    if os.path.exists(pdf_path):
        print(f"✓ 返回PDF文件(从outputs): {pdf_path}")
        return send_file(pdf_path, mimetype='application/pdf')
    print(f"⚠️ PDF请求失败: 任务 {task_id} 的PDF文件不存在")
    return jsonify({"error": "PDF文件不存在"}), 404


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

    provider_raw = data.get("provider", "deeplx")
    provider = (provider_raw or "").strip().lower()
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
    provider_raw = data.get("provider", "deeplx")
    provider = (provider_raw or "").strip().lower()
    source_lang = data.get("source_lang") or DEFAULT_SOURCE_LANG
    target_lang = data.get("target_lang") or DEFAULT_TARGET_LANG
    print(
        f"🌍 translate_batch provider_raw={provider_raw!r} provider={provider} chunks={len(texts)} source={source_lang} target={target_lang}",
        flush=True
    )

    try:
        if provider == "openai":
            print("➡️ using openai", flush=True)
            translations = batch_translate_via_openai(texts, data.get("config"), source_lang, target_lang)
        else:
            print("➡️ using deeplx", flush=True)
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
