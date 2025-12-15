"""
文档解析模块 - 调用MinerU API
"""
import requests
import os
import time
import json
import zipfile
import tempfile
from io import BytesIO

from config import MINERU_API_URL, MINERU_API_TOKEN


class DocumentParser:
    def __init__(self):
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {MINERU_API_TOKEN}"
        }

    def parse(self, file_path: str = None, file_url: str = None) -> dict:
        """
        解析文档并返回结果

        Args:
            file_path: 本地文件路径
            file_url: 文件URL（二选一）

        Returns:
            解析结果字典，包含markdown内容和位置信息
        """
        print(f"📄 开始解析文档...")

        if file_url:
            # 通过URL直接解析
            task_id = self._create_task_by_url(file_url)
            if not task_id:
                raise ValueError("创建任务失败")
            print(f"📋 任务ID: {task_id}")
            return self._poll_task(task_id)

        elif file_path:
            # 本地文件需要先上传
            batch_id = self._upload_file(file_path)
            if not batch_id:
                raise ValueError("文件上传失败")
            print(f"📋 批次ID: {batch_id}")
            return self._poll_batch(batch_id)

        else:
            raise ValueError("必须提供file_path或file_url")

    def _create_task_by_url(self, file_url: str) -> str:
        """通过URL创建解析任务"""
        url = "https://mineru.net/api/v4/extract/task"
        payload = {
            "url": file_url,
            "is_ocr": True,
            "enable_formula": True,
            "enable_table": True,
            "model_version": "vlm"
        }

        response = requests.post(url, headers=self.headers, json=payload)
        result = response.json()

        print(f"创建任务响应: code={result.get('code')}, msg={result.get('msg')}")

        if result.get("code") == 0:
            return result.get("data", {}).get("task_id")

        print(f"创建任务失败: {result}")
        return None

    def _upload_file(self, file_path: str) -> str:
        """上传本地文件"""
        url = "https://mineru.net/api/v4/file-urls/batch"
        filename = os.path.basename(file_path)

        payload = {
            "files": [{"name": filename}],
            "is_ocr": True,
            "enable_formula": True,
            "enable_table": True,
            "model_version": "vlm"
        }

        import sys
        print(f"🔑 当前Token: {MINERU_API_TOKEN[:30] if MINERU_API_TOKEN else '(未设置)'}...", flush=True)
        sys.stdout.flush()

        response = requests.post(url, headers=self.headers, json=payload)

        # 添加调试信息
        print(f"API响应状态码: {response.status_code}", flush=True)
        print(f"API响应内容: {response.text[:500] if response.text else '(空)'}", flush=True)
        sys.stdout.flush()

        if not response.text:
            raise ValueError(f"MinerU API返回空响应，状态码: {response.status_code}。请检查MINERU_API_TOKEN是否正确。")

        result = response.json()

        print(f"申请上传URL响应: code={result.get('code')}, msg={result.get('msg')}")

        if result.get("code") != 0:
            print(f"申请上传URL失败: {result}")
            return None

        batch_id = result.get("data", {}).get("batch_id")
        file_urls = result.get("data", {}).get("file_urls", [])

        if not file_urls:
            print("未获取到上传URL")
            return None

        upload_url = file_urls[0]
        print(f"📤 上传文件到: {upload_url[:50]}...")

        # 上传文件（注意：不需要设置Content-Type）
        with open(file_path, "rb") as f:
            upload_headers = {}  # 上传时不需要Content-Type
            upload_response = requests.put(upload_url, data=f, headers=upload_headers)

            if upload_response.status_code == 200:
                print(f"✅ 文件上传成功")
                return batch_id
            else:
                print(f"❌ 文件上传失败: {upload_response.status_code}")
                return None

    def _poll_task(self, task_id: str, timeout: int = 1800, interval: int = 5) -> dict:
        """轮询单个任务状态"""
        url = f"https://mineru.net/api/v4/extract/task/{task_id}"
        max_attempts = timeout // interval

        for attempt in range(1, max_attempts + 1):
            print(f"⏳ 检查任务状态 [{attempt}/{max_attempts}]...", end=" ")

            response = requests.get(url, headers=self.headers)
            result = response.json()

            if result.get("code") != 0:
                print(f"查询失败: {result.get('msg')}")
                time.sleep(interval)
                continue

            data = result.get("data", {})
            state = data.get("state", "unknown")

            # 显示进度
            progress = data.get("extract_progress", {})
            if progress:
                extracted = progress.get("extracted_pages", 0)
                total = progress.get("total_pages", 0)
                print(f"{state} ({extracted}/{total}页)")
            else:
                print(state)

            if state == "done":
                return self._download_result(data)
            elif state in ["failed", "error"]:
                raise ValueError(f"任务失败: {data.get('err_msg', 'Unknown error')}")

            time.sleep(interval)

        raise TimeoutError("任务超时")

    def _poll_batch(self, batch_id: str, timeout: int = 1800, interval: int = 5) -> dict:
        """轮询批量任务状态"""
        url = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"
        max_attempts = timeout // interval

        for attempt in range(1, max_attempts + 1):
            print(f"⏳ 检查批量任务状态 [{attempt}/{max_attempts}]...", end=" ")

            response = requests.get(url, headers=self.headers)
            result = response.json()

            if result.get("code") != 0:
                print(f"查询失败: {result.get('msg')}")
                time.sleep(interval)
                continue

            data = result.get("data", {})
            extract_results = data.get("extract_result", [])

            if not extract_results:
                print("等待文件处理...")
                time.sleep(interval)
                continue

            # 检查第一个文件的状态
            file_result = extract_results[0]
            state = file_result.get("state", "unknown")

            # 显示进度
            progress = file_result.get("extract_progress", {})
            if progress:
                extracted = progress.get("extracted_pages", 0)
                total = progress.get("total_pages", 0)
                print(f"{state} ({extracted}/{total}页)")
            else:
                print(state)

            if state == "done":
                return self._download_result(file_result)
            elif state in ["failed", "error"]:
                raise ValueError(f"任务失败: {file_result.get('err_msg', 'Unknown error')}")

            time.sleep(interval)

        raise TimeoutError("任务超时")

    def _download_result(self, data: dict) -> dict:
        """下载并解析结果"""
        full_zip_url = data.get("full_zip_url")

        if not full_zip_url:
            raise ValueError("未获取到结果文件URL")

        print(f"📥 下载结果: {full_zip_url[:60]}...")

        response = requests.get(full_zip_url)
        if response.status_code != 200:
            raise ValueError(f"下载失败: {response.status_code}")

        result = {
            "type": "mineru",
            "markdown": "",
            "content_list": [],
            "layout_info": [],
            "images": {},
            "page_mappings": {}
        }

        # 解压并读取内容
        with zipfile.ZipFile(BytesIO(response.content)) as zf:
            file_list = zf.namelist()
            print(f"📦 解压文件: {file_list}")

            for filename in file_list:
                # 读取markdown文件
                if filename.endswith('.md'):
                    result["markdown"] = zf.read(filename).decode('utf-8')
                    print(f"  ✓ 读取Markdown: {len(result['markdown'])} 字符")

                # 读取content_list.json（包含位置信息）
                elif filename.endswith('content_list.json'):
                    content = zf.read(filename).decode('utf-8')
                    result["content_list"] = json.loads(content)
                    print(f"  ✓ 读取内容列表: {len(result['content_list'])} 项")

                # 读取layout信息
                elif filename.endswith('layout.json') or filename.endswith('middle.json'):
                    content = zf.read(filename).decode('utf-8')
                    result["layout_info"] = json.loads(content)
                    print(f"  ✓ 读取布局信息")

                # 读取图片 - 保存basename作为键名
                elif filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    img_name = os.path.basename(filename)
                    result["images"][img_name] = zf.read(filename)
                    print(f"  ✓ 读取图片: {img_name}")

        # 计算每页坐标映射，便于前端精确定位
        result["page_mappings"] = self._build_page_mappings(
            result.get("content_list", []),
            result.get("layout_info")
        )

        print(f"✅ 解析完成")
        return result

    def _build_page_mappings(self, content_list, layout_info):
        """根据content_list和layout信息构建每页的坐标映射"""
        page_stats = {}

        for item in content_list or []:
            bbox = item.get("bbox")
            page_idx = item.get("page_idx")
            if not bbox or page_idx is None:
                continue
            stats = page_stats.setdefault(page_idx, {
                "norm_min_x": bbox[0],
                "norm_max_x": bbox[2],
                "norm_min_y": bbox[1],
                "norm_max_y": bbox[3]
            })
            stats["norm_min_x"] = min(stats["norm_min_x"], bbox[0])
            stats["norm_max_x"] = max(stats["norm_max_x"], bbox[2])
            stats["norm_min_y"] = min(stats["norm_min_y"], bbox[1])
            stats["norm_max_y"] = max(stats["norm_max_y"], bbox[3])

        pdf_info = []
        if isinstance(layout_info, dict):
            pdf_info = layout_info.get("pdf_info", [])
        elif isinstance(layout_info, list):
            pdf_info = layout_info

        for page in pdf_info or []:
            page_idx = page.get("page_idx")
            if page_idx is None:
                continue
            stats = page_stats.setdefault(page_idx, {})
            page_size = page.get("page_size") or []
            if isinstance(page_size, (list, tuple)) and len(page_size) >= 2:
                stats["page_width"], stats["page_height"] = page_size[:2]
            else:
                stats["page_width"] = page.get("width")
                stats["page_height"] = page.get("height")

            x_vals = []
            y_vals = []
            for block in page.get("para_blocks", []):
                bbox = block.get("bbox")
                if bbox:
                    x_vals.extend([bbox[0], bbox[2]])
                    y_vals.extend([bbox[1], bbox[3]])
            for block in page.get("discarded_blocks", []):
                bbox = block.get("bbox")
                if bbox:
                    x_vals.extend([bbox[0], bbox[2]])
                    y_vals.extend([bbox[1], bbox[3]])

            if x_vals:
                stats["actual_min_x"] = min(x_vals)
                stats["actual_max_x"] = max(x_vals)
            elif "page_width" in stats:
                stats.setdefault("actual_min_x", 0)
                stats.setdefault("actual_max_x", stats["page_width"])

            if y_vals:
                stats["actual_min_y"] = min(y_vals)
                stats["actual_max_y"] = max(y_vals)
            elif "page_height" in stats:
                stats.setdefault("actual_min_y", 0)
                stats.setdefault("actual_max_y", stats["page_height"])

        for page_idx, stats in page_stats.items():
            stats.setdefault("norm_min_x", 0)
            stats.setdefault("norm_max_x", stats.get("page_width", 1))
            stats.setdefault("norm_min_y", 0)
            stats.setdefault("norm_max_y", stats.get("page_height", 1))
            stats.setdefault("actual_min_x", 0)
            stats.setdefault("actual_max_x", stats.get("page_width", 0))
            stats.setdefault("actual_min_y", 0)
            stats.setdefault("actual_max_y", stats.get("page_height", 0))
            stats["page_idx"] = page_idx

        return page_stats


if __name__ == "__main__":
    # 测试
    parser = DocumentParser()

    # 测试本地文件上传
    result = parser.parse(file_path="/Users/enithz/Downloads/NC-2023_Sulfate_triple-oxygen-isotope_evidence_confirming_oceanic_oxygenation_570_million_years_ago_2.pdf")

    print(f"\n=== 解析结果 ===")
    print(f"类型: {result['type']}")
    print(f"Markdown长度: {len(result.get('markdown', ''))}")
    print(f"内容块数量: {len(result.get('content_list', []))}")
    print(f"图片数量: {len(result.get('images', {}))}")

    # 打印前500字符的markdown
    if result.get('markdown'):
        print(f"\n=== Markdown预览 ===")
        print(result['markdown'][:500])
