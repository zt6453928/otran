# otran

Flask 后端 + Vite React 前端的同仓库项目。前端通过同域路径调用后端接口：`/api/*`。

## Zeabur 部署（展示 React 页面）

1. Zeabur → New Project → Add Service → 选择该 GitHub 仓库
2. 在 Service 的构建方式里选择 **Dockerfile**（仓库根目录已提供 `Dockerfile`，会自动构建 `frontend/dist`）
3. 配置环境变量（Service → Variables）：
   - `MINERU_API_TOKEN`（必填）
   - `DEEPLX_API_URL`（必填）
   - （可选）`MINERU_API_URL`、`DEFAULT_SOURCE_LANG`、`DEFAULT_TARGET_LANG`
4. 部署完成后访问 Service Domain：`/` 即为 React 页面；页面请求 `/api/*` 会同域命中 Flask。

## 本地开发

- 后端：`pip install -r requirements.txt && python app.py`（默认 `http://localhost:8080`）
- 前端：`cd frontend && npm install && npm run dev`（默认 `http://localhost:5173`，已在 `frontend/vite.config.js` 里代理 `/api` 到后端）

