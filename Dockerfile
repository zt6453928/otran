FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* frontend/pnpm-lock.yaml* frontend/yarn.lock* ./
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim AS runtime
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./
COPY --from=frontend-build /app/frontend/dist /app/frontend/dist

CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 300"]

