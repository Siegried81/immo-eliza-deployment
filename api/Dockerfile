# Python 3.12: numpy 2.5 and scikit-learn 1.9 no longer ship wheels for 3.11.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip     pip install --upgrade pip &&     pip install --only-binary=:all: -r requirements.txt

COPY . .

EXPOSE 8000
# Render injects PORT; docker compose and local runs fall back to 8000.
CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
