FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --only-binary=:all: -r requirements.txt

COPY . .

RUN ls -la /app/models/

EXPOSE 1000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]