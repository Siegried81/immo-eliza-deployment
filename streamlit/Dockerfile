FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --only-binary=:all: -r requirements.txt

COPY . .

EXPOSE 10000

CMD ["streamlit", "run", "streamlit/app.py", "--server.port=10000", "--server.address=0.0.0.0"]