FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip     pip install --upgrade pip &&     pip install --only-binary=:all: -r requirements.txt

COPY . .

EXPOSE 10000

# Slim images ship no curl, so the probe uses Python. It reads $PORT like CMD,
# so the check follows the server when Render or a user moves the port.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '10000') + '/_stcore/health', timeout=4).status == 200 else 1)"]

CMD ["sh", "-c", "streamlit run streamlit/app.py --server.port=${PORT:-10000} --server.address=0.0.0.0"]
