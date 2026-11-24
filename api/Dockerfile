# Python 3.12: numpy 2.5 and scikit-learn 1.9 no longer ship wheels for 3.11.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --only-binary=:all: -r requirements.txt

# Only what the API imports at run time: no docs, tests, scripts or UI.
# The service writes its prediction log under monitoring/, so the app user owns /app.
RUN useradd --create-home --uid 1000 app && chown app:app /app
COPY --chown=app:app api/ api/
COPY --chown=app:app src/ src/
COPY --chown=app:app monitoring/ monitoring/
COPY --chown=app:app models/ models/
USER app

EXPOSE 8000

# Slim images ship no curl, so the probe uses Python. It reads $PORT like CMD,
# so the check follows the server when Render or a user moves the port.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/ping', timeout=4).status == 200 else 1)"]
# Render injects PORT; docker compose and local runs fall back to 8000.
# exec: uvicorn becomes PID 1 and gets the SIGTERM of `docker stop` itself.
CMD ["sh", "-c", "exec uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
