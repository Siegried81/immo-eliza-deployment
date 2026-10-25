# Hugging Face Space image: the API and the UI in one container.
# A Space exposes a single port (7860), so the API listens on localhost only
# and the UI reaches it there; the two compose images stay for local runs.
# Python 3.12: numpy 2.5 and scikit-learn 1.9 no longer ship wheels for 3.11.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Spaces run the container as uid 1000, which must own the app directory.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --only-binary=:all: -r requirements.txt

COPY --chown=user . .

ENV API_URL=http://127.0.0.1:8000
# The API is not reachable from outside the container, so the UI hides its docs link.
ENV API_DOCS_URL=
# The Space filesystem is wiped on restart; the drift log is scratch here.
ENV PREDICTION_LOG_PATH=/tmp/predictions.jsonl

EXPOSE 7860
CMD ["sh", "start.sh"]
