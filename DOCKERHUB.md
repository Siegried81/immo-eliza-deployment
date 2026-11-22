# Immo Eliza

## Quick reference

- **Source code, Dockerfile, issues:** https://github.com/Siegried81/immo-eliza-deployment
- **Maintained by:** Siegried81
- **Also on:** GitHub Container Registry, `ghcr.io/siegried81/<image>`
- **Architecture:** `linux/amd64`

## Supported tags

- `latest`: the current `main` branch
- `X.Y.Z` and `X`: released versions (git tags `vX.Y.Z`), the ones to pin for a stable deployment
- `sha-xxxxxxx`: one specific commit

## What is it?

Belgian residential property price prediction: a FastAPI model server and a Streamlit front end, with a dedicated model for the luxury segment.

Images from this repository:

- `immo-eliza-api`: FastAPI prediction API on 8000 (`/docs` for Swagger)
- `immo-eliza-ui`: Streamlit UI on 10000, talks to the API through `API_URL`

## How to use this image

```bash
docker network create immo
docker run -d --name api --network immo -p 8008:8000 siegried/immo-eliza-api
docker run -d --name ui --network immo -p 8508:10000 -e API_URL=http://api:8000 siegried/immo-eliza-ui
# open http://localhost:8508 (API docs: http://localhost:8008/docs)
```

## Environment variables

None are required and neither image needs a secret. Pass them with `-e NAME=value` or `--env-file .env` (`.env.example` in the repo lists them).

| Variable | Purpose |
|---|---|
| `API_URL` | UI only: where the API answers (default `http://localhost:8000`) |
| `API_DOCS_URL` | UI only: the "API documentation" link (default `$API_URL/docs`; empty hides it) |
| `PREDICTION_LOG_PATH` | API only: where predictions are logged (default `monitoring/logs.jsonl`) |
| `PORT` | Listening port; set automatically on Render |

The API image runs as a non-root user and has no volume: the prediction log lives inside the container unless you mount a path and point `PREDICTION_LOG_PATH` at it.

## License

See the LICENSE file in the source repository, if any. Check that your use complies with the licenses of the software the images contain.
