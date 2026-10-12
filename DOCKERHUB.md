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
docker run -d --name api --network immo -p 8003:8000 siegried/immo-eliza-api
docker run -d --network immo -p 8508:10000 -e API_URL=http://api:8000 siegried/immo-eliza-ui
# open http://localhost:8508 (API docs: http://localhost:8003/docs)
```

## Environment variables

None are required to start the UI. Secrets are **never** baked into the image: pass
them at run time with `-e NAME=value` or `--env-file .env` (see `.env.example` in the repo).

| Variable | Purpose |
|---|---|
| `API_URL` | UI only: where the API answers (default `http://localhost:8000`) |
| `PORT` | listening port, set automatically on Render |

## Volumes

- none: the image is self-contained

## License

See the LICENSE file in the source repository, if any. As with every image, it is
your responsibility to check that your use complies with the licenses of the
software it contains.
