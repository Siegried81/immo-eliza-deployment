#!/bin/sh
# Start the API in the background, wait until it answers, then run the UI in
# the foreground so the container lives as long as Streamlit does.
# The wait matters: the UI caches its /ping result for a minute, so a UI that
# started before the models finished loading would report the API as asleep.
uvicorn api.app:app --host 127.0.0.1 --port 8000 &

until python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ping', timeout=2)" 2>/dev/null; do
    sleep 1
done

exec streamlit run streamlit/app.py --server.port 7860 --server.address 0.0.0.0
