#!/bin/sh
# Start the API in the background, wait until it answers, then run the UI in
# the foreground so the container lives as long as Streamlit does.
# The wait matters: the UI caches its /ping result for a minute, so a UI that
# started before the models finished loading would report the API as asleep.
# If the API dies while loading (a model file that won't unpickle), stop with
# an error instead of waiting forever: the Space then shows a failed build log.
uvicorn api.app:app --host 127.0.0.1 --port 8000 &
API_PID=$!

tries=0
until python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ping', timeout=2)" 2>/dev/null; do
    if ! kill -0 "$API_PID" 2>/dev/null; then
        echo "The API exited during start-up." >&2
        exit 1
    fi
    tries=$((tries + 1))
    if [ "$tries" -ge 180 ]; then
        echo "The API did not answer /ping within 3 minutes." >&2
        exit 1
    fi
    sleep 1
done

exec streamlit run streamlit/app.py --server.port 7860 --server.address 0.0.0.0
