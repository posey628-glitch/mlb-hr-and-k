# LaunchCast — container for self-hosting (Growth Plan Phase 1)
# Works on Railway / Render / Fly / any Docker VPS.
# Build:  docker build -t launchcast .
# Run:    docker run -p 8501:8501 --env-file .env launchcast
# Secrets (OWNER key, GIST token, future OIDC) go in env vars / host secrets,
# never in the image.

FROM python:3.12-slim

WORKDIR /app

# System deps kept minimal; pandas/numpy ship wheels on 3.12-slim.
RUN pip install --no-cache-dir \
    streamlit>=1.42 \
    pandas \
    numpy \
    requests \
    pytz \
    openpyxl

# App code (all 12 modules + .streamlit theme)
COPY *.py /app/
COPY .streamlit/ /app/.streamlit/

EXPOSE 8501

# Health endpoint lets the host restart a wedged instance
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
