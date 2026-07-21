# LaunchCast — container for self-hosting (Growth Plan Phase 1)
# Works on Railway / Render / Fly / any Docker VPS.
# Build:  docker build -t launchcast .
# Run:    docker run -p 8501:8501 --env-file .env launchcast
# Secrets (OWNER key, GIST token, future OIDC) go in env vars / host secrets,
# never in the image.

FROM python:3.12-slim

WORKDIR /app

# v45.53 (reviewer pro-tip): copy requirements FIRST and install from it, so
# Docker caches the dependency layer. Editing a .py file then rebuilds only
# the fast COPY layer below — not the whole pip install. requirements.txt is
# also the single source of truth for deps (it carries lxml/html5lib/tzdata
# that the old inline list here was silently missing).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code (all 12 modules + .streamlit theme). This layer rebuilds on every
# code change but does NOT invalidate the cached pip layer above.
COPY *.py /app/
COPY .streamlit/ /app/.streamlit/

EXPOSE 8501

# Health endpoint lets the host restart a wedged instance
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# NOTE ON MEMORY: pandas can spike when building combined_all on big slates.
# On memory-constrained tiers (Render free/cheap, Fly shared-cpu-1x), give the
# container at least 1 GB to avoid OOM kills, e.g.:
#   docker run -p 8501:8501 --env-file .env --memory=1g --memory-swap=1g launchcast
# (Railway/Render set this in the dashboard; the flag is for local/VPS runs.)

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
