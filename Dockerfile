# ShelfCast: Streamlit app for explainable Walmart sales forecasting
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# LightGBM needs the OpenMP runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so this layer is cached when only the code changes
COPY requirements-app.txt .
RUN pip install -r requirements-app.txt

COPY app.py walmart_features.py walmart_explain.py walmart_agent.py walmart_charts.py walmart_lgbm_model.pkl model_validation.csv.gz ./

# The Kaggle data is not part of the image: mount it at /app/Data (see docker-compose.yml)
RUN useradd --create-home appuser && mkdir -p /app/Data && chown -R appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
