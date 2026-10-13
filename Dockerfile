FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md requirements.lock ./
COPY src ./src
RUN pip install --no-cache-dir -c requirements.lock . && useradd --create-home worker
USER worker
EXPOSE 8000
CMD ["evidenceforge", "--data-dir", "/artifacts/data/scifact", "--cache-dir", "/artifacts/cache", "serve", "--host", "0.0.0.0", "--model-dir", "/artifacts/results/scifact/model"]
