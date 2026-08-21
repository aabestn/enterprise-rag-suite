# Enterprise RAG Pipeline & Evaluation Suite

A production-ready enterprise document QA platform featuring multi-page PDF ingestion, dynamic chunking, vector retrieval with hybrid re-ranking, security guardrails, and automated evaluation metrics.

## Architecture Features

- **Ingestion**: Async PostgreSQL document tracking paired with PyMuPDF vision layout extraction and dynamic semantic chunking.
- **Retrieval Engine**: Dense vector search via Qdrant backed by Cohere hybrid re-ranking (`rerank-english-v3.0`).
- **Security**: Prompt injection defense and compliance enforcement powered by NeMo Guardrails.
- **Evaluation Suite**: Automated tracking of context precision, recall, and hallucination rates using Ragas and MLflow.
- **Telemetry & Microservices**: Fully dockerized service deployment with Prometheus & Grafana metrics monitoring.

## Quickstart

1. **Environment Configuration**:
   ```bash
   cp .env.example .env
   # Update .env with your Cohere and OpenAI API keys
   ```

2. **Launch Services via Docker Compose**:
    ```Bash

    docker-compose -f docker/docker-compose.yml up --build -d
    ```
3. **API Documentation**:
    Access FastAPI interactive docs at http://localhost:8000/docs.

4. **Monitoring & Metrics**:

    Grafana Dashboard: http://localhost:3000

    Prometheus Targets: http://localhost:9090

    MLflow Experiments: http://localhost:5000