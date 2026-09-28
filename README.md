# TokenPilot — AI Engineering Cost Intelligence

A Streamlit production-oriented SaaS foundation for observing, analyzing, optimizing and governing AI coding usage.

## Included
- Dark enterprise SaaS UI matching the TokenPilot product concept
- Overview, Usage, Developers, Costs, Optimizer, AI Agent, Policies, Budgets, Integrations and Settings
- SQLite by default; SQLAlchemy supports PostgreSQL/MySQL through `DATABASE_URL`
- Deterministic cost calculation from model pricing
- CSV telemetry ingestion with schema validation
- Optimization engine with evidence and savings estimates
- Policy simulation and persisted policy records
- Audit log
- OpenAI-compatible LLM agent integration, including Ollama
- Safe agent tool layer: read/analyze/simulate first; mutation requires explicit UI action
- Optional Streamlit OIDC authentication path
- Docker deployment

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app seeds realistic demo telemetry on first run. Data is stored in `data/tokenpilot.db`.

## LLM / Ollama

Install Ollama separately and pull a model, then set:

```env
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=llama3.1:8b
```

The agent also works without an LLM: it falls back to a deterministic analyst that queries the same application data.

## Production notes

1. Use PostgreSQL rather than SQLite for multiple workers/users.
2. Put secrets in Streamlit secrets or a secret manager, never in the database or source code.
3. Put TokenPilot behind HTTPS and an identity provider. Streamlit OIDC can be configured through `.streamlit/secrets.toml`.
4. For real provider telemetry, route company-managed AI traffic through an approved gateway or provider integration. Do not attempt to inspect employees' personal consumer accounts.
5. Replace demo seed data before connecting a real tenant.
6. Add a background queue (Celery/RQ/managed queue) for high-volume ingestion instead of doing large imports in a Streamlit request.
7. Add database row-level security or strict tenant scoping at the service layer before multi-tenant production deployment.

## Telemetry CSV schema

Required columns:
- timestamp
- developer_email
- team
- provider
- model
- input_tokens
- output_tokens
- task_type

Optional:
- repository
- latency_ms
- status
- request_id

## Test

```bash
pytest -q
```
