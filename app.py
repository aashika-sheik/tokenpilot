from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    func,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

load_dotenv()

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DEFAULT_DB = f"sqlite:///{(DATA_DIR / 'tokenpilot.db').as_posix()}"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DB)

# ---------- Database ----------
class Base(DeclarativeBase):
    pass


class Organization(Base):
    __tablename__ = "organizations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class Developer(Base):
    __tablename__ = "developers"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    organization_id: Mapped[int] = mapped_column(Integer, index=True)
    name: Mapped[str] = mapped_column(String(150))
    email: Mapped[str] = mapped_column(String(250), index=True)
    team: Mapped[str] = mapped_column(String(120))
    role: Mapped[str] = mapped_column(String(120), default="Software Engineer")
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class UsageEvent(Base):
    __tablename__ = "usage_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    organization_id: Mapped[int] = mapped_column(Integer, index=True)
    request_id: Mapped[str] = mapped_column(String(100), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    developer_email: Mapped[str] = mapped_column(String(250), index=True)
    team: Mapped[str] = mapped_column(String(120), index=True)
    provider: Mapped[str] = mapped_column(String(80), index=True)
    model: Mapped[str] = mapped_column(String(120), index=True)
    input_tokens: Mapped[int] = mapped_column(Integer)
    output_tokens: Mapped[int] = mapped_column(Integer)
    task_type: Mapped[str] = mapped_column(String(120))
    task_complexity: Mapped[str] = mapped_column(String(20), default="MEDIUM")
    repository: Mapped[str | None] = mapped_column(String(200), nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="success")
    input_cost: Mapped[float] = mapped_column(Float, default=0.0)
    output_cost: Mapped[float] = mapped_column(Float, default=0.0)
    total_cost: Mapped[float] = mapped_column(Float, default=0.0)


class Policy(Base):
    __tablename__ = "policies"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    organization_id: Mapped[int] = mapped_column(Integer, index=True)
    name: Mapped[str] = mapped_column(String(200))
    condition_json: Mapped[str] = mapped_column(Text)
    action_json: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(30), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    organization_id: Mapped[int] = mapped_column(Integer, index=True)
    actor: Mapped[str] = mapped_column(String(250))
    action: Mapped[str] = mapped_column(String(150))
    details: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


@st.cache_resource(show_spinner=False)
def get_engine():
    kwargs = {"pool_pre_ping": True}
    if DATABASE_URL.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(DATABASE_URL, **kwargs)


engine = get_engine()
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
Base.metadata.create_all(engine)

# Pricing is deliberately explicit and versionable. Values are USD / 1M tokens.
PRICING: dict[str, dict[str, float]] = {
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "claude-sonnet": {"input": 3.00, "output": 15.00},
    "claude-haiku": {"input": 0.80, "output": 4.00},
    "gemini-flash": {"input": 0.30, "output": 2.50},
    "gemini-pro": {"input": 1.25, "output": 10.00},
    "llama3.1:8b": {"input": 0.0, "output": 0.0},
}

MODEL_ALIASES = {
    "gpt-5": "gpt-5",
    "gpt-5-mini": "gpt-5-mini",
    "gpt-4.1": "gpt-4.1",
    "claude-sonnet": "claude-sonnet",
    "claude-3-5-sonnet": "claude-sonnet",
    "claude-haiku": "claude-haiku",
    "gemini-flash": "gemini-flash",
    "gemini-2.5-flash": "gemini-flash",
    "gemini-pro": "gemini-pro",
}


def normalize_model(model: str) -> str:
    key = model.strip().lower()
    return MODEL_ALIASES.get(key, key)


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> tuple[float, float, float]:
    p = PRICING.get(normalize_model(model), {"input": 1.0, "output": 4.0})
    ic = input_tokens / 1_000_000 * p["input"]
    oc = output_tokens / 1_000_000 * p["output"]
    return ic, oc, ic + oc


def classify_complexity(task_type: str, input_tokens: int, output_tokens: int) -> str:
    t = task_type.lower()
    if any(x in t for x in ["architecture", "large refactor", "security review", "migration"]):
        return "HIGH"
    total = input_tokens + output_tokens
    if total < 4_000 and any(x in t for x in ["explain", "typo", "format", "unit test", "simple", "documentation"]):
        return "LOW"
    if total > 30_000:
        return "HIGH"
    return "MEDIUM"


# ---------- Seed / ingestion ----------
def audit(session: Session, org_id: int, actor: str, action: str, details: dict[str, Any] | None = None):
    session.add(AuditLog(organization_id=org_id, actor=actor, action=action, details=json.dumps(details or {})))
    session.commit()


def seed_demo():
    with SessionLocal() as s:
        org = s.scalar(select(Organization).where(Organization.name == "Acme Technologies"))
        if org is None:
            org = Organization(name="Acme Technologies")
            s.add(org)
            s.flush()
            names = [
                ("Aashika J", "aashika@example.com", "Platform"), ("Rahul Menon", "rahul@example.com", "Backend"),
                ("Sarah Lee", "sarah@example.com", "Frontend"), ("Daniel Kim", "daniel@example.com", "Data"),
                ("Meera Iyer", "meera@example.com", "Platform"), ("Arjun Rao", "arjun@example.com", "Backend"),
                ("Nisha Patel", "nisha@example.com", "Data"), ("Owen Smith", "owen@example.com", "Product"),
            ]
            for n, e, t in names:
                s.add(Developer(organization_id=org.id, name=n, email=e, team=t))
            s.commit()

        count = s.scalar(select(func.count(UsageEvent.id)).where(UsageEvent.organization_id == org.id)) or 0
        if count > 0:
            return org.id

        rng = np.random.default_rng(42)
        devs = s.scalars(select(Developer).where(Developer.organization_id == org.id)).all()
        teams = [d.team for d in devs]
        models = ["gpt-5", "claude-sonnet", "gpt-5-mini", "gemini-flash"]
        tasks = ["Code generation", "Debugging", "Code explanation", "Documentation", "Unit testing", "Architecture"]
        providers = {"gpt-5": "OpenAI", "gpt-5-mini": "OpenAI", "claude-sonnet": "Anthropic", "gemini-flash": "Google"}
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        events = []
        for i in range(5000):
            days_ago = int(rng.integers(0, 180))
            ts = now - timedelta(days=days_ago, hours=int(rng.integers(0, 24)))
            d = devs[int(rng.integers(0, len(devs)))]
            model = str(rng.choice(models, p=[.34, .27, .18, .21]))
            task = str(rng.choice(tasks, p=[.31, .24, .18, .11, .09, .07]))
            base = int(rng.lognormal(mean=7.2, sigma=.8))
            if task == "Architecture":
                base *= 4
            inp = max(50, int(base))
            out = max(20, int(inp * float(rng.uniform(.15, .55))))
            complexity = classify_complexity(task, inp, out)
            ic, oc, total = calculate_cost(model, inp, out)
            events.append(UsageEvent(
                organization_id=org.id, request_id=f"demo-{i:06d}", timestamp=ts,
                developer_email=d.email, team=d.team, provider=providers[model], model=model,
                input_tokens=inp, output_tokens=out, task_type=task, task_complexity=complexity,
                repository=f"{d.team.lower()}-service", latency_ms=float(rng.integers(200, 4000)),
                status="success", input_cost=ic, output_cost=oc, total_cost=total,
            ))
        s.add_all(events)
        s.add_all([
            Policy(organization_id=org.id, name="Low Complexity Model Policy", condition_json=json.dumps({"complexity":"LOW"}), action_json=json.dumps({"route_to":"gpt-5-mini"})),
            Policy(organization_id=org.id, name="Engineering Monthly Budget", condition_json=json.dumps({"team":"Engineering"}), action_json=json.dumps({"budget":50000})),
        ])
        s.commit()
        audit(s, org.id, "system", "demo_seeded", {"events": len(events)})
        return org.id


ORG_ID = seed_demo()


def load_events(org_id: int) -> pd.DataFrame:
    with SessionLocal() as s:
        rows = s.scalars(select(UsageEvent).where(UsageEvent.organization_id == org_id).order_by(UsageEvent.timestamp.desc())).all()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([{
            "id": r.id, "timestamp": r.timestamp, "developer_email": r.developer_email, "team": r.team,
            "provider": r.provider, "model": r.model, "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
            "total_tokens": r.input_tokens + r.output_tokens, "task_type": r.task_type, "task_complexity": r.task_complexity,
            "repository": r.repository, "latency_ms": r.latency_ms, "status": r.status, "cost": r.total_cost,
        } for r in rows])


def ingest_csv(file, org_id: int, actor: str) -> tuple[bool, str]:
    required = {"timestamp", "developer_email", "team", "provider", "model", "input_tokens", "output_tokens", "task_type"}
    try:
        df = pd.read_csv(file)
        missing = required - set(df.columns)
        if missing:
            return False, f"Missing required columns: {', '.join(sorted(missing))}"
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
        if df["timestamp"].isna().any():
            return False, "One or more timestamps could not be parsed."
        df["input_tokens"] = pd.to_numeric(df["input_tokens"], errors="raise").astype(int)
        df["output_tokens"] = pd.to_numeric(df["output_tokens"], errors="raise").astype(int)
        if (df[["input_tokens", "output_tokens"]] < 0).any().any():
            return False, "Token counts cannot be negative."
        with SessionLocal() as s:
            existing = {x for x in s.scalars(select(UsageEvent.request_id).where(UsageEvent.organization_id == org_id)).all()}
            events = []
            for idx, r in df.iterrows():
                request_id = str(r.get("request_id") or hashlib.sha256(f"{r.developer_email}-{r.timestamp}-{idx}".encode()).hexdigest()[:24])
                if request_id in existing:
                    continue
                model = str(r.model)
                ic, oc, total = calculate_cost(model, int(r.input_tokens), int(r.output_tokens))
                events.append(UsageEvent(
                    organization_id=org_id, request_id=request_id, timestamp=r.timestamp.to_pydatetime(),
                    developer_email=str(r.developer_email), team=str(r.team), provider=str(r.provider), model=model,
                    input_tokens=int(r.input_tokens), output_tokens=int(r.output_tokens), task_type=str(r.task_type),
                    task_complexity=str(r.get("task_complexity") or classify_complexity(str(r.task_type), int(r.input_tokens), int(r.output_tokens))),
                    repository=str(r.get("repository")) if pd.notna(r.get("repository")) else None,
                    latency_ms=float(r.get("latency_ms")) if pd.notna(r.get("latency_ms")) else None,
                    status=str(r.get("status") or "success"), input_cost=ic, output_cost=oc, total_cost=total,
                ))
            s.add_all(events)
            s.commit()
            audit(s, org_id, actor, "telemetry_imported", {"rows": len(events)})
        return True, f"Imported {len(events)} new telemetry events."
    except Exception as exc:
        return False, f"Import failed safely: {type(exc).__name__}: {exc}"


# ---------- Analytics ----------
def period_df(df: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    if df.empty:
        return df
    cutoff = pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=days)
    return df[df.timestamp >= cutoff].copy()


def money(x: float) -> str:
    return f"${x:,.2f}" if abs(x) < 1000 else f"${x:,.0f}"


def compact(x: float) -> str:
    if x >= 1_000_000_000: return f"{x/1e9:.2f}B"
    if x >= 1_000_000: return f"{x/1e6:.2f}M"
    if x >= 1_000: return f"{x/1e3:.1f}K"
    return f"{x:,.0f}"


def optimization_candidates(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    out = []
    low = df[df.task_complexity == "LOW"]
    premium = low[low.model.isin(["gpt-5", "claude-sonnet", "gemini-pro"])]
    if len(premium):
        current = premium.cost.sum()
        cheap = premium.apply(lambda r: calculate_cost("gpt-5-mini", int(r.input_tokens), int(r.output_tokens))[2], axis=1).sum()
        save = max(0.0, current - cheap) * (30 / max(1, min(30, (df.timestamp.max() - df.timestamp.min()).days + 1)))
        out.append({"title":"Premium model overuse","severity":"HIGH IMPACT","saving":save,"confidence":.94,"evidence":f"{len(premium):,} low-complexity requests use premium models.","action":"Route low-complexity requests to an economical model."})
    repeated = df.groupby(["developer_email", "task_type", "repository"]).size().sort_values(ascending=False)
    dup_groups = repeated[repeated >= 5]
    if len(dup_groups):
        affected = int(dup_groups.sum())
        save = float(df[df.set_index(["developer_email","task_type","repository"]).index.isin(dup_groups.index)].cost.sum()) * .10
        out.append({"title":"Repeated request patterns","severity":"QUICK WIN","saving":save,"confidence":.97,"evidence":f"{affected:,} requests fall into repeated task/repository patterns.","action":"Introduce prompt/result caching or reusable workflows."})
    recent = df[df.timestamp >= df.timestamp.max() - pd.Timedelta(days=30)]
    prev = df[(df.timestamp < df.timestamp.max() - pd.Timedelta(days=30)) & (df.timestamp >= df.timestamp.max() - pd.Timedelta(days=60))]
    if len(recent) and len(prev):
        recent_avg = recent.total_tokens.mean(); prev_avg = prev.total_tokens.mean()
        if recent_avg > prev_avg * 1.15:
            save = max(0, (recent_avg-prev_avg)/recent_avg) * recent.cost.sum()
            out.append({"title":"Large context requests","severity":"OPPORTUNITY","saving":float(save),"confidence":.89,"evidence":f"Average tokens/request increased {(recent_avg/prev_avg-1)*100:.0f}% versus the previous period.","action":"Filter repository context and retrieve only relevant files."})
    return sorted(out, key=lambda x: x["saving"], reverse=True)


# ---------- Agent ----------
AGENT_SYSTEM = """You are TokenPilot's AI Cost Agent. You analyze company AI telemetry. Never invent metrics. Use the provided facts only. Distinguish observed facts from recommendations. Do not claim to have executed an action unless the application confirms it. Be concise, explain drivers, and recommend safe next steps."""


def deterministic_agent(question: str, df: pd.DataFrame) -> str:
    q = question.lower()
    d = period_df(df, 30)
    if d.empty:
        return "There is no telemetry in the selected period yet. Connect a provider or import telemetry CSV data first."
    total = d.cost.sum(); tokens = d.total_tokens.sum(); requests = len(d)
    if "spend" in q or "increase" in q or "cost" in q or "bill" in q:
        by_team = d.groupby("team").cost.sum().sort_values(ascending=False)
        by_model = d.groupby("model").cost.sum().sort_values(ascending=False)
        opp = optimization_candidates(df)
        lines = [f"AI spend for the last 30 days is **{money(total)}** across **{requests:,} requests** and **{compact(tokens)} tokens**."]
        if len(by_team): lines.append(f"The largest team cost is **{by_team.index[0]}** at **{money(by_team.iloc[0])}**.")
        if len(by_model): lines.append(f"The largest model cost is **{by_model.index[0]}** at **{money(by_model.iloc[0])}**.")
        if opp:
            lines.append(f"I found **{len(opp)} optimization opportunities** with estimated savings of **{money(sum(x['saving'] for x in opp))}/month**. These are estimates, not guaranteed savings.")
        return "\n\n".join(lines)
    if "save" in q or "optim" in q or "waste" in q:
        opp = optimization_candidates(df)
        if not opp: return "I did not find a material optimization candidate in the current telemetry window."
        bullets = [f"**{o['title']}** — estimated {money(o['saving'])}/month, {o['confidence']*100:.0f}% confidence. {o['evidence']}" for o in opp[:5]]
        return "Here are the strongest candidates:\n\n" + "\n\n".join(f"{i+1}. {b}" for i,b in enumerate(bullets))
    if "team" in q or "developer" in q:
        by = d.groupby("team").agg(spend=("cost","sum"), requests=("id","count")).sort_values("spend", ascending=False)
        return "\n".join([f"**{idx}** — {money(r.spend)} across {int(r.requests):,} requests." for idx, r in by.head(8).iterrows()])
    return f"I analyzed {requests:,} requests from the last 30 days. Spend is {money(total)} and token usage is {compact(tokens)}. Ask me about spend drivers, teams, developers, models, anomalies, or savings opportunities."


def llm_agent(question: str, facts: str) -> str | None:
    base = os.getenv("LLM_BASE_URL")
    key = os.getenv("LLM_API_KEY", "ollama")
    model = os.getenv("LLM_MODEL")
    if not base or not model:
        return None
    url = base.rstrip("/") + "/chat/completions"
    payload = {"model": model, "temperature": 0.1, "messages": [
        {"role":"system", "content":AGENT_SYSTEM},
        {"role":"user", "content":f"Company facts:\n{facts}\n\nQuestion: {question}"},
    ]}
    try:
        r = requests.post(url, headers={"Authorization": f"Bearer {key}", "Content-Type":"application/json"}, json=payload, timeout=45)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except Exception:
        return None


def agent_answer(question: str, df: pd.DataFrame) -> str:
    d = period_df(df, 30)
    opp = optimization_candidates(df)
    if d.empty: return deterministic_agent(question, df)
    by_team = d.groupby("team").cost.sum().sort_values(ascending=False).head(8).to_dict()
    by_model = d.groupby("model").cost.sum().sort_values(ascending=False).head(8).to_dict()
    facts = json.dumps({
        "period":"last_30_days", "spend":round(float(d.cost.sum()),2), "requests":len(d), "tokens":int(d.total_tokens.sum()),
        "teams":{k:round(float(v),2) for k,v in by_team.items()}, "models":{k:round(float(v),2) for k,v in by_model.items()},
        "optimization_candidates":[{k:(round(v,2) if isinstance(v,float) else v) for k,v in o.items() if k != "action"} for o in opp],
    }, default=str)
    return llm_agent(question, facts) or deterministic_agent(question, df)


# ---------- UI ----------
st.set_page_config(page_title="TokenPilot", page_icon="✦", layout="wide", initial_sidebar_state="expanded")

CSS = r"""
<style>
:root{--bg:#08090b;--surface:#101216;--surface2:#151820;--border:#252932;--text:#f5f7fa;--muted:#8f98a6;--accent:#8b5cf6;--green:#22c55e;--amber:#f59e0b;--red:#ef4444}
.stApp{background:var(--bg);color:var(--text)}
[data-testid="stHeader"]{background:rgba(8,9,11,.88)}
[data-testid="stSidebar"]{background:#0b0d10;border-right:1px solid var(--border)}
[data-testid="stSidebar"]>div{padding-top:1rem}
.block-container{max-width:1500px;padding:1.5rem 2.3rem 4rem}
.brand{padding:4px 8px 24px;border-bottom:1px solid var(--border);margin-bottom:18px}.brand-title{font-weight:800;letter-spacing:.08em}.brand-sub{font-size:9px;color:#69717d;letter-spacing:.13em;margin-top:3px}
.navlabel{font-size:9px;letter-spacing:.12em;color:#5f6874;margin:18px 8px 7px;text-transform:uppercase}
.kicker{font-size:10px;letter-spacing:.14em;color:#9b83df;font-weight:700;text-transform:uppercase}.title{font-size:30px;font-weight:650;letter-spacing:-.035em;margin:4px 0}.subtitle{color:var(--muted);font-size:13px}.section-title{font-size:16px;font-weight:650;margin:0}.section-sub{font-size:11px;color:#737c89}
.metric{background:var(--surface);border:1px solid var(--border);border-radius:11px;padding:17px 18px;min-height:105px}.metric-label{font-size:9px;color:#69717d;letter-spacing:.12em}.metric-value{font-size:27px;font-weight:650;letter-spacing:-.04em;margin-top:12px}.metric-change{font-size:10px;color:#71c98a;margin-top:4px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:11px;padding:18px}.card-head{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px}.small{font-size:11px;color:#707987}.insight{background:radial-gradient(circle at 80% 5%,#24173e 0,#101216 48%);border-color:#312747}.insight-text{font-size:14px;line-height:1.65;color:#b4bbc5}.big-number{font-size:30px;font-weight:650;letter-spacing:-.04em}.positive{color:#71c98a}.purple{color:#b69cff}
.stButton>button{border-radius:7px;border:1px solid var(--border);background:#12151a;color:#cbd0d8;font-size:12px}.stButton>button:hover{border-color:#6549a1;color:#fff}.primary-btn button{background:var(--accent)!important;border-color:var(--accent)!important;color:white!important}.danger button{border-color:#5a2929!important;color:#ef9999!important}
[data-testid="stMetric"]{background:transparent}.dataframe{font-size:12px}
div[data-testid="stVerticalBlockBorderWrapper"]{border-color:var(--border)!important;background:var(--surface)}
.agent-box{background:linear-gradient(135deg,#111318,#171323);border:1px solid #30264a;border-radius:11px;padding:20px}.agent-badge{color:#b69cff;font-size:10px;letter-spacing:.12em;font-weight:700}.agent-answer{font-size:13px;line-height:1.75;color:#c3c9d2}.source-pill{background:#15111e;border:1px solid #33284a;padding:5px 8px;border-radius:6px;font-size:10px;color:#a894d3}
.status-ok{color:#71c98a}.status-warn{color:#e3bb70}.status-off{color:#e08080}.tag{display:inline-block;font-size:8px;letter-spacing:.1em;color:#a88aff;border:1px solid #392e53;padding:4px 6px;border-radius:5px}.opportunity{min-height:250px}.opportunity h3{margin:14px 0 7px;font-size:17px}.opportunity p{color:#818a96;font-size:11px;line-height:1.6}.saving{font-size:20px;font-weight:650}.evidence{font-size:10px;color:#6f7885;line-height:1.6}
div[data-testid="stTabs"] button{font-size:12px}.stTextInput input,.stTextArea textarea,.stSelectbox div[data-baseweb="select"]>div,.stNumberInput input{background:#0c0e11!important;color:#e8ebef!important;border-color:#292e36!important}
hr{border-color:var(--border)}
@media(max-width:900px){.block-container{padding:1rem}.title{font-size:25px}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def ensure_session():
    if "org_id" not in st.session_state: st.session_state.org_id = ORG_ID
    if "actor" not in st.session_state: st.session_state.actor = "admin@acme.example"
    if "page" not in st.session_state: st.session_state.page = "Overview"
    if "agent_messages" not in st.session_state: st.session_state.agent_messages = []


def header(kicker: str, title: str, subtitle: str = ""):
    c1, c2 = st.columns([3,1])
    with c1:
        st.markdown(f'<div class="kicker">{kicker}</div><div class="title">{title}</div><div class="subtitle">{subtitle}</div>', unsafe_allow_html=True)
    with c2:
        st.caption("September 2026 · Last 30 days")
    st.write("")


def metric_cards(items):
    cols = st.columns(len(items))
    for col, (label, value, change) in zip(cols, items):
        with col:
            st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-change">{change}</div></div>', unsafe_allow_html=True)


def chart_theme(fig):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#9ba3af", size=11), margin=dict(l=0,r=0,t=10,b=0), legend=dict(orientation="h"))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#20242b", zeroline=False)
    return fig


def page_overview(df):
    d = period_df(df)
    spend = float(d.cost.sum()); requests = len(d); tokens = int(d.total_tokens.sum()); opp = optimization_candidates(df); savings = sum(x["saving"] for x in opp)
    header("COMMAND CENTER", "Good morning, Aashika.", "Here's what happened with AI usage this month.")
    metric_cards([("AI SPEND", money(spend), "↑ live telemetry"), ("AI REQUESTS", compact(requests), "last 30 days"), ("AI TOKENS", compact(tokens), "normalized usage"), ("SAVINGS FOUND", money(savings), f"{len(opp)} opportunities")])
    st.write("")
    c1,c2=st.columns([1.35,1])
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True); st.markdown('<div class="card-head"><div><div class="section-title">AI spend</div><div class="section-sub">Daily spend across connected providers</div></div></div>', unsafe_allow_html=True)
        daily=d.groupby(d.timestamp.dt.date).cost.sum().reset_index(name="cost") if not d.empty else pd.DataFrame(columns=["timestamp","cost"])
        fig=px.area(daily,x="timestamp",y="cost"); fig.update_traces(line_color="#8b5cf6",fillcolor="rgba(139,92,246,.16)"); st.plotly_chart(chart_theme(fig),use_container_width=True,config={"displayModeBar":False}); st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card insight"><div class="agent-badge">✦ TOKENPILOT INSIGHT</div>', unsafe_allow_html=True)
        if not d.empty:
            by_team=d.groupby("team").cost.sum().sort_values(ascending=False); by_model=d.groupby("model").cost.sum().sort_values(ascending=False)
            st.markdown(f'<h3>Largest cost driver: <span class="purple">{by_team.index[0]}</span></h3><div class="insight-text">The largest team cost is <b>{money(by_team.iloc[0])}</b>. The largest model cost is <b>{by_model.index[0]}</b> at <b>{money(by_model.iloc[0])}</b>.</div>', unsafe_allow_html=True)
        st.write("")
        if st.button("Investigate with AI Agent →", key="overview_agent"):
            st.session_state.page="AI Agent"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    st.write("")
    c1,c2=st.columns(2)
    with c1:
        st.markdown('<div class="card">',unsafe_allow_html=True); st.markdown('<div class="section-title">Where is AI being used?</div><div class="section-sub">Request share by team</div>',unsafe_allow_html=True)
        by=d.groupby("team").size().sort_values(ascending=False) if not d.empty else pd.Series(dtype=float)
        if not by.empty: st.bar_chart(by, height=220)
        st.markdown('</div>',unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">',unsafe_allow_html=True); st.markdown('<div class="section-title">Optimization opportunities</div><div class="section-sub">Potential monthly savings</div>',unsafe_allow_html=True)
        for o in opp[:3]:
            with st.container(border=True):
                st.markdown(f'<span class="tag">{o["severity"]}</span><h4>{o["title"]}</h4><div class="evidence">{o["evidence"]}</div><div class="saving">{money(o["saving"])}/mo</div>',unsafe_allow_html=True)
        if st.button("View all opportunities →", key="view_opps"): st.session_state.page="Optimizer"; st.rerun()
        st.markdown('</div>',unsafe_allow_html=True)


def page_usage(df):
    d=period_df(df); header("OBSERVABILITY","AI Usage","Understand how AI is being used across your organization.")
    metric_cards([("REQUESTS",compact(len(d)),"last 30 days"),("TOKENS",compact(d.total_tokens.sum()),"input + output"),("AVG TOKENS / REQUEST",f"{d.total_tokens.mean():,.0f}" if len(d) else "0","normalized"),("ACTIVE DEVELOPERS",str(d.developer_email.nunique()),"observed")])
    st.write("")
    c1,c2=st.columns(2)
    with c1:
        st.markdown('<div class="card"><div class="section-title">Model usage</div><div class="section-sub">Share of total requests</div>',unsafe_allow_html=True)
        x=d.groupby("model").size().sort_values(ascending=False) if not d.empty else pd.Series(dtype=float); st.bar_chart(x,height=260); st.markdown('</div>',unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><div class="section-title">Why are developers using AI?</div><div class="section-sub">Use-case classification</div>',unsafe_allow_html=True)
        x=d.groupby("task_type").size().sort_values(ascending=False) if not d.empty else pd.Series(dtype=float); st.bar_chart(x,height=260); st.markdown('</div>',unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="card"><div class="section-title">Recent AI activity</div><div class="section-sub">Latest normalized telemetry events</div>',unsafe_allow_html=True)
    st.dataframe(d[["timestamp","developer_email","team","task_type","model","total_tokens","cost"]].head(50),use_container_width=True,hide_index=True,column_config={"cost":st.column_config.NumberColumn(format="$%.4f")})
    st.markdown('</div>',unsafe_allow_html=True)


def page_developers(df):
    d=period_df(df); header("PEOPLE","Developers","See how individual developers use and spend on AI.")
    if d.empty:return
    x=d.groupby(["developer_email","team"]).agg(spend=("cost","sum"),requests=("id","count"),tokens=("total_tokens","sum")).reset_index().sort_values("spend",ascending=False)
    safe_requests = x["requests"].clip(lower=1)

    x["efficiency"] = (
    100 / (1 + (x["tokens"] / safe_requests / 1000))
    ).clip(0, 100).round(0)
    st.dataframe(x,use_container_width=True,hide_index=True,column_config={"spend":st.column_config.NumberColumn(format="$%.2f"),"tokens":st.column_config.NumberColumn(format="%d")})
    selected=st.selectbox("Open developer profile",["—"]+x.developer_email.tolist(),label_visibility="collapsed")
    if selected!="—": page_developer_profile(df,selected)


def page_developer_profile(df,email):
    d=period_df(df); x=d[d.developer_email==email]; header("DEVELOPER PROFILE",email,"Individual AI usage and optimization signals.")
    metric_cards([("AI SPEND",money(x.cost.sum()),"last 30 days"),("REQUESTS",compact(len(x)),"observed"),("TOKENS",compact(x.total_tokens.sum()),"input + output"),("AVG TOKENS",f"{x.total_tokens.mean():,.0f}" if len(x) else "0","per request")])
    c1,c2=st.columns(2)
    with c1:
        st.markdown('<div class="card"><div class="section-title">Usage by task</div>',unsafe_allow_html=True); st.bar_chart(x.groupby("task_type").size().sort_values(ascending=False),height=260); st.markdown('</div>',unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="agent-box"><div class="agent-badge">✦ AI OBSERVATION</div>',unsafe_allow_html=True)
        repeated=x.groupby(["task_type","repository"]).size().sort_values(ascending=False); repeated_count=int(repeated[repeated>=3].sum()) if len(repeated) else 0
        st.markdown(f'<h3>Context efficiency signal</h3><div class="agent-answer">{repeated_count:,} requests belong to repeated task/repository patterns. This is a candidate for caching or context filtering.</div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)


def page_costs(df):
    d=period_df(df); header("FINOPS","Costs","Understand where AI money is going.")
    total=d.cost.sum(); st.markdown(f'<div class="card insight"><div class="kicker">TOTAL AI SPEND</div><div class="big-number">{money(total)}</div><div class="positive">Observed from normalized telemetry</div></div>',unsafe_allow_html=True)
    st.write("")
    tabs=st.tabs(["By Team","By Model","By Developer","By Provider"])
    groups=["team","model","developer_email","provider"]
    for tab,g in zip(tabs,groups):
        with tab:
            x=d.groupby(g).agg(spend=("cost","sum"),requests=("id","count"),tokens=("total_tokens","sum")).reset_index().sort_values("spend",ascending=False)
            st.dataframe(x,use_container_width=True,hide_index=True,column_config={"spend":st.column_config.NumberColumn(format="$%.2f")})
            st.bar_chart(x.set_index(g)["spend"],height=280)
    st.subheader("Spend waterfall")
    prev=df[(df.timestamp<df.timestamp.max()-pd.Timedelta(days=30)) & (df.timestamp>=df.timestamp.max()-pd.Timedelta(days=60))] if not df.empty else pd.DataFrame()
    pc=prev.cost.sum() if not prev.empty else 0
    if pc: st.info(f"Previous 30-day spend: {money(pc)} → Current: {money(total)} ({(total/pc-1)*100:+.1f}%).")


def page_optimizer(df):
    header("OPTIMIZATION","Optimization Center","Find measurable AI cost opportunities with evidence.")
    opp=optimization_candidates(df); total=sum(o["saving"] for o in opp)
    metric_cards([("POTENTIAL MONTHLY SAVINGS",money(total),f"{len(opp)} candidates"),("HIGH IMPACT",str(sum(o["severity"]=="HIGH IMPACT" for o in opp)),"review required"),("AVG CONFIDENCE",f"{np.mean([o['confidence'] for o in opp])*100:.0f}%" if opp else "—","candidate confidence"),("STATUS","Analysis ready","safe mode")])
    st.write("")
    for i,o in enumerate(opp):
        with st.container(border=True):
            c1,c2,c3=st.columns([2.3,1,1])
            with c1:
                st.markdown(f'<span class="tag">{o["severity"]}</span><h3>{o["title"]}</h3><div class="evidence">{o["evidence"]}</div>',unsafe_allow_html=True)
            with c2: st.metric("Estimated saving",money(o["saving"]))
            with c3:
                st.metric("Confidence",f'{o["confidence"]*100:.0f}%')
                if st.button("Investigate",key=f"invest_{i}"): st.session_state.selected_opp=i; st.rerun()
    if "selected_opp" in st.session_state and st.session_state.selected_opp < len(opp):
        o=opp[st.session_state.selected_opp]; st.divider(); st.markdown(f"### Investigation · {o['title']}"); st.write(o["evidence"]); st.markdown(f"**Recommended action:** {o['action']}")
        st.warning("Estimated savings are analytical estimates. Validate quality and policy impact before activation.")
        if st.button("Create draft policy from this recommendation",key="draft_policy"):
            with SessionLocal() as s:
                s.add(Policy(organization_id=ORG_ID,name=o["title"],condition_json=json.dumps({"complexity":"LOW"}),action_json=json.dumps({"route_to":"gpt-5-mini"}),status="draft")); s.commit(); audit(s,ORG_ID,st.session_state.actor,"policy_drafted",{"name":o["title"]})
            st.success("Draft policy created. Review it under Policies before activation.")


def page_agent(df):
    header("INTELLIGENCE","AI Cost Agent","Ask questions about your actual company telemetry.")
    if not st.session_state.agent_messages:
        st.markdown('<div class="agent-box"><div class="agent-badge">✦ TOKENPILOT AI COST AGENT</div><h2>What would you like to investigate?</h2><div class="agent-answer">The agent uses the same normalized telemetry and optimization engine as the dashboard. It cannot claim actions it has not executed.</div></div>',unsafe_allow_html=True)
        suggestions=["Why did our AI spend increase?","Where can we save the most money?","Which teams use the most AI?","Show me the biggest optimization opportunities."]
        cols=st.columns(2)
        for c,q in zip(cols*2,suggestions):
            if c.button(q,key=f"suggest_{q}"): st.session_state.agent_messages.append(("user",q)); st.session_state.agent_messages.append(("assistant",agent_answer(q,df))); st.rerun()
    for role,msg in st.session_state.agent_messages:
        if role=="user":
            st.chat_message("user").write(msg)
        else:
            with st.chat_message("assistant"):
                st.markdown(msg)
                st.caption(f"Evidence: normalized telemetry · last 30 days · generated {datetime.now().strftime('%H:%M')}")
    prompt=st.chat_input("Ask why, where, what changed, or how to optimize…")
    if prompt:
        st.session_state.agent_messages.append(("user",prompt)); st.session_state.agent_messages.append(("assistant",agent_answer(prompt,df))); st.rerun()
    if st.session_state.agent_messages and st.button("Clear conversation"):
        st.session_state.agent_messages=[]; st.rerun()


def page_policies(df):
    header("GOVERNANCE","AI Policies","Turn analysis into controlled, auditable rules.")
    with SessionLocal() as s:
        policies=s.scalars(select(Policy).where(Policy.organization_id==ORG_ID).order_by(Policy.created_at.desc())).all()
    c1,c2=st.columns([3,1]);
    with c1: st.markdown(f"**{len(policies)} policies** · {sum(p.status=='active' for p in policies)} active")
    with c2:
        if st.button("+ Create policy",type="primary"): st.session_state.show_policy_form=True
    if st.session_state.get("show_policy_form"):
        with st.form("policy_form"):
            st.subheader("Create AI policy")
            name=st.text_input("Policy name","Low Complexity Model Routing")
            complexity=st.selectbox("WHEN task complexity is",["LOW","MEDIUM","HIGH"])
            team=st.selectbox("AND team is",["Any","Engineering","Backend","Frontend","Data","Platform"])
            route=st.selectbox("THEN route to",["gpt-5-mini","claude-haiku","gemini-flash","gpt-5","claude-sonnet"])
            submitted=st.form_submit_button("Create draft")
            if submitted:
                cond={"complexity":complexity,"team":None if team=="Any" else team}; act={"route_to":route}
                with SessionLocal() as s:
                    s.add(Policy(organization_id=ORG_ID,name=name,condition_json=json.dumps(cond),action_json=json.dumps(act),status="draft")); s.commit(); audit(s,ORG_ID,st.session_state.actor,"policy_created",{"name":name})
                st.session_state.show_policy_form=False; st.success("Draft policy created."); st.rerun()
    for p in policies:
        cond=json.loads(p.condition_json); act=json.loads(p.action_json)
        with st.container(border=True):
            c1,c2,c3=st.columns([2.3,2,1])
            with c1: st.markdown(f"**{p.name}**\n\n`{p.status.upper()}`")
            with c2: st.markdown(f"WHEN `{cond}`  \nTHEN `{act}`")
            with c3:
                if p.status=="draft":
                    if st.button("Activate",key=f"activate_{p.id}"):
                        with SessionLocal() as s:
                            obj=s.get(Policy,p.id); obj.status="active"; s.commit(); audit(s,ORG_ID,st.session_state.actor,"policy_activated",{"policy_id":p.id})
                        st.success("Policy activated."); st.rerun()


def page_budgets(df):
    header("GOVERNANCE","Budgets","Monitor organizational AI spend against explicit limits.")
    d=period_df(df); total=d.cost.sum(); budget=50000.0; pct=min(100,total/budget*100)
    st.progress(pct/100,text=f"Organization · {money(total)} / {money(budget)} · {pct:.1f}% used")
    metric_cards([("CURRENT SPEND",money(total),"last 30 days"),("MONTHLY BUDGET",money(budget),"configured"),("REMAINING",money(max(0,budget-total)),"before limit"),("STATUS","Healthy" if pct<80 else "Warning","budget control")])
    st.write("")
    x=d.groupby("team").cost.sum().sort_values(ascending=False)
    st.bar_chart(x,height=320)


def page_integrations(df):
    header("WORKSPACE","Integrations","Connect company-managed AI telemetry sources.")
    integrations=[("OpenAI","API / gateway","Connected"),("Anthropic","API / gateway","Connected"),("Google Gemini","API / gateway","Available"),("GitHub","Engineering metadata","Connected"),("GitHub Copilot","Coding telemetry","Available"),("Cursor","Coding telemetry","Available")]
    for name,kind,status in integrations:
        with st.container(border=True):
            c1,c2,c3=st.columns([2,2,1])
            c1.markdown(f"**{name}**")
            c2.caption(kind)
            c3.markdown(f'<span class="{ "status-ok" if status=="Connected" else "status-warn" }">● {status}</span>',unsafe_allow_html=True)
    st.info("For production, connect only company-managed accounts, gateways or approved telemetry sources. TokenPilot does not inspect personal consumer AI accounts.")
    st.subheader("Import telemetry CSV")
    file=st.file_uploader("CSV schema is documented in README.md",type=["csv"])
    if file and st.button("Validate and import"):
        ok,msg=ingest_csv(file,ORG_ID,st.session_state.actor); (st.success if ok else st.error)(msg); st.rerun() if ok else None


def page_settings(df):
    header("WORKSPACE","Settings","Workspace configuration and operational controls.")
    tabs=st.tabs(["Workspace","Security","Audit Log","Pricing"])
    with tabs[0]:
        st.text_input("Organization","Acme Technologies",disabled=True); st.text_input("Workspace ID",f"org_{ORG_ID:08d}",disabled=True)
    with tabs[1]:
        st.info("Production authentication should use an external identity provider through Streamlit OIDC. Secrets should live in Streamlit secrets or your cloud secret manager.")
        st.code("LLM_BASE_URL=http://localhost:11434/v1\nLLM_API_KEY=ollama\nLLM_MODEL=llama3.1:8b",language="bash")
    with tabs[2]:
        with SessionLocal() as s: logs=s.scalars(select(AuditLog).where(AuditLog.organization_id==ORG_ID).order_by(AuditLog.created_at.desc()).limit(100)).all()
        st.dataframe(pd.DataFrame([{"time":x.created_at,"actor":x.actor,"action":x.action,"details":x.details} for x in logs]),use_container_width=True,hide_index=True)
    with tabs[3]:
        st.dataframe(pd.DataFrame([{"model":k,"input_per_1m":v["input"],"output_per_1m":v["output"]} for k,v in PRICING.items()]),use_container_width=True,hide_index=True)


def sidebar():
    with st.sidebar:
        st.markdown('<div class="brand"><div class="brand-title">✦ TOKENPILOT</div><div class="brand-sub">AI COST INTELLIGENCE</div></div>',unsafe_allow_html=True)
        groups={"INTELLIGENCE":["Overview","Usage","Developers","Costs","Optimizer","AI Agent"],"GOVERNANCE":["Policies","Budgets"],"WORKSPACE":["Integrations","Settings"]}
        icons={"Overview":"◈","Usage":"◌","Developers":"◎","Costs":"$","Optimizer":"✦","AI Agent":"◉","Policies":"◇","Budgets":"▣","Integrations":"⊕","Settings":"⚙"}
        for group,pages in groups.items():
            st.markdown(f'<div class="navlabel">{group}</div>',unsafe_allow_html=True)
            for p in pages:
                if st.button(f'{icons[p]}  {p}',key=f"nav_{p}",use_container_width=True,type="primary" if st.session_state.page==p else "secondary"):
                    st.session_state.page=p; st.rerun()
        st.divider(); st.markdown('<span class="status-ok">●</span> All systems operational',unsafe_allow_html=True); st.caption("Acme Technologies · Admin")


def main():
    ensure_session(); sidebar(); df=load_events(ORG_ID)
    page=st.session_state.page
    if page=="Overview": page_overview(df)
    elif page=="Usage": page_usage(df)
    elif page=="Developers": page_developers(df)
    elif page=="Costs": page_costs(df)
    elif page=="Optimizer": page_optimizer(df)
    elif page=="AI Agent": page_agent(df)
    elif page=="Policies": page_policies(df)
    elif page=="Budgets": page_budgets(df)
    elif page=="Integrations": page_integrations(df)
    elif page=="Settings": page_settings(df)

if __name__ == "__main__":
    main()
