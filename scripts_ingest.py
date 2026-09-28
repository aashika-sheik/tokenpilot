"""CLI for importing JSONL telemetry into TokenPilot's database."""
import argparse, json
from datetime import datetime
from app import SessionLocal, UsageEvent, ORG_ID, calculate_cost, classify_complexity

p=argparse.ArgumentParser(); p.add_argument("file"); args=p.parse_args()
accepted=0
with SessionLocal() as s:
    existing={x for x in s.query(UsageEvent.request_id).filter(UsageEvent.organization_id==ORG_ID).all()}
    existing={x[0] for x in existing}
    for line in open(args.file, encoding="utf-8"):
        if not line.strip(): continue
        e=json.loads(line); rid=str(e["request_id"])
        if rid in existing: continue
        inp=int(e["input_tokens"]); out=int(e["output_tokens"]); ic,oc,total=calculate_cost(e["model"],inp,out)
        s.add(UsageEvent(organization_id=ORG_ID,request_id=rid,timestamp=datetime.fromisoformat(e["timestamp"].replace("Z","+00:00")).replace(tzinfo=None),developer_email=e["developer_email"],team=e["team"],provider=e["provider"],model=e["model"],input_tokens=inp,output_tokens=out,task_type=e["task_type"],task_complexity=e.get("task_complexity") or classify_complexity(e["task_type"],inp,out),repository=e.get("repository"),latency_ms=e.get("latency_ms"),status=e.get("status","success"),input_cost=ic,output_cost=oc,total_cost=total)); accepted+=1
    s.commit()
print(f"Imported {accepted} events")
