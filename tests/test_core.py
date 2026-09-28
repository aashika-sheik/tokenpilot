import os
os.environ["DATABASE_URL"] = "sqlite:///./test_tokenpilot.db"
from app import calculate_cost, classify_complexity, optimization_candidates
import pandas as pd

def test_cost_is_deterministic():
    a=calculate_cost("gpt-5-mini",1000000,1000000)
    assert a == (0.25,2.0,2.25)

def test_complexity():
    assert classify_complexity("Explain function",1000,500)=="LOW"
    assert classify_complexity("Architecture",1000,500)=="HIGH"

def test_optimizer_finds_premium_low_complexity():
    df=pd.DataFrame([
      {"timestamp":pd.Timestamp.now(),"developer_email":"a","team":"Eng","provider":"OpenAI","model":"gpt-5","input_tokens":1000,"output_tokens":1000,"total_tokens":2000,"task_type":"Explain function","task_complexity":"LOW","repository":"r","cost":.01125,"id":1},
      {"timestamp":pd.Timestamp.now(),"developer_email":"a","team":"Eng","provider":"OpenAI","model":"gpt-5","input_tokens":1000,"output_tokens":1000,"total_tokens":2000,"task_type":"Explain function","task_complexity":"LOW","repository":"r","cost":.01125,"id":2},
    ])
    out=optimization_candidates(df)
    assert out and out[0]["saving"] >= 0
