# ─────────────────────────────────────────────────────────
# STRICT MULTI-AGENT ARCHITECTURE using LangGraph + Mistral
#
# INPUT  : JSON task with topic, agent roles, ground truth
# FLOW   : Mother reads JSON → delegates to each agent
#          → collects output → evaluates vs ground truth
# LOGS   : Execution flow logs showing what is happening
# ─────────────────────────────────────────────────────────

import json
import time
from datetime import datetime
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_mistralai import ChatMistralAI

# ── SETUP ─────────────────────────────────────────────────
llm = ChatMistralAI(
    model="mistral-small-2506",
    api_key="szdc0RuzZlBjdpGNMn0k6bgoCvkROyB4"
)

LOG_FILE       = "pipeline_execution.log"
PIPELINE_START = None

# ─────────────────────────────────────────────────────────
# LOGGER — execution flow only
# ─────────────────────────────────────────────────────────
def log(agent: str, message: str):
    timestamp  = datetime.now().strftime("%H:%M:%S")
    elapsed    = f"+{round(time.time() - PIPELINE_START, 1)}s" if PIPELINE_START else ""
    log_line   = f"[{timestamp}] [{elapsed:>7}]  [{agent:<12}]  {message}"

    print(log_line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_line + "\n")


# ─────────────────────────────────────────────────────────
# INPUT JSON TASK
# ─────────────────────────────────────────────────────────
TASK_JSON = {
    "task_id": "investment_report_001",
    "topic"  : "Tesla Inc. as an investment opportunity in 2025",

    "agent_roles": {
        "researcher": """Research Tesla Inc. and return exactly 5 bullet points covering:
            - Latest revenue and profit margin
            - Current market cap and P/E ratio
            - Top 2 risks facing Tesla in 2025
            - Top 2 growth opportunities
            - Main competitor comparison
            Format each as a clear single-line bullet point starting with -""",

        "writer": """You are a financial report writer.
            Using the facts provided write a professional investment report with exactly these 4 paragraphs:
            Paragraph 1 - Financial Health Summary (must mention revenue, profit margin, market cap, valuation)
            Paragraph 2 - Risk Assessment (must mention competition, margins, regulatory)
            Paragraph 3 - Growth Potential (must mention autonomous, energy, expansion, EV)
            Paragraph 4 - Final Investment Recommendation (must include exactly one of: Buy, Hold, Sell
                          and mention portfolio, investors, risk tolerance)
            Separate each paragraph with a blank line.""",

        "editor": """You are a senior financial editor.
            Review the investment report and make sure ALL of these words appear naturally:
            - Financial : revenue, profit, margin, valuation, market cap, P/E, earnings
            - Risk      : competition, regulatory, margins, volatility, dilution
            - Growth    : autonomous, energy storage, expansion, EV market, Gigafactory
            - Action    : Buy/Hold/Sell (one must appear), portfolio, investors, risk tolerance
            Fix grammar, improve flow, keep professional tone.
            Title must be on the very first line.
            Return only the final polished report."""
    },

    "ground_truth": {
        "must_contain_keywords": [
            "revenue", "profit", "margin", "valuation", "market cap",
            "earnings", "competition", "regulatory", "volatility",
            "autonomous", "expansion", "Gigafactory", "EV",
            "portfolio", "investors", "risk tolerance"
        ],
        "min_paragraphs"      : 3,
        "must_have_title"     : True,
        "expected_fact_count" : 5
    }
}

# ── SHARED STATE ──────────────────────────────────────────
class BlogState(TypedDict):
    topic        : str
    agent_roles  : dict
    ground_truth : dict
    facts        : str
    draft        : str
    final        : str
    evaluation   : dict
    next_agent   : str

# ─────────────────────────────────────────────────────────
# 👑 MOTHER AGENT
# ─────────────────────────────────────────────────────────
def mother_agent(state: BlogState) -> BlogState:

    # Log what Mother received
    received_from = (
        "RESEARCHER" if state["facts"] and not state["draft"]  else
        "WRITER"     if state["draft"] and not state["final"]  else
        "EDITOR"     if state["final"] and not state["evaluation"] else
        "EVALUATOR"  if state["evaluation"] else
        "USER"
    )
    log("MOTHER", f"Received state from {received_from}")
    log("MOTHER", f"State snapshot → facts={'HAS CONTENT' if state['facts'] else 'EMPTY'} | draft={'HAS CONTENT' if state['draft'] else 'EMPTY'} | final={'HAS CONTENT' if state['final'] else 'EMPTY'} | evaluation={'HAS CONTENT' if state['evaluation'] else 'EMPTY'}")
    log("MOTHER", "Thinking... asking LLM to decide next agent")

    system = """You are the Mother Agent — the orchestrator of a blog writing team.
You have 4 sub-agents:
  - "researcher" : finds facts about the topic
  - "writer"     : writes a blog post using the facts
  - "editor"     : polishes the draft
  - "evaluator"  : compares final output with ground truth

Decision Rules — follow these strictly:
  1. facts is empty                          → call "researcher"
  2. facts exists, draft is empty            → call "writer"
  3. draft exists, final is empty            → call "editor"
  4. final exists, evaluation is empty       → call "evaluator"
  5. evaluation exists                       → call "done"

Respond ONLY in this exact JSON format with no extra text:
{
  "decision": "researcher" | "writer" | "editor" | "evaluator" | "done",
  "reason"  : "one line reason"
}"""

    state_summary = f"""
Current State:
- facts      : {"HAS CONTENT" if state["facts"]      else "EMPTY"}
- draft      : {"HAS CONTENT" if state["draft"]      else "EMPTY"}
- final      : {"HAS CONTENT" if state["final"]      else "EMPTY"}
- evaluation : {"HAS CONTENT" if state["evaluation"] else "EMPTY"}
What should I do next?
"""

    response = llm.invoke([
        ("system", system),
        ("human",  state_summary)
    ])

    raw = response.content.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    parsed   = json.loads(raw.strip())
    decision = parsed["decision"]
    reason   = parsed["reason"]

    log("MOTHER", f"Decision made → sending to {decision.upper()} | Reason: {reason}")

    return {**state, "next_agent": decision}


# ─────────────────────────────────────────────────────────
# SUB-AGENT 1: RESEARCHER
# ─────────────────────────────────────────────────────────
def researcher_agent(state: BlogState) -> BlogState:
    log("RESEARCHER", "Task received from Mother")
    log("RESEARCHER", f"Topic → {state['topic']}")
    log("RESEARCHER", "Started working...")

    t0       = time.time()
    response = llm.invoke([
        ("system", f"""You are a Research Agent.
Your role: {state['agent_roles']['researcher']}
Return exactly 5 bullet-point facts.
Format:
- Fact 1
- Fact 2
- Fact 3
- Fact 4
- Fact 5"""),
        ("human", f"Research this topic: {state['topic']}")
    ])
    elapsed    = round(time.time() - t0, 1)
    fact_lines = [l for l in response.content.split("\n") if l.strip().startswith("-")]

    log("RESEARCHER", f"Finished | Time taken: {elapsed}s | Output size: {len(response.content)} chars | Facts found: {len(fact_lines)}")
    log("RESEARCHER", "Reporting back to Mother")

    return {**state, "facts": response.content}


# ─────────────────────────────────────────────────────────
# SUB-AGENT 2: WRITER
# ─────────────────────────────────────────────────────────
def writer_agent(state: BlogState) -> BlogState:
    log("WRITER", "Task received from Mother")
    log("WRITER", f"Input → topic: {len(state['topic'])} chars | facts: {len(state['facts'])} chars")
    log("WRITER", "Started working...")

    t0       = time.time()
    response = llm.invoke([
        ("system", f"""You are a Writer Agent.
Your role: {state['agent_roles']['writer']}"""),
        ("human", f"Topic: {state['topic']}\n\nFacts:\n{state['facts']}")
    ])
    elapsed    = round(time.time() - t0, 1)
    paragraphs = [p.strip() for p in response.content.split("\n\n") if p.strip()]

    log("WRITER", f"Finished | Time taken: {elapsed}s | Output size: {len(response.content)} chars | Paragraphs: {len(paragraphs)}")
    log("WRITER", "Reporting back to Mother")

    return {**state, "draft": response.content}


# ─────────────────────────────────────────────────────────
# SUB-AGENT 3: EDITOR
# ─────────────────────────────────────────────────────────
def editor_agent(state: BlogState) -> BlogState:
    log("EDITOR", "Task received from Mother")
    log("EDITOR", f"Input → draft size: {len(state['draft'])} chars")
    log("EDITOR", "Started working...")

    t0       = time.time()
    response = llm.invoke([
        ("system", f"""You are an Editor Agent.
Your role: {state['agent_roles']['editor']}"""),
        ("human", f"Edit this blog post:\n\n{state['draft']}")
    ])
    elapsed = round(time.time() - t0, 1)

    log("EDITOR", f"Finished | Time taken: {elapsed}s | Output size: {len(response.content)} chars")
    log("EDITOR", "Reporting back to Mother")

    return {**state, "final": response.content}


# ─────────────────────────────────────────────────────────
# SUB-AGENT 4: EVALUATOR
# ─────────────────────────────────────────────────────────
def evaluator_agent(state: BlogState) -> BlogState:
    log("EVALUATOR", "Task received from Mother")
    log("EVALUATOR", f"Input → final report: {len(state['final'])} chars | Keywords to check: {len(state['ground_truth']['must_contain_keywords'])}")
    log("EVALUATOR", "Started evaluation against ground truth...")

    gt      = state["ground_truth"]
    final   = state["final"].lower()
    results = {}

    # Check 1 — Keywords
    found   = [kw for kw in gt["must_contain_keywords"] if kw.lower() in final]
    missing = [kw for kw in gt["must_contain_keywords"] if kw.lower() not in final]
    results["keyword_check"] = {
        "found"  : found,
        "missing": missing,
        "passed" : len(missing) == 0
    }
    log("EVALUATOR", f"Keyword check  → {len(found)}/{len(gt['must_contain_keywords'])} found | Missing: {missing if missing else 'None'} | {'✅ PASS' if not missing else '❌ FAIL'}")

    # Check 2 — Paragraphs
    paragraphs = [p.strip() for p in state["final"].split("\n\n") if p.strip()]
    results["paragraph_check"] = {
        "expected": gt["min_paragraphs"],
        "found"   : len(paragraphs),
        "passed"  : len(paragraphs) >= gt["min_paragraphs"]
    }
    log("EVALUATOR", f"Paragraph check → Expected: {gt['min_paragraphs']} | Found: {len(paragraphs)} | {'✅ PASS' if results['paragraph_check']['passed'] else '❌ FAIL'}")

    # Check 3 — Title
    first_line = state["final"].strip().split("\n")[0]
    results["title_check"] = {
        "title" : first_line,
        "passed": len(first_line) > 0 and len(first_line) < 120
    }
    log("EVALUATOR", f"Title check    → '{first_line[:60]}...' | {'✅ PASS' if results['title_check']['passed'] else '❌ FAIL'}")

    # Check 4 — Fact count
    fact_lines = [l for l in state["facts"].split("\n") if l.strip().startswith("-")]
    results["fact_count_check"] = {
        "expected": gt["expected_fact_count"],
        "found"   : len(fact_lines),
        "passed"  : len(fact_lines) >= gt["expected_fact_count"]
    }
    log("EVALUATOR", f"Fact count     → Expected: {gt['expected_fact_count']} | Found: {len(fact_lines)} | {'✅ PASS' if results['fact_count_check']['passed'] else '❌ FAIL'}")

    # Overall
    passed = sum(1 for r in results.values() if r["passed"])
    total  = len(results)
    score  = round((passed / total) * 100)
    results["overall"] = {
        "passed" : passed,
        "total"  : total,
        "score"  : f"{score}%",
        "verdict": "✅ PASS" if score >= 75 else "❌ FAIL"
    }

    log("EVALUATOR", f"Score: {score}% | Checks passed: {passed}/{total} | Verdict: {results['overall']['verdict']}")
    log("EVALUATOR", "Reporting back to Mother")

    return {**state, "evaluation": results}


# ─────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────
def router(state: BlogState) -> Literal["researcher", "writer", "editor", "evaluator", "done"]:
    return state["next_agent"]


# ── BUILD THE GRAPH ───────────────────────────────────────
graph_builder = StateGraph(BlogState)

graph_builder.add_node("mother",     mother_agent)
graph_builder.add_node("researcher", researcher_agent)
graph_builder.add_node("writer",     writer_agent)
graph_builder.add_node("editor",     editor_agent)
graph_builder.add_node("evaluator",  evaluator_agent)

graph_builder.set_entry_point("mother")

graph_builder.add_conditional_edges(
    "mother",
    router,
    {
        "researcher" : "researcher",
        "writer"     : "writer",
        "editor"     : "editor",
        "evaluator"  : "evaluator",
        "done"       : END
    }
)

graph_builder.add_edge("researcher", "mother")
graph_builder.add_edge("writer",     "mother")
graph_builder.add_edge("editor",     "mother")
graph_builder.add_edge("evaluator",  "mother")

pipeline = graph_builder.compile()


# ── RUN IT ────────────────────────────────────────────────
if __name__ == "__main__":

    PIPELINE_START = time.time()

    # Clear old log file and write header
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"PIPELINE EXECUTION LOG\n")
        f.write(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Task    : {TASK_JSON['task_id']}\n")
        f.write(f"Topic   : {TASK_JSON['topic']}\n")
        f.write("=" * 80 + "\n\n")

    log("PIPELINE", f"Started | Task: {TASK_JSON['task_id']}")
    log("PIPELINE", f"Topic: {TASK_JSON['topic']}")
    log("PIPELINE", f"Ground truth keywords to match: {len(TASK_JSON['ground_truth']['must_contain_keywords'])}")
    print()

    initial_state: BlogState = {
        "topic"       : TASK_JSON["topic"],
        "agent_roles" : TASK_JSON["agent_roles"],
        "ground_truth": TASK_JSON["ground_truth"],
        "facts"       : "",
        "draft"       : "",
        "final"       : "",
        "evaluation"  : {},
        "next_agent"  : ""
    }

    result = pipeline.invoke(initial_state)

    total_time = round(time.time() - PIPELINE_START, 1)
    log("PIPELINE", f"Finished | Total time: {total_time}s")

    # ── FINAL OUTPUT ──────────────────────────────────────
    print("\n" + "═"*55)
    print("📄  FINAL REPORT")
    print("═"*55)
    print(result["final"])

    print("\n" + "═"*55)
    print("🧪  GROUND TRUTH EVALUATION")
    print("═"*55)
    ev = result["evaluation"]

    print(f"\n  Keyword Check   : {'✅ PASS' if ev['keyword_check']['passed']    else '❌ FAIL'}")
    print(f"    Found   : {ev['keyword_check']['found']}")
    print(f"    Missing : {ev['keyword_check']['missing']}")

    print(f"\n  Paragraph Check : {'✅ PASS' if ev['paragraph_check']['passed']  else '❌ FAIL'}")
    print(f"    Expected {ev['paragraph_check']['expected']} | Found {ev['paragraph_check']['found']}")

    print(f"\n  Title Check     : {'✅ PASS' if ev['title_check']['passed']      else '❌ FAIL'}")
    print(f"    Title : {ev['title_check']['title']}")

    print(f"\n  Fact Count      : {'✅ PASS' if ev['fact_count_check']['passed'] else '❌ FAIL'}")
    print(f"    Expected {ev['fact_count_check']['expected']} | Found {ev['fact_count_check']['found']}")

    print(f"\n  ── OVERALL ──────────────────────────")
    print(f"  Checks Passed : {ev['overall']['passed']} / {ev['overall']['total']}")
    print(f"  Final Score   : {ev['overall']['score']}")
    print(f"  Verdict       : {ev['overall']['verdict']}")
    print("═"*55)
    print(f"\n📁 Execution log saved to → {LOG_FILE}")