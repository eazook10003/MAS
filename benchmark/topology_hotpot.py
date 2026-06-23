import json
import logging
import os
import time
from operator import add
from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from benchmark.tools import make_local_search, make_lookup


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME    = os.environ.get("VLLM_MODEL", "Qwen2.5-7B-Instruct")

RESEARCHERS = ["searcher", "reader", "hop_chainer"]
ACTIVE_RESEARCHERS = []
for _r in os.environ.get("ACTIVE_RESEARCHERS", "searcher,reader,hop_chainer").split(","):
    _r = _r.strip()
    if _r in RESEARCHERS:
        ACTIVE_RESEARCHERS.append(_r)

TOOL_CAPS = {"searcher": 3, "reader": 3, "hop_chainer": 3}

MA_LOG = os.environ.get("MA_LOG", "/nfs/hpc/share/kangdo/personal_ma/logs/bench_agents.log")
logging.basicConfig(
    filename=MA_LOG,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bench")

TOOL_LOG_CAP = 500


AGENTS = {
    "planner": {
        "role": "Planner",
        "system": (
            "You are the planner for a multi-hop QA team. "
            "Classify the question as 'bridge' (the answer requires chaining two facts, "
            "e.g. find X first, then ask about X) or 'comparison' (compare two entities). "
            "Then write one short hint for each researcher. Respond EXACTLY in this format:\n"
            "TYPE: bridge or comparison\n"
            "HINT searcher: <2-3 diverse search keyword sets>\n"
            "HINT reader: <which facts to look up in the given passages>\n"
            "HINT hop_chainer: <which entity chain to follow, step by step>"
        ),
    },
    "searcher": {
        "role": "Searcher",
        "system": (
            "You are the Searcher. You may ONLY use the local_search tool. "
            "Diversify your query keywords to collect relevant passages broadly. "
            "Do NOT chase clues from one result into the next — just gather evidence widely. "
            "You have a budget of 3 tool calls. Then conclude in this format:\n"
            "EVIDENCE:\n- <passage title>: <relevant fact>\n"
            "ANSWER_CANDIDATE: <short answer, or 'unknown'>"
        ),
    },
    "reader": {
        "role": "Reader",
        "system": (
            "You are the Reader. You may ONLY use the lookup tool, which finds sentences "
            "inside this question's 10 reference passages. Read carefully and extract facts "
            "relevant to the question (names, dates, places), always quoting the source sentence. "
            "You have a budget of 4 tool calls. Then conclude in this format:\n"
            "EVIDENCE:\n- [passage title] <quoted sentence>\n"
            "ANSWER_CANDIDATE: <short answer, or 'unknown'>"
        ),
    },
    "hop_chainer": {
        "role": "Hop-chainer",
        "system": (
            "You are the Hop-chainer. Use local_search and lookup in a ReAct loop: "
            "search, identify the bridge entity in the result, then search again with that "
            "entity. Repeat until you can answer the full question. "
            "You have a budget of 6 tool calls. Then conclude in this format:\n"
            "EVIDENCE:\n- <fact 1 (hop 1)>\n- <fact 2 (hop 2)>\n"
            "ANSWER_CANDIDATE: <short answer, or 'unknown'>"
        ),
    },
    "synthesizer": {
        "role": "Synthesizer",
        "system": (
            "Given the researchers' findings, output the single best answer to the "
            "original question. HotpotQA style: a short phrase, name, date, or 'yes'/'no'. "
            "Output only the answer, no explanation or punctuation."
        ),
    },
}


def _merge_dicts(a: dict, b: dict) -> dict:
    return {**a, **b}


class GraphState(TypedDict):
    question: str
    passages: list
    qtype: str
    hints: dict
    outputs: Annotated[dict, _merge_dicts]
    tool_calls: Annotated[int, add]
    timings: Annotated[dict, _merge_dicts]


def build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=VLLM_BASE_URL,
        api_key="dummy",
        model=MODEL_NAME,
        temperature=0.3,
    )


def _parse_plan(text: str) -> tuple[str, dict]:
    qtype = "bridge"
    hints = {}
    for line in text.splitlines():
        s = line.strip()
        if s.upper().startswith("TYPE:"):
            val = s.split(":", 1)[1].strip().lower()
            if "comparison" in val:
                qtype = "comparison"
        elif s.upper().startswith("HINT "):
            head, _, val = s.partition(":")
            words = head.split()
            if len(words) >= 2:
                name = words[1].strip().lower()
                if name in RESEARCHERS:
                    hints[name] = val.strip()
    return qtype, hints


def _parse_candidate(text: str) -> str:
    for line in text.splitlines():
        s = line.strip()
        if s.upper().startswith("ANSWER_CANDIDATE"):
            if ":" in s:
                return s.split(":", 1)[1].strip()
    return ""


def _make_planner(llm: ChatOpenAI, log_io: bool, qid: str):
    agent = AGENTS["planner"]

    def node(state: GraphState) -> dict:
        t_start = time.time()
        question = state["question"]
        bar = "=" * 70
        user_msg = (
            f"Question: {question}\n\n"
            f"Active researchers: {', '.join(ACTIVE_RESEARCHERS)}"
        )

        if log_io:
            log.info(
                "\n%s\n[q=%s | plan / %s] -- INPUT --\n[SYSTEM]\n%s\n[USER]\n%s\n%s",
                bar, qid, agent["role"], agent["system"], user_msg, bar,
            )

        t_llm = time.time()
        result = llm.invoke([
            SystemMessage(content=agent["system"]),
            HumanMessage(content=user_msg),
        ])
        llm_time = time.time() - t_llm
        text = result.content
        qtype, hints = _parse_plan(text)

        if log_io:
            log.info(
                "\n%s\n[q=%s | plan / %s] -- OUTPUT --\ntype=%s\n%s\n%s",
                bar, qid, agent["role"], qtype, text, bar,
            )

        wall = time.time() - t_start
        return {
            "outputs": {"plan": text},
            "qtype": qtype,
            "hints": hints,
            "timings": {"plan": {"wall": wall, "llm": llm_time, "tool": {}}},
        }

    return node


def _make_researcher(name: str, llm: ChatOpenAI, tool_registry: dict,
                     log_io: bool, qid: str):
    agent = AGENTS[name]
    cap = TOOL_CAPS[name]

    if name == "searcher":
        tool_names = ["local_search"]
    elif name == "reader":
        tool_names = ["lookup"]
    else:
        tool_names = ["local_search", "lookup"]

    my_tools = [tool_registry[n] for n in tool_names]
    llm_bound = llm.bind_tools(my_tools)
    tool_lookup = {t.name: t for t in my_tools}

    def node(payload: dict) -> dict:
        t_start = time.time()
        llm_time = 0.0
        tool_time = {}
        question = payload["question"]
        hint = payload.get("hints", {}).get(name, "")
        bar = "=" * 70

        parts = [f"Question: {question}"]
        if hint:
            parts.append(f"Planner's hint for you: {hint}")
        user_msg = "\n\n".join(parts)

        if log_io:
            log.info(
                "\n%s\n[q=%s | %s / %s] -- INPUT --\n[SYSTEM]\n%s\n[USER]\n%s\n%s",
                bar, qid, name, agent["role"], agent["system"], user_msg, bar,
            )

        messages = [
            SystemMessage(content=agent["system"]),
            HumanMessage(content=user_msg),
        ]

        tool_call_count = 0
        final_text = ""
        while True:
            t_llm = time.time()
            result = llm_bound.invoke(messages)
            llm_time += time.time() - t_llm
            messages.append(result)

            if log_io:
                log.info(
                    "\n[q=%s | %s] -- LLM RAW RESULT --\ncontent: %r\ntool_calls: %s",
                    qid, name,
                    result.content,
                    json.dumps(result.tool_calls, ensure_ascii=False, indent=2)
                    if result.tool_calls else "none",
                )

            tcs = getattr(result, "tool_calls", None)
            if not tcs:
                final_text = result.content
                break

            for tc in tcs:
                tool_name = tc["name"]
                tool_args = tc["args"]
                if tool_call_count >= cap:
                    output = ("(tool budget exhausted — no more tool calls allowed. "
                              "Provide your final answer.)")
                elif tool_name not in tool_lookup:
                    output = f"ERROR: unknown tool {tool_name}"
                    tool_call_count += 1
                else:
                    t_tool = time.time()
                    try:
                        output = tool_lookup[tool_name].invoke(tool_args)
                    except Exception as e:
                        output = f"ERROR running {tool_name}: {e}"
                    tool_time[tool_name] = tool_time.get(tool_name, 0.0) + (time.time() - t_tool)
                    tool_call_count += 1

                if log_io:
                    out_s = str(output)
                    if out_s.strip():
                        out_log = out_s[:TOOL_LOG_CAP]
                    else:
                        out_log = "(EMPTY OUTPUT — tool ran fine, stdout/stderr both empty)"
                    log.info(
                        "\n[q=%s | %s] -- TOOL CALL #%d --\n%s(%s)\n→ %s",
                        qid, name, tool_call_count,
                        tool_name, tool_args, out_log,
                    )

                messages.append(ToolMessage(
                    content=str(output),
                    tool_call_id=tc["id"],
                ))

            if tool_call_count >= cap:
                messages.append(HumanMessage(content=(
                    "Tool budget exhausted. Output your final EVIDENCE and "
                    "ANSWER_CANDIDATE now, without any more tool calls."
                )))
                t_llm = time.time()
                result = llm.invoke(messages)
                llm_time += time.time() - t_llm
                messages.append(result)
                final_text = result.content
                break

        if not final_text:
            for m in reversed(messages):
                if hasattr(m, "type") and m.type == "ai" and getattr(m, "content", ""):
                    final_text = m.content
                    break

        wall = time.time() - t_start
        result_dict = {
            "role": name,
            "evidence": final_text,
            "answer_candidate": _parse_candidate(final_text),
            "tool_calls": tool_call_count,
            "wall_time_sec": round(wall, 3),
        }

        if log_io:
            log.info(
                "\n%s\n[q=%s | %s / %s] -- OUTPUT (%.2fs, %d tool calls) --\n%s\n%s",
                bar, qid, name, agent["role"], wall, tool_call_count, final_text, bar,
            )

        return {
            "outputs": {name: result_dict},
            "tool_calls": tool_call_count,
            "timings": {name: {"wall": wall, "llm": llm_time, "tool": tool_time}},
        }

    return node


def _make_synthesizer(llm: ChatOpenAI, log_io: bool, qid: str):
    agent = AGENTS["synthesizer"]

    def node(state: GraphState) -> dict:
        t_start = time.time()
        question = state["question"]
        bar = "=" * 70

        parts = [f"Original question: {question}"]
        for r in ACTIVE_RESEARCHERS:
            out = state["outputs"].get(r)
            if out:
                parts.append(
                    f"[{r}]\nanswer_candidate: {out['answer_candidate'] or '(none)'}\n"
                    f"{out['evidence']}"
                )
        user_msg = "\n\n".join(parts)

        if log_io:
            log.info(
                "\n%s\n[q=%s | synthesize / %s] -- INPUT --\n[SYSTEM]\n%s\n[USER]\n%s\n%s",
                bar, qid, agent["role"], agent["system"], user_msg, bar,
            )

        t_llm = time.time()
        result = llm.invoke([
            SystemMessage(content=agent["system"]),
            HumanMessage(content=user_msg),
        ])
        llm_time = time.time() - t_llm
        text = result.content

        if log_io:
            log.info(
                "\n%s\n[q=%s | synthesize / %s] -- OUTPUT --\n%s\n%s",
                bar, qid, agent["role"], text, bar,
            )

        wall = time.time() - t_start
        return {
            "outputs": {"synthesize": text},
            "timings": {"synthesize": {"wall": wall, "llm": llm_time, "tool": {}}},
        }

    return node


def build_graph(question: str, passages: list[dict], llm: ChatOpenAI,
                log_io: bool = False, qid: str = "?"):
    tool_registry = {
        "local_search": make_local_search(),
        "lookup":       make_lookup(passages),
    }

    def dispatch(state: GraphState):
        sends = []
        for r in ACTIVE_RESEARCHERS:
            sends.append(Send(r, {
                "question": state["question"],
                "hints":    state.get("hints", {}),
            }))
        return sends

    graph = StateGraph(GraphState)

    graph.add_node("plan", _make_planner(llm, log_io, qid))
    for r in RESEARCHERS:
        graph.add_node(r, _make_researcher(r, llm, tool_registry, log_io, qid))
    graph.add_node("synthesize", _make_synthesizer(llm, log_io, qid), defer=True)

    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", dispatch, RESEARCHERS)
    for r in RESEARCHERS:
        graph.add_edge(r, "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()
