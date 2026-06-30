import logging
import operator
import os
import time
from operator import add
from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from fanout_mas.tools_fanout import make_local_search


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME    = os.environ.get("VLLM_MODEL", "Qwen2.5-7B-Instruct")

RESEARCHER_TOOL_CAP = int(os.environ.get("RESEARCHER_TOOL_CAP", "4"))
DISCOVER_TOOL_CAP = int(os.environ.get("DISCOVER_TOOL_CAP", "3"))
MAX_SUBQ = int(os.environ.get("MAX_SUBQ", "48"))

MA_LOG = os.environ.get("MA_LOG", "/nfs/hpc/share/kangdo/personal_ma/logs/bench_fanout_agents.log")
logging.basicConfig(filename=MA_LOG, filemode="w", level=logging.INFO,
                    format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("fanout")
TOOL_LOG_CAP = 500


AGENTS = {
    "decompose": {
        "role": "Decomposer",
        "system": (
            "You are the planner for a fan-out question-answering team. "
            "A fan-out question asks the same thing about MANY entities at once "
            "(e.g. 'the population of every country bordering France'). "
            "You have a local_search tool (BM25 over a local Wikipedia corpus) and a "
            "budget of {cap} searches. FIRST, if you do not already know the EXACT list "
            "of entities the question fans out over, use local_search to find that list "
            "(e.g. search 'countries bordering France') and read the returned passages to "
            "get the precise entities. THEN break the question into independent "
            "SUB-QUESTIONS, one per entity / item, so each can be researched on its own "
            "and in parallel. "
            "Output ONE sub-question per line, each starting EXACTLY with 'SUBQ: '. "
            "Do not number them."
        ),
    },
    "researcher": {
        "role": "Researcher",
        "system": (
            "You research ONE sub-question using the local_search tool, which does BM25 "
            "retrieval over a local Wikipedia corpus. Search with focused keywords, read the "
            "returned passages, and extract the specific fact asked for (a name, number, date, "
            "yes/no). You have a budget of {cap} tool calls. Then conclude EXACTLY:\n"
            "EVIDENCE: <the source passage title + the quoted fact>\n"
            "ANSWER: <short answer, or 'unknown'>"
        ),
    },
    "aggregate": {
        "role": "Aggregator",
        "system": (
            "You assemble the final answer to a fan-out question from the researchers' "
            "per-entity findings. Combine them into one complete, structured answer that "
            "directly answers the original question. Be concise: list each entity with its "
            "answer. Do not add commentary."
        ),
    },
}


def _merge_dicts(a: dict, b: dict) -> dict:
    return {**a, **b}


class GraphState(TypedDict):
    question: str
    sub_questions: list
    researcher_outputs: Annotated[list, operator.add]  
    outputs: Annotated[dict, _merge_dicts]              
    tool_calls: Annotated[int, add]
    timings: Annotated[dict, _merge_dicts]


def build_llm() -> ChatOpenAI:
    return ChatOpenAI(base_url=VLLM_BASE_URL, api_key="dummy",
                      model=MODEL_NAME, temperature=0.3)


def _parse_subquestions(text: str) -> list[str]:
    subs = []
    for line in text.splitlines():
        s = line.strip()
        if s.upper().startswith("SUBQ:"):
            q = s.split(":", 1)[1].strip()
            if q:
                subs.append(q)
    if not subs:                      
        subs = ["(answer directly) "]
    return subs[:MAX_SUBQ]


def _parse_answer(text: str) -> str:
    for line in text.splitlines():
        s = line.strip()
        if s.upper().startswith("ANSWER"):
            if ":" in s:
                return s.split(":", 1)[1].strip()
    return ""


def _fmt_args(args: dict) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in args.items())


def _fmt_tcs(tcs) -> str:
    if not tcs:
        return "none"
    return "; ".join(f"{tc['name']}({_fmt_args(tc['args'])})" for tc in tcs)


def _run_tool_loop(llm_bound, llm, messages, cap, tool_lookup,
                   log_io, qid, label, first_input_desc, budget_msg):
    llm_time = 0.0
    tool_time = {}
    tool_call_count = 0
    final_text = ""
    turn = 0
    llm_input_desc = first_input_desc 

    while True:
        turn += 1
        t_llm = time.time()
        result = llm_bound.invoke(messages)
        llm_time += time.time() - t_llm
        messages.append(result)

        tcs = getattr(result, "tool_calls", None)
        if log_io:
            log.info("\n[q=%s | %s] -- LLM CALL (turn %d) --\n[INPUT]  %s\n[OUTPUT] content: %r\n         tool_calls: %s",
                     qid, label, turn, llm_input_desc, result.content, _fmt_tcs(tcs))

        if not tcs:
            final_text = result.content
            break

        tool_results = []        
        for tc in tcs:
            tname, targs = tc["name"], tc["args"]
            if tool_call_count >= cap:
                output = "(tool budget exhausted — give your final answer now.)"
            elif tname not in tool_lookup:
                output = f"ERROR: unknown tool {tname}"
                tool_call_count += 1
            else:
                t_tool = time.time()
                try:
                    output = tool_lookup[tname].invoke(targs)
                except Exception as e:
                    output = f"ERROR running {tname}: {e}"
                tool_time[tname] = tool_time.get(tname, 0.0) + (time.time() - t_tool)
                tool_call_count += 1

            out_s = str(output)
            if log_io:
                out_log = out_s[:TOOL_LOG_CAP] if out_s.strip() else "(EMPTY OUTPUT)"
                log.info("\n[q=%s | %s] -- TOOL CALL #%d --\n[INPUT]  %s(%s)\n[OUTPUT] %s",
                         qid, label, tool_call_count, tname, _fmt_args(targs), out_log)

            snippet = out_s[:200].replace("\n", " ")
            tool_results.append(f"tool#{tool_call_count} ({tname}) → {snippet}")
            messages.append(ToolMessage(content=out_s, tool_call_id=tc["id"]))

        llm_input_desc = " | ".join(tool_results)   # 다음 turn 의 LLM INPUT

        if tool_call_count >= cap:
            messages.append(HumanMessage(content=budget_msg))
            turn += 1
            t_llm = time.time()
            result = llm.invoke(messages)
            llm_time += time.time() - t_llm
            final_text = result.content
            if log_io:
                log.info("\n[q=%s | %s] -- LLM CALL (turn %d, forced final) --\n[INPUT]  budget exhausted → %s\n[OUTPUT] content: %r",
                         qid, label, turn, llm_input_desc, result.content)
            break

    if not final_text:
        for m in reversed(messages):
            if getattr(m, "type", "") == "ai" and getattr(m, "content", ""):
                final_text = m.content
                break

    return final_text, llm_time, tool_time, tool_call_count


def _make_decomposer(llm: ChatOpenAI, tool_registry: dict, log_io: bool, qid: str):
    agent = AGENTS["decompose"]
    cap = DISCOVER_TOOL_CAP
    system_txt = agent["system"].format(cap=cap)

    my_tools = [tool_registry["local_search"]]
    llm_bound = llm.bind_tools(my_tools)
    tool_lookup = {t.name: t for t in my_tools}

    def node(state: GraphState) -> dict:
        t_start = time.time()
        question = state["question"]
        user_msg = f"Fan-out question: {question}"
        bar = "=" * 70
        label = "decompose"

        if log_io:
            log.info("\n%s\n[q=%s | %s / %s] -- INPUT --\n[SYSTEM]\n%s\n[USER]\n%s\n%s",
                     bar, qid, label, agent["role"], system_txt, user_msg, bar)

        messages = [SystemMessage(content=system_txt), HumanMessage(content=user_msg)]
        text, llm_time, tool_time, tool_call_count = _run_tool_loop(
            llm_bound, llm, messages, cap, tool_lookup, log_io, qid, label,
            first_input_desc=f"fan-out question: {question}",
            budget_msg="Search budget exhausted. Output the SUBQ lines now.")
        subs = _parse_subquestions(text)

        if log_io:
            log.info("\n%s\n[q=%s | %s / %s] -- OUTPUT (%d sub-questions) --\n%s\n\n[raw]\n%s\n%s",
                     bar, qid, label, agent["role"], len(subs),
                     "\n".join(f"  - {s}" for s in subs), text, bar)

        wall = time.time() - t_start
        return {
            "sub_questions": subs,
            "outputs": {"decompose": text},
            "tool_calls": tool_call_count,
            "timings": {"decompose": {"wall": wall, "llm": llm_time, "tool": tool_time}},
        }

    return node


def _make_researcher(llm: ChatOpenAI, tool_registry: dict, log_io: bool, qid: str):
    agent = AGENTS["researcher"]
    cap = RESEARCHER_TOOL_CAP
    system_txt = agent["system"].format(cap=cap)

    my_tools = [tool_registry["local_search"]]
    llm_bound = llm.bind_tools(my_tools)
    tool_lookup = {t.name: t for t in my_tools}

    def node(payload: dict) -> dict:
        t_start = time.time()
        sub_q = payload["sub_question"]
        idx = payload["idx"]
        user_msg = f"Sub-question: {sub_q}"
        bar = "=" * 70
        label = f"researcher#{idx}"

        if log_io:
            log.info("\n%s\n[q=%s | %s / %s] -- INPUT --\n[SYSTEM]\n%s\n[USER]\n%s\n%s",
                     bar, qid, label, agent["role"], system_txt, user_msg, bar)

        messages = [SystemMessage(content=system_txt), HumanMessage(content=user_msg)]
        final_text, llm_time, tool_time, tool_call_count = _run_tool_loop(
            llm_bound, llm, messages, cap, tool_lookup, log_io, qid, label,
            first_input_desc=f"sub-question: {sub_q}",
            budget_msg="Tool budget exhausted. Output your final EVIDENCE and ANSWER now.")

        wall = time.time() - t_start
        out = {
            "idx": idx,
            "sub_question": sub_q,
            "answer": _parse_answer(final_text),
            "evidence": final_text,
            "tool_calls": tool_call_count,
        }
        if log_io:
            log.info("\n%s\n[q=%s | %s / %s] -- OUTPUT (%.2fs, %d tools, answer=%r) --\n%s\n%s",
                     bar, qid, label, agent["role"], wall, tool_call_count, out["answer"], final_text, bar)

        return {
            "researcher_outputs": [out],
            "tool_calls": tool_call_count,
            "timings": {f"researcher_{idx}": {"wall": wall, "llm": llm_time, "tool": tool_time}},
        }

    return node


def _make_aggregator(llm: ChatOpenAI, log_io: bool, qid: str):
    agent = AGENTS["aggregate"]

    def node(state: GraphState) -> dict:
        t_start = time.time()
        question = state["question"]
        outs = sorted(state.get("researcher_outputs", []), key=lambda o: o["idx"])

        parts = [f"Original fan-out question: {question}", "", "Researcher findings:"]
        for o in outs:
            parts.append(f"- [{o['sub_question']}] → {o['answer'] or 'unknown'}")
        user_msg = "\n".join(parts)
        bar = "=" * 70

        if log_io:
            log.info("\n%s\n[q=%s | aggregate / %s] -- INPUT --\n[SYSTEM]\n%s\n[USER]\n%s\n%s",
                     bar, qid, agent["role"], agent["system"], user_msg, bar)

        t_llm = time.time()
        result = llm.invoke([SystemMessage(content=agent["system"]),
                             HumanMessage(content=user_msg)])
        llm_time = time.time() - t_llm
        text = result.content

        if log_io:
            log.info("\n%s\n[q=%s | aggregate / %s] -- OUTPUT --\n%s\n%s",
                     bar, qid, agent["role"], text, bar)

        wall = time.time() - t_start
        return {
            "outputs": {"aggregate": text},
            "timings": {"aggregate": {"wall": wall, "llm": llm_time, "tool": {}}},
        }

    return node


def build_graph(llm: ChatOpenAI, log_io: bool = False, qid: str = "?"):
    tool_registry = {"local_search": make_local_search()}

    def dispatch(state: GraphState):
        return [
            Send("researcher", {"question": state["question"],
                                "sub_question": sq, "idx": i})
            for i, sq in enumerate(state["sub_questions"])
        ]

    graph = StateGraph(GraphState)
    graph.add_node("decompose", _make_decomposer(llm, tool_registry, log_io, qid))
    graph.add_node("researcher", _make_researcher(llm, tool_registry, log_io, qid))
    graph.add_node("aggregate", _make_aggregator(llm, log_io, qid), defer=True)

    graph.add_edge(START, "decompose")
    graph.add_conditional_edges("decompose", dispatch, ["researcher"])
    graph.add_edge("researcher", "aggregate")
    graph.add_edge("aggregate", END)

    return graph.compile()
