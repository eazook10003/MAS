import itertools
import re
from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, TypedDict

import rouge_score.scoring
from rouge_score.rouge_scorer import RougeScorer

from fanoutqa.models import AnswerType, DevQuestion
from fanoutqa.norm import normalize


def str_answer(ans: AnswerType) -> str:
    """Ensure the answer is a string for string-based metrics like ROUGE. Don't normalize it otherwise."""
    if isinstance(ans, list):
        return "\n".join(map(str_answer, ans))
    elif isinstance(ans, dict):
        return "\n".join(f"{k} - {str_answer(v)}" for k, v in ans.items())
    elif isinstance(ans, bool):
        return "yes" if ans else "no"
    elif ans is None:
        return ""
    return str(ans)


AccuracyResult = namedtuple("AccuracyResult", "found score missing")


def answer_in_text(reference: AnswerType, candidate: str) -> AccuracyResult:
    """What proportion of answer strings found in the reference can also be found in the candidate?"""
    if isinstance(reference, list):
        missing = []
        for a in reference:
            result = answer_in_text(a, candidate)
            missing.extend(result.missing)
        n_found = len(reference) - len(missing)
        return AccuracyResult(found=n_found == len(reference), score=n_found / len(reference), missing=missing)
    elif isinstance(reference, dict):
        missing = []
        vals = itertools.chain(reference.keys(), reference.values())
        for a in vals:
            result = answer_in_text(a, candidate)
            missing.extend(result.missing)
        n_ref = len(reference) * 2
        n_found = n_ref - len(missing)  # kvs
        return AccuracyResult(found=n_found == n_ref, score=n_found / n_ref, missing=missing)
    else:
        if isinstance(reference, bool):
            reference = "yes" if reference else "no"
        # primitive
        norm_ans = normalize(reference)
        norm_cand = normalize(candidate)
        # ensure the answer is surrounded by word boundaries
        if not re.search(rf"\b{re.escape(norm_ans)}\b", norm_cand):
            return AccuracyResult(found=False, score=0, missing=[norm_ans])
    return AccuracyResult(found=True, score=1, missing=[])


@dataclass
class AccuracyScore:
    loose: float
    """Loose accuracy: The mean proportion of reference strings found in the generation."""

    strict: float
    """Strict accuracy: The proportion of questions with a loose accuracy of 1.0."""


@dataclass
class RougeScorePart:
    precision: float
    recall: float
    fscore: float


@dataclass
class RougeScore:
    rouge1: RougeScorePart
    rouge2: RougeScorePart
    rougeL: RougeScorePart


class Answer(TypedDict):
    """A dictionary of the form ``{"id": "...", "answer": "..."}``."""

    id: str
    answer: str


ROUGE_TYPES = ("rouge1", "rouge2", "rougeL")


class Scorer:
    def __init__(
        self, questions: list[DevQuestion], answers: list[Answer], only_score_answered=False, llm_cache_key: str = None
    ):
        """
        :param questions: The questions and reference answers, as loaded by the dataset
        :param answers: The generated answers to score
        :param only_score_answered: Whether to only score questions that have an answer (True), or consider unanswered
            questions to have 0 score (False, default).
        :param llm_cache_key: If this is provided, cache the LLM-as-judge generations with this key. We recommend
            setting this to a human-readable key for each system under test.
        """
        self.questions = questions
        self.questions_by_id = {q.id: q for q in self.questions}
        self.answers = answers
        self.answers_by_id = {r["id"]: r for r in self.answers}

        self.only_score_answered = only_score_answered
        if self.only_score_answered:
            self.eval_len = len(self.answers)
        else:
            self.eval_len = len(self.questions)

        self.llm_cache_key = llm_cache_key

        self.rouge = RougeScorer(ROUGE_TYPES, use_stemmer=True)

    def get_qa_pairs(self) -> Iterable[tuple[DevQuestion, Optional[Answer]]]:
        """Yield pairs of questions and answers to score.
        The answer may be None if there is no answer for a given question and ``only_score_answered`` is False.
        """
        if self.only_score_answered:
            for a in self.answers:
                q = self.questions_by_id.get(a["id"])
                yield q, a
        else:
            for q in self.questions:
                a = self.answers_by_id.get(q.id)
                yield q, a

    def score_accuracy(self) -> Tuple[AccuracyScore, Dict[str, float]]:
        """Get the loose and strict accuracy scores for the loaded qs and as."""
        raw_scores = {}  
        accs = []
        n_perfect = 0
        for q, a in self.get_qa_pairs():
            if a is None:
                accs.append(0)
                raw_scores[q.id] = 0
                continue
            result = answer_in_text(q.answer, a["answer"])
            accs.append(result.score)
            raw_scores[q.id] = result.score
            if result.found:
                n_perfect += 1

        assert len(accs) == self.eval_len
        assert len(raw_scores) == self.eval_len
        avg_acc = sum(accs) / self.eval_len
        pct_perfect = n_perfect / self.eval_len
        return AccuracyScore(loose=avg_acc, strict=pct_perfect), raw_scores

    def score_rouge(self) -> Tuple[RougeScore, Dict[str, RougeScore]]:
        """Get the ROUGE-1, ROUGE-2, and ROUGE-L scores (P/R/F1) for the loaded qs and as."""
        raw_scores = {}  
        scores = {t: [] for t in ROUGE_TYPES}  
        for q, a in self.get_qa_pairs():
            if a is None:
                for score in scores.values():
                    score.append(rouge_score.scoring.Score(0, 0, 0))
                raw_scores[q.id] = RougeScore(
                    **{k: RougeScorePart(precision=0, recall=0, fscore=0) for k in ROUGE_TYPES}
                )
                continue
            results = self.rouge.score(str_answer(q.answer), str_answer(a["answer"]))
            for k, v in results.items():
                scores[k].append(v)
            raw_scores[q.id] = RougeScore(**{
                k: RougeScorePart(precision=v.precision, recall=v.recall, fscore=v.fmeasure) for k, v in results.items()
            })

        assert all(len(v) == self.eval_len for v in scores.values())
        assert len(raw_scores) == self.eval_len
        out = {}
        for k, v in scores.items():
            avg_precision = sum(s.precision for s in v) / self.eval_len
            avg_recall = sum(s.recall for s in v) / self.eval_len
            avg_fscore = sum(s.fmeasure for s in v) / self.eval_len
            out[k] = RougeScorePart(precision=avg_precision, recall=avg_recall, fscore=avg_fscore)
        return RougeScore(**out), raw_scores
