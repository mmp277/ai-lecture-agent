import os
from typing import Dict, List, Tuple
from rouge_score import rouge_scorer

from .loaders import read_pdf, read_docx, read_doc
from .text_utils import normalize_whitespace


def _read_ref(path_base: str) -> str:
	if os.path.exists(path_base + ".docx"):
		return read_docx(path_base + ".docx")
	if os.path.exists(path_base + ".doc"):
		return read_doc(path_base + ".doc")
	return ""


def rouge_f1(reference: str, generated: str) -> Dict[str, float]:
	scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
	scores = scorer.score(reference, generated)
	return {"rouge1": scores["rouge1"].fmeasure, "rougeL": scores["rougeL"].fmeasure}


def flashcard_answerability(cards: List[Tuple[str, str]], source_text: str) -> float:
	"""Heuristic: percent of answers whose unigrams appear at least k times in source."""
	source_tokens = [t.lower() for t in source_text.split()]
	freq = Counter(source_tokens)
	ok = 0
	for _, answer in cards:
		ans_tokens = [t.lower() for t in answer.split() if len(t) > 2]
		if not ans_tokens:
			continue
		covered = sum(1 for t in ans_tokens if freq[t] > 0)
		if covered / max(1, len(ans_tokens)) >= 0.3:
			ok += 1
	return ok / max(1, len(cards))


def formula_consistency(equations: List[str], source_text: str) -> float:
	"""Heuristic: fraction of equations whose variable symbols appear near equation text."""
	if not equations:
		return 0.0
	hits = 0
	for eq in equations:
		vars = [v for v in eq.replace("=", " ").split() if v.isalpha() and len(v) <= 3]
		match = sum(1 for v in vars if source_text.count(v) > 0)
		if match >= max(1, len(vars) // 2):
			hits += 1
	return hits / len(equations)


def evaluate(lectures_dir: str, summaries_dir: str, flashcards_dir: str, gen_sum: str, gen_fc: str) -> Dict[str, float]:
	"""Compute average ROUGE between generated and references across aligned lX files.
	gen_sum/gen_fc are paths to generated DOCX files (not parsed here). Caller should pass text content.
	"""
	# This function expects caller to provide already combined generated text for simplicity
	avg: Dict[str, float] = {"rouge1": 0.0, "rougeL": 0.0}
	count = 0
	for i in range(1, 100):
		lx = f"l{i}"
		lec_pdf = os.path.join(lectures_dir, f"{lx}.pdf")
		if not os.path.exists(lec_pdf):
			continue
		ref_sum = _read_ref(os.path.join(summaries_dir, f"{lx}_s"))
		if not ref_sum.strip():
			continue
		# For simplicity, use overall generated summary text vs reference (macro check). In practice, map per-lecture.
		s = rouge_f1(normalize_whitespace(ref_sum), normalize_whitespace(gen_sum))
		avg["rouge1"] += s["rouge1"]
		avg["rougeL"] += s["rougeL"]
		count += 1
	if count:
		avg = {k: v / count for k, v in avg.items()}
	return avg
