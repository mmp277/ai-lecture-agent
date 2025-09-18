#!/usr/bin/env python3
"""
Minimal validator for LoRA fine-tuned model on the validate split.

Reads PDFs from "lecture notes/validate/lectures" and reference summaries from
"lecture notes/validate/summary", generates LoRA summaries, and reports averaged ROUGE.

Usage (run from repo root):
  python validate_lora.py
Optional args:
  --lora_dir models/tinyllama-lora
  --lectures_dir "lecture notes/validate/lectures"
  --summaries_dir "lecture notes/validate/summary"
  --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent.loaders import read_pdf, read_docx, read_doc  # type: ignore
from agent.text_utils import normalize_whitespace, chunk_text  # type: ignore
from agent.nlp import TinyLlamaLoRA  # type: ignore
from rouge_score import rouge_scorer  # type: ignore


def _read_ref_any(base_path_no_ext: str) -> str:
	"""Read reference summary given a base path without extension.
	Supports both *_s.(doc|docx) and *_summary.(doc|docx)."""
	candidates = [
		base_path_no_ext + suffix
		for suffix in ("_s.docx", "_s.doc", "_summary.docx", "_summary.doc")
	]
	for path in candidates:
		if os.path.exists(path):
			if path.endswith(".docx"):
				return read_docx(path)
			if path.endswith(".doc"):
				return read_doc(path)
	return ""


def compute_rouge(reference: str, generated: str) -> Dict[str, float]:
	scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
	scores = scorer.score(reference, generated)
	return {"rouge1": scores["rouge1"].fmeasure, "rougeL": scores["rougeL"].fmeasure}


def evaluate(lora_dir: str, lectures_dir: str, summaries_dir: str, base_model: str) -> Dict[str, float]:
	model = TinyLlamaLoRA(base_model=base_model, lora_dir=lora_dir)
	avg: Dict[str, float] = {"rouge1": 0.0, "rougeL": 0.0}
	count = 0

	# Iterate l1..l99 if present
	for i in range(1, 100):
		lx = f"l{i}"
		lec_pdf = os.path.join(lectures_dir, f"{lx}.pdf")
		if not os.path.exists(lec_pdf):
			continue
		ref_text = _read_ref_any(os.path.join(summaries_dir, lx))
		if not ref_text.strip():
			continue

		lecture_text = read_pdf(lec_pdf)
		clean = normalize_whitespace(lecture_text)
		chunks = chunk_text(clean, max_tokens=300)
		joined = "\n\n".join(chunks[:2]) if chunks else clean[:1500]

		try:
			gen = model.summarize(joined)
		except Exception as e:
			print(f"[WARN] Generation failed for {lx}: {e}")
			continue

		s = compute_rouge(normalize_whitespace(ref_text), normalize_whitespace(gen))
		avg["rouge1"] += s["rouge1"]
		avg["rougeL"] += s["rougeL"]
		count += 1
		print(f"{lx}: ROUGE-1={s['rouge1']:.3f}, ROUGE-L={s['rougeL']:.3f}")

	if count:
		avg = {k: v / count for k, v in avg.items()}
	print(f"\nFiles evaluated: {count}")
	return avg


def main() -> None:
	parser = argparse.ArgumentParser(description="Validate LoRA model on validate split")
	parser.add_argument("--lora_dir", default="models/tinyllama-lora")
    parser.add_argument("--lectures_dir", default="lecture notes/validate/lectures")
    parser.add_argument("--summaries_dir", default="lecture notes/validate/summary")
	parser.add_argument("--base_model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
	args = parser.parse_args()

	# Resolve to absolute
	root = Path(__file__).parent
	lora_dir = str((root / args.lora_dir).resolve())
	lectures_dir = str((root / args.lectures_dir).resolve())
	summaries_dir = str((root / args.summaries_dir).resolve())

	# Sanity checks
	missing = []
	for p, label in [(lora_dir, "LoRA dir"), (lectures_dir, "lectures dir"), (summaries_dir, "summaries dir")]:
		if not os.path.exists(p):
			missing.append(f"{label}: {p}")
	if missing:
		print("Missing paths:\n  - " + "\n  - ".join(missing))
		return

	print("Validating LoRA model...")
	print(f"  LoRA: {lora_dir}")
	print(f"  Lectures: {lectures_dir}")
	print(f"  Summaries: {summaries_dir}")

	avg = evaluate(lora_dir, lectures_dir, summaries_dir, args.base_model)
	print("\nAveraged ROUGE on validate set:")
	print(f"  ROUGE-1: {avg.get('rouge1', 0.0):.3f}")
	print(f"  ROUGE-L: {avg.get('rougeL', 0.0):.3f}")


if __name__ == "__main__":
	main()
