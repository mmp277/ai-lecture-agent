import argparse
import os
from typing import List, Tuple, Dict

from .agent import PlannerAgent


def main() -> None:
	parser = argparse.ArgumentParser(description="AI Lecture Agent (Planner)")
	parser.add_argument("input_dir", help="Directory containing lecture notes (.pdf, .txt, .docx)")
	parser.add_argument("--out", dest="outputs_dir", default=None, help="Output directory (defaults to input directory)")
	parser.add_argument("--engine", default="tinyllama", choices=["tinyllama", "gemini"], help="Model engine")
	parser.add_argument("--lora", dest="lora_dir", default="models/tinyllama-lora", help="LoRA adapter dir")
	args = parser.parse_args()

	outputs_dir = args.outputs_dir or args.input_dir
	os.makedirs(outputs_dir, exist_ok=True)

	agent = PlannerAgent(engine=args.engine, lora_dir=args.lora_dir)
	agent.plan([])
	file_summaries, flashcards_by_file, formulas_by_file = agent.execute(args.input_dir, outputs_dir)
	agent.write_outputs(outputs_dir, file_summaries, flashcards_by_file, formulas_by_file)
	print("[INFO] Done.")


if __name__ == "__main__":
	main()
