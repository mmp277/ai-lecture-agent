import os
from typing import Dict, List, Tuple

from .loaders import load_documents_from_dir
from .text_utils import normalize_whitespace, chunk_text
from .nlp import TinyLlamaLoRA, GeminiClient
from .flashcards import merge_pairs
from .formulas import extract_equations, guess_symbol_definitions, format_equation_latex
from .output import write_summary_docx, write_flashcards_docx, write_formula_sheet_docx, write_formula_sheet_latex


class PlannerAgent:
    def __init__(self, engine: str = "tinyllama", lora_dir: str | None = None, gemini_key: str | None = None) -> None:
        self.engine = engine

        # Select provider based on engine
        if engine == "tinyllama":
            self.llm = TinyLlamaLoRA(lora_dir=lora_dir)
        elif engine == "gemini":
            # Use API key from environment or provided gemini_key
            self.llm = GeminiClient(api_key=gemini_key or os.getenv("GEMINI_API_KEY"))
        else:
            raise ValueError(f"Unknown engine: {engine}")

    def plan(self, doc_paths: List[str]) -> List[str]:
        steps = [
            "Load and clean documents",
            "Chunk long texts",
            "Summarize chunks",
            "Generate flashcards",
            "Extract formulas and variable definitions",
            "Write Word and LaTeX formula documents",
        ]
        print("[PLAN] Steps:")
        for i, s in enumerate(steps, 1):
            print(f"  {i}. {s}")
        return steps

    def reason(self, context: str) -> str:
        prompt = (
            "You are an agent that decides how to process technical lecture notes. "
            "Given the text, decide key points and glossary to preserve. Return bullet points.\n\n" + context[:2000]
        )
        return self.llm.summarize(prompt)

    def execute(self, input_dir: str, outputs_dir: str) -> Tuple[List[Tuple[str, List[str]]], List[Tuple[str, List[Tuple[str, str]]]], List[Tuple[str, List[Tuple[str, Dict[str, str]]]]]]:
        os.makedirs(outputs_dir, exist_ok=True)
        docs = load_documents_from_dir(input_dir)

        file_summaries: List[Tuple[str, List[str]]] = []
        flashcards_by_file: List[Tuple[str, List[Tuple[str, str]]]] = []
        formulas_by_file: List[Tuple[str, List[Tuple[str, Dict[str, str]]]]] = []

        for path, text in docs:
            print(f"[EXEC] Processing {path}")
            clean = normalize_whitespace(text)
            chunks = chunk_text(clean, max_tokens=400)
            joined = "\n\n".join(chunks[:4]) if chunks else clean[:3000]
            summary_text = self.llm.summarize(joined)
            cards = self.llm.flashcards(clean, num_cards=20)

            file_summaries.append((path, [summary_text] if summary_text else []))
            flashcards_by_file.append((path, merge_pairs(cards, max_cards=40)))

            eqs = extract_equations(joined)
            items: List[Tuple[str, Dict[str, str]]] = []
            for eq in eqs:
                defs = guess_symbol_definitions(clean, eq)
                items.append((format_equation_latex(eq), defs))
            formulas_by_file.append((path, items))

        return file_summaries, flashcards_by_file, formulas_by_file

    def write_outputs(self, outputs_dir: str, file_summaries, flashcards_by_file, formulas_by_file) -> None:
        write_summary_docx(os.path.join(outputs_dir, "summaries.docx"), file_summaries)
        write_flashcards_docx(os.path.join(outputs_dir, "flashcards.docx"), flashcards_by_file)
        write_formula_sheet_docx(os.path.join(outputs_dir, "formula_sheet.docx"), formulas_by_file)
        write_formula_sheet_latex(os.path.join(outputs_dir, "formula_sheet.tex"), formulas_by_file, compile_pdf=True)
