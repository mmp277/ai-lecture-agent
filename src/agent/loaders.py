import os
from typing import List, Tuple

from pypdf import PdfReader
from docx import Document


def read_txt(file_path: str) -> str:
	with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
		return f.read()


def read_pdf(file_path: str) -> str:
	reader = PdfReader(file_path)
	texts: List[str] = []
	for page in reader.pages:
		texts.append(page.extract_text() or "")
	return "\n".join(texts)


def read_docx(file_path: str) -> str:
	doc = Document(file_path)
	paras: List[str] = []
	for p in doc.paragraphs:
		text = p.text.strip()
		if text:
			paras.append(text)
	return "\n".join(paras)


def read_doc(file_path: str) -> str:
	try:
		import win32com.client  # type: ignore
		word = win32com.client.Dispatch("Word.Application")
		word.Visible = False
		doc = word.Documents.Open(file_path)
		text = doc.Content.Text
		doc.Close(False)
		word.Quit()
		return text
	except Exception:
		return ""


def load_documents_from_dir(directory: str) -> List[Tuple[str, str]]:
	"""
	Return list of (filename, text) for supported file types.
	"""
	results: List[Tuple[str, str]] = []
	for root, _, files in os.walk(directory):
		for name in files:
			path = os.path.join(root, name)
			lower = name.lower()
			try:
				if lower.endswith(".txt"):
					results.append((path, read_txt(path)))
				elif lower.endswith(".pdf"):
					results.append((path, read_pdf(path)))
				elif lower.endswith(".docx"):
					results.append((path, read_docx(path)))
				elif lower.endswith(".doc"):
					results.append((path, read_doc(path)))
				else:
					# Unsupported format; skip silently
					pass
			except Exception:
				results.append((path, f""))
	return results


def load_triplets(lectures_dir: str, summaries_dir: str, flashcards_dir: str) -> List[Tuple[str, str, str]]:
    """Return list of (lecture_text, reference_summary, reference_flashcards).

    Supports file name variants: lX_s.*, lX_summary.*, lX_f.*, lX_flashcards.* (.docx/.doc)
    """
    triplets: List[Tuple[str, str, str]] = []
    for i in range(1, 100):
        lx = f"l{i}"
        lec_pdf = os.path.join(lectures_dir, f"{lx}.pdf")
        if not os.path.exists(lec_pdf):
            continue
        lecture_text = read_pdf(lec_pdf)

        # Summary candidates
        sum_candidates = [
            os.path.join(summaries_dir, f"{lx}_s.docx"),
            os.path.join(summaries_dir, f"{lx}_s.doc"),
            os.path.join(summaries_dir, f"{lx}_summary.docx"),
            os.path.join(summaries_dir, f"{lx}_summary.doc"),
        ]
        ref_sum = ""
        for p in sum_candidates:
            if os.path.exists(p):
                ref_sum = read_docx(p) if p.endswith(".docx") else read_doc(p)
                break

        # Flashcards candidates
        fc_candidates = [
            os.path.join(flashcards_dir, f"{lx}_f.docx"),
            os.path.join(flashcards_dir, f"{lx}_f.doc"),
            os.path.join(flashcards_dir, f"{lx}_flashcards.docx"),
            os.path.join(flashcards_dir, f"{lx}_flashcards.doc"),
        ]
        ref_fc = ""
        for p in fc_candidates:
            if os.path.exists(p):
                ref_fc = read_docx(p) if p.endswith(".docx") else read_doc(p)
                break

        triplets.append((lecture_text, ref_sum, ref_fc))
    return triplets
