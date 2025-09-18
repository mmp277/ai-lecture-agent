
### Mithil Modi
### Indian Institute of Technology, Guwahati
### Mechanical Engineering

# AI Lecture Agent (FineTuned Model)

Read lecture notes (.pdf, .txt, .docx), summarize with Gemini, generate flashcards, and extract a formula sheet into Word documents. Only the input directory is required.

## Requirements
- Python 3.9+

## Install
```powershell
cd ai-lecture-agent
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configure API key
Create a `.env` file in the project root or set the env var in the shell:
```
GEMINI_API_KEY=<your_key>
```
Or in PowerShell for the current session:
```powershell
$env:GEMINI_API_KEY = "<your_key>"
```

## Run
```powershell
python -m src.agent.cli "lecture notes/input" --out outputs
```
Outputs: `outputs\summaries.docx`, `outputs\flashcards.docx`, `outputs\formula_sheet.docx`.

## Train/Validate LoRA (optional)
- Train TinyLlama LoRA on your `lecture notes/train/...` split:
```powershell
python -m src.agent.train_lora "lecture notes/train/lectures" "lecture notes/train/summary" "lecture notes/train/flashcards" --out models/tinyllama-lora
```
- Validate on `lecture notes/validate/...`:
```powershell
python validate_lora.py --lora_dir models/tinyllama-lora \
  --lectures_dir "lecture notes/validate/lectures" \
  --summaries_dir "lecture notes/validate/summary"
```
- Use the trained LoRA for generation:
```powershell
python -m src.agent.cli "lecture notes/input" --out outputs --engine tinyllama --lora models/tinyllama-lora
```

## Notes
- Uses Gemini (`gemini-1.5-flash`) via `google-generativeai`. No paid libraries are required, but API usage may incur costs on your account.
- Formula extraction is heuristic. For scanned PDFs, add OCR if needed (e.g., Tesseract).
