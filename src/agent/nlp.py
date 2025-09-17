import os

from typing import List, Tuple
from pathlib import Path

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel  # type: ignore


class Summarizer:
	def __init__(self, model_name: str = "t5-small") -> None:
		self.tokenizer = AutoTokenizer.from_pretrained(model_name)
		self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
		self.pipe = pipeline("summarization", model=self.model, tokenizer=self.tokenizer)

	def summarize_chunks(self, chunks: List[str], max_length: int = 180, min_length: int = 60) -> List[str]:
		outputs: List[str] = []
		for ch in chunks:
			text = ch.strip()
			if not text:
				continue
			res = self.pipe(text, truncation=True, max_length=max_length, min_length=min_length)
			outputs.append(res[0]["summary_text"].strip())
		return outputs


class SimpleGenerator:
	def __init__(self, model_name: str = "distilgpt2") -> None:
		self.tokenizer = AutoTokenizer.from_pretrained(model_name)
		self.model = AutoModelForCausalLM.from_pretrained(model_name)

	def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
		inputs = self.tokenizer(prompt, return_tensors="pt")
		outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
		return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class TinyLlamaLoRA:
	def __init__(self, base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", lora_dir: str | None = None) -> None:
		try:
			self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
		except Exception as e:
			print(f"Warning: Failed to load tokenizer from {base_model}: {e}")
			# Fallback to a simpler tokenizer
			self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
		
		if self.tokenizer.pad_token is None:
			self.tokenizer.pad_token = self.tokenizer.eos_token
			
		base = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, torch_dtype="auto")

		if lora_dir:
			adapter_path = Path(lora_dir)
			# Normalize and ensure directory
			adapter_path = adapter_path.resolve()
			cfg = adapter_path / "adapter_config.json"
			bin1 = adapter_path / "adapter_model.safetensors"
			bin2 = adapter_path / "adapter_model.bin"
			if not cfg.exists() or (not bin1.exists() and not bin2.exists()):
				raise FileNotFoundError(
					f"LoRA adapter directory is missing required files: {adapter_path}. "
					"Expected 'adapter_config.json' and 'adapter_model.safetensors' (or 'adapter_model.bin')."
				)
			# Load local adapter directory
			self.model = PeftModel.from_pretrained(base, str(adapter_path))
		else:
			self.model = base
		self.model.eval()

	def chat(self, system: str, user: str, max_new_tokens: int = 256) -> str:
		prompt = f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"
		# Ensure prompt fits model context window
		model_max = getattr(self.tokenizer, "model_max_length", 2048)
		if not isinstance(model_max, int) or model_max <= 0 or model_max > 32768:
			model_max = 2048
		inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=model_max)
		outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
		return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

	def summarize(self, text: str) -> str:
		return self.chat(
			"You are a precise academic summarizer.",
			"Summarize in 5-10 bullet points, faithful and concise.\n\n" + text,
		)

	def flashcards(self, text: str, num_cards: int = 15) -> List[Tuple[str, str]]:
		raw = self.chat(
			"Generate flashcards.",
			(
				f"Create strictly {num_cards} Q/A flashcards. Format lines exactly as 'Q: ...' and next line 'A: ...'. "
				"No extra commentary.\n\n" + text
			),
		)
		pairs: List[Tuple[str, str]] = []
		q, a = None, None
		for line in raw.splitlines():
			line = line.strip()
			if line.lower().startswith("q:"):
				q = line[2:].strip()
			elif line.lower().startswith("a:"):
				a = line[2:].strip()
				if q:
					pairs.append((q, a))
					q, a = None, None
		return pairs


# Gemini and Perplexity kept for previous flow
class GeminiClient:
	def __init__(self, model_name: str = "gemini-1.5-flash", api_key: str | None = "AIzaSyAMfUaM6Pa72wS3qPRMLEHrxv8YEI54xss") -> None:
		key = api_key or os.getenv("GEMINI_API_KEY")
		if not key:
			raise RuntimeError("Gemini API key missing")
		import google.generativeai as genai  # type: ignore
		genai.configure(api_key=key)
		self.model = genai.GenerativeModel(model_name)

	def summarize(self, text: str) -> str:
		prompt = (
			"You are a precise academic summarizer. Summarize the following lecture notes in 5-10 bullet points. "
			"Be concise, keep terminology, and avoid hallucinations.\n\n" + text
		)
		resp = self.model.generate_content(prompt)
		return (resp.text or "").strip()

	def flashcards(self, text: str, num_cards: int = 15) -> List[Tuple[str, str]]:
		prompt = (
			"Generate strictly "
			f"{num_cards} question-answer flashcards from the lecture text. "
			"Return as lines formatted exactly as: Q: <question>\nA: <answer>. "
			"Do not include extra commentary.\n\n" + text
		)
		resp = self.model.generate_content(prompt)
		raw = (resp.text or "").strip()
		pairs: List[Tuple[str, str]] = []
		q, a = None, None
		for line in raw.splitlines():
			line = line.strip()
			if line.lower().startswith("q:"):
				q = line[2:].strip()
			elif line.lower().startswith("a:"):
				a = line[2:].strip()
				if q:
					pairs.append((q, a))
					q, a = None, None
		return pairs


class PerplexityClient:
	def __init__(self, model_name: str = "llama-3.1-sonar-small-128k-online") -> None:
		self.api_key = os.getenv("PERPLEXITY_API_KEY")
		if not self.api_key:
			raise RuntimeError("PERPLEXITY_API_KEY is not set")
		self.model_name = model_name

	def _chat(self, system: str, user: str) -> str:
		import requests  # type: ignore
		url = "https://api.perplexity.ai/chat/completions"
		payload = {
			"model": self.model_name,
			"messages": [
				{"role": "system", "content": system},
				{"role": "user", "content": user},
			],
			"temperature": 0.2,
		}
		r = requests.post(url, headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, json=payload, timeout=120)
		r.raise_for_status()
		data = r.json()
		return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

	def summarize(self, text: str) -> str:
		return self._chat(
			"You are a precise academic summarizer.",
			"Summarize the following lecture notes in 5-10 bullet points. Be concise and faithful.\n\n" + text,
		)

	def flashcards(self, text: str, num_cards: int = 15) -> List[Tuple[str, str]]:
		raw = self._chat(
			"You generate flashcards.",
			(
				"Create strictly "
				f"{num_cards} question-answer flashcards from the lecture text. "
				"Return lines formatted exactly as 'Q: <question>' then next line 'A: <answer>'. "
				"No extra commentary.\n\n" + text
			),
		)
		pairs: List[Tuple[str, str]] = []
		q, a = None, None
		for line in raw.splitlines():
			line = line.strip()
			if line.lower().startswith("q:"):
				q = line[2:].strip()
			elif line.lower().startswith("a:"):
				a = line[2:].strip()
				if q:
					pairs.append((q, a))
					q, a = None, None
		return pairs


def get_provider(provider: str, model_name: str = ""):
	p = (provider or "local").lower()
	if p == "gemini":
		return GeminiClient(model_name or "gemini-1.5-flash")
	if p == "perplexity":
		return PerplexityClient(model_name or "llama-3.1-sonar-small-128k-online")
	return None  # local
