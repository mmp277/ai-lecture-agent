import os
import json
from typing import List, Dict

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

from .loaders import load_triplets


SYSTEM_PROMPT = "You are a helpful study assistant that summarizes lectures and creates flashcards."


def build_dataset(lectures_dir: str, summaries_dir: str, flashcards_dir: str) -> Dataset:
	records: List[Dict[str, str]] = []
	for lecture_text, ref_sum, ref_fc in load_triplets(lectures_dir, summaries_dir, flashcards_dir):
		if not lecture_text.strip():
			continue
		target = ""
		if ref_sum.strip():
			target += "Summary (bullets):\n" + ref_sum.strip() + "\n\n"
		if ref_fc.strip():
			target += "Flashcards (Q/A):\n" + ref_fc.strip() + "\n"
		if not target:
			continue
		prompt = (
			f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n"
			"Given the lecture notes below, produce a concise 5-10 bullet summary and 15 Q/A flashcards in the exact format.\n\n"
			+ lecture_text[:6000]
			+ "\n<|assistant|>\n"
		)
		records.append({"text": prompt + target})
	return Dataset.from_list(records)


def _save_adapter_weights(peft_model, output_dir: str) -> None:
	os.makedirs(output_dir, exist_ok=True)
	state_dict = get_peft_model_state_dict(peft_model)  # type: ignore
	try:
		from safetensors.torch import save_file  # type: ignore
		save_file(state_dict, os.path.join(output_dir, "adapter_model.safetensors"))
		return
	except Exception:
		pass
	import torch  # type: ignore
	torch.save(state_dict, os.path.join(output_dir, "adapter_model.bin"))


def train_lora(lectures_dir: str, summaries_dir: str, flashcards_dir: str, output_dir: str, base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0") -> str:
	os.makedirs(output_dir, exist_ok=True)
	# Limit CPU threads to keep UI responsive
	try:
		import torch  # type: ignore
		torch.set_num_threads(max(1, os.cpu_count() // 2 or 1))
	except Exception:
		pass

	tokenizer = AutoTokenizer.from_pretrained(base_model)
	if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
		tokenizer.pad_token = tokenizer.eos_token
	model = AutoModelForCausalLM.from_pretrained(base_model)
	try:
		model.config.use_cache = False  # type: ignore
	except Exception:
		pass

	lora = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.05, target_modules=["q_proj", "v_proj"])  # type: ignore
	model = get_peft_model(model, lora)

	# Ensure adapter config exists even if training is interrupted
	lora.save_pretrained(output_dir)

	ds = build_dataset(lectures_dir, summaries_dir, flashcards_dir)
	if len(ds) == 0:
		raise RuntimeError("No training data constructed from the provided directories.")

	def tokenize(batch):
		return tokenizer(batch["text"], truncation=True, max_length=512)

	tok = ds.map(tokenize, batched=True, remove_columns=["text"])  # type: ignore
	collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

	args = TrainingArguments(
		per_device_train_batch_size=1,
		gradient_accumulation_steps=1,
		learning_rate=2e-4,
		max_steps=20,
		logging_steps=1,
		report_to=[],
		output_dir=output_dir,
		save_steps=10,
		save_total_limit=1,
		dataloader_num_workers=0,
		dataloader_pin_memory=False,
		no_cuda=True,
	)

	trainer = Trainer(model=model, args=args, train_dataset=tok, data_collator=collator)
	trainer.train()

	# Save adapters definitively
	lora.save_pretrained(output_dir)
	_save_adapter_weights(model, output_dir)

	if not os.path.exists(os.path.join(output_dir, "adapter_config.json")):
		raise RuntimeError(f"Adapter not saved correctly at {output_dir} (missing adapter_config.json)")
	if not (os.path.exists(os.path.join(output_dir, "adapter_model.safetensors")) or os.path.exists(os.path.join(output_dir, "adapter_model.bin"))):
		raise RuntimeError(f"Adapter not saved correctly at {output_dir} (missing adapter model file)")
	return output_dir


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("lectures_dir")
	parser.add_argument("summaries_dir")
	parser.add_argument("flashcards_dir")
	parser.add_argument("--out", default="models/tinyllama-lora")
	args = parser.parse_args()

	out = train_lora(args.lectures_dir, args.summaries_dir, args.flashcards_dir, args.out)
	print(out)
