"""
Comprehensive Evaluation Matrix for LoRA Fine-tuned TinyLlama Model
Evaluates model performance across multiple dimensions including:
- Summary quality (ROUGE metrics)
- Flashcard generation quality
- Formula extraction accuracy
- Model comparison (base vs LoRA)
- Performance metrics
"""

import os
import json
import time
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
from pathlib import Path
import statistics

from rouge_score import rouge_scorer
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

from .nlp import TinyLlamaLoRA, GeminiClient
from .loaders import load_documents_from_dir, read_pdf, read_docx
from .text_utils import normalize_whitespace, chunk_text
from .formulas import extract_equations, guess_symbol_definitions, format_equation_latex
from .flashcards import merge_pairs


class EvaluationMatrix:
    """Comprehensive evaluation matrix for LoRA fine-tuned model"""
    
    def __init__(self, lora_dir: str, base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """
        Initialize evaluation matrix
        
        Args:
            lora_dir: Path to LoRA adapter directory
            base_model: Base model name for comparison
        """
        self.lora_dir = lora_dir
        self.base_model = base_model
        self.lora_model = None
        self.base_model_instance = None
        self.results = {}
        
        # Initialize models
        self._load_models()
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL", "rougeLsum"], 
            use_stemmer=True
        )
    
    def _load_models(self):
        """Load both LoRA and base models for comparison"""
        try:
            print("Loading LoRA model...")
            self.lora_model = TinyLlamaLoRA(
                base_model=self.base_model, 
                lora_dir=self.lora_dir
            )
            print("✓ LoRA model loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load LoRA model: {e}")
            self.lora_model = None
        
        try:
            print("Loading base model...")
            self.base_model_instance = TinyLlamaLoRA(
                base_model=self.base_model, 
                lora_dir=None
            )
            print("✓ Base model loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load base model: {e}")
            self.base_model_instance = None
    
    def evaluate_summary_quality(self, lectures_dir: str, summaries_dir: str) -> Dict[str, Any]:
        """
        Evaluate summary quality using ROUGE metrics
        
        Args:
            lectures_dir: Directory containing lecture PDFs
            summaries_dir: Directory containing reference summaries
            
        Returns:
            Dictionary with ROUGE scores for both models
        """
        print("\n=== EVALUATING SUMMARY QUALITY ===")
        
        results = {
            "lora_scores": {"rouge1": [], "rouge2": [], "rougeL": [], "rougeLsum": []},
            "base_scores": {"rouge1": [], "rouge2": [], "rougeL": [], "rougeLsum": []},
            "comparison": {}
        }
        
        # Get all lecture files
        lecture_files = [f for f in os.listdir(lectures_dir) if f.endswith('.pdf')]
        lecture_files.sort()
        
        for lecture_file in lecture_files:
            lecture_num = lecture_file.replace('.pdf', '')
            print(f"Processing {lecture_num}...")
            
            # Load lecture content
            lecture_path = os.path.join(lectures_dir, lecture_file)
            lecture_text = read_pdf(lecture_path)
            if not lecture_text.strip():
                continue
            
            # Load reference summary
            ref_summary_path = os.path.join(summaries_dir, f"{lecture_num}_s.docx")
            if not os.path.exists(ref_summary_path):
                continue
            
            ref_summary = read_docx(ref_summary_path)
            if not ref_summary.strip():
                continue
            
            # Generate summaries with both models
            lecture_chunks = chunk_text(normalize_whitespace(lecture_text), max_tokens=400)
            lecture_input = "\n\n".join(lecture_chunks[:4]) if lecture_chunks else lecture_text[:3000]
            
            # LoRA model summary
            if self.lora_model:
                try:
                    lora_summary = self.lora_model.summarize(lecture_input)
                    lora_scores = self._calculate_rouge_scores(ref_summary, lora_summary)
                    for metric, score in lora_scores.items():
                        results["lora_scores"][metric].append(score)
                except Exception as e:
                    print(f"  ✗ LoRA summary failed for {lecture_num}: {e}")
            
            # Base model summary
            if self.base_model_instance:
                try:
                    base_summary = self.base_model_instance.summarize(lecture_input)
                    base_scores = self._calculate_rouge_scores(ref_summary, base_summary)
                    for metric, score in base_scores.items():
                        results["base_scores"][metric].append(score)
                except Exception as e:
                    print(f"  ✗ Base summary failed for {lecture_num}: {e}")
        
        # Calculate average scores
        for model_type in ["lora_scores", "base_scores"]:
            for metric in results[model_type]:
                if results[model_type][metric]:
                    avg_score = statistics.mean(results[model_type][metric])
                    results[model_type][metric] = {
                        "scores": results[model_type][metric],
                        "average": avg_score,
                        "count": len(results[model_type][metric])
                    }
        
        # Calculate improvement
        if results["lora_scores"]["rouge1"]["scores"] and results["base_scores"]["rouge1"]["scores"]:
            for metric in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
                lora_avg = results["lora_scores"][metric]["average"]
                base_avg = results["base_scores"][metric]["average"]
                improvement = ((lora_avg - base_avg) / base_avg) * 100 if base_avg > 0 else 0
                results["comparison"][metric] = {
                    "improvement_percent": improvement,
                    "lora_avg": lora_avg,
                    "base_avg": base_avg
                }
        
        return results
    
    def evaluate_flashcard_quality(self, lectures_dir: str, flashcards_dir: str) -> Dict[str, Any]:
        """
        Evaluate flashcard generation quality
        
        Args:
            lectures_dir: Directory containing lecture PDFs
            flashcards_dir: Directory containing reference flashcards
            
        Returns:
            Dictionary with flashcard quality metrics
        """
        print("\n=== EVALUATING FLASHCARD QUALITY ===")
        
        results = {
            "lora_metrics": {"answerability": [], "coverage": [], "count_accuracy": []},
            "base_metrics": {"answerability": [], "coverage": [], "count_accuracy": []},
            "comparison": {}
        }
        
        # Get all lecture files
        lecture_files = [f for f in os.listdir(lectures_dir) if f.endswith('.pdf')]
        lecture_files.sort()
        
        for lecture_file in lecture_files:
            lecture_num = lecture_file.replace('.pdf', '')
            print(f"Processing {lecture_num}...")
            
            # Load lecture content
            lecture_path = os.path.join(lectures_dir, lecture_file)
            lecture_text = read_pdf(lecture_path)
            if not lecture_text.strip():
                continue
            
            # Load reference flashcards
            ref_flashcards_path = os.path.join(flashcards_dir, f"{lecture_num}_f.docx")
            if not os.path.exists(ref_flashcards_path):
                continue
            
            ref_flashcards_text = read_docx(ref_flashcards_path)
            if not ref_flashcards_text.strip():
                continue
            
            # Parse reference flashcards
            ref_cards = self._parse_flashcards(ref_flashcards_text)
            
            # Generate flashcards with both models
            clean_text = normalize_whitespace(lecture_text)
            
            # LoRA model flashcards
            if self.lora_model:
                try:
                    lora_cards = self.lora_model.flashcards(clean_text, num_cards=15)
                    lora_metrics = self._evaluate_flashcards(lora_cards, ref_cards, lecture_text)
                    for metric, value in lora_metrics.items():
                        results["lora_metrics"][metric].append(value)
                except Exception as e:
                    print(f"  ✗ LoRA flashcards failed for {lecture_num}: {e}")
            
            # Base model flashcards
            if self.base_model_instance:
                try:
                    base_cards = self.base_model_instance.flashcards(clean_text, num_cards=15)
                    base_metrics = self._evaluate_flashcards(base_cards, ref_cards, lecture_text)
                    for metric, value in base_metrics.items():
                        results["base_metrics"][metric].append(value)
                except Exception as e:
                    print(f"  ✗ Base flashcards failed for {lecture_num}: {e}")
        
        # Calculate average metrics
        for model_type in ["lora_metrics", "base_metrics"]:
            for metric in results[model_type]:
                if results[model_type][metric]:
                    avg_value = statistics.mean(results[model_type][metric])
                    results[model_type][metric] = {
                        "values": results[model_type][metric],
                        "average": avg_value,
                        "count": len(results[model_type][metric])
                    }
        
        # Calculate improvement
        if results["lora_metrics"]["answerability"]["values"] and results["base_metrics"]["answerability"]["values"]:
            for metric in ["answerability", "coverage", "count_accuracy"]:
                lora_avg = results["lora_metrics"][metric]["average"]
                base_avg = results["base_metrics"][metric]["average"]
                improvement = ((lora_avg - base_avg) / base_avg) * 100 if base_avg > 0 else 0
                results["comparison"][metric] = {
                    "improvement_percent": improvement,
                    "lora_avg": lora_avg,
                    "base_avg": base_avg
                }
        
        return results
    
    def evaluate_formula_extraction(self, lectures_dir: str) -> Dict[str, Any]:
        """
        Evaluate formula extraction quality
        
        Args:
            lectures_dir: Directory containing lecture PDFs
            
        Returns:
            Dictionary with formula extraction metrics
        """
        print("\n=== EVALUATING FORMULA EXTRACTION ===")
        
        results = {
            "lora_metrics": {"consistency": [], "extraction_rate": []},
            "base_metrics": {"consistency": [], "extraction_rate": []},
            "comparison": {}
        }
        
        # Get all lecture files
        lecture_files = [f for f in os.listdir(lectures_dir) if f.endswith('.pdf')]
        lecture_files.sort()
        
        for lecture_file in lecture_files:
            lecture_num = lecture_file.replace('.pdf', '')
            print(f"Processing {lecture_num}...")
            
            # Load lecture content
            lecture_path = os.path.join(lectures_dir, lecture_file)
            lecture_text = read_pdf(lecture_path)
            if not lecture_text.strip():
                continue
            
            clean_text = normalize_whitespace(lecture_text)
            lecture_chunks = chunk_text(clean_text, max_tokens=400)
            lecture_input = "\n\n".join(lecture_chunks[:4]) if lecture_chunks else clean_text[:3000]
            
            # Extract equations from text
            equations = extract_equations(lecture_input)
            
            if equations:
                # Calculate consistency for both models
                lora_consistency = self._calculate_formula_consistency(equations, clean_text)
                base_consistency = self._calculate_formula_consistency(equations, clean_text)
                
                # Calculate extraction rate (equations found / total text length)
                extraction_rate = len(equations) / max(1, len(clean_text.split()))
                
                results["lora_metrics"]["consistency"].append(lora_consistency)
                results["lora_metrics"]["extraction_rate"].append(extraction_rate)
                results["base_metrics"]["consistency"].append(base_consistency)
                results["base_metrics"]["extraction_rate"].append(extraction_rate)
        
        # Calculate average metrics
        for model_type in ["lora_metrics", "base_metrics"]:
            for metric in results[model_type]:
                if results[model_type][metric]:
                    avg_value = statistics.mean(results[model_type][metric])
                    results[model_type][metric] = {
                        "values": results[model_type][metric],
                        "average": avg_value,
                        "count": len(results[model_type][metric])
                    }
        
        return results
    
    def evaluate_performance_metrics(self, test_texts: List[str]) -> Dict[str, Any]:
        """
        Evaluate performance metrics (speed, memory usage)
        
        Args:
            test_texts: List of test texts for performance evaluation
            
        Returns:
            Dictionary with performance metrics
        """
        print("\n=== EVALUATING PERFORMANCE METRICS ===")
        
        results = {
            "lora_performance": {"avg_time": 0, "total_time": 0},
            "base_performance": {"avg_time": 0, "total_time": 0},
            "comparison": {}
        }
        
        # Test LoRA model performance
        if self.lora_model and test_texts:
            print("Testing LoRA model performance...")
            lora_times = []
            for i, text in enumerate(test_texts[:5]):  # Test with first 5 texts
                start_time = time.time()
                try:
                    _ = self.lora_model.summarize(text)
                    end_time = time.time()
                    lora_times.append(end_time - start_time)
                    print(f"  LoRA test {i+1}: {end_time - start_time:.2f}s")
                except Exception as e:
                    print(f"  ✗ LoRA test {i+1} failed: {e}")
            
            if lora_times:
                results["lora_performance"]["avg_time"] = statistics.mean(lora_times)
                results["lora_performance"]["total_time"] = sum(lora_times)
        
        # Test base model performance
        if self.base_model_instance and test_texts:
            print("Testing base model performance...")
            base_times = []
            for i, text in enumerate(test_texts[:5]):  # Test with first 5 texts
                start_time = time.time()
                try:
                    _ = self.base_model_instance.summarize(text)
                    end_time = time.time()
                    base_times.append(end_time - start_time)
                    print(f"  Base test {i+1}: {end_time - start_time:.2f}s")
                except Exception as e:
                    print(f"  ✗ Base test {i+1} failed: {e}")
            
            if base_times:
                results["base_performance"]["avg_time"] = statistics.mean(base_times)
                results["base_performance"]["total_time"] = sum(base_times)
        
        # Calculate performance comparison
        if (results["lora_performance"]["avg_time"] > 0 and 
            results["base_performance"]["avg_time"] > 0):
            speed_ratio = results["lora_performance"]["avg_time"] / results["base_performance"]["avg_time"]
            results["comparison"]["speed_ratio"] = speed_ratio
            results["comparison"]["lora_slower_by"] = (speed_ratio - 1) * 100
        
        return results
    
    def run_comprehensive_evaluation(self, lectures_dir: str, summaries_dir: str, 
                                   flashcards_dir: str) -> Dict[str, Any]:
        """
        Run comprehensive evaluation across all metrics
        
        Args:
            lectures_dir: Directory containing lecture PDFs
            summaries_dir: Directory containing reference summaries
            flashcards_dir: Directory containing reference flashcards
            
        Returns:
            Complete evaluation results
        """
        print("🚀 Starting Comprehensive Evaluation Matrix")
        print("=" * 60)
        
        # Load some test texts for performance evaluation
        test_texts = []
        for i in range(1, 6):  # Use first 5 lectures for performance testing
            lecture_path = os.path.join(lectures_dir, f"l{i}.pdf")
            if os.path.exists(lecture_path):
                text = read_pdf(lecture_path)
                if text.strip():
                    test_texts.append(text[:1000])  # Use first 1000 chars for speed
        
        # Run all evaluations
        evaluation_results = {
            "summary_quality": self.evaluate_summary_quality(lectures_dir, summaries_dir),
            "flashcard_quality": self.evaluate_flashcard_quality(lectures_dir, flashcards_dir),
            "formula_extraction": self.evaluate_formula_extraction(lectures_dir),
            "performance_metrics": self.evaluate_performance_metrics(test_texts),
            "model_info": {
                "lora_dir": self.lora_dir,
                "base_model": self.base_model,
                "lora_loaded": self.lora_model is not None,
                "base_loaded": self.base_model_instance is not None
            }
        }
        
        # Calculate overall score
        evaluation_results["overall_score"] = self._calculate_overall_score(evaluation_results)
        
        return evaluation_results
    
    def _calculate_rouge_scores(self, reference: str, generated: str) -> Dict[str, float]:
        """Calculate ROUGE scores between reference and generated text"""
        scores = self.rouge_scorer.score(reference, generated)
        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
            "rougeLsum": scores["rougeLsum"].fmeasure
        }
    
    def _parse_flashcards(self, text: str) -> List[Tuple[str, str]]:
        """Parse flashcards from text"""
        cards = []
        lines = text.split('\n')
        q, a = None, None
        
        for line in lines:
            line = line.strip()
            if line.lower().startswith('q:') or line.lower().startswith('question:'):
                if q and a:
                    cards.append((q, a))
                q = line.split(':', 1)[1].strip() if ':' in line else line
                a = None
            elif line.lower().startswith('a:') or line.lower().startswith('answer:'):
                a = line.split(':', 1)[1].strip() if ':' in line else line
                if q:
                    cards.append((q, a))
                    q, a = None, None
        
        if q and a:
            cards.append((q, a))
        
        return cards
    
    def _evaluate_flashcards(self, generated_cards: List[Tuple[str, str]], 
                           reference_cards: List[Tuple[str, str]], 
                           source_text: str) -> Dict[str, float]:
        """Evaluate flashcard quality"""
        # Answerability: percentage of answers that can be found in source text
        source_tokens = [t.lower() for t in source_text.split()]
        freq = Counter(source_tokens)
        
        answerable_count = 0
        for _, answer in generated_cards:
            ans_tokens = [t.lower() for t in answer.split() if len(t) > 2]
            if ans_tokens:
                covered = sum(1 for t in ans_tokens if freq[t] > 0)
                if covered / len(ans_tokens) >= 0.3:
                    answerable_count += 1
        
        answerability = answerable_count / max(1, len(generated_cards))
        
        # Coverage: percentage of reference concepts covered
        ref_concepts = set()
        for q, a in reference_cards:
            ref_concepts.update(q.lower().split())
            ref_concepts.update(a.lower().split())
        
        gen_concepts = set()
        for q, a in generated_cards:
            gen_concepts.update(q.lower().split())
            gen_concepts.update(a.lower().split())
        
        coverage = len(ref_concepts.intersection(gen_concepts)) / max(1, len(ref_concepts))
        
        # Count accuracy: how close is the number of generated cards to reference
        count_accuracy = 1 - abs(len(generated_cards) - len(reference_cards)) / max(1, len(reference_cards))
        
        return {
            "answerability": answerability,
            "coverage": coverage,
            "count_accuracy": count_accuracy
        }
    
    def _calculate_formula_consistency(self, equations: List[str], source_text: str) -> float:
        """Calculate formula consistency score"""
        if not equations:
            return 0.0
        
        hits = 0
        for eq in equations:
            # Extract variable symbols from equation
            vars = [v for v in eq.replace("=", " ").split() if v.isalpha() and len(v) <= 3]
            # Check if variables appear in source text
            match = sum(1 for v in vars if source_text.count(v) > 0)
            if match >= max(1, len(vars) // 2):
                hits += 1
        
        return hits / len(equations)
    
    def _calculate_overall_score(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall evaluation score"""
        overall = {
            "lora_score": 0.0,
            "base_score": 0.0,
            "improvement": 0.0
        }
        
        # Summary quality weight: 40%
        if "summary_quality" in results and results["summary_quality"]["lora_scores"]["rouge1"]["scores"]:
            lora_rouge1 = results["summary_quality"]["lora_scores"]["rouge1"]["average"]
            base_rouge1 = results["summary_quality"]["base_scores"]["rouge1"]["average"]
            overall["lora_score"] += lora_rouge1 * 0.4
            overall["base_score"] += base_rouge1 * 0.4
        
        # Flashcard quality weight: 35%
        if "flashcard_quality" in results and results["flashcard_quality"]["lora_metrics"]["answerability"]["values"]:
            lora_answerability = results["flashcard_quality"]["lora_metrics"]["answerability"]["average"]
            base_answerability = results["flashcard_quality"]["base_metrics"]["answerability"]["average"]
            overall["lora_score"] += lora_answerability * 0.35
            overall["base_score"] += base_answerability * 0.35
        
        # Formula extraction weight: 15%
        if "formula_extraction" in results and results["formula_extraction"]["lora_metrics"]["consistency"]["values"]:
            lora_consistency = results["formula_extraction"]["lora_metrics"]["consistency"]["average"]
            base_consistency = results["formula_extraction"]["base_metrics"]["consistency"]["average"]
            overall["lora_score"] += lora_consistency * 0.15
            overall["base_score"] += base_consistency * 0.15
        
        # Performance weight: 10% (inverse of time, normalized)
        if "performance_metrics" in results:
            lora_time = results["performance_metrics"]["lora_performance"]["avg_time"]
            base_time = results["performance_metrics"]["base_performance"]["avg_time"]
            if lora_time > 0 and base_time > 0:
                # Normalize performance (lower time = higher score)
                max_time = max(lora_time, base_time)
                lora_perf = 1 - (lora_time / max_time)
                base_perf = 1 - (base_time / max_time)
                overall["lora_score"] += lora_perf * 0.1
                overall["base_score"] += base_perf * 0.1
        
        # Calculate improvement
        if overall["base_score"] > 0:
            overall["improvement"] = ((overall["lora_score"] - overall["base_score"]) / overall["base_score"]) * 100
        
        return overall
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n📊 Results saved to: {output_path}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary"""
        print("\n" + "=" * 60)
        print("📊 EVALUATION MATRIX SUMMARY")
        print("=" * 60)
        
        # Model info
        model_info = results.get("model_info", {})
        print(f"LoRA Model: {'✓ Loaded' if model_info.get('lora_loaded') else '✗ Failed'}")
        print(f"Base Model: {'✓ Loaded' if model_info.get('base_loaded') else '✗ Failed'}")
        print(f"LoRA Directory: {model_info.get('lora_dir', 'N/A')}")
        
        # Overall score
        overall = results.get("overall_score", {})
        print(f"\n🎯 OVERALL SCORES:")
        print(f"  LoRA Model: {overall.get('lora_score', 0):.3f}")
        print(f"  Base Model: {overall.get('base_score', 0):.3f}")
        print(f"  Improvement: {overall.get('improvement', 0):+.1f}%")
        
        # Summary quality
        if "summary_quality" in results:
            print(f"\n📝 SUMMARY QUALITY (ROUGE-1):")
            lora_rouge1 = results["summary_quality"]["lora_scores"]["rouge1"]["average"]
            base_rouge1 = results["summary_quality"]["base_scores"]["rouge1"]["average"]
            print(f"  LoRA: {lora_rouge1:.3f}")
            print(f"  Base: {base_rouge1:.3f}")
            if "comparison" in results["summary_quality"]:
                improvement = results["summary_quality"]["comparison"]["rouge1"]["improvement_percent"]
                print(f"  Improvement: {improvement:+.1f}%")
        
        # Flashcard quality
        if "flashcard_quality" in results:
            print(f"\n🃏 FLASHCARD QUALITY:")
            lora_answerability = results["flashcard_quality"]["lora_metrics"]["answerability"]["average"]
            base_answerability = results["flashcard_quality"]["base_metrics"]["answerability"]["average"]
            print(f"  Answerability - LoRA: {lora_answerability:.3f}, Base: {base_answerability:.3f}")
            if "comparison" in results["flashcard_quality"]:
                improvement = results["flashcard_quality"]["comparison"]["answerability"]["improvement_percent"]
                print(f"  Improvement: {improvement:+.1f}%")
        
        # Performance
        if "performance_metrics" in results:
            print(f"\n⚡ PERFORMANCE:")
            lora_time = results["performance_metrics"]["lora_performance"]["avg_time"]
            base_time = results["performance_metrics"]["base_performance"]["avg_time"]
            print(f"  Avg Time - LoRA: {lora_time:.2f}s, Base: {base_time:.2f}s")
            if "comparison" in results["performance_metrics"]:
                slower_by = results["performance_metrics"]["comparison"]["lora_slower_by"]
                print(f"  LoRA is {slower_by:.1f}% slower")
        
        print("\n" + "=" * 60)


def main():
    """Main function for running evaluation matrix"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Evaluation Matrix for LoRA Model")
    parser.add_argument("--lora_dir", required=True, help="Path to LoRA adapter directory")
    parser.add_argument("--lectures_dir", required=True, help="Directory containing lecture PDFs")
    parser.add_argument("--summaries_dir", required=True, help="Directory containing reference summaries")
    parser.add_argument("--flashcards_dir", required=True, help="Directory containing reference flashcards")
    parser.add_argument("--output", default="evaluation_results.json", help="Output file for results")
    parser.add_argument("--base_model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Base model name")
    
    args = parser.parse_args()
    
    # Initialize evaluation matrix
    evaluator = EvaluationMatrix(args.lora_dir, args.base_model)
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation(
        args.lectures_dir,
        args.summaries_dir,
        args.flashcards_dir
    )
    
    # Save and display results
    evaluator.save_results(results, args.output)
    evaluator.print_summary(results)


if __name__ == "__main__":
    main()










