"""
CLI script for running the comprehensive evaluation matrix
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.evaluation_matrix import EvaluationMatrix


def main():
    """Main CLI function for evaluation matrix"""
    
    # Default paths based on your project structure
    default_lora_dir = "models/tinyllama-lora"
    default_lectures_dir = "../../lecture notes/lectures"
    default_summaries_dir = "../../lecture notes/summary"
    default_flashcards_dir = "../../lecture notes/flashcard"
    
    print("🚀 AI Lecture Agent - LoRA Model Evaluation Matrix")
    print("=" * 60)
    
    # Get current working directory
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    # Check if we're in the right directory
    if not os.path.exists("src/agent"):
        print("❌ Please run this script from the ai-lecture-agent root directory")
        return
    
    # Construct absolute paths
    lora_dir = os.path.abspath(default_lora_dir)
    lectures_dir = os.path.abspath(default_lectures_dir)
    summaries_dir = os.path.abspath(default_summaries_dir)
    flashcards_dir = os.path.abspath(default_flashcards_dir)
    
    print(f"LoRA Directory: {lora_dir}")
    print(f"Lectures Directory: {lectures_dir}")
    print(f"Summaries Directory: {summaries_dir}")
    print(f"Flashcards Directory: {flashcards_dir}")
    
    # Check if directories exist
    missing_dirs = []
    if not os.path.exists(lora_dir):
        missing_dirs.append(f"LoRA directory: {lora_dir}")
    if not os.path.exists(lectures_dir):
        missing_dirs.append(f"Lectures directory: {lectures_dir}")
    if not os.path.exists(summaries_dir):
        missing_dirs.append(f"Summaries directory: {summaries_dir}")
    if not os.path.exists(flashcards_dir):
        missing_dirs.append(f"Flashcards directory: {flashcards_dir}")
    
    if missing_dirs:
        print("\n❌ Missing directories:")
        for missing in missing_dirs:
            print(f"  - {missing}")
        print("\nPlease ensure all directories exist before running evaluation.")
        return
    
    # Check if LoRA model files exist
    required_files = ["adapter_config.json", "adapter_model.safetensors"]
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(lora_dir, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ Missing LoRA model files in {lora_dir}:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease train the LoRA model first using train_lora.py")
        return
    
    print("\n✅ All directories and files found!")
    print("\nStarting evaluation...")
    
    try:
        # Initialize evaluation matrix
        evaluator = EvaluationMatrix(lora_dir)
        
        # Run comprehensive evaluation
        results = evaluator.run_comprehensive_evaluation(
            lectures_dir,
            summaries_dir,
            flashcards_dir
        )
        
        # Save results
        output_file = "evaluation_results.json"
        evaluator.save_results(results, output_file)
        
        # Print summary
        evaluator.print_summary(results)
        
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"📊 Detailed results saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()










