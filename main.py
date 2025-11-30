import argparse
import sys
from data_loader import InputLoader
from chatbot import FoodOrderingChatbot
from evaluator import ChatbotEvaluator


def main():
    parser = argparse.ArgumentParser(
        description="Vietnamese Food Ordering Chatbot with LLM + RAG + Reranker"
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['batch', 'interactive'],
        default='batch',
        help='Run mode: batch (process queries from file) or interactive (chat mode)'
    )
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate chatbot performance after batch processing'
    )
    
    args = parser.parse_args()
    
    try:
        chatbot = FoodOrderingChatbot()
        
        if args.mode == 'interactive':
            chatbot.interactive_mode()
        
        else:
            queries = InputLoader.load_queries()
            print(f"Loaded {len(queries)} queries from input/queries.txt\n")
            
            answers = InputLoader.load_answers()
            print(f"Loaded {len(answers)} answers from input/answers.txt\n")
            
            results = chatbot.process_queries(queries)
            chatbot.save_results(results)
            
            if args.evaluate:
                evaluator = ChatbotEvaluator()
                metrics = evaluator.evaluate_responses(results, answers)
                evaluator.print_metrics()
                evaluator.save_metrics()
            
            print("\n" + "=" * 60)
            print(f"\nResults saved to 'output/' directory:")
            print("  - results.json: Full results with context")
            print("  - answers.txt: Answers only")
            print("  - formatted_output.txt: Human-readable format")
            print("  - query_response_pairs.txt: Q&A pairs")
            if args.evaluate:
                print("  - evaluation_metrics.json: Performance metrics")
            print()
    
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
        sys.exit(0)
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()