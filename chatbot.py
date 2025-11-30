import json
import os

from typing import List, Dict

from data_loader import MenuDataLoader
from rag_system import RAGSystem
from llm_generator import LLMGenerator
import config


class FoodOrderingChatbot:
    
    def __init__(self):
        """Initialize chatbot components"""
        print("=" * 60)
        print("Vietnamese Food Ordering Chatbot")
        print("LLM + RAG + Reranker System")
        print("=" * 60)
        
        self.menu_loader = MenuDataLoader()
        documents = self.menu_loader.get_documents_for_rag()
        print(f"Loaded {len(documents)} menu items")
        
        self.rag_system = RAGSystem(documents)
        self.llm_generator = LLMGenerator()
        
        print("Chatbot initialization complete!")
    
    def process_query(self, query: str) -> Dict:
        context = self.rag_system.get_context(query)
        response = self.llm_generator.generate(query, context)
        
        return {
            "query": query,
            "context": context,
            "response": response
        }
    
    def process_queries(self, queries: List[str]) -> List[Dict]:
        results = []
        print(f"Processing {len(queries)} queries...\n")
        
        for i, query in enumerate(queries, 1):
            print(f"[{i}/{len(queries)}] Processing: {query}")
            
            result = self.process_query(query)
            results.append(result)
            
            print(f"Response: {result['response']}...")
            print()
        
        return results
    
    def save_results(self, results: List[Dict]):
        print("Saving results to output/...")
        
        # Save full results as JSON
        output_json = os.path.join(config.OUTPUT_DIR, "results.json")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved: {output_json}")
        
        # Save answers only (for evaluation)
        output_answers = os.path.join(config.OUTPUT_DIR, "answers.txt")
        with open(output_answers, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(result['response'] + '\n')
        print(f"Saved: {output_answers}")
        
        # Save formatted output
        output_formatted = os.path.join(config.OUTPUT_DIR, "formatted_output.txt")
        with open(output_formatted, 'w', encoding='utf-8') as f:
            for i, result in enumerate(results, 1):
                f.write(f"{'=' * 80}\n")
                f.write(f"Câu hỏi {i}: {result['query']}\n")
                f.write(f"{'=' * 80}\n\n")
                
                if result['context']:
                    f.write("Ngữ cảnh được truy xuất:\n")
                    f.write(f"{result['context']}\n\n")
                
                f.write("Câu trả lời:\n")
                f.write(f"{result['response']}\n\n")
        print(f"Saved: {output_formatted}")
        
        # Save query-response pairs
        output_pairs = os.path.join(config.OUTPUT_DIR, "query_response_pairs.txt")
        with open(output_pairs, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(f"Q: {result['query']}\n")
                f.write(f"A: {result['response']}\n")
                f.write("-" * 80 + "\n")
        print(f"Saved: {output_pairs}")
        
        print("\nAll results saved successfully!")
    
    def interactive_mode(self):
        print("\n" + "=" * 60)
        print("Interactive Mode - Vietnamese Food Ordering Chatbot")
        print("Type 'exit' or 'quit' to stop")
        print("=" * 60 + "\n")
        
        while True:
            try:
                query = input("Bạn: ").strip()
                
                if query.lower() in ['exit', 'quit', 'thoát']:
                    print("\nCảm ơn bạn đã sử dụng dịch vụ! Hẹn gặp lại!")
                    break
                
                if not query:
                    continue
                
                result = self.process_query(query)
                print(f"\nChatbot: {result['response']}\n")
                
            except KeyboardInterrupt:
                print("\n\nCảm ơn bạn đã sử dụng dịch vụ! Hẹn gặp lại!")
                break
            except Exception as e:
                print(f"\nLỗi: {e}\n")