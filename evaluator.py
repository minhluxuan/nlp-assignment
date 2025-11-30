import json
import os
import re
from typing import List, Dict
from collections import Counter
import config


class ChatbotEvaluator:
    
    def __init__(self):
        self.metrics = {}
    
    def normalize_text(self, text: str) -> str:
        text = text.lower()
        text = ' '.join(text.split())
        # Remove punctuation except currency
        text = re.sub(r'[^\w\s\dđ₫]', '', text)
        return text
    
    def calculate_exact_match(self, predicted: str, reference: str) -> float:
        pred_norm = self.normalize_text(predicted)
        ref_norm = self.normalize_text(reference)
        return 1.0 if pred_norm == ref_norm else 0.0
    
    def calculate_f1_score(self, predicted: str, reference: str) -> float:
        pred_tokens = set(self.normalize_text(predicted).split())
        ref_tokens = set(self.normalize_text(reference).split())
        
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        common = pred_tokens & ref_tokens
        
        if not common:
            return 0.0
        
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def calculate_bleu(self, predicted: str, reference: str, n: int = 2) -> float:
        def get_ngrams(text: str, n: int) -> List[tuple]:
            tokens = self.normalize_text(text).split()
            if len(tokens) < n:
                return []
            return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        
        pred_ngrams = get_ngrams(predicted, n)
        ref_ngrams = get_ngrams(reference, n)
        
        if not pred_ngrams or not ref_ngrams:
            return 0.0
        
        pred_counter = Counter(pred_ngrams)
        ref_counter = Counter(ref_ngrams)
        
        overlap = sum((pred_counter & ref_counter).values())
        total = sum(pred_counter.values())
        
        return overlap / total if total > 0 else 0.0
    
    def calculate_rouge_l(self, predicted: str, reference: str) -> float:
        def lcs_length(s1: List[str], s2: List[str]) -> int:
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i-1] == s2[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n]
        
        pred_tokens = self.normalize_text(predicted).split()
        ref_tokens = self.normalize_text(reference).split()
        
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        lcs = lcs_length(pred_tokens, ref_tokens)
        
        precision = lcs / len(pred_tokens) if pred_tokens else 0
        recall = lcs / len(ref_tokens) if ref_tokens else 0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    def evaluate_responses(self, results: List[Dict], ground_truth: List[str] = None) -> Dict:
        print("\n" + "=" * 60)
        print("Evaluating Chatbot Performance")
        print("=" * 60 + "\n")
        
        metrics = {
            "total_queries": len(results),
            "avg_response_length": 0,
            "context_retrieval_rate": 0,
        }
        
        # Calculate basic statistics
        response_lengths = []
        contexts_retrieved = 0
        
        for result in results:
            response_lengths.append(len(result['response']))
            if result.get('context'):
                contexts_retrieved += 1
        
        metrics["avg_response_length"] = sum(response_lengths) / len(response_lengths) if response_lengths else 0
        metrics["context_retrieval_rate"] = contexts_retrieved / len(results) if results else 0
        
        # If ground truth is provided, calculate quality metrics
        if ground_truth and len(ground_truth) == len(results):
            print("Calculating quality metrics with ground truth...\n")
            
            exact_matches = []
            f1_scores = []
            bleu_scores = []
            rouge_scores = []
            
            for result, ref in zip(results, ground_truth):
                pred = result['response']
                
                exact_matches.append(self.calculate_exact_match(pred, ref))
                f1_scores.append(self.calculate_f1_score(pred, ref))
                bleu_scores.append(self.calculate_bleu(pred, ref))
                rouge_scores.append(self.calculate_rouge_l(pred, ref))
            
            metrics["exact_match"] = sum(exact_matches) / len(exact_matches)
            metrics["avg_f1_score"] = sum(f1_scores) / len(f1_scores)
            metrics["avg_bleu_score"] = sum(bleu_scores) / len(bleu_scores)
            metrics["avg_rouge_l"] = sum(rouge_scores) / len(rouge_scores)
        
        self.metrics = metrics
        return metrics
    
    def print_metrics(self):
        if not self.metrics:
            print("No metrics to display. Run evaluate_responses first.")
            return
        
        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60 + "\n")
        
        print(f"Total Queries: {self.metrics['total_queries']}")
        print(f"Average Response Length: {self.metrics['avg_response_length']:.2f} characters")
        print(f"Context Retrieval Rate: {self.metrics['context_retrieval_rate']:.2%}")
        
        if "avg_f1_score" in self.metrics:
            print(f"\nQuality Metrics:")
            print(f"  Exact Match: {self.metrics['exact_match']:.2%}")
            print(f"  Average F1 Score: {self.metrics['avg_f1_score']:.4f}")
            print(f"  Average BLEU Score: {self.metrics['avg_bleu_score']:.4f}")
            print(f"  Average ROUGE-L: {self.metrics['avg_rouge_l']:.4f}")
        
        print("\n" + "=" * 60)
    
    def save_metrics(self, filename: str = "evaluation_metrics.json"):
        if not self.metrics:
            print("No metrics to save.")
            return
        
        output_file = os.path.join(config.OUTPUT_DIR, filename)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2)
        
        print(f"\nMetrics saved to: {output_file}")