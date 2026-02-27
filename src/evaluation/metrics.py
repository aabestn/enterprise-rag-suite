import logging
from typing import List, Dict, Any
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy
)
from datasets import Dataset

logger = logging.getLogger(__name__)

class RagasEvaluationSuite:
    def __init__(self):
        self.metrics = [
            context_precision,
            context_recall,
            faithfulness,      # Detects hallucination rates
            answer_relevancy
        ]

    def evaluate_batch(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str]
    ) -> Dict[str, float]:
        """
        Runs automated evaluation across context precision, recall, and hallucination rates.
        """
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        dataset = Dataset.from_dict(data)

        logger.info("Executing Ragas evaluation pipeline across batch sample...")
        results = evaluate(
            dataset=dataset,
            metrics=self.metrics
        )

        scores = {
            "context_precision": results["context_precision"],
            "context_recall": results["context_recall"],
            "faithfulness": results["faithfulness"],
            "answer_relevancy": results["answer_relevancy"]
        }
        
        logger.info(f"Evaluation completed successfully: {scores}")
        return scores