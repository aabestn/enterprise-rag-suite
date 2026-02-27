import mlflow
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class MLflowEvalLogger:
    def __init__(self, tracking_uri: str, experiment_name: str = "enterprise_rag_eval"):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def log_evaluation_run(
        self,
        run_name: str,
        parameters: Dict[str, Any],
        metrics: Dict[str, float]
    ):
        """
        Logs RAG pipeline hyperparameters and evaluation metrics into MLflow.
        """
        with mlflow.start_run(run_name=run_name):
            # Log setup hyperparameters (chunk size, model configs, reranker settings)
            mlflow.log_params(parameters)

            # Log Ragas evaluation metrics
            mlflow.log_metrics(metrics)

            logger.info(f"Logged evaluation run '{run_name}' to MLflow tracking server.")