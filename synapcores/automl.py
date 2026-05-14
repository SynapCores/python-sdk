"""
AutoML client for SynapCores Python SDK.

v0.2.0: paths migrated from /ai/* to /automl/* to match gateway v1.5.0-ce.
"""

from typing import Dict, Any, List, Optional, Union, TYPE_CHECKING
import pandas as pd

from .models import ModelInfo

if TYPE_CHECKING:
    from .client import SynapCores


class AutoMLModel:
    """Represents a trained AutoML model."""

    def __init__(self, client: "AutoMLClient", model_info: ModelInfo):
        self.client = client
        self.info = model_info
        self.id = model_info.id
        self.name = model_info.name

    def predict(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
    ) -> Union[Any, List[Any]]:
        """Make predictions using the model."""
        if isinstance(data, pd.DataFrame):
            data = data.to_dict("records")

        is_single = isinstance(data, dict)
        inputs = [data] if is_single else data

        response = self.client.client._client.post(
            f"/automl/models/{self.id}/predict",
            json={"inputs": inputs},
        )
        result = self.client.client._handle_response(response)

        predictions = result.get("predictions") or result
        return predictions[0] if is_single else predictions

    def evaluate(
        self,
        test_data: Union[str, pd.DataFrame],
        target: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate model performance."""
        payload: Dict[str, Any] = {}
        if isinstance(test_data, str):
            payload["collection"] = test_data
        else:
            payload["data"] = test_data.to_dict("records")
            if target:
                payload["target"] = target

        response = self.client.client._client.post(
            f"/automl/models/{self.id}/evaluate",
            json=payload,
        )
        return self.client.client._handle_response(response)

    def delete(self) -> None:
        """Delete the model."""
        response = self.client.client._client.delete(
            f"/automl/models/{self.id}"
        )
        self.client.client._handle_response(response)


class AutoMLClient:
    """Client for AutoML operations."""

    def __init__(self, client: "SynapCores"):
        self.client = client

    def train(
        self,
        collection: str,
        target: str,
        features: Optional[List[str]] = None,
        task: str = "auto",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        validation_split: float = 0.2,
        max_trials: int = 10,
        timeout_minutes: int = 60,
    ) -> AutoMLModel:
        """Train an AutoML model (POST /automl/train)."""
        payload = {
            "collection": collection,
            "target": target,
            "features": features,
            "task": task,
            "name": name or f"{collection}_{target}_model",
            "config": config or {},
            "validation_split": validation_split,
            "max_trials": max_trials,
            "timeout_minutes": timeout_minutes,
        }

        response = self.client._client.post(
            "/automl/train",
            json=payload,
        )
        data = self.client._handle_response(response)

        # The gateway often returns a "job" wrapper for long-running runs.
        # Surface a fully-fledged ModelInfo when ``id``/``name`` are present.
        model_info = ModelInfo(
            id=data.get("id") or data.get("model_id") or "",
            name=data.get("name") or payload["name"],
            task=data.get("task") or task,
            status=data.get("status") or "training",
            accuracy=data.get("accuracy"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            config=data.get("config") or {},
        )
        return AutoMLModel(self, model_info)

    def get_model(self, model_id: str) -> AutoMLModel:
        """Fetch a trained model by id."""
        response = self.client._client.get(f"/automl/models/{model_id}")
        data = self.client._handle_response(response)
        model_info = ModelInfo(
            id=data.get("id") or model_id,
            name=data.get("name") or "",
            task=data.get("task") or "auto",
            status=data.get("status") or "ready",
            accuracy=data.get("accuracy"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            config=data.get("config") or {},
        )
        return AutoMLModel(self, model_info)

    def list_models(
        self,
        task: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[ModelInfo]:
        """List AutoML models."""
        params: Dict[str, str] = {}
        if task:
            params["task"] = task
        if status:
            params["status"] = status

        response = self.client._client.get(
            "/automl/models",
            params=params,
        )
        data = self.client._handle_response(response)

        models = data.get("models") or data or []
        return [
            ModelInfo(
                id=m.get("id") or m.get("model_id") or "",
                name=m.get("name") or "",
                task=m.get("task") or "auto",
                status=m.get("status") or "ready",
                accuracy=m.get("accuracy"),
                created_at=m.get("created_at"),
                updated_at=m.get("updated_at"),
                config=m.get("config") or {},
            )
            for m in models
        ]

    # --- Async training jobs --------------------------------------------------

    def get_training_job(self, job_id: str) -> Dict[str, Any]:
        """GET /automl/jobs/:id"""
        response = self.client._client.get(f"/automl/jobs/{job_id}")
        return self.client._handle_response(response)

    def list_training_jobs(
        self,
        status: Optional[str] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """GET /automl/jobs"""
        params: Dict[str, str] = {}
        if status:
            params["status"] = status
        if page:
            params["page"] = str(page)
        if page_size:
            params["page_size"] = str(page_size)
        response = self.client._client.get("/automl/jobs", params=params)
        data = self.client._handle_response(response)
        return data.get("jobs") or data or []

    def cancel_training_job(self, job_id: str) -> None:
        """POST /automl/jobs/:id/stop"""
        response = self.client._client.post(f"/automl/jobs/{job_id}/stop")
        self.client._handle_response(response)
