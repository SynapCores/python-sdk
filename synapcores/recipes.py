"""
Recipe management client for SynapCores Python SDK (v1.5.0-ce).
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .client import SynapCores


class RecipeClient:
    def __init__(self, client: "SynapCores") -> None:
        self.client = client

    def create(
        self,
        name: str,
        content: str,
        description: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        parameters: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"name": name, "content": content}
        if description:
            body["description"] = description
        if category:
            body["category"] = category
        if tags is not None:
            body["tags"] = tags
        if parameters is not None:
            body["parameters"] = parameters
        response = self.client._client.post("/recipes", json=body)
        return self.client._handle_response(response)

    def list(
        self,
        category: Optional[str] = None,
        search: Optional[str] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, str] = {}
        if category:
            params["category"] = category
        if search:
            params["search"] = search
        if page is not None:
            params["page"] = str(page)
        if page_size is not None:
            params["page_size"] = str(page_size)
        if tags:
            params["tags"] = ",".join(tags)
        response = self.client._client.get("/recipes", params=params)
        data = self.client._handle_response(response)
        return data.get("recipes") or data or []

    def get(self, id: str) -> Dict[str, Any]:
        response = self.client._client.get(f"/recipes/{id}")
        return self.client._handle_response(response)

    def update(self, id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        response = self.client._client.put(f"/recipes/{id}", json=updates)
        return self.client._handle_response(response)

    def delete(self, id: str) -> None:
        response = self.client._client.delete(f"/recipes/{id}")
        self.client._handle_response(response)

    def execute(
        self,
        recipe: str,
        parameters: Optional[Dict[str, Any]] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        response = self.client._client.post(
            f"/recipes/{recipe}/execute",
            json={"parameters": parameters or {}, "dry_run": dry_run},
        )
        return self.client._handle_response(response)

    def validate(self, body: Dict[str, Any]) -> Dict[str, Any]:
        response = self.client._client.post("/recipes/validate", json=body)
        return self.client._handle_response(response)

    def get_history(self, id: str) -> List[Dict[str, Any]]:
        response = self.client._client.get(f"/recipes/{id}/history")
        data = self.client._handle_response(response)
        return data.get("history") or data.get("executions") or data or []

    def list_categories(self) -> List[Dict[str, Any]]:
        response = self.client._client.get("/recipes/categories/counts")
        data = self.client._handle_response(response)
        return data.get("categories") or data or []

    def list_templates(self) -> List[Dict[str, Any]]:
        response = self.client._client.get("/recipes/templates")
        data = self.client._handle_response(response)
        return data.get("templates") or data or []

    def get_template(self, id: str) -> Dict[str, Any]:
        response = self.client._client.get(f"/recipes/templates/{id}")
        return self.client._handle_response(response)

    def execute_template(
        self,
        id: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        response = self.client._client.post(
            f"/recipes/templates/{id}/execute",
            json={"parameters": params or {}},
        )
        return self.client._handle_response(response)

    def list_executions(
        self,
        limit: Optional[int] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, str] = {}
        if limit is not None:
            params["limit"] = str(limit)
        if status:
            params["status"] = status
        response = self.client._client.get("/recipes/executions", params=params)
        data = self.client._handle_response(response)
        return data.get("executions") or data or []

    def get_execution(self, id: str) -> Dict[str, Any]:
        response = self.client._client.get(f"/recipes/executions/{id}")
        return self.client._handle_response(response)

    def generate(
        self,
        intent: str,
        category: Optional[str] = None,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"intent": intent}
        if category:
            body["category"] = category
        if context:
            body["context"] = context
        response = self.client._client.post("/ai/generate-recipe", json=body)
        return self.client._handle_response(response)
