"""
NLP client for SynapCores Python SDK.

v0.2.0: replaced /ai/analyze (which never existed in v1.5.0-ce) with a
client-side fan-out across /ai/sentiment + /ai/entities + /ai/summarize.
Added qa() for question-answering against /ai/qa.
"""

from typing import List, Optional, Union, Dict, Any, TYPE_CHECKING

from .models import NLPAnalysis

if TYPE_CHECKING:
    from .client import SynapCores


class NLPClient:
    """Client for NLP operations."""

    def __init__(self, client: "SynapCores"):
        self.client = client

    def analyze(
        self,
        text: Union[str, List[str]],
        tasks: Optional[List[str]] = None,
        language: Optional[str] = None,
    ) -> Union[NLPAnalysis, List[NLPAnalysis]]:
        """Run multiple NLP tasks and merge the results."""
        is_batch = isinstance(text, list)
        texts = text if is_batch else [text]
        tasks = tasks or ["sentiment", "entities", "summarize"]

        want_sentiment = "sentiment" in tasks
        want_entities = "entities" in tasks
        want_summary = "summary" in tasks or "summarize" in tasks

        results: List[NLPAnalysis] = []
        for t in texts:
            sentiment = None
            entities: List[NLPAnalysis.Entity] = []
            summary: Optional[str] = None

            if want_sentiment:
                try:
                    payload = {"texts": [t], "language": language}
                    response = self.client._client.post("/ai/sentiment", json=payload)
                    data = self.client._handle_response(response)
                    arr = data.get("sentiments") or []
                    if arr:
                        s = arr[0]
                        sentiment = NLPAnalysis.Sentiment(
                            label=s.get("label", "neutral"),
                            score=s.get("score", 0.0),
                            confidence=s.get("confidence", 0.0),
                        )
                except Exception:
                    pass

            if want_entities:
                try:
                    response = self.client._client.post(
                        "/ai/entities", json={"text": t, "language": language}
                    )
                    data = self.client._handle_response(response)
                    entities = [
                        NLPAnalysis.Entity(
                            text=e.get("text", ""),
                            type=e.get("type", ""),
                            start=e.get("start", 0),
                            end=e.get("end", 0),
                            score=e.get("score", 0.0),
                        )
                        for e in data.get("entities", [])
                    ]
                except Exception:
                    pass

            if want_summary:
                try:
                    response = self.client._client.post(
                        "/ai/summarize", json={"text": t}
                    )
                    data = self.client._handle_response(response)
                    summary = data.get("summary")
                except Exception:
                    pass

            results.append(
                NLPAnalysis(
                    sentiment=sentiment,
                    entities=entities or None,
                    summary=summary,
                    language=language,
                )
            )

        return results if is_batch else results[0]

    def summarize(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 30,
    ) -> str:
        """Generate text summary."""
        payload = {
            "text": text,
            "max_length": max_length,
            "min_length": min_length,
        }
        response = self.client._client.post("/ai/summarize", json=payload)
        data = self.client._handle_response(response)
        return data.get("summary", "")

    def extract_entities(
        self,
        text: str,
        entity_types: Optional[List[str]] = None,
    ) -> List[NLPAnalysis.Entity]:
        """Extract named entities from text."""
        payload = {"text": text, "entity_types": entity_types}
        response = self.client._client.post("/ai/entities", json=payload)
        data = self.client._handle_response(response)
        return [
            NLPAnalysis.Entity(
                text=e.get("text", ""),
                type=e.get("type", ""),
                start=e.get("start", 0),
                end=e.get("end", 0),
                score=e.get("score", 0.0),
            )
            for e in data.get("entities", [])
        ]

    def sentiment(
        self,
        text: Union[str, List[str]],
    ) -> Union[NLPAnalysis.Sentiment, List[NLPAnalysis.Sentiment]]:
        """Analyze sentiment of text."""
        is_batch = isinstance(text, list)
        texts = text if is_batch else [text]
        response = self.client._client.post("/ai/sentiment", json={"texts": texts})
        data = self.client._handle_response(response)
        results = [
            NLPAnalysis.Sentiment(
                label=s.get("label", "neutral"),
                score=s.get("score", 0.0),
                confidence=s.get("confidence", 0.0),
            )
            for s in data.get("sentiments", [])
        ]
        return results if is_batch else (results[0] if results else NLPAnalysis.Sentiment(label="neutral", score=0.0, confidence=0.0))

    def classify(
        self,
        text: Union[str, List[str]],
        categories: List[str],
        multi_label: bool = False,
    ) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """Classify text into categories."""
        is_batch = isinstance(text, list)
        texts = text if is_batch else [text]
        payload = {
            "texts": texts,
            "categories": categories,
            "multi_label": multi_label,
        }
        response = self.client._client.post("/ai/classify", json=payload)
        data = self.client._handle_response(response)
        results = data.get("classifications") or []
        return results if is_batch else (results[0] if results else {})

    def qa(
        self,
        question: str,
        context: Optional[str] = None,
        max_answer_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Question answering against /ai/qa."""
        payload: Dict[str, Any] = {"question": question}
        if context is not None:
            payload["context"] = context
        if max_answer_tokens is not None:
            payload["max_answer_tokens"] = max_answer_tokens
        response = self.client._client.post("/ai/qa", json=payload)
        data = self.client._handle_response(response)
        return {
            "answer": data.get("answer") or data.get("text") or "",
            "score": data.get("score") or data.get("confidence"),
            "start": data.get("start"),
            "end": data.get("end"),
        }
