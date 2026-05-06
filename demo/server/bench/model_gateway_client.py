"""模型网关客户端：透传 OpenAI 兼容 Chat Completions 请求。"""

from __future__ import annotations

from typing import Any

import requests


class ModelGatewayClient:
    """封装对上游模型接口的调用。"""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        default_model_name: str,
        timeout_seconds: float,
    ) -> None:
        self._api_url = api_url
        self._api_key = api_key
        self._default_model_name = default_model_name
        self._timeout_seconds = timeout_seconds
        self._session = requests.Session()
        self._session.trust_env = False

    def chat_completion(self, payload: dict[str, Any]) -> requests.Response:
        """发起非流式请求并返回原始响应对象。"""
        body = self._build_payload(payload)
        return self._session.post(
            self._api_url,
            headers=self._build_headers(),
            json=body,
            timeout=self._timeout_seconds,
        )

    def _build_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = dict(payload)
        body.setdefault("model", self._default_model_name)
        return body

    def _build_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

    def close(self) -> None:
        """释放底层连接（评测进程退出前建议调用）。"""
        self._session.close()
