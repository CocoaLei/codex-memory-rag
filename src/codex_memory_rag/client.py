import httpx

from .schemas import CmrError


def request(config, method, path, payload=None, params=None):
    transport = httpx.HTTPTransport(uds=str(config.storage.runtime_dir / "run/daemon.sock"))
    try:
        with httpx.Client(transport=transport, base_url="http://cmr", trust_env=False,
                          timeout=httpx.Timeout(25, connect=2)) as client:
            response = client.request(method, path, json=payload, params=params)
            result = response.json()
            if response.status_code >= 400:
                raise CmrError(result.get("error", "INDEX_NOT_READY"))
            return result
    except (httpx.TransportError, ValueError):
        raise CmrError("MODEL_UNAVAILABLE") from None
