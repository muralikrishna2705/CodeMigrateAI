import httpx


class ValidatorClient:
    def __init__(self, base_url: str, timeout_sec: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout_sec)

    async def validate(self, *, code: str, language: str, version: str) -> dict:
        response = await self._client.post(
            f"{self.base_url}/validate",
            json={
                "code": code,
                "language": language,
                "version": version,
            },
        )
        response.raise_for_status()
        return response.json()

    async def close(self):
        await self._client.aclose()
