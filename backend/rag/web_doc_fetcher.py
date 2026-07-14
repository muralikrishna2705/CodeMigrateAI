import asyncio
import hashlib
import logging
import os
import time
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify as md

from config import get_settings

log = logging.getLogger("CodeMigrateAI.WebDocFetcher")

REFERENCE_DIR = Path(__file__).resolve().parent / "reference"


class WebDocFetcher:
    def __init__(self):
        self.settings = get_settings()
        self.client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)

    async def fetch_for_language(self, language: str, urls: list[str]) -> list[Path]:
        cache_dir = REFERENCE_DIR / language / "_fetched"
        cache_dir.mkdir(parents=True, exist_ok=True)

        saved_files = []
        for url in urls:
            cache_path = cache_dir / f"{self._url_to_filename(url)}.md"
            if cache_path.exists():
                age_hours = (os.path.getmtime(cache_path) - time.time()) / 3600
                if abs(age_hours) < self.settings.web_docs_refresh_days * 24:
                    saved_files.append(cache_path)
                    continue

            try:
                content = await self._fetch_and_convert(url)
                cache_path.write_text(content, encoding="utf-8")
                saved_files.append(cache_path)
                log.info("Fetched %s -> %s", url, cache_path.name)
            except Exception as e:
                log.warning("Failed to fetch %s: %s", url, e)

            await asyncio.sleep(0.5)  # Rate limiting

        return saved_files

    async def _fetch_and_convert(self, url: str) -> str:
        response = await self.client.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")
        # Remove nav, footer, script, style
        for tag in soup(["nav", "footer", "script", "style", "header", "aside"]):
            tag.decompose()
        # Try main content area first
        main = soup.find("main") or soup.find("article") or soup.find("body")
        html = str(main) if main else response.text
        return md(html, heading_style="ATX", strip=["img"])

    def _url_to_filename(self, url: str) -> str:
        return hashlib.sha256(url.encode()).hexdigest()[:16]

    async def close(self):
        await self.client.aclose()
