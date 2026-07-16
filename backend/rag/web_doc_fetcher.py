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
        """Fetch version-agnostic official docs into ``<lang>/_fetched/`` (flat)."""
        cache_dir = REFERENCE_DIR / language / "_fetched"
        return await self._fetch_into(cache_dir, urls)

    async def fetch_versioned(
        self,
        language: str,
        versioned: dict[str, list[str]],
        doc_type: str,
    ) -> list[Path]:
        """Fetch per-version official docs into ``<lang>/_fetched/<version>/<doc_type>/``.

        The nested layout is what the corpus loader keys on: the ``<version>``
        segment stamps a real version and the ``<doc_type>`` segment marks the
        chunk as an authoritative migration/release-notes doc, activating the
        version-aware retrieval ladder and the metadata ranking boosts.
        """
        saved_files: list[Path] = []
        for version, urls in versioned.items():
            cache_dir = REFERENCE_DIR / language / "_fetched" / version / doc_type
            saved_files.extend(await self._fetch_into(cache_dir, urls))
        return saved_files

    async def _fetch_into(self, cache_dir: Path, urls: list[str]) -> list[Path]:
        """Fetch ``urls`` into ``cache_dir`` with freshness caching + rate limiting."""
        cache_dir.mkdir(parents=True, exist_ok=True)

        saved_files = []
        for url in urls:
            cache_path = cache_dir / f"{self._url_to_filename(url)}.md"
            if cache_path.exists():
                age_hours = (time.time() - os.path.getmtime(cache_path)) / 3600
                if age_hours < self.settings.web_docs_refresh_days * 24:
                    saved_files.append(cache_path)
                    continue

            try:
                content = await self._fetch_and_convert(url)
                cache_path.write_text(content, encoding="utf-8")
                saved_files.append(cache_path)
                log.info("Fetched %s -> %s", url, cache_path)
            except Exception as e:
                log.warning("Failed to fetch %s: %s", url, e)

            await asyncio.sleep(0.5)  # Rate limiting

        return saved_files

    async def fetch_as_markdown(self, url: str) -> str:
        """Fetch one URL and return its main content as markdown.

        Public because the WebSearchTool reads a result page through the same
        boilerplate-stripping extraction the corpus ingestion uses, rather than
        reimplementing it.
        """
        return await self._fetch_and_convert(url)

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
