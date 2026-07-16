"""WebSearchTool — search official language documentation on the live web.

Scope note: this searches the *official documentation domains* already catalogued
in ``rag.url_index``, not the open web. Two reasons. The corpus is only as fresh
as the last ingest, so the genuine gap this fills is "an API the corpus has never
seen" — and for that, authoritative docs are the only answer worth grounding on;
an open-web result is as likely to be a 2011 forum post as a spec. Restricting to
known-good domains also means a hallucinated-API check gets a trustworthy
yes/no instead of SEO noise.

Implementation uses DuckDuckGo's HTML endpoint via httpx + BeautifulSoup — both
already dependencies, so no new package. That endpoint rate-limits and can block
datacenter IPs; every failure degrades to a ToolResult with ``success=False`` and
the caller carries on, so an unavailable search never breaks a migration.

Off by default (``tool_web_search_enabled``): it is the only tool that reaches
the public internet, so it stays opt-in.
"""

import logging
import re
from urllib.parse import parse_qs, urlparse

import httpx
from bs4 import BeautifulSoup
from rag.url_index import OFFICIAL_DOC_URLS, VERSIONED_DOC_URLS

from agents.tools.base import AgentTool, ToolResult

log = logging.getLogger("CodeMigrateAI.Tools.WebSearch")

_DDG_HTML_ENDPOINT = "https://html.duckduckgo.com/html/"

# DDG serves the HTML endpoint differently (or not at all) to obvious bots.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
}

_MAX_CONTENT_CHARS = 4000


def official_domains(language: str) -> list[str]:
    """Documentation domains for a language, from the existing URL registry.

    Derived rather than hardcoded so adding a language to ``url_index`` extends
    search coverage automatically.
    """
    urls = list(OFFICIAL_DOC_URLS.get(language, []))
    for version_urls in (VERSIONED_DOC_URLS.get(language) or {}).values():
        urls.extend(version_urls)

    domains: list[str] = []
    for url in urls:
        host = urlparse(url).netloc
        if host and host not in domains:
            domains.append(host)
    return domains


class WebSearchTool(AgentTool):
    name = "web_search"
    description = (
        "Search official language documentation on the web for an API, symbol, "
        "or migration question. Use to check whether an API actually exists, or "
        "to find current guidance the local corpus lacks."
    )
    parameters = {
        "query": "what to search for",
        "language": "language whose official docs to search (optional)",
        "fetch_content": "true to also fetch the top result's page text (optional)",
    }

    timeout_sec = 30.0

    def __init__(self, timeout_sec: float | None = None, max_results: int = 5) -> None:
        super().__init__(timeout_sec)
        self._max_results = max_results

    async def run(
        self,
        query: str = "",
        language: str = "",
        fetch_content: bool = False,
        **_,
    ) -> ToolResult:
        if not query.strip():
            return ToolResult(tool=self.name, success=False, error="query is required")

        search_query = self._build_query(query, language)
        try:
            results = await self._search(search_query)
        except httpx.HTTPStatusError as exc:
            # 202/403 here is DDG's rate-limit / bot block, the expected failure.
            return ToolResult(
                tool=self.name,
                success=False,
                error=f"Search unavailable (HTTP {exc.response.status_code})",
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                tool=self.name, success=False, error=f"Search failed: {exc}"
            )

        if not results:
            return ToolResult(
                tool=self.name,
                success=True,
                data={"query": search_query, "results": [], "content": ""},
                summary=f"No documentation results for {query!r}",
            )

        content = ""
        if fetch_content:
            content = await self._fetch_top_content(results)

        return ToolResult(
            tool=self.name,
            success=True,
            data={"query": search_query, "results": results, "content": content},
            summary=f"{len(results)} documentation result(s) for {query!r}",
        )

    def _build_query(self, query: str, language: str) -> str:
        """Restrict to the language's official doc domains when we know them."""
        domains = official_domains(language) if language else []
        if not domains:
            return f"{language} {query}".strip() if language else query
        sites = " OR ".join(f"site:{domain}" for domain in domains)
        return f"{query} ({sites})"

    async def _search(self, search_query: str) -> list[dict]:
        async with httpx.AsyncClient(
            timeout=self.timeout_sec, follow_redirects=True, headers=_HEADERS
        ) as client:
            response = await client.post(
                _DDG_HTML_ENDPOINT, data={"q": search_query}
            )
            response.raise_for_status()
            return self._parse_results(response.text)

    def _parse_results(self, html: str) -> list[dict]:
        soup = BeautifulSoup(html, "lxml")
        results: list[dict] = []
        for node in soup.select(".result"):
            link = node.select_one("a.result__a")
            if not link:
                continue
            url = self._unwrap_url(link.get("href", ""))
            if not url:
                continue
            snippet_node = node.select_one(".result__snippet")
            results.append(
                {
                    "title": link.get_text(strip=True),
                    "url": url,
                    "snippet": snippet_node.get_text(" ", strip=True)
                    if snippet_node
                    else "",
                }
            )
            if len(results) >= self._max_results:
                break
        return results

    @staticmethod
    def _unwrap_url(href: str) -> str:
        """Resolve DDG's ``/l/?uddg=<encoded>`` redirect wrapper to the real URL."""
        if not href:
            return ""
        if "uddg=" in href:
            query = urlparse(href).query
            target = parse_qs(query).get("uddg", [""])[0]
            if target:
                return target
        if href.startswith("//"):
            return f"https:{href}"
        return href if href.startswith("http") else ""

    async def _fetch_top_content(self, results: list[dict]) -> str:
        """Pull the top result's readable text, reusing the ingestion extractor."""
        from rag.web_doc_fetcher import WebDocFetcher

        fetcher = WebDocFetcher()
        try:
            markdown = await fetcher.fetch_as_markdown(results[0]["url"])
        except Exception as exc:  # noqa: BLE001 — content is a bonus, not the result
            log.warning("Could not fetch %s: %s", results[0]["url"], exc)
            return ""
        finally:
            await fetcher.close()
        return re.sub(r"\n{3,}", "\n\n", markdown)[:_MAX_CONTENT_CHARS]
