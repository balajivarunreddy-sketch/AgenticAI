"""DuckDuckGo search integration with bounded, text-only results."""

from ddgs import DDGS
from ddgs.exceptions import DDGSException
from langchain_core.tools import tool

from config.guardrails import redact_sensitive_data, untrusted_data_block

_MAX_QUERY_CHARS = 500


@tool
def web_search(query: str) -> str:
    """Search the public web for current information about a specific query."""
    query = query.strip()
    if not query:
        raise ValueError("Search query cannot be empty.")
    if len(query) > _MAX_QUERY_CHARS:
        raise ValueError(f"Search query cannot exceed {_MAX_QUERY_CHARS} characters.")

    try:
        results = list(DDGS(timeout=15).text(query, max_results=5))
    except DDGSException as exc:
        return untrusted_data_block(
            "WEB SEARCH ERROR",
            f"Web search failed: {exc}. Continue without web results or try again.",
        )
    if not results:
        return "No search results were returned."

    formatted = []
    for item in results:
        title = redact_sensitive_data(str(item.get("title", "Untitled"))[:300])
        url = redact_sensitive_data(str(item.get("href", ""))[:1000])
        snippet = redact_sensitive_data(str(item.get("body", "")))
        formatted.append(f"- {title}\n  URL: {url}\n  Summary: {snippet[:1200]}")
    return untrusted_data_block("WEB SEARCH RESULTS", "\n".join(formatted))
