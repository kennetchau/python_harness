"""Web tools: search (ddgs) and fetch (httpx + trafilatura).

Both are lazily imported so a missing dependency degrades to an
error result instead of breaking the tools package import.
"""

from __future__ import annotations

import httpx

from .registry import ToolContext, cap, err, register, res

FETCH_TIMEOUT = 30.0
USER_AGENT = "Mozilla/5.0 (Fedora; Linux x86_64) harness/0.1"
SEARCH_SNIPPET_CHARS = 300
SEARCH_RESULT_CHARS = 6000


def _web_search(args: dict, ctx: ToolContext):
    query = args.get("query", "")
    if not query:
        return err("error: query is required")
    try:
        from ddgs import DDGS
    except ImportError:
        return err("error: ddgs is not installed in the container")
    max_results = min(int(args.get("max_results", ctx.cfg.search.max_results)), 10)
    try:
        hits = DDGS().text(query, max_results=max_results)
    except Exception as e:
        return err(f"error: search failed: {type(e).__name__}: {e}")
    if not hits:
        return res(f"no results for {query!r}")
    lines: list[str] = []
    for i, hit in enumerate(hits, 1):
        title = hit.get("title", "").strip()
        url = hit.get("href") or hit.get("url") or ""
        body = (hit.get("body") or "").strip().replace("\n", " ")
        if len(body) > SEARCH_SNIPPET_CHARS:
            body = body[:SEARCH_SNIPPET_CHARS] + "..."
        lines.append(f"{i}. {title}\n   {url}\n   {body}")
    return res(cap("\n".join(lines), SEARCH_RESULT_CHARS))


def _web_fetch(args: dict, ctx: ToolContext):
    url = args.get("url", "")
    if not url:
        return err("error: url is required")
    if not url.startswith(("http://", "https://")):
        return err("error: url must start with http:// or https://")
    try:
        resp = httpx.get(url, timeout=FETCH_TIMEOUT, follow_redirects=True,
                         headers={"User-Agent": USER_AGENT})
    except httpx.HTTPError as e:
        return err(f"error: fetch failed: {type(e).__name__}: {e}")
    if resp.status_code >= 400:
        return err(f"error: fetch failed: HTTP {resp.status_code} for {url}")
    try:
        from trafilatura import extract
    except ImportError:
        return err("error: trafilatura is not installed in the container")
    text = extract(resp.text) or resp.text
    text = "\n".join(line.rstrip() for line in text.strip().splitlines())
    return res(cap(text, ctx.limits.web_max_chars))


register(
    "web_search",
    "Search the web. Returns numbered results with title, URL, and snippet.",
    {"type": "object",
     "properties": {
         "query": {"type": "string", "description": "Search query"},
         "max_results": {"type": "integer", "minimum": 1, "maximum": 10,
                         "description": "Number of results (default: configured max_results)"}},
     "required": ["query"]},
    _web_search)

register(
    "web_fetch",
    "Fetch a URL and return the main content as plain text (docs, READMEs, "
    "issue pages). Use after web_search to read a result in full.",
    {"type": "object",
     "properties": {"url": {"type": "string",
                            "description": "Absolute http(s) URL"}},
     "required": ["url"]},
    _web_fetch)

