from __future__ import annotations

import argparse
import html
import json
import random
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ATLAS Page Inspector</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --border: #d9dee7;
      --text: #17202a;
      --muted: #5e6b7a;
      --accent: #275f9f;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
    }
    header {
      display: grid;
      grid-template-columns: minmax(220px, 420px) auto auto 1fr;
      gap: 8px;
      align-items: center;
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      background: var(--panel);
      position: sticky;
      top: 0;
      z-index: 2;
    }
    input {
      width: 100%;
      height: 34px;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 0 10px;
      color: var(--text);
      background: #fff;
    }
    button, select {
      height: 34px;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 0 10px;
      background: #fff;
      color: var(--text);
    }
    button {
      cursor: pointer;
      font-weight: 600;
    }
    button:hover { border-color: var(--accent); color: var(--accent); }
    main {
      display: grid;
      grid-template-columns: 280px 1fr 1fr;
      gap: 10px;
      padding: 10px;
      height: calc(100vh - 55px);
    }
    aside, section {
      min-height: 0;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
    }
    aside {
      display: flex;
      flex-direction: column;
    }
    .status {
      justify-self: end;
      color: var(--muted);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: 100%;
    }
    .list {
      overflow: auto;
      padding: 6px;
    }
    .item {
      width: 100%;
      height: auto;
      min-height: 34px;
      display: block;
      text-align: left;
      border: 0;
      border-radius: 5px;
      padding: 7px 8px;
      color: var(--text);
      background: transparent;
      font-weight: 500;
    }
    .item:hover, .item.active { background: #e9f1fb; color: #174b82; }
    .item small {
      display: block;
      color: var(--muted);
      font-weight: 400;
      margin-top: 2px;
    }
    .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      height: 42px;
      padding: 0 12px;
      border-bottom: 1px solid var(--border);
      font-weight: 700;
    }
    .panel-head span:last-child {
      color: var(--muted);
      font-weight: 500;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    pre {
      margin: 0;
      padding: 12px;
      height: calc(100% - 42px);
      overflow: auto;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      line-height: 1.45;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      font-size: 12px;
    }
    @media (max-width: 980px) {
      header { grid-template-columns: 1fr auto auto; }
      .status { grid-column: 1 / -1; justify-self: start; }
      main { grid-template-columns: 1fr; height: auto; }
      aside, section { min-height: 420px; }
    }
  </style>
</head>
<body>
  <header>
    <input id="search" placeholder="Search title or Wikipedia id">
    <button id="random">Random</button>
    <select id="limit">
      <option value="100">100</option>
      <option value="500" selected>500</option>
      <option value="1000">1000</option>
    </select>
    <div id="status" class="status"></div>
  </header>
  <main>
    <aside>
      <div class="panel-head"><span>Pages</span><span id="count"></span></div>
      <div id="list" class="list"></div>
    </aside>
    <section>
      <div class="panel-head"><span>Raw page</span><span id="rawTitle"></span></div>
      <pre id="raw"></pre>
    </section>
    <section>
      <div class="panel-head"><span>Structured JSON</span><span id="structuredTitle"></span></div>
      <pre id="structured"></pre>
    </section>
  </main>
  <script>
    const search = document.getElementById("search");
    const limit = document.getElementById("limit");
    const list = document.getElementById("list");
    const statusEl = document.getElementById("status");
    const countEl = document.getElementById("count");
    const rawEl = document.getElementById("raw");
    const structuredEl = document.getElementById("structured");
    const rawTitle = document.getElementById("rawTitle");
    const structuredTitle = document.getElementById("structuredTitle");
    let activeId = null;
    let timer = null;

    function setStatus(text) {
      statusEl.textContent = text;
    }

    async function fetchJson(url) {
      const response = await fetch(url);
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || response.statusText);
      }
      return response.json();
    }

    function renderList(pages) {
      countEl.textContent = pages.length;
      list.innerHTML = "";
      for (const page of pages) {
        const button = document.createElement("button");
        button.className = "item" + (page.wikipedia_id === activeId ? " active" : "");
        button.innerHTML = `${escapeHtml(page.title)}<small>${escapeHtml(page.wikipedia_id)}</small>`;
        button.onclick = () => loadPage(page.wikipedia_id);
        list.appendChild(button);
      }
    }

    async function loadList() {
      const q = encodeURIComponent(search.value.trim());
      const n = encodeURIComponent(limit.value);
      setStatus("Loading list...");
      const data = await fetchJson(`/api/pages?q=${q}&limit=${n}`);
      renderList(data.pages);
      setStatus(`Loaded ${data.pages.length} of ${data.total_pages} pages`);
      if (!activeId && data.pages.length) {
        loadPage(data.pages[0].wikipedia_id);
      }
    }

    async function loadPage(id) {
      activeId = id;
      setStatus("Loading page...");
      const data = await fetchJson(`/api/page?id=${encodeURIComponent(id)}`);
      rawTitle.textContent = `${data.title} (${data.wikipedia_id})`;
      structuredTitle.textContent = `${data.title} (${data.wikipedia_id})`;
      rawEl.textContent = data.raw_text || "";
      structuredEl.textContent = JSON.stringify(data.structured, null, 2);
      for (const node of list.querySelectorAll(".item")) {
        node.classList.remove("active");
        if (node.textContent.includes(id)) node.classList.add("active");
      }
      setStatus("Ready");
    }

    function escapeHtml(text) {
      return String(text)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    search.addEventListener("input", () => {
      clearTimeout(timer);
      timer = setTimeout(loadList, 250);
    });
    limit.addEventListener("change", loadList);
    document.getElementById("random").onclick = async () => {
      const data = await fetchJson("/api/random");
      await loadPage(data.wikipedia_id);
    };

    loadList().catch(error => setStatus(error.message));
  </script>
</body>
</html>
"""


class StructuredPageStore:
    def __init__(self, raw_db: Path, structured_jsonl: Path) -> None:
        self.raw_db = raw_db
        self.structured_jsonl = structured_jsonl
        self.lock = threading.Lock()
        self.pages: list[dict[str, str]] = []
        self.offsets: dict[str, int] = {}
        self._build_index()

    def _build_index(self) -> None:
        with self.structured_jsonl.open("rb") as fin:
            while True:
                offset = fin.tell()
                line = fin.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                record = json.loads(line)
                wikipedia_id = str(record["wikipedia_id"])
                title = str(record.get("title", ""))
                self.offsets[wikipedia_id] = offset
                self.pages.append({"wikipedia_id": wikipedia_id, "title": title})

    def search_pages(self, query: str, limit: int) -> list[dict[str, str]]:
        query = query.strip().lower()
        if not query:
            return self.pages[:limit]
        matches = []
        for page in self.pages:
            if query in page["wikipedia_id"].lower() or query in page["title"].lower():
                matches.append(page)
                if len(matches) >= limit:
                    break
        return matches

    def random_page_id(self) -> str:
        if not self.pages:
            raise LookupError("No structured pages loaded")
        return random.choice(self.pages)["wikipedia_id"]

    def get_structured(self, wikipedia_id: str) -> dict:
        offset = self.offsets.get(str(wikipedia_id))
        if offset is None:
            raise LookupError(f"Structured page not found: {wikipedia_id}")
        with self.lock, self.structured_jsonl.open("rb") as fin:
            fin.seek(offset)
            return json.loads(fin.readline())

    def get_raw_text(self, wikipedia_id: str) -> str:
        conn = sqlite3.connect(self.raw_db)
        try:
            row = conn.execute(
                "SELECT text FROM pages WHERE wikipedia_id = ?",
                (str(wikipedia_id),),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return ""
        return row[0] or ""

    def get_page(self, wikipedia_id: str) -> dict:
        structured = self.get_structured(wikipedia_id)
        return {
            "wikipedia_id": structured["wikipedia_id"],
            "title": structured.get("title", ""),
            "raw_text": self.get_raw_text(wikipedia_id),
            "structured": structured,
        }


def make_handler(store: StructuredPageStore) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    self.send_html(INDEX_HTML)
                elif parsed.path == "/api/pages":
                    params = parse_qs(parsed.query)
                    query = params.get("q", [""])[0]
                    limit = int(params.get("limit", ["500"])[0])
                    pages = store.search_pages(query, max(1, min(limit, 5000)))
                    self.send_json({"pages": pages, "total_pages": len(store.pages)})
                elif parsed.path == "/api/page":
                    params = parse_qs(parsed.query)
                    page_id = params.get("id", [""])[0]
                    self.send_json(store.get_page(page_id))
                elif parsed.path == "/api/random":
                    self.send_json({"wikipedia_id": store.random_page_id()})
                else:
                    self.send_error(404)
            except Exception as exc:
                self.send_error(500, html.escape(str(exc)))

        def log_message(self, format: str, *args: object) -> None:
            return

        def send_html(self, body: str) -> None:
            payload = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def send_json(self, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Open a local UI to compare raw and structured Wikipedia pages.")
    parser.add_argument("--raw-db", type=Path, default=Path("data/wikipedia_pages_50k.sqlite"))
    parser.add_argument("--structured-jsonl", type=Path, default=Path("data/structured_pages_sample_20.jsonl"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    store = StructuredPageStore(args.raw_db, args.structured_jsonl)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    print(f"loaded_structured_pages={len(store.pages)}")
    print(f"open http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
