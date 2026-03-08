import json

import os
import time
import sqlite3
import subprocess
from contextlib import contextmanager
from typing import Optional, Iterator, BinaryIO
from lxml import etree


def strip_tag(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def find_decompressor() -> Optional[list[str]]:
    """
    Return the fastest available bzip2 decompressor command.
    Preference:
      1) lbzip2 -dc
      2) pbzip2 -dc
      3) bzip2 -dc
    If nothing is found, return None and fallback to Python bz2 reader.
    """
    import shutil

    for cmd in ("lbzip2", "pbzip2", "bzip2"):
        path = shutil.which(cmd)
        if path is not None:
            return [path, "-dc"]
    return None


@contextmanager
def open_bz2_stream(dump_path: str) -> Iterator[BinaryIO]:
    """
    Open .bz2 dump as a binary stream.
    Uses external decompressor if available, otherwise falls back to bz2.
    """
    decompressor = find_decompressor()

    if decompressor is not None:
        proc = subprocess.Popen(
            decompressor + [dump_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1024 * 1024,
        )
        try:
            assert proc.stdout is not None
            yield proc.stdout
        finally:
            if proc.stdout:
                proc.stdout.close()
            stderr = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
            ret = proc.wait()
            if ret != 0:
                raise RuntimeError(
                    f"Decompressor failed with exit code {ret}.\n"
                    f"Command: {' '.join(decompressor)} {dump_path}\n"
                    f"stderr:\n{stderr}"
                )
    else:
        import bz2
        f = bz2.open(dump_path, "rb")
        try:
            yield f
        finally:
            f.close()


def init_db(db_path: str) -> sqlite3.Connection:
    """
    SQLite settings for maximum ingestion speed.
    These settings prioritize speed over crash safety during loading.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=OFF;")
    conn.execute("PRAGMA synchronous=OFF;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA locking_mode=EXCLUSIVE;")
    conn.execute("PRAGMA cache_size=-500000;")  # ~500 MB page cache
    conn.execute("PRAGMA page_size=65536;")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pages (
            wikipedia_id TEXT PRIMARY KEY,
            title        TEXT NOT NULL,
            ns           INTEGER,
            text         TEXT
        )
        """
    )
    conn.commit()
    return conn


def is_redirect_text(text: Optional[str]) -> bool:
    if not text:
        return False
    return text.lstrip().upper().startswith("#REDIRECT")


def parse_dump(
    dump_path: str,
    db_path: str,
    batch_size: int = 20000,
    only_main_namespace: bool = False,
    log_every: int = 100000,
) -> None:
    """
    Fast streaming parser for enwiki-pages-articles.xml.bz2.

    Saves:
      - wikipedia_id
      - title
      - ns
      - text

    Skips:
      - redirect pages
      - optionally non-main namespace pages

    Notes:
      - No percent progress, because that requires a full extra pass.
      - This is optimized for throughput.
    """
    start_time = time.time()

    conn = init_db(db_path)
    cur = conn.cursor()

    processed = 0
    inserted = 0
    skipped_redirect = 0
    skipped_other_ns = 0
    batch = []

    cur.execute("BEGIN")

    with open_bz2_stream(dump_path) as stream:
        context = etree.iterparse(
            stream,
            events=("end",),
            tag="{*}page",
            huge_tree=True,
        )

        for _, elem in context:
            processed += 1

            title = None
            ns = None
            wikipedia_id = None
            text = None
            has_redirect_tag = False

            # Fast direct child scan
            for child in elem:
                tag = strip_tag(child.tag)

                if tag == "title":
                    title = child.text

                elif tag == "ns":
                    t = child.text
                    if t is not None:
                        try:
                            ns = int(t)
                        except ValueError:
                            ns = None

                elif tag == "id" and wikipedia_id is None:
                    # first <id> under <page> is page id; later ones are revision ids
                    wikipedia_id = child.text

                elif tag == "redirect":
                    has_redirect_tag = True

                elif tag == "revision":
                    for rev_child in child:
                        if strip_tag(rev_child.tag) == "text":
                            text = rev_child.text
                            break

            # Skip redirects
            if has_redirect_tag or is_redirect_text(text):
                skipped_redirect += 1

            # Optional namespace filter
            elif only_main_namespace and ns != 0:
                skipped_other_ns += 1

            # Save valid page
            elif wikipedia_id and title:
                batch.append((wikipedia_id, title, ns, text))

            # Flush batch
            if len(batch) >= batch_size:
                cur.executemany(
                    "INSERT INTO pages (wikipedia_id, title, ns, text) VALUES (?, ?, ?, ?)",
                    batch,
                )
                inserted += len(batch)
                batch.clear()

            # Log progress
            if processed % log_every == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0.0
                print(
                    f"processed={processed:,} | inserted={inserted:,} | "
                    f"redirects={skipped_redirect:,} | ns_skipped={skipped_other_ns:,} | "
                    f"speed={rate:,.0f} pages/s | elapsed={elapsed/60:.1f} min"
                )

            # Aggressive memory cleanup
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]

    # Final flush
    if batch:
        cur.executemany(
            "INSERT INTO pages (wikipedia_id, title, ns, text) VALUES (?, ?, ?, ?)",
            batch,
        )
        inserted += len(batch)

    conn.commit()

    # Create indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pages_title ON pages(title)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pages_ns ON pages(ns)")
    conn.commit()
    conn.close()

    elapsed = time.time() - start_time
    rate = processed / elapsed if elapsed > 0 else 0.0

    print("\nFinished")
    print(f"Processed:         {processed:,}")
    print(f"Inserted:          {inserted:,}")
    print(f"Redirects skipped: {skipped_redirect:,}")
    print(f"NS skipped:        {skipped_other_ns:,}")
    print(f"Elapsed:           {elapsed/60:.2f} min")
    print(f"Average speed:     {rate:,.0f} pages/s")


def get_page_by_wikipedia_id(db_path: str, wikipedia_id: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """
        SELECT wikipedia_id, title, ns, text
        FROM pages
        WHERE wikipedia_id = ?
        """,
        (str(wikipedia_id),),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def process_jsonl(input_path: str, output_path: str):
    """
    Reads a jsonl file and writes a new jsonl file with fields:
      - id
      - input
      - wikipedia_id (array)
      - answer (array)

    Example is skipped if:
      - wikipedia_id array is empty
      - answer array is empty
    """

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            data = json.loads(line)

            answers = []
            wikipedia_ids = set()

            for item in data.get("output", []):

                # collect answers
                ans = item.get("answer")
                if ans:
                    answers.append(ans)

                # collect wikipedia ids
                for prov in item.get("provenance", []):
                    wid = prov.get("wikipedia_id")
                    if wid:
                        wikipedia_ids.add(wid)

            wikipedia_ids = list(wikipedia_ids)

            # filtering condition
            if len(answers) == 0 or len(wikipedia_ids) == 0:
                continue

            result = {
                "id": data.get("id"),
                "input": data.get("input"),
                "wikipedia_id": wikipedia_ids,
                "answer": answers
            }

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    input_file = "data/nq-train-kilt.jsonl"
    output_file = "data/train.jsonl"

    process_jsonl(input_file, output_file)

    dump_file = "data/enwiki-pages-articles.xml.bz2"
    db_file = "data/wikipedia_pages.sqlite"

    parse_dump(
        dump_path=dump_file,
        db_path=db_file,
        batch_size=20000,
        only_main_namespace=True,  # только обычные статьи (ns=0)
        log_every=100000,
    )