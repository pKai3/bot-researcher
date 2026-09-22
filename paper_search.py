"""Literal paper filtering using filenames and locally available PDF text."""
from pathlib import Path
import json
import re
import sqlite3
import unicodedata

import research as r

CACHE_DB = r.DATA / 'paper-search.sqlite3'


def normalize(text):
    text = unicodedata.normalize('NFKC', text).replace('\u00ad', '')
    text = re.sub(r'(?<=\w)-\s+(?=\w)', '', text)
    return ' '.join(text.casefold().split())


def filter_papers(files, query, db=r.DB, cache_db=CACHE_DB, progress=None):
    """Search all query words literally, regardless of the AI indexing state.

    A separate cache avoids holding the embedding database's writer lock while
    reading PDFs. File signatures invalidate text when source files change.
    """
    paths = [Path(p).resolve() for p in files]
    words = normalize(query).split()
    if not words:
        return {'paths': [str(p) for p in paths], 'searched': 0, 'unavailable': []}
    cached_docs = {d['path']: d for d in r.documents(db) if r.current_document(d)} if Path(db).exists() else {}
    cache_db = Path(cache_db)
    cache_db.parent.mkdir(parents=True, exist_ok=True)
    matches, unavailable, searched = [], [], 0
    with sqlite3.connect(cache_db, timeout=30) as con:
        con.execute('''CREATE TABLE IF NOT EXISTS paper_text (
            path TEXT PRIMARY KEY, signature TEXT NOT NULL,
            text TEXT NOT NULL, error TEXT NOT NULL)''')
        for i, path in enumerate(paths):
            try:
                stat = path.stat()
                cache = path.parent / '.zotero-ft-cache'
                siblings = [p for p in path.parent.iterdir() if p.is_file() and p.suffix.lower() == '.pdf'] if cache.is_file() else []
                cache_stat = cache.stat() if len(siblings) == 1 else None
                # Zotero may replace a PDF before regenerating its text cache.
                fresh_cache = cache_stat is not None and cache_stat.st_mtime_ns >= stat.st_mtime_ns
                doc = cached_docs.get(str(path))
                signature = json.dumps(['v1', stat.st_size, stat.st_mtime_ns,
                                        [cache_stat.st_size, cache_stat.st_mtime_ns] if fresh_cache else None,
                                        doc['indexed_at'] if doc else None])
                saved = con.execute('SELECT text,error FROM paper_text WHERE path=? AND signature=?',
                                    (str(path), signature)).fetchone()
                if saved:
                    text, error = saved
                else:
                    text, error = '', ''
                    if fresh_cache:
                        try:
                            text = cache.read_text(encoding='utf-8', errors='replace')
                        except OSError:
                            pass
                    if not text.strip() and doc:
                        with r.connect(db) as source:
                            text = '\n'.join(row[0] for row in source.execute(
                                'SELECT text FROM chunks WHERE path=? ORDER BY id', (str(path),)))
                    if not text.strip():
                        try:
                            chunks, _, _ = r.extract(path)
                            text = '\n'.join(c['text'] for c in chunks)
                        except Exception as exc:
                            error = str(exc)
                    text = normalize(text)
                    after = path.stat()
                    if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                        text, error = '', 'PDF changed while its text was being read; search again.'
                    else:
                        con.execute('INSERT OR REPLACE INTO paper_text VALUES (?,?,?,?)',
                                    (str(path), signature, text, error))
                        con.commit()
                if text:
                    searched += 1
                else:
                    unavailable.append({'file': path.name, 'reason': error or 'No extractable text'})
                haystack = normalize(path.name) + '\n' + text
                if all(word in haystack for word in words):
                    matches.append(str(path))
            except OSError as exc:
                unavailable.append({'file': path.name, 'reason': str(exc)})
                if all(word in normalize(path.name) for word in words):
                    matches.append(str(path))
            if progress:
                progress(i + 1, len(paths))
    return {'paths': matches, 'searched': searched, 'unavailable': unavailable}
