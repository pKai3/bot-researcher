"""Local, page-level OCR. Originals are read-only; only recognized text is cached."""
from dataclasses import dataclass
from contextlib import closing
import math
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time

CACHE_DB = Path(__file__).resolve().parent / '.data' / 'ocr.sqlite3'
VERSION = 'tesseract-300dpi-v1'
# PDFium must never be called concurrently, including from separate UI sessions.
_RENDER_LOCK = threading.Lock()


class OCRError(RuntimeError):
    pass


@dataclass(frozen=True)
class Options:
    languages: tuple = ('eng',)
    force: bool = False

    @property
    def key(self):
        return VERSION + ':' + '+'.join(self.languages)

    @property
    def policy(self):
        return self.key + (':all' if self.force else ':sparse')


def executable():
    candidates = [os.environ.get('TESSERACT_CMD'), shutil.which('tesseract'),
                  '/opt/homebrew/bin/tesseract', '/usr/local/bin/tesseract',
                  str(Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'Tesseract-OCR/tesseract.exe')]
    return next((str(p) for p in candidates if p and Path(p).is_file()), None)


def available_languages():
    command = executable()
    if not command:
        raise OCRError('Tesseract is missing. On macOS, install it with brew install tesseract; then refresh the app.')
    try:
        result = subprocess.run([command, '--list-langs'], capture_output=True, text=True,
                                encoding='utf-8', errors='replace', timeout=10, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise OCRError(f'Tesseract could not start: {exc}') from exc
    return sorted(line.strip() for line in result.stdout.splitlines()
                  if line.strip() and not line.startswith('List of ') and line.strip() not in {'osd', 'snum'})


def validate(options):
    installed = available_languages()
    if not options.languages:
        raise OCRError('Choose at least one OCR language.')
    missing = set(options.languages) - set(installed)
    if missing:
        raise OCRError('OCR language data is not installed: ' + ', '.join(sorted(missing)))


def initialize(db=CACHE_DB):
    db = Path(db)
    db.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db, timeout=30)) as con, con:
        con.execute('PRAGMA journal_mode=WAL')
        con.execute('''CREATE TABLE IF NOT EXISTS pages (
            path TEXT NOT NULL, page INTEGER NOT NULL, size INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL, settings TEXT NOT NULL, text TEXT NOT NULL,
            updated_at INTEGER NOT NULL, PRIMARY KEY(path,page))''')


def cached_pages(path, db=CACHE_DB, options=None):
    path, db = Path(path).resolve(), Path(db)
    if not db.exists():
        return {}
    stat = path.stat()
    with closing(sqlite3.connect(db, timeout=30)) as con:
        rows = con.execute('SELECT page,text,settings FROM pages WHERE path=? AND size=? AND mtime_ns=? ORDER BY page',
                           (str(path), stat.st_size, stat.st_mtime_ns))
        return {page: text for page, text, settings in rows if options is None or settings == options.key}


def revisions(db=CACHE_DB):
    """Small signature map; filters never start OCR as a side effect."""
    if not Path(db).exists():
        return {}
    with closing(sqlite3.connect(db, timeout=30)) as con:
        return dict(con.execute('SELECT path,MAX(updated_at) FROM pages GROUP BY path'))


def render_page(path, number, output):
    import pypdfium2 as pdfium
    with _RENDER_LOCK:
        with pdfium.PdfDocument(str(path)) as pdf:
            page = pdf[number - 1]
            try:
                width, height = page.get_size()
                # Bound memory for oversized engineering drawings (about 80 MB RGB).
                scale = min(300 / 72, math.sqrt(25_000_000 / max(width * height, 1)))
                bitmap = page.render(scale=scale)
                try:
                    pil_image = bitmap.to_pil()
                    try:
                        pil_image.save(output, format='PNG')
                    finally:
                        pil_image.close()
                finally:
                    bitmap.close()
            finally:
                page.close()
    return max(70, round(scale * 72))


def recognize_page(path, number, options, db=CACHE_DB):
    """Save each successful page so interruption or an embedding failure can resume."""
    path = Path(path).resolve()
    before = path.stat()
    saved = cached_pages(path, db, options)
    if number in saved:
        return saved[number]
    validate(options)
    try:
        with tempfile.TemporaryDirectory(prefix='research-ocr-') as temporary:
            image = Path(temporary) / 'page.png'
            dpi = render_page(path, number, image)
            result = subprocess.run(
                [executable(), str(image), 'stdout', '-l', '+'.join(options.languages),
                 '--psm', '3', '--dpi', str(dpi)],
                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=120)
            if result.returncode:
                raise OCRError(result.stderr.strip()[-600:] or 'Tesseract returned an error.')
            text = result.stdout.strip()
    except subprocess.TimeoutExpired as exc:
        raise OCRError(f'OCR timed out on PDF page {number}. Completed pages are saved; try again.') from exc
    except Exception as exc:
        raise OCRError(f'OCR failed on PDF page {number}: {exc}') from exc
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise OCRError('The PDF changed during OCR. Please try again.')
    initialize(db)
    with closing(sqlite3.connect(db, timeout=30)) as con, con:
        con.execute('INSERT OR REPLACE INTO pages VALUES (?,?,?,?,?,?,?)',
                    (str(path), number, before.st_size, before.st_mtime_ns, options.key, text, time.time_ns()))
    return text
