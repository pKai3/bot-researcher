import hashlib
import io
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

import ocr
import paper_search
import research as r
from test_research import FakeEmbeddings


def mixed_pdf(path):
    """Native text, a scan, and an empty page, with no authoring dependency."""
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({NameObject('/Type'): NameObject('/Font'),
                             NameObject('/Subtype'): NameObject('/Type1'),
                             NameObject('/BaseFont'): NameObject('/Helvetica')})
    page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
    stream = DecodedStreamObject()
    stream.set_data(b'BT /F1 12 Tf 50 700 Td (Native aluminium research paper with enough readable text to avoid OCR. Results and experimental observations from the laboratory.) Tj ET')
    page[NameObject('/Contents')] = writer._add_object(stream)
    image = Image.new('RGB', (1400, 1800), 'white')
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=44)
    for y, line in enumerate(['Titanium grain refinement', 'Scanned research evidence', 'Cooling rate controls grain size.', 'Experimental results from the laboratory.']):
        draw.text((80, 180 + 80*y), line, fill='black', font=font)
    data = io.BytesIO()
    image.save(data, format='PDF', resolution=150)
    writer.add_page(PdfReader(io.BytesIO(data.getvalue())).pages[0])
    writer.add_blank_page(width=612, height=792)
    with path.open('wb') as output:
        writer.write(output)


class OCRTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.path = self.root / 'paper.pdf'
        mixed_pdf(self.path)
        self.db = self.root / 'ocr.sqlite3'
        self.index_db = self.root / 'library.sqlite3'
        self.search_db = self.root / 'search.sqlite3'
        r.initialize(self.index_db)
        self.options = ocr.Options()

    def tearDown(self):
        self.temp.cleanup()

    def save_page(self, text, page=2, options=None):
        ocr.initialize(self.db)
        stat = self.path.stat()
        with sqlite3.connect(self.db) as con:
            con.execute('INSERT OR REPLACE INTO pages VALUES (?,?,?,?,?,?,?)',
                        (str(self.path), page, stat.st_size, stat.st_mtime_ns,
                         (options or self.options).key, text, 123))

    def test_real_ocr_preserves_page_numbers_source_and_resumes(self):
        if not ocr.executable() or 'eng' not in ocr.available_languages():
            self.skipTest('Tesseract English is not installed')
        before = hashlib.sha256(self.path.read_bytes()).hexdigest()
        chunks, pages, blank = r.extract(self.path, self.options, ocr_db=self.db)
        self.assertEqual((pages, blank), (3, 1))
        recognized = [c for c in chunks if c['ocr']]
        self.assertEqual({c['page'] for c in recognized}, {2})
        self.assertIn('titanium', ' '.join(c['text'] for c in recognized).lower())
        self.assertEqual(before, hashlib.sha256(self.path.read_bytes()).hexdigest())
        with patch('ocr.render_page', side_effect=AssertionError('Cached pages must not render again')):
            self.assertEqual(r.extract(self.path, self.options, ocr_db=self.db), (chunks, pages, blank))
        self.assertEqual(set(ocr.cached_pages(self.path, self.db)), {2, 3})

    def test_auto_ocr_skips_native_pages_and_force_includes_them(self):
        def recognized(path, number, options, db):
            return 'Titanium research scanned experimental evidence for grain refinement.' if number < 3 else ''
        with patch('ocr.recognize_page', side_effect=recognized) as recognize:
            r.extract(self.path, self.options, ocr_db=self.db)
            self.assertEqual([c.args[1] for c in recognize.call_args_list], [2, 3])
            recognize.reset_mock()
            r.extract(self.path, ocr.Options(force=True), ocr_db=self.db)
            self.assertEqual([c.args[1] for c in recognize.call_args_list], [1, 2, 3])

    def test_cache_invalidates_on_source_or_language_change(self):
        self.save_page('Titanium')
        self.assertEqual(ocr.cached_pages(self.path, self.db, self.options), {2: 'Titanium'})
        self.assertEqual(ocr.cached_pages(self.path, self.db, ocr.Options(('deu',))), {})
        self.path.write_bytes(self.path.read_bytes() + b'\n')
        self.assertEqual(ocr.cached_pages(self.path, self.db), {})

    def test_missing_engine_and_language_have_actionable_errors(self):
        with patch('ocr.executable', return_value=None):
            with self.assertRaisesRegex(ocr.OCRError, 'Tesseract is missing'):
                ocr.available_languages()
        with patch('ocr.available_languages', return_value=['eng']):
            with self.assertRaisesRegex(ocr.OCRError, 'not installed: deu'):
                ocr.validate(ocr.Options(('deu',)))

    def test_timeout_does_not_poison_cache(self):
        with patch('ocr.validate'), patch('ocr.render_page', return_value=300), patch('ocr.executable', return_value='tesseract'), patch('ocr.subprocess.run', side_effect=subprocess.TimeoutExpired('tesseract', 120)):
            with self.assertRaisesRegex(ocr.OCRError, 'page 2'):
                ocr.recognize_page(self.path, 2, self.options, self.db)
        self.assertEqual(ocr.cached_pages(self.path, self.db), {})

    def test_source_change_during_ocr_is_not_cached(self):
        def changed(*args, **kwargs):
            self.path.write_bytes(self.path.read_bytes() + b'\n')
            return subprocess.CompletedProcess([], 0, 'Recognized text', '')
        with patch('ocr.validate'), patch('ocr.render_page', return_value=300), patch('ocr.executable', return_value='tesseract'), patch('ocr.subprocess.run', side_effect=changed):
            with self.assertRaisesRegex(ocr.OCRError, 'changed during OCR'):
                ocr.recognize_page(self.path, 2, self.options, self.db)
        self.assertEqual(ocr.cached_pages(self.path, self.db), {})

    def search(self, word):
        return paper_search.filter_papers([self.path], word, self.index_db, self.search_db, ocr_db=self.db)

    def test_filter_sees_ocr_added_after_zotero_cache_and_never_starts_ocr(self):
        (self.root / '.zotero-ft-cache').write_text('Native aluminium research cover')
        self.assertEqual(self.search('titanium')['paths'], [])
        self.save_page('Titanium is found on the scanned second page.')
        with patch('ocr.recognize_page', side_effect=AssertionError('Filtering must not start OCR')):
            self.assertEqual(self.search('titanium')['paths'], [str(self.path)])
            self.assertEqual(self.search('aluminium')['paths'], [str(self.path)])
        self.path.write_bytes(self.path.read_bytes() + b'\n')
        self.assertEqual(self.search('titanium')['paths'], [])

    def test_index_upgrades_old_blank_pages_preserves_ocr_sources_and_skips_repeats(self):
        client = FakeEmbeddings()
        r.index_document(self.path, self.root, 'v1', client, self.index_db, ocr_db=self.db)
        self.assertEqual(r.documents(self.index_db)[0]['blank_pages'], 2)
        # Simulate an index written before OCR support existed.
        with r.connect(self.index_db) as con, con:
            con.execute('DELETE FROM document_extraction')
        self.save_page('Titanium research scanned experimental evidence for grain refinement.')
        self.save_page('', page=3)
        result = r.index_document(self.path, self.root, 'v1', client, self.index_db, ocr_options=self.options, ocr_db=self.db)
        self.assertEqual(result['ocr_pages'], 1)
        self.assertEqual(result['blank_pages'], 1)
        corpus = r.load_corpus(self.index_db, self.root, 'v1')
        found = r.retrieve('titanium', corpus, client)
        self.assertTrue(next(s for s in found if s['page'] == 2)['ocr'])
        calls = client.calls
        self.assertEqual(r.index_document(self.path, self.root, 'v1', client, self.index_db, ocr_options=self.options, ocr_db=self.db)['status'], 'unchanged')
        self.assertEqual(client.calls, calls)

    def test_filter_recovers_a_previously_unreadable_scan(self):
        writer = PdfWriter()
        writer.add_page(PdfReader(self.path).pages[1])
        with self.path.open('wb') as output:
            writer.write(output)
        before = self.search('titanium')
        self.assertEqual(before['searched'], 0)
        self.assertEqual(len(before['unavailable']), 1)
        self.save_page('Titanium research evidence from this previously unreadable scan.', page=1)
        after = self.search('titanium')
        self.assertEqual(after['paths'], [str(self.path)])
        self.assertEqual(after['searched'], 1)
        self.assertEqual(after['unavailable'], [])

    def test_ocr_failure_preserves_existing_index(self):
        r.index_document(self.path, self.root, 'v1', FakeEmbeddings(), self.index_db, ocr_db=self.db)
        before = r.revision(self.index_db)
        with patch('ocr.recognize_page', side_effect=ocr.OCRError('OCR failed')):
            with self.assertRaisesRegex(r.AssistantError, 'OCR failed'):
                r.index_document(self.path, self.root, 'v1', FakeEmbeddings(), self.index_db, ocr_options=self.options, ocr_db=self.db)
        self.assertEqual(r.revision(self.index_db), before)


if __name__ == '__main__':
    unittest.main()
