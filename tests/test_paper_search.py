from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

import paper_search as s
import research as r


class PaperSearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.db = self.root / 'library.sqlite3'
        self.cache = self.root / 'search.sqlite3'
        r.initialize(self.db)
        self.paper = self.root / 'ABCD1234' / 'Truncated title.pdf'
        self.paper.parent.mkdir()
        self.paper.write_bytes(b'PDF fixture')

    def tearDown(self):
        self.temp.cleanup()

    def search(self, query, files=None):
        return s.filter_papers(files or [self.paper], query, self.db, self.cache)

    def text_cache(self, text):
        p = self.paper.parent / '.zotero-ft-cache'
        p.write_text(text)
        stamp = max(p.stat().st_mtime_ns, self.paper.stat().st_mtime_ns + 1000)
        os.utime(p, ns=(stamp, stamp))
        return p

    def test_unindexed_paper_matches_content_and_normalizes_pdf_words(self):
        self.text_cache('Results on Tita-\nnium alloys and grain reﬁnement.')
        with patch('paper_search.r.extract') as extract:
            result = self.search('TITANIUM refinement')
            self.assertEqual(result['paths'], [str(self.paper)])
            self.assertEqual(result['searched'], 1)
            extract.assert_not_called()
        self.assertEqual(r.documents(self.db), [])

    def test_query_matches_across_filename_and_body_and_is_literal(self):
        self.text_cache('Titanium alloys')
        self.assertEqual(self.search('truncated titanium')['paths'], [str(self.paper)])
        self.assertEqual(self.search('titanium missing')['paths'], [])
        self.assertEqual(self.search('% OR _')['paths'], [])

    def test_uncached_pdf_is_extracted_once_and_refreshed_on_change(self):
        with patch('paper_search.r.extract', return_value=([{'text': 'titanium', 'page': 1}], 1, 0)) as extract:
            self.assertEqual(self.search('titanium')['paths'], [str(self.paper)])
            self.search('titanium')
            self.assertEqual(extract.call_count, 1)
            self.paper.write_bytes(b'changed PDF fixture')
            self.search('titanium')
            self.assertEqual(extract.call_count, 2)

    def test_updated_zotero_cache_is_seen(self):
        self.text_cache('aluminium')
        self.assertEqual(self.search('titanium')['paths'], [])
        self.text_cache('titanium and aluminium')
        self.assertEqual(self.search('titanium')['paths'], [str(self.paper)])

    def test_old_or_ambiguous_zotero_cache_is_not_used(self):
        cache = self.text_cache('incorrect titanium')
        os.utime(cache, ns=(1, 1))
        with patch('paper_search.r.extract', return_value=([{'text': 'aluminium'}], 1, 0)):
            self.assertEqual(self.search('titanium')['paths'], [])
        self.text_cache('incorrect titanium')
        (self.paper.parent / 'different.pdf').write_bytes(b'different source')
        with patch('paper_search.r.extract', return_value=([{'text': 'aluminium'}], 1, 0)):
            self.assertEqual(self.search('titanium')['paths'], [])

    def test_no_text_is_reported_and_filename_still_matches(self):
        with patch('paper_search.r.extract', side_effect=r.AssistantError('OCR needed')):
            result = self.search('truncated')
            self.assertEqual(result['paths'], [str(self.paper)])
            self.assertEqual(result['searched'], 0)
            self.assertEqual(result['unavailable'][0]['reason'], 'OCR needed')

    def test_empty_filter_does_not_extract_pdfs(self):
        with patch('paper_search.r.extract') as extract:
            self.assertEqual(self.search('  ')['paths'], [str(self.paper)])
            extract.assert_not_called()
