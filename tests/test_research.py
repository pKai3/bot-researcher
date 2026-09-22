import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import research as r


class FakeEmbeddings:
    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    def embed(self, texts, query=False):
        self.calls += 1
        if self.fail:
            raise r.AssistantError('model unavailable')
        return np.array([[1.0, 0.0] if 'titanium' in t else [0.0, 1.0] for t in texts], dtype=np.float32)


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.db = self.root / 'test.sqlite3'
        r.initialize(self.db)
        self.paper = self.root / 'titanium.PDF'
        self.paper.write_bytes(b'placeholder source for transactional indexing tests')
        self.extraction = ([{'page': 3, 'text': 'titanium grain refinement results'}], 4, 1)

    def tearDown(self):
        self.temp.cleanup()

    def index(self, client=None, key='model-v1'):
        with patch('research.extract', return_value=self.extraction):
            return r.index_document(self.paper, self.root, key, client or FakeEmbeddings(), self.db)

    def test_index_survives_reopening_and_skips_unchanged_file(self):
        client = FakeEmbeddings()
        self.index(client)
        self.assertEqual(self.index(client)['status'], 'unchanged')
        self.assertEqual(client.calls, 1)
        corpus = r.load_corpus(self.db, self.root, 'model-v1')
        found = r.retrieve('titanium', corpus, client)
        self.assertEqual(found[0]['page'], 3)
        self.assertEqual(found[0]['path'], str(self.paper))

    def test_failed_reindex_keeps_committed_data_and_rejects_stale_source(self):
        self.index()
        self.paper.write_bytes(b'changed PDF source')
        with self.assertRaises(r.AssistantError):
            self.index(FakeEmbeddings(fail=True))
        self.assertEqual(r.documents(self.db)[0]['chunks'], 1)
        self.assertEqual(r.load_corpus(self.db, self.root, 'model-v1').rows, [])
        self.index()
        self.assertEqual(r.documents(self.db)[0]['chunks'], 1)

    def test_model_change_requires_fresh_vectors(self):
        self.index()
        self.assertEqual(r.load_corpus(self.db, self.root, 'model-v2').rows, [])
        self.assertEqual(self.index(key='model-v2')['status'], 'indexed')
        self.assertEqual(len(r.load_corpus(self.db, self.root, 'model-v2').rows), 1)

    def test_retrieval_filters_before_ranking_and_deduplicates(self):
        other = self.root / 'other.pdf'
        other.write_bytes(b'other')
        corpus = r.Corpus([
            {'path': str(self.paper), 'page': 1, 'text': 'titanium'},
            {'path': str(self.paper), 'page': 2, 'text': 'titanium'},
            {'path': str(other), 'page': 7, 'text': 'recycling'},
        ], np.array([[1,0],[1,0],[0,1]], dtype=np.float32))
        self.assertEqual(len(r.retrieve('titanium', corpus, FakeEmbeddings())), 2)
        self.assertEqual(r.retrieve('titanium', corpus, FakeEmbeddings(), paths=[]), [])
        found = r.retrieve('titanium', corpus, FakeEmbeddings(), paths=[str(other)])
        self.assertEqual(found[0]['page'], 7)
        self.assertEqual(len(found), 1)
        other.unlink()
        self.assertEqual(r.retrieve('titanium', corpus, FakeEmbeddings(), paths=[str(other)]), [])

    def test_case_insensitive_pdf_discovery_and_root_boundary(self):
        (self.root / 'ignore.txt').write_text('ignore')
        self.assertEqual(r.discover(self.root), [self.paper])
        with self.assertRaises(r.AssistantError):
            r.index_document(self.paper, self.root / 'different', 'v1', FakeEmbeddings(), self.db)

    def test_chunking_terminates_and_keeps_tail(self):
        text = ' '.join(f'word{i}' for i in range(500))
        chunks = list(r.split_text(text, 120, 20))
        self.assertTrue(all(len(c) <= 120 for c in chunks))
        self.assertTrue(chunks[-1].endswith('word499'))
        self.assertEqual(list(r.split_text('')), [])
        with self.assertRaises(ValueError):
            list(r.split_text(text, 20, 20))

    def test_unreadable_pdf_symbols_are_not_presented_as_valid_units(self):
        chunk = next(r.split_text('Cooling at 120 /C14C/s and (cid:12).'))
        self.assertNotIn('/C14', chunk)
        self.assertEqual(chunk.count('[unreadable PDF symbol]'), 2)

    def test_symbol_warning_upgrade_reuses_compatible_vectors(self):
        old = 'nomic:digest:page-chunks-1200-180-nomic-prefix-fonttools-v2'
        new = 'nomic:digest:page-chunks-1200-180-nomic-prefix-fonttools-symbols-v3'
        self.assertTrue(r.index_compatible(old, new))
        self.assertFalse(r.index_compatible(old, new.replace('digest', 'other-model')))
        self.assertNotIn('/C14', r.clean_pdf_text('120 /C14C/s'))

    def test_citation_checks_and_zotero_page_link(self):
        sources = [{'path': '/library/ABCD2345/test.pdf', 'page': 9}]
        self.assertIsNone(r.citation_warning('Evidence [1].', sources))
        self.assertIsNotNone(r.citation_warning('120 [unreadable PDF symbol]C/s [1]', sources))
        self.assertIsNotNone(r.citation_warning('Evidence [1, 2].', sources))
        self.assertIsNotNone(r.citation_warning('Unsupported answer.', sources))
        self.assertEqual(r.zotero_uri(sources[0]), 'zotero://open-pdf/library/items/ABCD2345?page=9')


if __name__ == '__main__':
    unittest.main()
