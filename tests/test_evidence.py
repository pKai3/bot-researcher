from pathlib import Path
import unittest

import numpy as np

import evidence
import research as r
import test_research as fixtures


REFS = ('[1] A. Smith, B. Jones, Lanthanum additions in titanium, Acta Mater. 20 (2020) 100. '
        'https://doi.org/10.1000/example. [2] J. Yang, Microstructure evolution, J. Alloys Compd. 827 (2021) 154170.')


class EvidenceTests(unittest.TestCase):
    def rows(self, *texts):
        return [{'path': '/library/paper.pdf', 'page': i + 1, 'text': text} for i, text in enumerate(texts)]

    def test_reference_section_continues_across_chunks_and_pages(self):
        body = 'We measured the titanium specimen and observed grain refinement after adding the particles.'
        rows = self.rows(body, '30 of 41 References ' + REFS,
                         'The tail of an entry whose author and number occurred on the previous page.',
                         'Appendix A. Additional experiments confirmed the measurements with a second specimen.')
        kept = evidence.filter_rows(rows)
        self.assertEqual([i for i, _ in kept], [0, 3])
        self.assertEqual(rows[1]['text'], '30 of 41 References ' + REFS)

    def test_mixed_conclusion_and_references_keeps_only_the_conclusion(self):
        body = 'Conclusions. Our measurements demonstrate a reduction in grain size. The measured change persisted across all specimens.'
        kept = evidence.filter_rows(self.rows(body + ' References ' + REFS))
        self.assertEqual(kept[0][1]['text'], body)
        self.assertTrue(kept[0][1]['references_removed'])

    def test_bibliography_without_heading_and_truncated_first_entry_are_excluded(self):
        tail = ('commercially pure titanium through lanthanum oxide addition, Mater. Charact. 176 (2021) 111074. '
                'https://doi.org/10.1016/j.matchar.2021.111074. ')
        self.assertEqual(evidence.filter_rows(self.rows(tail + REFS)), [])
        self.assertEqual(evidence.filter_rows(self.rows(REFS)), [])
        surname_first = ('1. Moiseyev, V. N. Titanium Alloys (Press, 2005). '
                         '2. Donachie, M. J. Titanium Guide (Press, 2000).')
        self.assertEqual(evidence.filter_rows(self.rows(surname_first)), [])

    def test_inline_citations_and_word_references_do_not_remove_body_evidence(self):
        texts = [
            'Earlier work [1, 2] found grain refinement. Our study tested a different alloy in 2020 and 2021.',
            'For interpretation of the references to colour in this figure legend, see the online article.',
            'The references [1] and [2] discuss titanium; our measurement was different.',
        ]
        rows = self.rows(*texts)
        self.assertEqual([row for _, row in evidence.filter_rows(rows)], rows)

    def test_reference_state_does_not_leak_between_documents(self):
        rows = self.rows('References ' + REFS)
        rows += [{'path': '/library/other.pdf', 'page': 1, 'text': 'A different paper remains searchable.'}]
        self.assertEqual([i for i, _ in evidence.filter_rows(rows)], [1])

    def test_prompt_labels_internal_references_without_altering_measurements(self):
        text = 'Others reported this [29e31]. We measured 120 MPa [7, 8].'
        labeled = evidence.label_internal_citations(text)
        self.assertIn('⟦29e31⟧', labeled)
        self.assertIn('120 MPa', labeled)
        self.assertNotIn('[7, 8]', labeled)
        self.assertIn('⟦100⟧', evidence.label_internal_citations('A [100] crystal direction.'))



class EvidenceIndexTests(unittest.TestCase):
    setUp = fixtures.ResearchTests.setUp
    tearDown = fixtures.ResearchTests.tearDown
    index = fixtures.ResearchTests.index

    def test_existing_index_excludes_references_without_reembedding_and_keeps_vectors_aligned(self):
        self.extraction = ([
            {'page': 1, 'text': 'titanium experimental findings'},
            {'page': 2, 'text': 'References ' + REFS},
            {'page': 3, 'text': 'Continuation of a bibliography entry.'},
            {'page': 4, 'text': 'Appendix A. recycling results'},
        ], 4, 0)
        client = fixtures.FakeEmbeddings()
        self.index(client)
        self.assertEqual(r.documents(self.db)[0]['chunks'], 4)
        corpus = r.load_corpus(self.db, self.root, 'model-v1')
        self.assertEqual([row['page'] for row in corpus.rows], [1, 4])
        np.testing.assert_array_equal(corpus.matrix, [[1, 0], [0, 1]])
        self.assertEqual(client.calls, 1)
        found = r.retrieve('titanium', corpus, client)
        self.assertEqual({row['page'] for row in found}, {1})

    def test_reference_only_index_returns_empty_corpus(self):
        self.extraction = ([{'page': 1, 'text': 'References ' + REFS}], 1, 0)
        self.index()
        corpus = r.load_corpus(self.db, self.root, 'model-v1')
        self.assertEqual(corpus.rows, [])
        self.assertEqual(corpus.matrix.shape, (0, 0))

    def test_retrieval_expands_surrounding_context_and_merges_overlapping_hits(self):
        text = ('Earlier work by Smith reported this titanium mechanism. ' +
                'The earlier experiment used a different alloy and its conclusions have that limitation. ' * 12 +
                'In the present study we instead measured the response of a second titanium alloy. ' * 10)
        chunks = [{'page': 2, 'text': part} for part in r.split_text(text)]
        self.extraction = (chunks, 2, 0)
        self.index()
        corpus = r.load_corpus(self.db, self.root, 'model-v1')
        found = r.retrieve('titanium', corpus, fixtures.FakeEmbeddings(), k=6)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]['text'], r.clean_pdf_text(text))
        self.assertEqual(found[0]['page'], 2)
        self.assertTrue(found[0]['context_expanded'])
        self.assertGreater(len(found[0]['text']), 1200)
        self.assertLessEqual(len(found[0]['text']), evidence.CONTEXT_CHAR_BUDGET // 6)

    def test_expansion_never_crosses_a_pdf_page_boundary(self):
        self.extraction = ([{'page': 1, 'text': 'titanium evidence on the first page'},
                            {'page': 2, 'text': 'titanium evidence on the next page'}], 2, 0)
        self.index()
        found = r.retrieve('titanium', r.load_corpus(self.db, self.root, 'model-v1'), fixtures.FakeEmbeddings())
        self.assertEqual(len(found), 2)
        self.assertFalse(any('first page' in row['text'] and 'next page' in row['text'] for row in found))


if __name__ == '__main__':
    unittest.main()
