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

    def test_only_explicit_paper_summary_requests_use_per_paper_format(self):
        for question in ['Summarize these papers.', 'Give me a summary of both studies.']:
            self.assertTrue(evidence.asks_for_paper_summaries(question))
        for question in ['Lanthanum behaviour in titanium?', 'Summarize lanthanum behaviour.', 'Compare the mechanisms reported by these papers.']:
            self.assertFalse(evidence.asks_for_paper_summaries(question))


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


class GroundingTests(unittest.TestCase):
    def setUp(self):
        self.sources = [
            {'path': '/a.pdf', 'page': 2, 'text': 'We measured a reduction in grain size in the current experiment.'},
            {'path': '/b.pdf', 'page': 31, 'text': 'Several researchers have reported grain refinement through lanthanum oxide additions.'},
        ]

    def claim(self, passage=1, quote=None):
        return {'statement': 'Grain refinement was observed.', 'passage': passage,
                'quote': quote or self.sources[0]['text'], 'attribution': 'own_result', 'relevance': 'direct'}

    def test_app_assigns_paper_and_page_from_the_matched_passage(self):
        answer, support = evidence.grounded_answer({'claims': [self.claim(2, self.sources[1]['text'])]}, self.sources)
        self.assertIn('[1, p. 31]', answer)
        self.assertNotIn('[2, p. 31]', answer)
        self.assertEqual(support[0]['path'], '/b.pdf')
        self.assertIn('Prior work discussed in the source', answer)
        self.assertEqual(support[0]['attribution'], 'prior_work')

    def test_wrong_source_and_invented_quotes_are_omitted(self):
        claims = [self.claim(), self.claim(2), self.claim(99),
                  self.claim(1, 'An invented quotation about a different and unsupported result.')]
        answer, support = evidence.grounded_answer({'claims': claims}, self.sources)
        self.assertEqual(len(support), 1)
        self.assertIn('Some draft statements were omitted', answer)
        self.assertNotIn('[2, p. 31]', answer)

    def test_failed_grounding_abstains_instead_of_falling_back_to_raw_model_text(self):
        answer, support = evidence.grounded_answer({'claims': [self.claim(99)]}, self.sources)
        self.assertEqual(support, [])
        self.assertIn('could not find sufficiently relevant evidence', answer)

    def test_wrong_passage_id_is_resolved_only_by_a_unique_matching_quote(self):
        answer, support = evidence.grounded_answer({'claims': [self.claim(2)]}, self.sources)
        self.assertEqual(len(support), 1)
        self.assertIn('[1, p. 2]', answer)
        self.assertNotIn('[2, p. 31]', answer)
        self.sources.extend([{**self.sources[0], 'path': '/duplicate.pdf'}])
        _, support = evidence.grounded_answer({'claims': [self.claim(2)]}, self.sources)
        self.assertEqual(support, [])

    def test_quote_normalization_preserves_values_but_accepts_pdf_ligatures_and_spacing(self):
        self.sources[0]['text'] = 'The re ﬁned grain size was 120 micrometres in this test [ 21].'
        _, support = evidence.grounded_answer({'claims': [self.claim(1, 'The refined grain size was 120 micrometres in this test ⟦21⟧.')]}, self.sources)
        self.assertEqual(len(support), 1)
        self.assertEqual(support[0]['quote'], self.sources[0]['text'])
        _, support = evidence.grounded_answer({'claims': [self.claim(1, 'The refined grain size was 140 micrometres in this test ⟦21⟧.')]}, self.sources)
        self.assertEqual(support, [])

    def test_summaries_skip_papers_without_relevant_claims_and_renumber_sources(self):
        answer, claims = evidence.grounded_answer({'claims': [self.claim(2, self.sources[1]['text'])]}, self.sources, summaries=True)
        self.assertNotIn('a.pdf', answer)
        self.assertNotIn('did not yield', answer)
        self.assertIn('### [1] b', answer)
        self.assertEqual(evidence.supporting_sources(self.sources, claims), [self.sources[1]])

    def test_background_and_no_mention_filler_are_not_returned_as_claims(self):
        background = {**self.claim(), 'relevance': 'background'}
        no_mention = {**self.claim(), 'statement': 'No information about lanthanum was found in this paper.'}
        answer, claims = evidence.grounded_answer({'claims': [background, no_mention]}, self.sources, summaries=True)
        self.assertEqual(claims, [])
        self.assertNotIn('###', answer)
        self.assertEqual(evidence.supporting_sources(self.sources, claims), [])

    def test_measured_negative_findings_are_not_treated_as_absence_filler(self):
        statement = 'Lanthanum did not significantly reduce grain size in this experiment.'
        self.sources[0]['text'] = statement
        claim = {**self.claim(1, statement), 'statement': statement}
        answer, claims = evidence.grounded_answer({'claims': [claim]}, self.sources)
        self.assertEqual(len(claims), 1)
        self.assertIn(statement, answer)


if __name__ == '__main__':
    unittest.main()
