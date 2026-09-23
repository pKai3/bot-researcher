import unittest
from unittest.mock import patch

import research as r


class PaperCitationTests(unittest.TestCase):
    def setUp(self):
        self.first = '/library/FIRST001/paper.pdf'
        self.second = '/library/SECOND02/paper.pdf'
        self.sources = [
            {'path': self.first, 'page': 2, 'text': 'First paper, excerpt one.'},
            {'path': self.second, 'page': 5, 'text': 'Second paper, excerpt one.'},
            {'path': self.first, 'page': 2, 'text': 'First paper, excerpt two.'},
            {'path': self.first, 'page': 3, 'text': 'First paper, excerpt three.'},
            {'path': self.second, 'page': 5, 'text': 'Second paper, excerpt two.'},
            {'path': self.second, 'page': 3, 'text': 'Second paper, excerpt three.', 'ocr': True},
        ]
        for source in self.sources:
            source['text'] += ' This is a complete supporting sentence for the supplied evidence.'

    def model_response(self):
        return {'message': {'content': 'The first paper reports a finding [1, p. 2].'}}

    def test_six_passages_keep_two_paper_identities_even_with_equal_filenames(self):
        papers = r.group_sources(self.sources)
        self.assertEqual([p['number'] for p in papers], [1, 2])
        self.assertEqual([p['path'] for p in papers], [self.first, self.second])
        self.assertEqual([len(p['passages']) for p in papers], [3, 3])
        self.assertEqual([s['page'] for s in papers[0]['passages']], [2, 2, 3])

    def test_model_receives_two_grouped_papers_and_page_citations(self):
        client = r.Ollama()
        with patch.object(client, 'request', return_value=self.model_response()) as request:
            answer = client.answer('Summarize these papers.', self.sources)
        messages = request.call_args.args[1]['messages']
        self.assertIn('exactly 2 distinct papers represented by 6 excerpts', messages[0]['content'])
        evidence = messages[1]['content']
        self.assertEqual(evidence.count('PAPER ['), 2)
        first, second = evidence.split('PAPER [2]:')
        self.assertIn('First paper, excerpt three.', first)
        self.assertNotIn('Second paper, excerpt one.', first)
        self.assertIn('Second paper, excerpt three.', second)
        self.assertNotIn('First paper, excerpt', second)
        self.assertIn('cite [1, p. 2]', first)
        self.assertIn('cite [2, p. 5]', second)
        self.assertIn('OCR text', second)
        self.assertIsNone(r.citation_warning(answer, self.sources))
        self.assertEqual(answer, self.model_response()['message']['content'])
        self.assertNotIn('format', request.call_args.args[1])

    def test_citations_must_reference_an_actual_paper_and_its_supplied_page(self):
        self.assertIsNone(r.citation_warning('Findings [1, p. 2] [1, p. 3] [2, p. 5].', self.sources))
        self.assertIsNotNone(r.citation_warning('An imaginary third paper [3, p. 2].', self.sources))
        self.assertIsNotNone(r.citation_warning('A page from the wrong paper [2, p. 2].', self.sources))
        self.assertIsNotNone(r.citation_warning('An unavailable page [1, p. 99].', self.sources))
        self.assertIsNone(r.citation_warning('Two supplied pages [1, pp. 2–3].', self.sources))
        self.assertIsNotNone(r.citation_warning('A range containing an unavailable page [2, p. 3-4].', self.sources))
        self.assertIsNotNone(r.citation_warning('A reversed range [1, pp. 3-2].', self.sources))
        self.assertIsNotNone(r.citation_warning('Uncited claims.', self.sources))
        self.assertIn('omits PDF page', r.citation_warning('Paper [1]: Findings. Paper [2]: Findings.', self.sources))
        self.assertIn('ambiguous page', r.citation_warning('Confused internal references [1, p. 2, 31].', self.sources))

    def test_prose_is_preserved_without_claim_or_quote_checks(self):
        client = r.Ollama()
        # The app must not turn paraphrases, a limitation, or even a bad citation
        # into a rejected/rewritten answer. Citation issues remain advisory.
        prose = ('The experiment suggests a mechanism [1, p. 2].\n\n'
                 'There is not enough information here to establish its limits.\n\n'
                 'A conclusion with a wrong page [2, p. 99].')
        with patch.object(client, 'request', return_value={'message': {'content': prose}}) as request:
            answer = client.answer('Explain the findings.', self.sources)
        self.assertEqual(answer, prose)
        self.assertNotIn('format', request.call_args.args[1])
        self.assertIsNotNone(r.citation_warning(answer, self.sources))

    def test_empty_model_response_is_an_error_not_an_evidence_judgment(self):
        client = r.Ollama()
        for content in ['', '   ', None]:
            with self.subTest(content=content), patch.object(client, 'request', return_value={'message': {'content': content}}):
                with self.assertRaisesRegex(r.AssistantError, 'empty answer'):
                    client.answer('Explain the findings.', self.sources)

    def test_empty_retrieval_does_not_call_the_answer_model(self):
        client = r.Ollama()
        with patch.object(client, 'request') as request:
            answer = client.answer('Explain the findings.', [])
        request.assert_not_called()
        self.assertIn('could not find', answer)

    def test_legacy_answers_retain_their_six_passage_citations(self):
        self.assertIsNone(r.citation_warning('Earlier answer [6].', self.sources, 'passages'))
        self.assertIsNotNone(r.citation_warning('New answer [6].', self.sources))

    def test_selected_answer_model_does_not_change_the_embedding_model(self):
        client = r.Ollama()
        with patch.object(client, 'request', side_effect=[
            self.model_response(), {'embeddings': [[1.0, 0.0]]},
        ]) as request:
            client.answer('What was found?', self.sources, model='another-model:14b')
            client.embed(['What was found?'], query=True)
        self.assertEqual(request.call_args_list[0].args[1]['model'], 'another-model:14b')
        self.assertEqual(request.call_args_list[1].args[1]['model'], r.EMBED_MODEL)


if __name__ == '__main__':
    unittest.main()
