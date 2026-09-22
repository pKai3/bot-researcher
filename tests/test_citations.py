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

    def test_six_passages_keep_two_paper_identities_even_with_equal_filenames(self):
        papers = r.group_sources(self.sources)
        self.assertEqual([p['number'] for p in papers], [1, 2])
        self.assertEqual([p['path'] for p in papers], [self.first, self.second])
        self.assertEqual([len(p['passages']) for p in papers], [3, 3])
        self.assertEqual([s['page'] for s in papers[0]['passages']], [2, 2, 3])

    def test_model_receives_two_grouped_papers_and_page_citations(self):
        client = r.Ollama()
        with patch.object(client, 'request', return_value={'message': {'content': 'Grounded summary [1, p. 2].'}}) as request:
            answer = client.answer('Summarize these papers.', self.sources)
        messages = request.call_args.args[1]['messages']
        self.assertIn('exactly 2 distinct papers represented by 6 excerpts', messages[0]['content'])
        self.assertIn('ONE summary per PAPER', messages[0]['content'])
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

    def test_legacy_answers_retain_their_six_passage_citations(self):
        self.assertIsNone(r.citation_warning('Earlier answer [6].', self.sources, 'passages'))
        self.assertIsNotNone(r.citation_warning('New answer [6].', self.sources))


if __name__ == '__main__':
    unittest.main()
