from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import streamlit as st
from streamlit.testing.v1 import AppTest

import research as r
import zotero
from test_zotero import collection, item


class LocalModels:
    def models(self):
        return {r.CHAT_MODEL: {}}

    def embedding_key(self):
        return 'test-model'

    def embed(self, texts, query=False):
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)


class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        root = Path(self.stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.files = []
        for key in ['ATT00001', 'ATT00002', 'ATT00003']:
            path = root / key / 'paper.pdf'
            path.parent.mkdir()
            path.write_bytes(b'PDF fixture')
            self.files.append(path)
        self.catalog = {
            'collections': [collection('ROOT0001', 'Research'), collection('CHILD001', 'Titanium', 'ROOT0001'), collection('EMPTY001', 'Empty')],
            'items': [item('PARENT01', collections=['ROOT0001']), item('PARENT02', collections=['CHILD001']), item('PARENT03'),
                      *[item(f'ATT0000{i}', 'attachment', parentItem=f'PARENT0{i}', filename='paper.pdf') for i in range(1, 4)]]}
        docs = [{'path': str(p), 'embedding_key': 'test-model', 'chunks': 1, 'pages': 1, 'blank_pages': 0, 'ocr_pages': []} for p in self.files]
        corpus = r.Corpus([{'path': str(p), 'page': 1, 'text': f'Unique evidence {i}'} for i, p in enumerate(self.files)],
                          np.array([[1, 0]] * 3, dtype=np.float32))
        for name, replacement in [
            ('research.DEFAULT_LIBRARY', root), ('research.Ollama', LocalModels),
        ]:
            self.stack.enter_context(patch(name, replacement))
        for name, result in [
            ('runtime.refresh_answer_modules', None),
            ('research.initialize', None), ('research.discover', self.files),
            ('research.documents', docs), ('research.current_document', True),
            ('research.load_corpus', corpus), ('research.revision', (3, 1)),
            ('ocr.available_languages', ['eng']),
            ('zotero.LocalZotero.libraries', [{'id': 'users/0', 'name': 'My Library', 'group_id': None}]),
            ('zotero.LocalZotero.catalog', self.catalog),
        ]:
            self.stack.enter_context(patch(name, return_value=result))
        st.cache_data.clear()
        st.cache_resource.clear()
        self.app = AppTest.from_file(str(Path(r.__file__).parent / 'app.py'), default_timeout=15).run()

    def widget(self, kind, label):
        return next(x for x in getattr(self.app, kind) if x.label == label)

    def select_collection(self, key):
        self.widget('checkbox', 'Limit to a Zotero collection').check().run()
        self.widget('selectbox', 'Zotero collection').set_value(key).run()
        self.assertFalse(self.app.exception)

    def test_ocr_indexing_and_questions_use_only_selected_collection(self):
        self.select_collection('ROOT0001')
        choices = self.widget('multiselect', 'Choose PDFs')
        self.assertEqual(len(choices.options), 2)
        self.widget('radio', 'Papers to index').set_value('All matching papers').run()
        with patch('research.prepare_ocr', return_value={'status': 'text ready'}) as run_ocr:
            self.widget('button', 'OCR selected papers').click().run()
            self.assertEqual([c.args[0] for c in run_ocr.call_args_list], [str(p) for p in self.files[:2]])
        with patch('research.index_document', return_value={'status': 'unchanged'}) as index:
            self.widget('button', 'Index selected papers').click().run()
            self.assertEqual([c.args[0] for c in index.call_args_list], [str(p) for p in self.files[:2]])
        self.widget('checkbox', 'Find passages without generating an answer').check().run()
        self.app.chat_input[0].set_value('What evidence is available?').run()
        self.assertFalse(self.app.exception)
        sources = self.app.session_state['messages'][-1]['sources']
        self.assertEqual({s['path'] for s in sources}, {str(p) for p in self.files[:2]})
        self.widget('checkbox', 'Include subcollections').uncheck().run()
        self.assertEqual(len(self.widget('multiselect', 'Focus on specific indexed papers (optional)').options), 1)
        self.app.chat_input[0].set_value('What about the parent collection alone?').run()
        self.assertEqual({s['path'] for s in self.app.session_state['messages'][-1]['sources']}, {str(self.files[0])})

    def test_empty_and_unavailable_collections_cannot_search_the_full_index(self):
        self.select_collection('EMPTY001')
        self.assertTrue(self.app.chat_input[0].disabled)
        self.assertTrue(self.widget('button', 'OCR selected papers').disabled)
        self.assertEqual(self.widget('multiselect', 'Choose PDFs').options, [])
        self.catalog['collections'] = [c for c in self.catalog['collections'] if c['key'] != 'EMPTY001']
        st.cache_data.clear()
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertTrue(any('no longer available' in e.value for e in self.app.error))
        self.assertEqual(len(self.app.chat_input), 0)
        self.widget('selectbox', 'Zotero collection').set_value('ROOT0001').run()
        with patch('zotero.LocalZotero.catalog', side_effect=zotero.ZoteroError('Open Zotero')):
            st.cache_data.clear()
            self.app.run()
            self.assertTrue(any('Open Zotero' in e.value for e in self.app.error))
            self.assertEqual(len(self.app.chat_input), 0)

    def test_new_sources_group_six_excerpts_into_two_papers_and_preserve_old_answers(self):
        sources = [
            {'path': str(self.files[paper]), 'page': page, 'text': f'Excerpt {number}'}
            for number, (paper, page) in enumerate([(0, 2), (1, 5), (0, 2), (0, 3), (1, 5), (1, 3)])
        ]
        self.app.session_state['messages'] = [
            {'question': 'New summary', 'answer': 'First paper [1, p. 2]. Second paper [2, p. 5].',
             'sources': sources, 'citation_style': r.CITATION_STYLE,
             'grounding': [{'citation': '[1, p. 2]', 'statement': 'The measured finding.',
                            'quote': 'Exact supporting text from the source.'}]},
            {'question': 'Earlier summary', 'answer': 'Earlier passage [6].', 'sources': sources},
        ]
        self.app.run()
        self.assertFalse(self.app.exception)
        labels = [e.label for e in self.app.expander if e.label.startswith('[')]
        self.assertEqual(labels[:2], ['[1] paper · 3 passages', '[2] paper · 3 passages'])
        self.assertEqual(len(labels), 8)  # Two new paper groups plus six legacy passages.
        self.assertEqual(labels[-1], '[6] paper · PDF page 3')
        self.assertTrue(any('6 passages from 2 papers' in c.value for c in self.app.caption))
        page_headings = [m.value for m in self.app.markdown if m.value.startswith('**[') and '· PDF page' in m.value]
        self.assertEqual(page_headings, [
            '**[1, p. 2] · PDF page 2**', '**[1, p. 3] · PDF page 3**',
            '**[2, p. 3] · PDF page 3**', '**[2, p. 5] · PDF page 5**',
        ])
        self.assertFalse(any('invalid source' in w.value for w in self.app.warning))
        self.assertTrue(any(e.label == 'Check supporting quotes' for e in self.app.expander))
        self.assertTrue(any(t.value == 'Exact supporting text from the source.' for t in self.app.get('text')))

    def test_answer_displays_only_the_sources_used_by_supported_claims(self):
        def answer(client, question, sources):
            client.answer_sources = sources[:1]
            client.answer_evidence = []
            return 'A supported finding [1, p. 1].'
        with patch.object(LocalModels, 'answer', answer, create=True):
            self.app.chat_input[0].set_value('What evidence answers this question?').run()
        self.assertFalse(self.app.exception)
        message = self.app.session_state['messages'][-1]
        self.assertEqual(len(message['sources']), 1)
        self.assertEqual(len([e for e in self.app.expander if e.label.startswith('[')]), 1)


if __name__ == '__main__':
    unittest.main()
