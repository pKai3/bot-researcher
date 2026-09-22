import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit

import zotero as z


def collection(key, name, parent=False):
    return {'key': key, 'data': {'name': name, 'parentCollection': parent}}


def item(key, kind='journalArticle', collections=(), **extra):
    return {'key': key, 'data': {'itemType': kind, 'collections': list(collections), **extra}}


class ZoteroTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.files = []
        for key in ['ATT00001', 'ATT00002', 'ATT00003', 'ATT00004', 'ATT00005']:
            path = self.root / key / 'paper.pdf'
            path.parent.mkdir()
            path.write_bytes(b'PDF fixture')
            self.files.append(path)
        self.catalog = {
            'collections': [collection('ROOT0001', 'Research'), collection('CHILD001', 'Titanium', 'ROOT0001'),
                            collection('DEEP0001', 'Heat treatment', 'CHILD001'),
                            collection('OTHER001', 'Other'), collection('EMPTY001', 'Empty')],
            'items': [item('PARENT01', collections=['ROOT0001', 'OTHER001']),
                      item('PARENT02', collections=['CHILD001']),
                      item('PARENT03', collections=['OTHER001']),
                      item('ATT00001', 'attachment', parentItem='PARENT01', filename='paper.pdf', contentType='application/pdf'),
                      item('ATT00002', 'attachment', parentItem='PARENT02', filename='paper.pdf', contentType='application/pdf'),
                      item('ATT00003', 'attachment', parentItem='PARENT03', filename='paper.pdf', contentType='application/pdf'),
                      item('ATT00004', 'attachment', ['DEEP0001'], filename='paper.pdf', contentType='application/pdf'),
                      item('ATT00005', 'attachment', ['ROOT0001'], filename='paper.pdf', contentType='application/pdf', deleted=1)]}

    def tearDown(self):
        self.temp.cleanup()

    def test_nested_selection_inherits_parent_membership_and_deduplicates(self):
        selected = z.select_pdfs(self.files, self.catalog, 'ROOT0001')
        self.assertEqual(selected['paths'], [self.files[0], self.files[1], self.files[3]])
        self.assertEqual(selected['unavailable'], 0)
        self.assertEqual(z.select_pdfs(self.files, self.catalog, 'ROOT0001', False)['paths'], [self.files[0]])

    def test_empty_and_removed_collections_do_not_broaden_scope(self):
        self.assertEqual(z.select_pdfs(self.files, self.catalog, 'EMPTY001')['paths'], [])
        with self.assertRaisesRegex(z.ZoteroError, 'no longer available'):
            z.select_pdfs(self.files, self.catalog, 'MISSING1')

    def test_hierarchy_labels_distinguish_duplicate_names(self):
        labels = z.collection_labels(self.catalog['collections'])
        self.assertEqual(labels['DEEP0001'], 'Research / Titanium / Heat treatment')
        duplicate = [collection('A', 'Same'), collection('B', 'Same')]
        self.assertEqual(len(set(z.collection_labels(duplicate).values())), 2)
        with self.assertRaises(z.ZoteroError):
            z.collection_labels([collection('A', 'Cycle', 'A')])

    def test_only_available_matching_pdf_files_are_selected(self):
        sibling = self.files[0].parent / 'unrelated.pdf'
        sibling.write_bytes(b'Other PDF in same folder')
        result = z.select_pdfs([self.files[0], sibling], self.catalog, 'ROOT0001')
        self.assertEqual(result['paths'], [self.files[0]])
        self.assertEqual(result['unavailable'], 2)

    def test_trashed_parent_and_non_pdf_attachments_are_excluded(self):
        self.catalog['items'][0]['data']['deleted'] = 1
        self.catalog['items'].append(item('HTML0001', 'attachment', ['ROOT0001'], filename='index.html', contentType='text/html'))
        result = z.select_pdfs(self.files, self.catalog, 'ROOT0001', False)
        self.assertEqual(result, {'paths': [], 'unavailable': 0})

    def test_unfiled_papers_only_appear_with_entire_library_selected(self):
        self.catalog['items'].append(item('UNFILED1', 'attachment', filename='unfiled.pdf', contentType='application/pdf'))
        unfiled = self.root / 'UNFILED1' / 'unfiled.pdf'
        unfiled.parent.mkdir()
        unfiled.write_bytes(b'Unfiled PDF')
        self.assertIn(unfiled, z.select_pdfs(self.files + [unfiled], self.catalog)['paths'])
        self.assertNotIn(unfiled, z.select_pdfs(self.files + [unfiled], self.catalog, 'ROOT0001')['paths'])

    def test_linked_files_must_already_be_inside_available_folder(self):
        self.catalog['items'].append(item('LINK0001', 'attachment', ['ROOT0001'], linkMode='linked_file',
                                          path=str(self.root / 'outside.pdf'), contentType='application/pdf'))
        self.assertEqual(z.select_pdfs(self.files, self.catalog, 'ROOT0001', False)['unavailable'], 1)
        self.catalog['items'][-1]['data']['path'] = str(self.files[2])
        self.assertEqual(z.select_pdfs(self.files, self.catalog, 'ROOT0001', False)['paths'], [self.files[0], self.files[2]])

    def test_pagination_reads_more_than_one_hundred_and_detects_duplicates(self):
        client = z.LocalZotero()
        first = [collection(str(n), f'Collection {n}') for n in range(100)]
        with patch.object(client, 'request', side_effect=[first, [collection('last', 'Last')]]) as request:
            self.assertEqual(len(client.all('users/0/collections')), 101)
            self.assertEqual(request.call_args_list[1].kwargs['start'], 100)
        with patch.object(client, 'request', side_effect=[first, first]):
            with self.assertRaises(z.ZoteroError):
                client.all('users/0/collections')

    def test_disabled_or_closed_zotero_is_actionable(self):
        client = z.LocalZotero()
        with patch.object(client.opener, 'open', side_effect=HTTPError(z.BASE_URL, 403, 'Forbidden', {}, io.BytesIO())):
            with self.assertRaisesRegex(z.ZoteroError, 'Allow other applications'):
                client.libraries()
        with patch.object(client.opener, 'open', side_effect=URLError('Connection refused')):
            with self.assertRaisesRegex(z.ZoteroError, 'Open Zotero'):
                client.libraries()

    def test_requests_stay_local_read_only_and_pin_server_identity(self):
        client = z.LocalZotero()
        response = io.BytesIO(json.dumps([collection('A', 'Test')]).encode())
        response.headers = {'Zotero-Server-ID': 'local-instance'}
        with patch.object(client.opener, 'open', return_value=response) as request:
            client.request('users/0/collections', format='json')
            sent = request.call_args.args[0]
            self.assertEqual(sent.get_method(), 'GET')
            self.assertEqual(urlsplit(sent.full_url).netloc, '127.0.0.1:23119')
        self.assertEqual(client.server_id, 'local-instance')
        response = io.BytesIO(b'[]')
        response.headers = {'Zotero-Server-ID': 'different-instance'}
        with patch.object(client.opener, 'open', return_value=response):
            with self.assertRaisesRegex(z.ZoteroError, 'library changed'):
                client.request('users/0/collections')


if __name__ == '__main__':
    unittest.main()
