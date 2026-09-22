"""Read collection membership through Zotero's local API. No Zotero writes."""
import json
from collections import Counter
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

BASE_URL = 'http://127.0.0.1:23119/api/'


class ZoteroError(RuntimeError):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class LocalZotero:
    def __init__(self):
        self.opener = build_opener(ProxyHandler({}), NoRedirect())
        self.server_id = None

    def request(self, path, **params):
        headers = {'Zotero-API-Version': '3', 'Accept': 'application/json'}
        if self.server_id:
            headers['Zotero-Server-ID'] = self.server_id
        request = Request(BASE_URL + path + '?' + urlencode(params), headers=headers)
        try:
            with self.opener.open(request, timeout=15) as response:
                server_id = response.headers.get('Zotero-Server-ID')
                if self.server_id and server_id and self.server_id != server_id:
                    raise ZoteroError('The active Zotero library changed. Refresh library and select the collection again.')
                self.server_id = server_id or self.server_id
                result = json.load(response)
        except HTTPError as exc:
            if exc.code == 403:
                raise ZoteroError('In Zotero → Settings → Advanced, enable “Allow other applications on this computer to communicate with Zotero”, then click Refresh library.') from exc
            if exc.code == 412:
                raise ZoteroError('The active Zotero library changed. Click Refresh library.') from exc
            raise ZoteroError(f'Zotero could not read the collection (HTTP {exc.code}). Refresh library; Zotero 7 or newer is required.') from exc
        except (URLError, TimeoutError, ConnectionError) as exc:
            raise ZoteroError('Open Zotero to use collection filtering, then click Refresh library.') from exc
        except (ValueError, UnicodeError) as exc:
            raise ZoteroError('Zotero returned unreadable collection data. Click Refresh library.') from exc
        if not isinstance(result, list):
            raise ZoteroError('Zotero returned an unexpected collection response.')
        return result

    def all(self, path, **params):
        rows, seen = [], set()
        while True:
            batch = self.request(path, format='json', limit=100, start=len(rows), **params)
            for row in batch:
                if not isinstance(row, dict) or not isinstance(row.get('data'), dict):
                    raise ZoteroError('Zotero returned unexpected collection data. Click Refresh library.')
                key = row.get('key', row.get('id'))
                if key is None or key in seen:
                    raise ZoteroError('Zotero data changed while loading. Click Refresh library to try again.')
                seen.add(key)
            rows.extend(batch)
            if len(batch) < 100:
                return rows

    def libraries(self):
        groups = self.all('users/0/groups')
        return [{'id': 'users/0', 'name': 'My Library', 'group_id': None}] + [
            {'id': f'groups/{int(g["id"])}', 'name': g['data']['name'], 'group_id': int(g['id'])}
            for g in groups]

    def catalog(self, library):
        if not re.fullmatch(r'users/0|groups/\d+', library):
            raise ZoteroError('Choose a valid Zotero library.')
        collections = self.all(library + '/collections')
        # Parent items hold membership; their child PDF attachments inherit it.
        items = self.all(library + '/items', itemType='-note')
        return {'collections': collections, 'items': items}


def collection_labels(collections, compact=False):
    by_key = {c['key']: c['data'] for c in collections}
    labels = {}
    for key, data in by_key.items():
        parts, seen, current = [], set(), key
        while current:
            if current in seen or current not in by_key:
                raise ZoteroError('Zotero’s collection hierarchy changed. Click Refresh library.')
            seen.add(current)
            node = by_key[current]
            parts.append(node['name'])
            current = node.get('parentCollection')
        labels[key] = ' / '.join(reversed(parts))
    # Zotero permits duplicate collection names; keep those choices distinguishable.
    counts = Counter(labels.values())
    for key, label in list(labels.items()):
        if counts[label] > 1:
            labels[key] = f'{label} [{key}]'
    if compact:
        names = Counter(data['name'] for data in by_key.values())
        return {key: data['name'] if names[data['name']] == 1 else labels[key]
                for key, data in by_key.items()}
    return labels


def collection_keys(collections, selected, include_children=True):
    keys = {c['key'] for c in collections}
    if selected is None:
        return keys
    if selected not in keys:
        raise ZoteroError('The selected collection is no longer available. Choose a collection again.')
    result = {selected}
    if include_children:
        while True:
            children = {c['key'] for c in collections if c['data'].get('parentCollection') in result}
            if children <= result:
                break
            result.update(children)
    return result


def select_pdfs(files, catalog, selected=None, include_children=True):
    """Intersect collection membership with actual PDFs inside the configured folder."""
    allowed = collection_keys(catalog['collections'], selected, include_children)
    items = {row['key']: row['data'] for row in catalog['items'] if not row['data'].get('deleted')}
    members = {key for key, data in items.items()
               if selected is None or allowed.intersection(data.get('collections', []))}
    files = [Path(p).resolve() for p in files]
    stored = {(p.parent.name, p.name): p for p in files}
    available = set(files)
    matched, unavailable = set(), 0
    for key, data in items.items():
        if data.get('itemType') != 'attachment':
            continue
        parent = data.get('parentItem')
        if parent and parent not in items:
            continue  # A trashed or otherwise unavailable parent is outside the scope.
        if key not in members and parent not in members:
            continue
        filename = data.get('filename', '')
        linked = data.get('path', '')
        if data.get('contentType') != 'application/pdf' and Path(filename or linked).suffix.lower() != '.pdf':
            continue
        path = stored.get((key, filename))
        if data.get('linkMode') == 'linked_file' and Path(linked).is_absolute():
            candidate = Path(linked).resolve()
            path = candidate if candidate in available else None
        if path is None:
            unavailable += 1
        else:
            matched.add(path)
    return {'paths': [p for p in files if p in matched], 'unavailable': unavailable}
