"""Optional command-line indexing and diagnostics."""
import argparse
import json
import research as r


def main():
    parser = argparse.ArgumentParser(description='Manage the local research library')
    parser.add_argument('command', choices=['index', 'status'])
    parser.add_argument('--folder', default=str(r.DEFAULT_LIBRARY))
    parser.add_argument('--match', default='', help='Case-insensitive filename substring')
    parser.add_argument('--limit', type=int, default=0, help='Maximum files; 0 means all')
    args = parser.parse_args()
    r.initialize()
    files = [p for p in r.discover(args.folder) if args.match.casefold() in p.name.casefold()]
    if args.command == 'status':
        print(json.dumps({'pdfs': len(files), 'indexed': len(r.documents(root=args.folder))}, indent=2))
        return
    client = r.Ollama()
    key = client.embedding_key()
    selected = files[:args.limit] if args.limit > 0 else files
    failures = 0
    for i, path in enumerate(selected, 1):
        try:
            result = r.index_document(path, args.folder, key, client)
        except Exception as exc:
            result = {'status': 'failed', 'error': str(exc)}
            failures += 1
        print(json.dumps({'file': path.name, 'progress': f'{i}/{len(selected)}', **result}), flush=True)
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()
