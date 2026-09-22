"""Optional command-line indexing and diagnostics."""
import argparse
import json
import research as r
import ocr


def main():
    parser = argparse.ArgumentParser(description='Manage the local research library')
    parser.add_argument('command', choices=['index', 'ocr', 'status'])
    parser.add_argument('--folder', default=str(r.DEFAULT_LIBRARY))
    parser.add_argument('--match', default='', help='Case-insensitive filename substring')
    parser.add_argument('--limit', type=int, default=0, help='Maximum files; 0 means all')
    parser.add_argument('--no-ocr', action='store_true', help='Do not run OCR when indexing; already cached OCR text remains usable')
    parser.add_argument('--ocr-languages', default='eng', help='Installed Tesseract languages, such as eng or eng+deu')
    parser.add_argument('--force-ocr', action='store_true', help='OCR every page, including pages with an existing text layer')
    args = parser.parse_args()
    r.initialize()
    files = [p for p in r.discover(args.folder) if args.match.casefold() in p.name.casefold()]
    if args.command == 'status':
        print(json.dumps({'pdfs': len(files), 'indexed': len(r.documents(root=args.folder))}, indent=2))
        return
    if args.command == 'ocr' and args.no_ocr:
        parser.error('The ocr command cannot be combined with --no-ocr.')
    options = None if args.no_ocr else ocr.Options(tuple(args.ocr_languages.split('+')), args.force_ocr)
    client = r.Ollama() if args.command == 'index' else None
    key = client.embedding_key() if client else None
    selected = files[:args.limit] if args.limit > 0 else files
    failures = 0
    for i, path in enumerate(selected, 1):
        try:
            if args.command == 'ocr':
                result = r.prepare_ocr(path, options)
            else:
                result = r.index_document(path, args.folder, key, client, ocr_options=options)
        except Exception as exc:
            result = {'status': 'failed', 'error': str(exc)}
            failures += 1
        print(json.dumps({'file': path.name, 'progress': f'{i}/{len(selected)}', **result}), flush=True)
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()
