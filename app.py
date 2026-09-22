from pathlib import Path
import html
import streamlit as st
import research as r
import evidence
from runtime import refresh_answer_modules
import ocr
import zotero
from paper_search import filter_papers

refresh_answer_modules(evidence, r)
# Read signatures here as well: app.py reruns even if an older refresh helper
# remains imported during an update of the helper itself.
answer_code_revision = tuple((module.__name__, Path(module.__file__).stat().st_mtime_ns)
                             for module in (evidence, r))

st.set_page_config(page_title='Research Desk', page_icon='📚', layout='wide')
# These two menus need more room than their sidebar controls. Keep the fixed
# option heights used by Streamlit's virtual list; widen instead of wrapping.
st.html('''<style>
[data-testid="stSelectboxVirtualDropdown"]:has([role="listbox"][aria-label="Zotero library"]),
[data-testid="stSelectboxVirtualDropdown"]:has([role="listbox"][aria-label="Zotero collection"]) {
    width: min(44rem, calc(100vw - 2.5rem)) !important;
    max-width: calc(100vw - 2.5rem);
}
</style>''')
r.initialize()
client = r.Ollama()

@st.cache_resource(max_entries=2, show_spinner=False)
def cached_corpus(db, root, key, revision, valid_paths, evidence_version):
    return r.load_corpus(db, root, key)


@st.cache_data(ttl=30, show_spinner=False)
def ocr_languages():
    try:
        return ocr.available_languages(), None
    except ocr.OCRError as exc:
        return [], str(exc)


@st.cache_data(ttl=30, show_spinner=False)
def zotero_libraries():
    return zotero.LocalZotero().libraries()


@st.cache_data(ttl=30, max_entries=8, show_spinner=False)
def zotero_catalog(library_id):
    return zotero.LocalZotero().catalog(library_id)


def show_passage_sources(sources, prefix):
    # Older answers used passage numbers; preserve their original citation map.
    for i, source in enumerate(sources, 1):
        path = Path(source['path'])
        with st.expander(f'[{i}] {path.stem} · PDF page {source["page"]}'):
            st.write(source['text'])
            if source.get('ocr'):
                st.caption('Read using OCR. Check numbers, units and equations against the original page.')
            if '[unreadable PDF symbol]' in source['text']:
                st.warning('This passage contains a symbol the PDF extractor could not read. Check the original PDF for units, equations and numerical claims.')
            st.caption(f'File: {path.name} · Page counted from the start of the PDF')
            link = r.zotero_uri(source)
            if link:
                st.markdown(f'<a href="{html.escape(link, quote=True)}" target="_self">Open this page in Zotero ↗</a>', unsafe_allow_html=True)
            if st.button('Show PDF download', key=f'{prefix}-prepare-{i}'):
                try:
                    st.download_button('Download source PDF', path.read_bytes(), file_name=path.name,
                                       mime='application/pdf', key=f'{prefix}-download-{i}')
                except OSError:
                    st.warning('This PDF has moved. Refresh the library index.')


def show_sources(sources, prefix, citation_style):
    if not sources:
        return
    if citation_style != r.CITATION_STYLE:
        show_passage_sources(sources, prefix)
        return
    papers = r.group_sources(sources)
    passage_label = 'passage' if len(sources) == 1 else 'passages'
    paper_label = 'paper' if len(papers) == 1 else 'papers'
    st.caption(f'{len(sources)} {passage_label} from {len(papers)} {paper_label}. Citation numbers identify papers; page numbers locate the evidence.')
    for paper in papers:
        path, number = Path(paper['path']), paper['number']
        label = 'passage' if len(paper['passages']) == 1 else 'passages'
        with st.expander(f'[{number}] {path.stem} · {len(paper["passages"])} {label}'):
            pages = sorted({s['page'] for s in paper['passages']})
            for page in pages:
                excerpts = [s for s in paper['passages'] if s['page'] == page]
                st.markdown(f'**[{number}, p. {page}] · PDF page {page}**')
                for excerpt in excerpts:
                    st.write(excerpt['text'])
                if any(s.get('ocr') for s in excerpts):
                    st.caption('Read using OCR. Check numbers, units and equations against the original page.')
                if any('[unreadable PDF symbol]' in s['text'] for s in excerpts):
                    st.warning('This page contains a symbol the PDF extractor could not read. Check the original PDF for units, equations and numerical claims.')
                link = r.zotero_uri(excerpts[0])
                if link:
                    st.markdown(f'<a href="{html.escape(link, quote=True)}" target="_self">Open this page in Zotero ↗</a>', unsafe_allow_html=True)
            st.caption(f'File: {path.name} · Pages counted from the start of the PDF')
            if st.button('Show PDF download', key=f'{prefix}-paper-prepare-{number}'):
                try:
                    st.download_button('Download source PDF', path.read_bytes(), file_name=path.name,
                                       mime='application/pdf', key=f'{prefix}-paper-download-{number}')
                except OSError:
                    st.warning('This PDF has moved. Refresh the library index.')


with st.sidebar:
    st.title('Research Desk')
    st.caption('YOUR PAPERS, CLOSE AT HAND')
    library = st.text_input('PDF library folder', str(r.DEFAULT_LIBRARY))
    root = str(Path(library).expanduser().resolve())
    if st.button('Refresh library'):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()
    limit_collection = st.checkbox('Limit to a Zotero collection')
    collection_controls = st.container()
    st.divider()
    st.caption('Mistral 7B · Local answers\n\nNomic Embed · Local document search')
    st.caption('The app reads PDFs without changing them. Its search index stays in this project’s .data folder.')
    st.caption('Zotero collections can limit your paper selection. Notes and annotations are not imported.')

try:
    files = r.discover(root)
except r.AssistantError as exc:
    st.error(str(exc))
    st.stop()

scope_label, group_id = 'All PDFs in the selected folder', None
if limit_collection:
    with collection_controls:
        try:
            with st.spinner('Reading Zotero collections…'):
                libraries = zotero_libraries()
                library_names = {entry['id']: entry['name'] for entry in libraries}
                # Keep a removed selection visible until the user chooses another;
                # refreshing must never silently widen the scope.
                old_library = st.session_state.get('zotero_library')
                if old_library and old_library not in library_names:
                    library_names[old_library] = 'Unavailable library — choose another'
                library_id = st.selectbox('Zotero library', list(library_names),
                                          format_func=library_names.get, key='zotero_library')
                st.caption(library_names[library_id])
                if library_id not in {entry['id'] for entry in libraries}:
                    raise zotero.ZoteroError('Choose an available Zotero library.')
                catalog = zotero_catalog(library_id)
                labels = zotero.collection_labels(catalog['collections'])
                menu_labels = zotero.collection_labels(catalog['collections'], compact=True)
                selection_key = f'zotero_collection_{library_id}'
                old_collection = st.session_state.get(selection_key)
                if old_collection and old_collection not in labels:
                    labels[old_collection] = 'Unavailable collection — choose another'
                    menu_labels[old_collection] = labels[old_collection]
                collection = st.selectbox('Zotero collection', [None] + sorted(labels, key=lambda k: labels[k].casefold()),
                                          format_func=lambda key: menu_labels[key] if key else 'Entire Zotero library',
                                          key=selection_key)
                if collection:
                    st.caption(labels[collection])
                include_children = st.checkbox('Include subcollections', value=True, disabled=collection is None)
                narrowed = zotero.select_pdfs(files, catalog, collection, include_children)
                files = narrowed['paths']
                group_id = next(entry['group_id'] for entry in libraries if entry['id'] == library_id)
                scope_label = library_names[library_id] + (f' / {labels[collection]}' if collection else '')
                if collection and include_children:
                    scope_label += ' (including subcollections)'
                st.caption(f'{len(files)} local PDFs in this selection. Applies to OCR, indexing and questions.')
                if narrowed['unavailable']:
                    st.caption(f'{narrowed["unavailable"]} PDF attachments are not available inside the selected PDF folder. Download them in Zotero or check the folder location.')
        except zotero.ZoteroError as exc:
            st.error(str(exc))
            st.stop()

visible_paths = {str(p) for p in files}
all_docs = [d for d in r.documents(root=root) if d['path'] in visible_paths]
key, online, chat_ready = None, False, False
try:
    models = client.models()
    key = client.embedding_key()
    online = True
    chat_ready = r.CHAT_MODEL in models
except r.AssistantError as exc:
    st.warning(str(exc))
if online and not chat_ready:
    st.warning(f'The answer model is missing. Run: ollama pull {r.CHAT_MODEL}')
ready = [d for d in all_docs if key and r.current_document(d) and r.index_compatible(d['embedding_key'], key)]

st.title('Explore your research library')
st.write('Ask a focused question. Follow the evidence back to the page.')
st.caption(f'Paper selection: {scope_label}')
a, b, c = st.columns(3)
a.metric('PDFs in selection', len(files))
b.metric('Papers ready to search', len(ready))
c.metric('Searchable passages', sum(d['chunks'] for d in ready))
ask_tab, library_tab = st.tabs(['Ask your papers', 'Library'])

with library_tab:
    st.subheader('Build your searchable library')
    st.write('Indexing reads each PDF and saves a local search index. Unchanged papers are skipped on later runs. You can stop and resume; completed papers are saved individually.')
    query = st.text_input('Find PDFs by filename or content', placeholder='e.g. titanium, microwave, recycling', help='Case-insensitive keyword search. Every word must occur in the filename or available paper text. No AI indexing is required.')
    with st.spinner('Searching paper text… The first search may take a little longer.'):
        results = filter_papers(files, query)
    matching = results['paths']
    if query.strip():
        st.caption(f'Content available for {results["searched"]} of {len(files)} PDFs. All filenames are searched too.')
        if results['unavailable']:
            with st.expander(f'{len(results["unavailable"])} PDFs could only be searched by filename'):
                st.write('These papers may need OCR or an unlocked PDF before their contents can be searched.')
                st.dataframe(results['unavailable'], hide_index=True, width='stretch')
    mode = st.radio('Papers to index', ['Choose papers', 'All matching papers'], horizontal=True)
    selected = st.multiselect('Choose PDFs', matching, format_func=lambda p: f'{Path(p).name} · {Path(p).parent.name}') if mode == 'Choose papers' else matching
    st.caption(f'{len(selected)} selected · {len(matching)} matching PDFs')
    languages, ocr_error = ocr_languages()
    with st.expander('OCR for scanned papers'):
        st.write('OCR reads text from page images locally. It saves the recognized text separately and never changes your PDFs. Completed pages are reused when you resume.')
        if ocr_error:
            st.warning(ocr_error)
        use_ocr = st.checkbox('Use OCR when indexing', value=bool(languages), disabled=not languages)
        chosen_languages = st.multiselect('OCR languages', languages,
                                         default=['eng'] if 'eng' in languages else languages[:1],
                                         format_func=lambda code: 'English (eng)' if code == 'eng' else code,
                                         disabled=not use_ocr)
        force_ocr = st.checkbox('Read every page with OCR', disabled=not use_ocr,
                                help='Use for PDFs with a broken text layer or scanned sections that automatic detection misses. This takes longer.')
        st.caption('Normally, only pages with little readable text use OCR. Select the languages that match the paper. For additional languages on macOS, install tesseract-lang with Homebrew, then refresh.')
    ocr_options = ocr.Options(tuple(chosen_languages), force_ocr) if use_ocr and chosen_languages else None
    if use_ocr and not chosen_languages:
        st.warning('Choose an OCR language to continue.')
    if st.button('OCR selected papers', disabled=not selected or ocr_options is None,
                 help='Prepare scanned text for content filters without running the AI index.'):
        progress = st.progress(0.0)
        detail = st.empty()
        report = []
        for number, path in enumerate(selected):
            name = Path(path).name
            def update_page(page, total, action):
                detail.caption(f'{number+1}/{len(selected)} · {name} · {action} page {page}/{total}')
            try:
                result = r.prepare_ocr(path, ocr_options, update_page)
                report.append({'file': name, **result})
            except Exception as exc:
                report.append({'file': name, 'status': 'failed', 'error': str(exc)})
            progress.progress((number + 1) / len(selected))
        st.session_state['ocr_report'] = report
        st.rerun()
    if 'ocr_report' in st.session_state:
        report = st.session_state['ocr_report']
        failures = sum(row['status'] == 'failed' for row in report)
        st.info(f'Last OCR run: {len(report)-failures} completed, {failures} failed. Recognized text is available to content filters. Index these papers to include it in AI answers.')
        st.dataframe(report, hide_index=True, width='stretch')
    if st.button('Index selected papers', type='primary', disabled=not selected or not online or (use_ocr and not chosen_languages)):
        progress = st.progress(0.0)
        detail = st.empty()
        report = []
        for number, path in enumerate(selected):
            name = Path(path).name
            def update(done, total):
                detail.caption(f'{number+1}/{len(selected)} · {name} · passage {done}/{total}')
            def update_page(page, total, action):
                detail.caption(f'{number+1}/{len(selected)} · {name} · {action} page {page}/{total}')
            detail.caption(f'{number+1}/{len(selected)} · Reading {name}')
            try:
                result = r.index_document(path, root, key, client, progress=update,
                                          ocr_options=ocr_options, page_progress=update_page)
                report.append({'file': name, **result})
            except Exception as exc:
                report.append({'file': name, 'status': 'failed', 'error': str(exc)})
            progress.progress((number + 1) / len(selected))
        st.session_state['index_report'] = report
        st.cache_resource.clear()
        st.rerun()
    if 'index_report' in st.session_state:
        report = st.session_state['index_report']
        failures = sum(row['status'] == 'failed' for row in report)
        st.info(f'Last indexing run: {len(report)-failures} completed or unchanged, {failures} failed.')
        st.dataframe(report, hide_index=True, width='stretch')
    st.caption('OCR can misread numbers, units and equations. Images, plots and complex tables are not interpreted reliably. Pages still without enough readable text are reported as blank_pages.')
    with st.expander('Indexed papers', expanded=False):
        st.dataframe([{'Paper': Path(d['path']).name, 'Pages': d['pages'], 'Passages': d['chunks'],
                       'Pages without text': d['blank_pages'],
                       'OCR pages': len(d['ocr_pages']),
                       'Status': 'Ready' if d in ready else 'Needs reindexing / file missing'} for d in all_docs],
                     hide_index=True, width='stretch')

with ask_tab:
    if not ready:
        st.info('Start in Library: choose a few PDFs and click Index selected papers.')
    scope_query = st.text_input('Filter indexed papers by filename or content', placeholder='e.g. titanium', key='scope_query')
    with st.spinner('Filtering indexed papers…'):
        scope_matches = filter_papers([d['path'] for d in ready], scope_query)['paths']
    scope = st.multiselect('Focus on specific indexed papers (optional)', scope_matches,
                          format_func=lambda p: f'{Path(p).stem} · {Path(p).parent.name}')
    # Always pass an explicit set of allowed papers. None would search the full
    # on-disk index and could leak results from outside the chosen collection.
    effective_scope = scope or scope_matches
    if scope_query.strip():
        st.caption(f'{len(scope_matches)} matching indexed papers. Questions use all matches unless you select specific papers above.')
        if not scope_matches:
            st.info('No indexed papers match this filter. Change the filter, or find and index more papers in Library.')
    passages = st.slider('Maximum source passages', 1, 8, 6,
                         help='An upper limit, not a target. Answers may use fewer passages or none when evidence is insufficient.')
    st.caption(f'Searching {len(effective_scope)} papers · Up to {passages} passages total. A paper can contribute more than one passage.')
    st.caption('Passages include surrounding text. Recognizable reference lists are excluded from answer evidence.')
    search_only = st.checkbox('Find passages without generating an answer')
    if not search_only:
        st.caption('Only passages supporting the answer are shown. Papers without relevant findings are omitted.')
    st.caption('Ask each question with its full context. Earlier answers are displayed for reference but are not sent to the model.')
    if st.button('Clear conversation'):
        st.session_state['messages'] = []
    messages = st.session_state.setdefault('messages', [])
    for number, message in enumerate(messages):
        with st.chat_message('user'):
            st.write(message['question'])
        with st.chat_message('assistant'):
            if message.get('paper_selection'):
                st.caption(f'Paper selection: {message["paper_selection"]}')
            st.markdown(message['answer'])
            citation_style = message.get('citation_style', 'passages')
            warning = r.citation_warning(message['answer'], message['sources'], citation_style) if message['sources'] and not message.get('search_only') else None
            if warning:
                st.warning(warning)
            if message.get('grounding'):
                with st.expander('Check supporting quotes'):
                    st.caption('Each quote was matched to its cited passage. Check whether it supports the claim; matching text does not verify the interpretation.')
                    for support in message['grounding']:
                        st.markdown(f'**{support["citation"]}** {support["statement"]}')
                        st.text(support['quote'])
            show_sources(message['sources'], f'history-{number}', citation_style)
    question = st.chat_input('What do these papers say about…?', disabled=not ready or not online or (bool(scope_query.strip()) and not scope_matches) or (not chat_ready and not search_only), max_chars=2000)
    if question:
        try:
            with st.spinner('Finding relevant passages in your papers…'):
                valid_paths = tuple(sorted(d['path'] for d in ready))
                corpus = cached_corpus(str(r.DB), root, key, r.revision(), valid_paths,
                                       (evidence.VERSION, answer_code_revision))
                sources = r.retrieve(question, corpus, client, paths=effective_scope, k=passages)
                if group_id:
                    sources = [{**source, 'group_id': group_id} for source in sources]
            with st.spinner('Reading the evidence and drafting an answer locally…'):
                answer = ('Here are the closest matching passages.' if sources else 'No passages found.') if search_only else client.answer(question, sources)
                if not search_only:
                    sources = client.answer_sources
            warning = r.citation_warning(answer, sources) if not search_only and sources else None
            messages.append({'question': question, 'answer': answer, 'sources': sources, 'warning': warning,
                             'search_only': search_only, 'paper_selection': scope_label,
                             'grounding': getattr(client, 'answer_evidence', []) if not search_only else [],
                             'citation_style': r.CITATION_STYLE})
            st.rerun()
        except r.AssistantError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f'The request could not be completed: {exc}')
    st.caption('Answers are generated from selected excerpts, not a complete literature review. Check the linked passages before using claims in your work.')
