from pathlib import Path
import html
import streamlit as st
import research as r
import ocr
from paper_search import filter_papers

st.set_page_config(page_title='Research Desk', page_icon='📚', layout='wide')
r.initialize()
client = r.Ollama()

@st.cache_resource(max_entries=2, show_spinner=False)
def cached_corpus(db, root, key, revision, valid_paths):
    return r.load_corpus(db, root, key)


@st.cache_data(ttl=30, show_spinner=False)
def ocr_languages():
    try:
        return ocr.available_languages(), None
    except ocr.OCRError as exc:
        return [], str(exc)


def show_sources(sources, prefix):
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


with st.sidebar:
    st.title('Research Desk')
    st.caption('YOUR PAPERS, CLOSE AT HAND')
    library = st.text_input('PDF library folder', str(r.DEFAULT_LIBRARY))
    root = str(Path(library).expanduser().resolve())
    if st.button('Refresh library'):
        st.cache_resource.clear()
        st.rerun()
    st.divider()
    st.caption('Mistral 7B · Local answers\n\nNomic Embed · Local document search')
    st.caption('The app reads PDFs without changing them. Its search index stays in this project’s .data folder.')
    st.caption('This version reads attachment files; it does not import Zotero collections, notes or annotations.')

try:
    files = r.discover(root)
except r.AssistantError as exc:
    st.error(str(exc))
    st.stop()

all_docs = r.documents(root=root)
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
a, b, c = st.columns(3)
a.metric('PDFs in library', len(files))
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
    effective_scope = scope or (scope_matches if scope_query.strip() else None)
    if scope_query.strip():
        st.caption(f'{len(scope_matches)} matching indexed papers. Questions use all matches unless you select specific papers above.')
        if not scope_matches:
            st.info('No indexed papers match this filter. Change the filter, or find and index more papers in Library.')
    passages = st.slider('Source passages per answer', 3, 8, 6)
    search_only = st.checkbox('Find passages without generating an answer')
    st.caption('Ask each question with its full context. Earlier answers are displayed for reference but are not sent to the model.')
    if st.button('Clear conversation'):
        st.session_state['messages'] = []
    messages = st.session_state.setdefault('messages', [])
    for number, message in enumerate(messages):
        with st.chat_message('user'):
            st.write(message['question'])
        with st.chat_message('assistant'):
            st.markdown(message['answer'])
            warning = r.citation_warning(message['answer'], message['sources']) if message['sources'] and not message.get('search_only') else None
            if warning:
                st.warning(warning)
            show_sources(message['sources'], f'history-{number}')
    question = st.chat_input('What do these papers say about…?', disabled=not ready or not online or (bool(scope_query.strip()) and not scope_matches) or (not chat_ready and not search_only), max_chars=2000)
    if question:
        try:
            with st.spinner('Finding relevant passages in your papers…'):
                valid_paths = tuple(sorted(d['path'] for d in ready))
                corpus = cached_corpus(str(r.DB), root, key, r.revision(), valid_paths)
                sources = r.retrieve(question, corpus, client, paths=effective_scope, k=passages)
            with st.spinner('Reading the evidence and drafting an answer locally…'):
                answer = ('Here are the closest matching passages.' if sources else 'No passages found.') if search_only else client.answer(question, sources)
            warning = r.citation_warning(answer, sources) if not search_only and sources else None
            messages.append({'question': question, 'answer': answer, 'sources': sources, 'warning': warning, 'search_only': search_only})
            st.rerun()
        except r.AssistantError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f'The request could not be completed: {exc}')
    st.caption('Answers are generated from selected excerpts, not a complete literature review. Check the linked passages before using claims in your work.')
