# Research Desk / bot-researcher

A local research assistant for a Zotero PDF library. Search paper excerpts, ask questions, and follow numbered sources to their PDF pages. Built for macOS with Ollama, Mistral 7B, Nomic Embed, FAISS, SQLite, and Streamlit.

## Open the app

Double-click **Start Research Assistant.command**, then use **http://127.0.0.1:8501**. The launcher starts Ollama and the app when needed and stays active in Terminal. Keep that window open while using Research Desk. Server output appears there; press **Ctrl+C** to shut down. It does not install a login service.

1. Open **Library** and search by filename or paper content, then choose PDFs or select **All matching papers**. Search is case-insensitive and matches every entered word. It uses Zotero’s available text cache, saved index text, or direct PDF extraction, and works before AI indexing. The first search builds a local text cache; later searches reuse it. Papers without readable text are listed and can still match by filename.
2. Click **Index selected papers**. OCR is enabled by default when Tesseract is installed and reads pages with little extractable text. Each completed paper is saved; indexing again skips unchanged PDFs.
3. Open **Ask your papers**. Optionally filter indexed papers by filename or content and choose papers to focus on, then ask a specific question. With a filter active, questions search all matching papers unless you choose a smaller selection.
4. Expand the numbered papers and choose **Open this page in Zotero** to check the evidence. Citations identify a paper and its PDF page, for example `[1, p. 2]`.

The default library is `~/Zotero/storage`. You can change the folder in the sidebar. The app searches attachment PDFs. Zotero collections can narrow the selection; notes, annotations, linked files outside the selected folder, and bibliographic metadata are not indexed. Source PDFs and Zotero's database are never modified.

Choose **Answer model** in the sidebar to use another installed local Ollama model. The list includes models that support text generation and excludes embedding-only and cloud-backed models. After installing a model in Ollama, click **Refresh models**. Mistral 7B is the default when installed; other installed models, including Ministral 3 8B, remain selectable. Your selection lasts for the browser session. Switching affects new answers only, and each answer records its model. The separate Nomic search model and existing index are unchanged, so no reindexing is needed. Model file size and Ollama's currently loaded model memory are displayed separately; loaded memory does not include the rest of the app or macOS. Refresh models to update the reading. Models load when answering, not when selecting the dropdown. Mistral now [retires 7B and recommends Ministral 3 8B](https://docs.mistral.ai/models/mistral-7b-0-3). Its Ollama tag is `ministral-3:8b`; [Ollama 0.13.1 or newer is required](https://ollama.com/library/ministral-3:8b).

Example: “What mechanisms cause grain refinement in titanium alloys? Distinguish nucleation from growth restriction and cite the evidence.”

**Maximum source passages** limits how many candidate excerpts the model receives, not how many findings or papers it should discuss. Weak similarity matches are excluded. The model is asked to focus on relevant findings and leave unrelated papers out of the answer. Six passages can come from two papers. Summaries cover retrieved evidence, not necessarily the whole paper.

Search still uses small indexed chunks to find relevant text, but answers receive larger surrounding passages from the same PDF page (up to 3,600 characters each, sharing an 18,000-character budget). Overlapping hits are combined instead of being repeated. Recognizable reference lists are excluded before ranking, including continuations on later pages. Text before a references heading is retained; recognizable appendix or methods headings resume body evidence. This works with existing indexes without reindexing. Topic questions receive a direct synthesis; only explicit paper-summary requests use a summary per paper. The model is instructed to distinguish the paper's own findings from earlier work it discusses.

Answers are generated as ordinary prose with paper/page citations. There is no forced JSON claim list, verbatim-quote requirement, relevance classifier, or automatic claim deletion. The app displays the model's answer directly. Basic citation warnings flag unknown papers or pages without hiding or rewriting the answer; they do not verify its scientific accuracy.

Open **Passages supplied to the model** to inspect the grouped excerpts. These are search candidates, not a declaration that every passage supports the answer. Paper numbers stay consistent with the supplied context even when the answer cites only some papers. Earlier messages keep their original answer, sources, and any quote-check panels.

After an app update, refresh the browser and ask a new question. The app checks for changed answer-code modules on rerun, so a missed file-watcher event cannot keep an older answer engine silently active. Existing chat answers retain their original content. These retrieval and answer changes do not require reindexing.

## Stop the app

Press **Ctrl+C** in the launcher’s Terminal window. Closing that window also requests shutdown. The launcher stops Research Desk and its Ollama server; an Ollama server started elsewhere is left running. Completed paper indexes and downloaded models remain saved. Any current indexing run ends; select the papers and index again after restarting to resume.

If a server was left running by an older launcher, the new launcher connects to it and gives you the same Ctrl+C control without restarting it. **Stop Research Assistant.command** remains available as a fallback. It verifies each server’s owner, working directory, and command before stopping it. An Ollama server started elsewhere is left running. To preview what would stop without stopping it, run `.venv/bin/python stop.py --dry-run`.

## Installation on a new Mac

Use Python 3.12 or newer and [Ollama](https://ollama.com/download). On this Mac, the project uses Python 3.12 in `.venv`.

```sh
brew install ollama tesseract
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
OLLAMA_NO_CLOUD=1 ollama serve
```

In another terminal:

```sh
ollama pull nomic-embed-text:v1.5
ollama pull mistral:7b
./Start\ Research\ Assistant.command
```

The two model downloads total roughly 4.7 GB. An Apple Silicon Mac with 18 GB memory was used for the initial setup. Indexing the entire library can take time and uses additional disk space proportional to extracted text.

## Limit papers to a Zotero collection

Enable **Limit to a Zotero collection** in the sidebar. Choose **Zotero library** (My Library or a group library), then choose a **Zotero collection**. The dropdown shows short collection names; the full selected path appears underneath, such as `UQ / ENGG7341 Thesis / Ti-Al-Fe`. Duplicate names retain their full paths in the menu. **Include subcollections** is on by default; turn it off for only items directly in the chosen collection.

This selection applies to the PDF list, keyword filters, OCR, indexing, paper counts and AI questions. Existing indexes remain saved, but answers only retrieve passages from the current selection. Earlier answers keep their original selection label. An empty or unavailable collection never falls back to the entire library. Disable the collection filter to return to all PDFs in the selected disk folder.

Keep Zotero running. In **Zotero → Settings → Advanced**, enable **Allow other applications on this computer to communicate with Zotero**. Research Desk uses read-only requests to Zotero's local API at `127.0.0.1:23119`; it does not need a cloud API key or internet access, and does not modify Zotero's database. That Zotero setting allows other local applications to read library data too.

Click **Refresh library** after reorganizing collections or downloading attachments. Collection data is also refreshed within 30 seconds on the next app interaction. Only PDFs already present inside **PDF library folder** are included; missing downloads and attachments outside that folder are counted separately. Normal collections are supported; saved searches and special views such as Trash are not collection choices.

## OCR for scanned papers

In **Library**, select PDFs and open **OCR for scanned papers**. **Use OCR when indexing** is enabled by default when Tesseract is available. The default language is English (`eng`). Select the languages that match the paper; additional language data can be installed on macOS with `brew install tesseract-lang`. Refresh the app afterward (language availability refreshes within 30 seconds).

- **Index selected papers** automatically reads pages with fewer than 80 letters or digits using OCR, then adds their text to the AI index. Previously indexed papers with pages missing text are upgraded when selected again.
- **OCR selected papers** prepares text for filename/content filters without running Ollama. Index these papers afterward to make their new passages available to AI answers.
- **Read every page with OCR** handles broken text layers or scanned sections that automatic detection misses. It takes longer; ordinary text extraction is usually more accurate for native PDFs.

Completed OCR pages are cached, including empty pages, so resuming does not repeat them. Failed pages are retried; a failed OCR or embedding run leaves the previously saved AI index intact. Changes to the PDF or selected languages invalidate the applicable cached OCR. Existing readable papers do not need to be reindexed just to enable this feature. Searches reuse cached text and never launch OCR automatically.

OCR operates locally on the CPU. It is intended for printed text, not reliable transcription of handwriting, plots, equations or complex tables. Very poor or sideways scans may still need cleaning or rotation in a separate copy. PDFs with passwords must be unlocked first. Original PDF files and Zotero metadata are never rewritten.

Tesseract is a separate system dependency. `TESSERACT_CMD` can specify its executable path; the app also checks PATH, common Homebrew locations and the standard Windows installation folder. The OCR code uses portable PDFium/Tesseract components; the current launcher and shutdown scripts remain macOS-specific.

## How it works

- `pypdf` extracts text page by page, preserving one-based **PDF** page numbers (which may differ from printed page labels).
- PDFium renders scanned pages for local Tesseract OCR. Rendering targets 300 DPI, with a 25-megapixel cap for unusually large pages. Temporary images are deleted after recognition; recognized text is cached one page at a time.
- Text is split into overlapping passages and embedded with Nomic's `search_document:` prefix. Questions use `search_query:`.
- SQLite stores text, vectors, source paths, file signatures, and embedding model identity. Updates commit one complete document at a time. Changed or missing files are excluded until reindexed.
- FAISS ranks normalized vectors by similarity. Exact duplicate passages are removed from retrieved results, though duplicate attachments remain visible as separate indexed files.
- Reference filtering uses section headings and bibliographic entry structure. It does not remove ordinary inline citations. It can miss unusual reference layouts or OCR errors; filtering is not proof that every remaining statement is an original finding. Internal reference numbers are labeled separately in the model context so they are not confused with PDF pages.
- The selected answer model reads up to eight expanded passages grouped by source PDF and writes a cited prose answer. The app preserves that answer and offers the supplied passages for inspection. Citation warnings check source/page identifiers, not whether a claim logically follows from its source.
- The app calls Ollama only at `127.0.0.1:11434`, bypasses HTTP proxies, binds the interface to `127.0.0.1`, and disables Streamlit usage telemetry. Answer models and the fixed embedding model run locally. Internet access is needed to install software/download models, not to ask questions afterward.

The implementation uses Ollama's current embedding/chat HTTP interfaces directly, keeping the application small without a LangChain dependency. Embeddings are persisted in SQLite; the FAISS search structure is built in memory, so there is no pickle deserialization.

## Local data and limitations

`.data/library.sqlite3` contains extracted text and embeddings. `.data/paper-search.sqlite3` caches normalized text for keyword filtering. `.data/ocr.sqlite3` stores recognized text with PDF page numbers, source file signatures and OCR settings. Content filters include cached OCR text, even for papers with both scanned and native pages. `.data/` and `.venv/` are ignored by Git. No PDFs are copied into the project. Chat history lasts only for the browser session; it is not saved to disk by this app. Each question is independent: include the context it needs.

Scanned pages can be read with the built-in OCR controls. OCR can misread numbers, units and equations; OCR sources are labeled so you can check them against the original. Figures, equations, complex tables, and poor reading order can lose information during text extraction. Recognizable broken glyph codes are replaced with `[unreadable PDF symbol]`, shown with a warning, and the model is instructed not to infer affected units or values. Not every extraction error can be detected. A “blank_pages” count reports pages with too little extractable text. Retrieval selects a small set of excerpts, so answers are not exhaustive literature reviews and may still be wrong. Check sources before using findings in research.

Zotero links use the attachment folder key and target the personal library. When a group library is selected, links target that group. Without collection filtering, group-library attachments may require opening the PDF manually. PDF downloads are available as a fallback.

There is no background watcher. Use **Refresh library** and run indexing again after adding or editing PDFs. Failed papers are listed in the indexing report. If the embedding model changes, reindex before searching.

## Development and diagnostics

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python manage.py status
.venv/bin/python manage.py index --match titanium --limit 5
.venv/bin/python manage.py ocr --match Westly --limit 1
# Optional: --force-ocr, --ocr-languages eng+deu, or index --no-ocr
.venv/bin/python -m streamlit run app.py
```

`requirements.txt` declares supported dependency ranges; `requirements.lock.txt` records the tested environment. App output appears in the launcher’s Terminal; Ollama logs are in `.data/ollama.log`. Older detached runs may also have `.data/app.log`. Press **Ctrl+C** in the launcher window to stop the servers. **Stop Research Assistant.command** is available as a fallback. Models unload from memory after five idle minutes.

## References

- [Original article](https://medium.com/@itzcharles03/build-a-local-llm-powered-research-assistant-in-minutes-76ac70b0b64f)
- [Ollama embedding API](https://docs.ollama.com/api/embed) and [chat API](https://docs.ollama.com/api/chat)
- [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs)
- [Nomic model and required prefixes](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- [pypdf text-extraction limitations](https://pypdf.readthedocs.io/en/stable/user/extract-text.html)
- [Tesseract OCR usage](https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html)
- [PDFium Python renderer](https://pypdfium2.readthedocs.io/en/stable/python_api.html)

- [Zotero collections](https://www.zotero.org/support/collections_and_tags) and [local API](https://www.zotero.org/support/dev/web_api/v3/local_api)
