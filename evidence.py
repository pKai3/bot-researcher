"""Separate recognizable bibliographies from evidence in existing PDF chunks.

This runs on the saved text, so upgrading retrieval does not require new
embeddings. It is deliberately based on reference-entry structure, not topic
words or the mere presence of an inline citation.
"""
from __future__ import annotations

import re
from collections import defaultdict

VERSION = 'body-evidence-v1'
CONTEXT_CHAR_BUDGET = 18000
MAX_PASSAGE_CHARS = 3600
# A coarse floor for embedding candidates, not a probability of relevance.
MIN_RETRIEVAL_SCORE = 0.55

_INITIALS = r'(?:[A-Z]\s*\.\s*){1,4}'
_SURNAME = r"[A-ZÀ-ÖØ-Þ][\w’'−-]+"
_AUTHOR = rf'(?:{_INITIALS}{_SURNAME}|{_SURNAME},\s*{_INITIALS})'
_ENTRY = re.compile(rf'(?<!\w)(?:\[(\d{{1,3}})\]|(\d{{1,3}})[.)])\s*(?={_AUTHOR})')
_YEAR = re.compile(r'\b(?:19|20)\d{2}\b')
_DOI = re.compile(r'\b10\.\d{4,9}/\S+', re.I)
_HEADING = re.compile(r'\b(?:references(?:\s+and\s+notes)?|bibliography|literature\s+cited)\b\s*:?\s*', re.I)
# A following appendix/methods section is evidence again. Only accept headings
# at the start of a chunk, allowing an extracted page number before the heading.
_RESUME = re.compile(
    r'^\s*(?:\d+(?:\s+of\s+\d+)?\s+)?(?:'
    r'appendix(?:\s+[A-Z0-9](?=[. :]))?|supplement(?:ary|al)\s+(?:information|material|methods|results|note)'
    r'|materials\s+and\s+methods|methods|experimental\s+(?:methods|procedure))\b', re.I)


def reference_start(text):
    """Return a bibliography boundary, or None when it is not recognizable."""
    for heading in _HEADING.finditer(text):
        following = text[heading.end():heading.end() + 700]
        entries = list(_ENTRY.finditer(following))
        # "references to colour" and prose mentioning references are not headings.
        if entries and entries[0].start() < 25 and _YEAR.search(following):
            return heading.start()
        if re.match(_AUTHOR, following) and _YEAR.search(following):
            return heading.start()
    entries = list(_ENTRY.finditer(text))
    numbers = [int(m[1] or m[2]) for m in entries]
    if (len(entries) >= 2 and any(b == a + 1 for a, b in zip(numbers, numbers[1:]))
            and (len(_YEAR.findall(text)) >= 2 or _DOI.search(text))):
        # A continuation fragment before the first complete entry is also a
        # reference, even when its entry number was in the preceding chunk.
        prefix = text[:entries[0].start()]
        if _YEAR.search(prefix) or _DOI.search(prefix):
            return 0
        return entries[0].start()
    return None


def filter_rows(rows):
    """Return (original row index, safe row) pairs, retaining vector alignment.

    Follow reference sections across page/chunk boundaries. Keep body text
    before a references heading on a mixed page; resume at appendix/methods
    headings. Do not alter the stored index or original PDFs.
    """
    by_paper = defaultdict(list)
    for index, row in enumerate(rows):
        by_paper[row['path']].append((index, row))
    kept = []
    for paper_rows in by_paper.values():
        in_references = False
        for index, row in sorted(paper_rows, key=lambda pair: (pair[1]['page'], pair[0])):
            text = row['text']
            if in_references and _RESUME.match(text):
                in_references = False
            start = reference_start(text)
            if in_references:
                continue
            if start is not None:
                in_references = True
                text = text[:start].strip()
                if len(text) < 80:
                    continue
                row = {**row, 'text': text, 'references_removed': True}
            kept.append((index, row))
    return sorted(kept, key=lambda pair: pair[0])


def label_internal_citations(text):
    """Distinguish source brackets from app citations without assigning meaning.

    Numeric brackets can also be scientific notation, so do not relabel every
    bracket as a bibliographic reference or change the numbers inside it.
    """
    return re.sub(r'\[\s*(\d+(?:\s*[,;–—e-]\s*\d+)*)\s*\]',
                  lambda match: f'⟦{match[1]}⟧', text)


def page_context(rows):
    """Reconstruct one page from overlapping indexed chunks; retain hit offsets."""
    text, spans = '', {}
    for index, row in rows:
        part = row['text']
        overlap = next((size for size in range(min(250, len(text), len(part)), 19, -1)
                        if text.endswith(part[:size])), 0)
        if text and not overlap:
            text += '\n\n'
        start = len(text) - overlap
        text += part[overlap:]
        spans[index] = (start, len(text))
    return text, spans


def surrounding_passage(text, span, width):
    """Center a bounded window on the hit, preferring sentence boundaries."""
    first, last = span
    start = max(0, min(first - max(0, width - (last - first)) // 2, len(text) - width))
    end = min(len(text), start + width)
    # Keep the complete anchor; avoid introducing word fragments around it.
    if start:
        sentence = re.search(r'(?<=[.!?])\s+(?=[A-Z])', text[start:min(first, start + 300)])
        if sentence:
            start += sentence.end()
        else:
            space = text.find(' ', start, first)
            if space != -1:
                start = space + 1
    if end < len(text):
        sentences = list(re.finditer(r'(?<=[.!?])\s+(?=[A-Z])', text[max(last, end - 300):end]))
        if sentences:
            end = max(last, end - 300) + sentences[-1].start()
        else:
            space = text.rfind(' ', last, end)
            if space != -1:
                end = space
    return start, end
