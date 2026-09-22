"""Separate recognizable bibliographies from evidence in existing PDF chunks.

This runs on the saved text, so upgrading retrieval does not require new
embeddings. It is deliberately based on reference-entry structure, not topic
words or the mere presence of an inline citation.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from collections import defaultdict

VERSION = 'body-evidence-v1'
CONTEXT_CHAR_BUDGET = 18000
MAX_PASSAGE_CHARS = 3600
# A coarse floor for embedding candidates, not a probability of relevance.
# Direct relevance is checked separately when generating supported claims.
MIN_RETRIEVAL_SCORE = 0.55

ANSWER_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {'claims': {'type': 'array', 'maxItems': 6, 'items': {
        'type': 'object', 'additionalProperties': False,
        'properties': {
            'statement': {'type': 'string'},
            'passage': {'type': 'integer'},
            'quote': {'type': 'string'},
            'attribution': {'type': 'string', 'enum': ['own_result', 'prior_work', 'interpretation', 'unclear']},
            'relevance': {'type': 'string', 'enum': ['direct', 'background', 'unrelated']},
        }, 'required': ['statement', 'passage', 'quote', 'attribution', 'relevance'],
    }}}, 'required': ['claims'],
}

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


def asks_for_paper_summaries(question):
    # Only explicit paper-summary requests get a per-paper template. Topic
    # questions (including "summarize lanthanum behaviour") need synthesis.
    return bool(re.search(
        r'\b(?:summari[sz]e|summar(?:y|ies)|overview)\b.{0,60}\b(?:papers?|articles?|studies|documents?)\b',
        question, flags=re.I | re.S))


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


def _matching_text(text):
    text = text.replace('⟦', '[').replace('⟧', ']')
    text = re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip()
    return re.sub(r'\s+\]', ']', re.sub(r'\[\s+', '[', text))


def _quote_location(quote, text):
    # PDF extraction splits words around ligatures and subscripts. Ignore only
    # spacing/ligature differences, preserving every other character and digit.
    # Return the ORIGINAL source span, not the model's version of the quote.
    def characters(value):
        compact, offsets = [], []
        for offset, char in enumerate(value):
            char = {'⟦': '[', '⟧': ']'}.get(char, char)
            for normalized in unicodedata.normalize('NFKC', char):
                if not normalized.isspace():
                    compact.append(normalized)
                    offsets.append(offset)
        return ''.join(compact), offsets
    needle, _ = characters(quote)
    haystack, offsets = characters(text)
    start = haystack.find(needle)
    if start < 0 or not needle:
        return None
    return text[offsets[start]:offsets[start + len(needle) - 1] + 1]


def grounded_answer(payload, sources, summaries=False):
    """Reject unmatched quotes and render citations from source metadata.

    This verifies quote location, not semantic entailment. A real quote can
    still be misinterpreted; expose the quotes so the reader can check that.
    """
    claims = payload.get('claims', []) if isinstance(payload, dict) else []
    if not isinstance(claims, list):
        claims = []
    verified, rejected, seen = [], 0, set()
    for claim in claims[:6]:
        if not isinstance(claim, dict):
            rejected += 1
            continue
        if claim.get('relevance') != 'direct':
            continue
        passage, quote, statement = claim.get('passage'), claim.get('quote'), claim.get('statement')
        if (type(passage) is not int or not 1 <= passage <= len(sources)
                or not isinstance(quote, str) or not isinstance(statement, str)):
            rejected += 1
            continue
        # An excerpt cannot establish that an entire paper never mentions a
        # subject. These are retrieval reports, not scientific findings.
        if is_absence_filler(statement):
            continue
        source = sources[passage - 1]
        normalized_quote = _matching_text(quote)
        if (len(normalized_quote) < 40 or len(normalized_quote) > 900
                or not statement.strip() or len(statement) > 1000
                or '[unreadable PDF symbol]' in quote
                or re.search(r'\[\d+\s*,\s*p', statement, flags=re.I)):
            rejected += 1
            continue
        original_quote = _quote_location(quote, source['text'])
        if original_quote is None:
            # The model can confuse passage IDs with paper IDs. Resolve only
            # when the verbatim quote identifies a unique paper/page itself.
            matches = {}
            for candidate in sources:
                found = _quote_location(quote, candidate['text'])
                if found is not None:
                    matches[(candidate['path'], candidate['page'])] = (candidate, found)
            if len(matches) != 1:
                rejected += 1
                continue
            source, original_quote = next(iter(matches.values()))
        quote_key = (source['path'], source['page'], original_quote)
        if quote_key in seen:
            continue
        seen.add(quote_key)
        attribution = claim.get('attribution', 'unclear')
        if attribution not in {'own_result', 'prior_work', 'interpretation', 'unclear'}:
            attribution = 'unclear'
        # A quote that explicitly attributes evidence to other researchers must
        # not be presented as the containing paper's own experiment.
        if re.search(r'\b(?:et\s+al\b|(?:earlier|recent)\s+work|previous\s+(?:work|stud|research)|'
                     r'(?:researchers|studies)\s+have\s+reported|has\s+been\s+reported)',
                     normalized_quote, flags=re.I):
            attribution = 'prior_work'
        verified.append({'statement': statement.strip(), 'quote': original_quote,
                         'path': source['path'], 'page': source['page'],
                         'attribution': attribution})
    if not verified:
        return ('I could not find sufficiently relevant evidence to answer this question '
                'in the retrieved passages. This does not establish that the topic is absent from your library.'), []
    paper_numbers = {}
    for source in supporting_sources(sources, verified):
        paper_numbers.setdefault(source['path'], len(paper_numbers) + 1)
    for claim in verified:
        claim['citation'] = f'[{paper_numbers[claim["path"]]}, p. {claim["page"]}]'
    prefixes = {'own_result': '', 'prior_work': '**Prior work discussed in the source:** ',
                'interpretation': '**Interpretation in the source:** ', 'unclear': '**Attribution unclear:** '}

    def line(claim):
        return f'- {prefixes[claim["attribution"]]}{claim["statement"]} {claim["citation"]}'

    if summaries:
        sections = []
        for path, number in paper_numbers.items():
            supported = [claim for claim in verified if claim['path'] == path]
            body = '\n\n'.join(map(line, supported))
            sections.append(f'### [{number}] {Path(path).stem}\n\n{body}')
        answer = '\n\n'.join(sections)
    else:
        answer = '\n\n'.join(map(line, verified))
    if rejected:
        answer += '\n\n*Some draft statements were omitted because their supporting quotes could not be matched to the supplied passages.*'
    return answer, verified


def is_absence_filler(statement):
    return bool(re.search(
        r'\bno\s+(?:relevant\s+)?(?:information|mention|discussion)\b|'
        r'\b(?:does|do|did)\s+not\s+(?:mention|discuss|address|provide\s+(?:information|details))\b|'
        r'\b(?:not|never)\s+(?:mentioned|discussed|addressed)\b|'
        r'\b(?:insufficient|not\s+enough)\s+(?:information|evidence)\b',
        statement, flags=re.I))


def supporting_sources(sources, claims):
    """Keep only passages containing the retained claims' supporting quotes."""
    return [source for source in sources if any(
        claim['path'] == source['path'] and claim['page'] == source['page']
        and claim['quote'] in source['text'] for claim in claims)]
