"""Offline demo course generator.

Builds a five-lesson draft course from the selected pages by fixed rules. No AI model is
used and no network requests are made. All course content that states something about
the subject is quoted from the extracted text with its page number; the remaining text
is clearly worded as instructions or templates. The same input always gives the same
output (pass `created_at` to fix the timestamp too).

When the pages do not contain enough usable text for five meaningful lessons, the
generator raises InsufficientSourceError instead of padding the course.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

from courseforge.pdf_extract import PageStatus, Selection, format_page_list
from courseforge.schema import LESSON_COUNT, MAX_QUOTE_CHARS, Course
from courseforge.validate import CitationReport, check_citations, parse_course

MIN_SENTENCES_PER_LESSON = 2
MIN_SENTENCES = MIN_SENTENCES_PER_LESSON * LESSON_COUNT
MAX_EXPLANATION_SENTENCES = 4
MAX_QUIZ_QUESTIONS_PER_LESSON = 2
MAX_DISTRACTORS = 3
BLANK = "_____"

DEMO_NOTICE = ("Demo mode: this course was assembled from the extracted textbook text by fixed "
               "rules. No AI model was used. It is a draft and must be reviewed by a teacher.")

STOPWORDS = frozenset("""
a about above after again against all also although always among an and another any are
around as at be because been before being below between both but by can cannot could did
do does doing down during each either else even every few for from further had has have
having he her here hers him his how however if in into is it its itself just less like made
make many may might more most much must near neither never no nor not now of off often on
once one only onto or other others our ours out over own per perhaps rather same several
shall she should since so some such than that the their theirs them then there therefore
these they this those though through thus to too toward under until up upon us usually very
was we were what when where whether which while who whom whose why will with within without
would yet you your example examples instance called known defined refers page pages chapter
section figure table first second third also used using use uses across along behind beyond
inside outside towards beneath besides despite still quite whole various certain become
becomes became given gives take takes taken
""".split())

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"“(]?[A-Z0-9])")
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*[A-Za-z]")
_EXAMPLE_MARKERS = re.compile(r"\b(for example|for instance|such as|e\.g\.)", re.IGNORECASE)
_DEFINITION = re.compile(
    r"^(?:an?\s+|the\s+)?([a-z][a-z -]{2,40}?)\s+(?:is|are)\s+(?:defined as|called|known as|an?|the)\s",
    re.IGNORECASE)


class DemoGenerationError(Exception):
    """The demo course could not be generated. The message is safe to show to a teacher."""


class InvalidSelectionError(DemoGenerationError):
    pass


class InsufficientSourceError(DemoGenerationError):
    pass


@dataclass(frozen=True)
class Sentence:
    page: int
    text: str

    @property
    def quote(self) -> str:
        if len(self.text) <= MAX_QUOTE_CHARS:
            return self.text
        return self.text[:MAX_QUOTE_CHARS].rsplit(" ", 1)[0]

    def ref(self) -> dict:
        return {"page": self.page, "quote": self.quote}

    def words(self) -> list[str]:
        return [w.lower() for w in _WORD.findall(self.text)]


@dataclass
class DemoResult:
    course: Course
    citations: CitationReport
    notices: list[str] = field(default_factory=list)

    @property
    def verified(self) -> bool:
        return self.citations.all_verified

    @property
    def status_message(self) -> str:
        citation_line = (f"All {len(self.citations.checks)} citations were found on their cited pages."
                         if self.verified else
                         f"Warning: {len(self.citations.problems)} citation(s) could not be verified; "
                         "see the citation report.")
        return f"{DEMO_NOTICE} {citation_line}"


def generate_demo_course(selection: Selection, source_filename: str, *,
                         chapter_label: str | None = None,
                         document_page_count: int | None = None,
                         created_at: datetime | None = None) -> DemoResult:
    """Build a five-lesson draft course from the selected pages, without AI."""
    _check_selection(selection, document_page_count)
    sentences = extract_sentences(selection)
    if len(sentences) < MIN_SENTENCES:
        raise InsufficientSourceError(_insufficient_message(selection, len(sentences)))

    groups = _split_into_lessons(sentences)
    vocabulary = _vocabulary(sentences)
    group_words = [set(w for s in g for w in s.words()) for g in groups]
    notices = list(selection.warnings)
    lessons = []
    for n, group in enumerate(groups, start=1):
        lesson, lesson_notices = _build_lesson(n, group, vocabulary, group_words)
        lessons.append(lesson)
        notices += lesson_notices

    label = chapter_label or selection.label
    metadata = {
        "title": f"{label} (demo course)",
        "source_filename": source_filename,
        "chapter_label": label,
        "start_page": selection.start_page,
        "end_page": selection.end_page,
        "generation_mode": "demo",
        "model": None,
        "review_status": "draft",
    }
    if created_at is not None:
        metadata["created_at"] = created_at

    course, errors = parse_course({"metadata": metadata, "lessons": lessons})
    if course is None:
        raise DemoGenerationError("The demo course failed schema validation: " + "; ".join(errors))
    report = check_citations(course, selection, document_page_count)
    return DemoResult(course=course, citations=report, notices=notices)


def _check_selection(selection: Selection, document_page_count: int | None) -> None:
    start, end = selection.start_page, selection.end_page
    if start < 1 or end < start:
        raise InvalidSelectionError(f"Invalid page range {start}-{end}.")
    if document_page_count is not None and end > document_page_count:
        raise InvalidSelectionError(
            f"Page range {start}-{end} goes past the end of the PDF ({document_page_count} pages).")
    numbers = [p.page_number for p in selection.pages]
    outside = [n for n in numbers if not start <= n <= end]
    if outside:
        raise InvalidSelectionError(
            f"Pages {format_page_list(outside)} are outside the selected range {start}-{end}.")
    if len(set(numbers)) != len(numbers) or numbers != sorted(numbers):
        raise InvalidSelectionError("The selected pages are duplicated or out of order.")
    missing = sorted(set(range(start, end + 1)) - set(numbers))
    if missing:
        raise InvalidSelectionError(
            f"Extracted text is missing for pages {format_page_list(missing)} of the selection.")


def _insufficient_message(selection: Selection, found: int) -> str:
    if not selection.has_usable_text:
        reason = "The selected pages contain no extractable text"
        if any(p.status == PageStatus.LIKELY_SCANNED for p in selection.pages):
            reason += " (they appear to be scanned images, and text recognition is not supported)"
        return reason + ". Select pages with selectable text."
    return (f"The selected pages contain {found} usable sentence(s); at least {MIN_SENTENCES} are "
            f"needed for {LESSON_COUNT} lessons. Select a longer chapter or a wider page range.")


def extract_sentences(selection: Selection) -> list[Sentence]:
    """Complete, well-formed sentences from the selected pages, in reading order."""
    seen, result = set(), []
    for page in selection.text_pages:
        for block in _blocks(page.text):
            text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", block)  # same rule as the validator
            text = re.sub(r"\s+", " ", text).strip()
            for raw in _SENTENCE_SPLIT.split(text):
                sentence = raw.strip()
                key = sentence.casefold()
                if _usable(sentence) and key not in seen:
                    seen.add(key)
                    result.append(Sentence(page.page_number, sentence))
    return result


def _blocks(text: str) -> list[str]:
    """Split page text at standalone heading lines so headings are not glued to sentences.

    A heading is a short line without end punctuation that starts with a capital, follows
    the end of a sentence (or starts the page) and is followed by a capitalized line.
    """
    lines = text.split("\n")
    blocks, current = [], []
    for i, line in enumerate(lines):
        stripped = line.strip()
        prev_ends = i == 0 or re.search(r"[.!?:][\"”)]?$", lines[i - 1].strip() or ".")
        next_starts = i + 1 < len(lines) and re.match(r"[\"“(]?[A-Z]", lines[i + 1].strip())
        heading = (stripped and len(stripped.split()) <= 8 and stripped[:1].isupper()
                   and not re.search(r"[.!?,;\-]$", stripped) and prev_ends and next_starts)
        if heading:
            blocks.append("\n".join(current))
            current = []
        else:
            current.append(line)
    blocks.append("\n".join(current))
    return [b for b in blocks if b.strip()]


def _usable(sentence: str) -> bool:
    words = sentence.split()
    if not 6 <= len(words) <= 60 or len(sentence) < 40:
        return False
    if not re.match(r"[\"“(]?[A-Z]", sentence) or not re.search(r"[.!?][\"”)]?$", sentence):
        return False
    letters = sum(c.isalpha() for c in sentence)
    return letters / len(sentence) >= 0.6


def _split_into_lessons(sentences: list[Sentence]) -> list[list[Sentence]]:
    size, extra = divmod(len(sentences), LESSON_COUNT)
    groups, i = [], 0
    for n in range(LESSON_COUNT):
        count = size + (1 if n < extra else 0)
        groups.append(sentences[i:i + count])
        i += count
    return groups


def _content_words(words) -> list[str]:
    return [w for w in words if len(w) >= 4 and w not in STOPWORDS and "'" not in w
            and not (len(w) >= 6 and w.endswith("ly"))]  # skip adverbs such as "slowly"


def _vocabulary(sentences: list[Sentence]) -> Counter:
    """Content words that appear in lowercase in the text, with their chapter frequency."""
    lowercase_forms = {w for s in sentences for w in _WORD.findall(s.text) if w.islower()}
    return Counter(w for s in sentences for w in _content_words(s.words()) if w in lowercase_forms)


def _pages_label(group: list[Sentence]) -> str:
    pages = sorted({s.page for s in group})
    return f"page {pages[0]}" if len(pages) == 1 else f"pages {format_page_list(pages)}"


def _build_lesson(n: int, group: list[Sentence], vocabulary: Counter,
                  group_words: list[set[str]]) -> tuple[dict, list[str]]:
    notices = []
    pages = _pages_label(group)
    keywords = _lesson_keywords(group, group_words)
    topic = " and ".join(keywords) if keywords else "key ideas"

    explained = group[:MAX_EXPLANATION_SENTENCES]
    explanation = "\n\n".join(
        [f"Source passages from {pages}, quoted in their original order:"]
        + [f"“{s.text}” (page {s.page})" for s in explained])

    example_sentence = next((s for s in group if _EXAMPLE_MARKERS.search(s.text)), None)
    if example_sentence:
        examples = [{"title": f"Example from the text (page {example_sentence.page})",
                     "explanation": f"“{example_sentence.text}”", "sources": [example_sentence.ref()]}]
    else:
        passage = max(group, key=lambda s: (len(_content_words(s.words())), -group.index(s)))
        examples = [{"title": f"Source passage for discussion (page {passage.page})",
                     "explanation": (f"“{passage.text}” Discuss with students what this "
                                     "passage shows. The textbook pages do not mark a worked example here."),
                     "sources": [passage.ref()]}]
        notices.append(f"Lesson {n}: no worked example was found in the text, "
                       "so a source passage is offered for discussion instead.")

    practice = []
    for s in group:
        m = _DEFINITION.match(s.text)
        if m and len(m.group(1).split()) <= 4:
            practice.append({"prompt": f"According to the textbook (page {s.page}), what is {m.group(1).strip()}?",
                             "suggested_answer": f"The textbook states: “{s.text}”",
                             "sources": [s.ref()]})
            break
    focus = max(group, key=lambda s: (len(set(_content_words(s.words()))), -group.index(s)))
    practice.append({"prompt": f"In your own words, explain this passage from page {focus.page}: "
                               f"“{focus.text}”",
                     "suggested_answer": ("Answers will vary. A complete answer restates the passage "
                                          "accurately without copying it word for word."),
                     "sources": [focus.ref()]})

    quiz = []
    for s in group:
        question = _completion_question(s, vocabulary)
        if question:
            quiz.append(question)
        if len(quiz) == MAX_QUIZ_QUESTIONS_PER_LESSON:
            break
    if not quiz:
        raise InsufficientSourceError(
            f"Lesson {n} ({pages}) has no sentence with a key term that appears elsewhere in the "
            "chapter, so no reliable quiz question can be made. Select a longer chapter or a wider "
            "page range.")

    lesson = {
        "number": n,
        "title": f"Part {n}: {topic[:1].upper() + topic[1:]} ({pages})",
        "objectives": [
            {"text": f"Explain the main ideas of the textbook passages on {pages}, including {topic}."},
            {"text": "Support answers with evidence quoted from the textbook."},
        ],
        "explanation": explanation,
        "examples": examples,
        "practice": practice,
        "quiz": quiz,
        "sources": [s.ref() for s in explained],
    }
    return lesson, notices


def _lesson_keywords(group: list[Sentence], group_words: list[set[str]]) -> list[str]:
    """The two terms most specific to this lesson: frequent here, rare in other lessons."""
    counts = Counter(w for s in group for w in _content_words(s.words()))
    lowercase = {w for s in group for w in _WORD.findall(s.text) if w.islower()}

    def score(word):
        spread = sum(word in words for words in group_words)
        return counts[word] * (LESSON_COUNT + 1 - spread)

    ranked = sorted((w for w in counts if w in lowercase), key=lambda w: (-score(w), w))
    return ranked[:2]


def _completion_question(s: Sentence, vocabulary: Counter) -> dict | None:
    """A fill-in-the-blank question about the exact wording of one source sentence.

    The answer is a key term that appears in lowercase in the sentence and elsewhere in
    the chapter. The other options are real terms from the same chapter; the question asks
    for the word the textbook uses, so only the original word is correct.
    """
    lowercase_in_sentence = {w for w in _WORD.findall(s.text) if w.islower() and len(w) >= 5}
    candidates = [w for w in _content_words(lowercase_in_sentence) if vocabulary[w] >= 2]
    if not candidates:
        return None
    answer = sorted(candidates, key=lambda w: (-vocabulary[w], w))[0]

    # Distractors must not be, or resemble, any word in the sentence or each other.
    sentence_words = set(s.words())
    distractors = []
    for w, _ in sorted(vocabulary.items(), key=lambda kv: (-kv[1], kv[0])):
        if len(distractors) == MAX_DISTRACTORS:
            break
        if len(w) >= 5 and not any(_similar(w, other) for other in sentence_words | set(distractors)):
            distractors.append(w)
    if not distractors:
        return None

    blanked = re.sub(rf"\b{re.escape(answer)}\b", BLANK, s.text, flags=re.IGNORECASE)
    options = sorted([answer] + distractors)
    ids = "ABCDEF"
    return {
        "question": (f"Source-based completion: which word does the textbook use on page {s.page} "
                     f"to complete this sentence? “{blanked}”"),
        "options": [{"id": ids[i], "text": text} for i, text in enumerate(options)],
        "correct_option": ids[options.index(answer)],
        "explanation": f"The textbook sentence on page {s.page} reads: “{s.text}”",
        "sources": [s.ref()],
    }


def _similar(a: str, b: str) -> bool:
    return a[:4] == b[:4] or a.rstrip("s") == b.rstrip("s")
