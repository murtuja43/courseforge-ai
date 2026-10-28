"""Tests for courseforge.pdf_extract, using small PDFs generated in memory."""

import pymupdf
import pytest

from courseforge.pdf_extract import (
    Chapter, PageRangeError, PageStatus, PDFExtractionError, extract_pdf, format_page_list,
    list_chapters, parse_toc, select_chapter, select_page_range,
)

BODY = "Photosynthesis converts light energy into chemical energy stored in glucose."


def make_pdf(page_texts, toc=None, scanned_pages=(), labels=None, encrypt=False) -> bytes:
    """Build a PDF. None in page_texts makes a blank page; scanned_pages get a full-page image."""
    doc = pymupdf.open()
    for i, text in enumerate(page_texts, start=1):
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
        if i in scanned_pages:
            pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 40), False)
            pix.set_rect(pix.irect, (210, 210, 210))
            page.insert_image(page.rect, pixmap=pix)
    if toc:
        doc.set_toc(toc)
    if labels:
        doc.set_page_labels(labels)
    kwargs = {}
    if encrypt:
        kwargs = dict(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner")
    data = doc.tobytes(**kwargs)
    doc.close()
    return data


# ---------------------------------------------------------------- page numbers and text

def test_pages_are_one_based_and_text_matches_its_page():
    texts = [f"Page marker {n}. {BODY}" for n in range(1, 6)]
    doc = extract_pdf(make_pdf(texts), "book.pdf")
    assert doc.filename == "book.pdf"
    assert doc.page_count == 5
    assert [p.page_number for p in doc.pages] == [1, 2, 3, 4, 5]
    for p in doc.pages:
        assert f"Page marker {p.page_number}." in p.text
        assert p.status == PageStatus.OK
    assert doc.page(3).page_number == 3
    assert doc.warnings == []


def test_metadata_is_returned():
    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 72), BODY)
    pdf.set_metadata({"title": "Biology Basics", "author": "A. Author"})
    doc = extract_pdf(pdf.tobytes())
    assert doc.metadata["title"] == "Biology Basics"
    assert doc.metadata["author"] == "A. Author"


def test_extract_from_path(tmp_path):
    path = tmp_path / "sample.pdf"
    path.write_bytes(make_pdf([BODY]))
    doc = extract_pdf(path)
    assert doc.filename == "sample.pdf"
    assert BODY in doc.pages[0].text


def test_printed_page_labels_are_kept_separately():
    labels = [{"startpage": 0, "prefix": "", "style": "r", "firstpagenum": 1},
              {"startpage": 2, "prefix": "", "style": "D", "firstpagenum": 1}]
    doc = extract_pdf(make_pdf([BODY] * 4, labels=labels))
    assert [p.label for p in doc.pages] == ["i", "ii", "1", "2"]
    assert [p.page_number for p in doc.pages] == [1, 2, 3, 4]
    assert doc.pages[2].display_name == "PDF page 3 (printed page 1)"


# ---------------------------------------------------------------- empty and scanned pages

def test_empty_and_scanned_pages_are_flagged_not_treated_as_text():
    doc = extract_pdf(make_pdf([BODY, None, None, "Hi", BODY], scanned_pages={3}))
    statuses = [p.status for p in doc.pages]
    assert statuses == [PageStatus.OK, PageStatus.EMPTY, PageStatus.LIKELY_SCANNED,
                        PageStatus.LOW_TEXT, PageStatus.OK]
    assert not doc.pages[1].has_text
    assert not doc.pages[2].has_text
    assert doc.pages[3].has_text  # a short page is still real text
    assert doc.text_page_count == 3
    joined = " ".join(doc.warnings)
    assert "Page 3 appears to be scanned" in joined
    assert "Page 2 has no extractable text" in joined


def test_mostly_scanned_document_gets_an_overall_warning():
    doc = extract_pdf(make_pdf([None, None, BODY], scanned_pages={1, 2}))
    assert any("Most of this PDF appears to be scanned" in w for w in doc.warnings)


# ---------------------------------------------------------------- table of contents

TOC = [[1, "Chapter 1: Cells", 1], [2, "1.1 Membranes", 2], [1, "Chapter 2: Energy", 4]]


def test_toc_is_read_with_titles_and_pages():
    doc = extract_pdf(make_pdf([BODY] * 6, toc=TOC))
    assert doc.has_toc
    assert [(e.level, e.title, e.page_number) for e in doc.toc] == [tuple(t) for t in TOC]


def test_chapter_ranges_come_from_the_toc():
    doc = extract_pdf(make_pdf([BODY] * 6, toc=TOC))
    chapters = list_chapters(doc)
    assert chapters == [Chapter("Chapter 1: Cells", 1, 1, 3),
                        Chapter("1.1 Membranes", 2, 2, 3),
                        Chapter("Chapter 2: Energy", 1, 4, 6)]
    assert [c.title for c in list_chapters(doc, max_level=1)] == ["Chapter 1: Cells",
                                                                  "Chapter 2: Energy"]


def test_entries_on_the_same_page_share_that_page():
    toc = [[1, "Preface", 1], [1, "Chapter 1", 1], [1, "Chapter 2", 3]]
    doc = extract_pdf(make_pdf([BODY] * 3, toc=toc))
    assert [(c.start_page, c.end_page) for c in list_chapters(doc)] == [(1, 1), (1, 2), (3, 3)]


def test_select_chapter_returns_exactly_its_pages():
    texts = [f"Page marker {n}. {BODY}" for n in range(1, 7)]
    doc = extract_pdf(make_pdf(texts, toc=TOC))
    chapter = list_chapters(doc)[2]
    sel = select_chapter(doc, chapter)
    assert sel.source == "toc"
    assert sel.label == "Chapter 2: Energy"
    assert [p.page_number for p in sel.pages] == [4, 5, 6]
    assert "Page marker 4." in sel.pages[0].text


def test_select_chapter_rejects_a_chapter_not_in_the_toc():
    doc = extract_pdf(make_pdf([BODY] * 6, toc=TOC))
    with pytest.raises(PageRangeError):
        select_chapter(doc, Chapter("Invented chapter", 1, 2, 5))


def test_no_toc_means_no_chapters():
    doc = extract_pdf(make_pdf([BODY] * 3))
    assert not doc.has_toc
    assert list_chapters(doc) == []


def test_invalid_toc_entries_are_dropped_with_a_warning():
    entries, warnings = parse_toc([[1, "Good", 2], [1, "Past the end", 99], [1, "Unresolved", -1],
                                   [0, "Bad level", 1], [1, "   ", 1], ["x"]], page_count=5)
    assert entries == [entries[0]] and entries[0].title == "Good"
    assert warnings == ["5 table of contents entries were skipped because they did not point "
                        "to a valid page."]


# ---------------------------------------------------------------- section boundaries (fictional layout)

PARAGRAPH = ("Lumen crystals store the light of the twin suns and release it slowly after dusk, "
             "which keeps the valley warm through the long night of the planet Orvane. ")


def make_wrapped_pdf(page_texts, toc, labels=None) -> bytes:
    doc = pymupdf.open()
    for text in page_texts:
        doc.new_page().insert_textbox(pymupdf.Rect(60, 60, 540, 780), text, fontsize=10)
    doc.set_toc(toc)
    if labels:
        doc.set_page_labels(labels)
    return doc.tobytes()


# Pages 1-3 of a fictional chapter laid out like a printed textbook: 4.2 starts partway down
# page 1, 4.3 starts partway down page 3, and running headers repeat titles at page ends.
SECTION_PAGES = [
    "4.1\nCRYSTAL BASICS\n" + PARAGRAPH * 3 + "\n4.2\nCRYSTAL GROWTH\n" + PARAGRAPH + "\n4.2\nCRYSTAL GROWTH\n41",
    PARAGRAPH * 4 + "\n42\nCRYSTALS OF ORVANE",
    PARAGRAPH * 3 + "\n4.3\nCRYSTAL HARVESTING\n" + PARAGRAPH + "\n4.3\nCRYSTAL HARVESTING\n43",
    "CHAPTER 5\nSKY RIVERS\n" + PARAGRAPH * 4,
]
SECTION_TOC = [[1, "CHAPTER 4: Crystals of Orvane", 1], [2, "4.1 Crystal Basics", 1],
               [2, "4.2 Crystal Growth", 1], [2, "4.3 Crystal Harvesting", 3],
               [1, "CHAPTER 5: Sky Rivers", 4]]


def section_doc():
    labels = [{"startpage": 0, "prefix": "", "style": "D", "firstpagenum": 41}]
    return extract_pdf(make_wrapped_pdf(SECTION_PAGES, SECTION_TOC, labels))


def test_section_extends_to_the_page_where_the_next_heading_starts_partway_down():
    chapters = {c.title: c for c in list_chapters(section_doc())}
    growth = chapters["4.2 Crystal Growth"]
    assert (growth.start_page, growth.end_page, growth.end_shared) == (1, 3, True)


def test_entries_starting_on_the_same_page_share_it():
    basics = {c.title: c for c in list_chapters(section_doc())}["4.1 Crystal Basics"]
    assert (basics.start_page, basics.end_page, basics.end_shared) == (1, 1, True)


def test_section_ends_before_a_chapter_that_starts_on_a_fresh_page():
    chapters = {c.title: c for c in list_chapters(section_doc())}
    harvesting = chapters["4.3 Crystal Harvesting"]
    assert (harvesting.start_page, harvesting.end_page, harvesting.end_shared) == (3, 3, False)
    chapter4 = chapters["CHAPTER 4: Crystals of Orvane"]
    assert (chapter4.start_page, chapter4.end_page, chapter4.end_shared) == (1, 3, False)


def test_heading_that_cannot_be_found_keeps_the_page_before_rule():
    toc = SECTION_TOC[:3] + [[2, "4.3 A Title That Is Not Printed", 3], SECTION_TOC[4]]
    doc = extract_pdf(make_wrapped_pdf(SECTION_PAGES, toc))
    growth = {c.title: c for c in list_chapters(doc)}["4.2 Crystal Growth"]
    assert (growth.end_page, growth.end_shared) == (2, False)


def test_title_inside_a_sentence_is_not_taken_as_a_heading():
    pages = ["Intro text.\n" + PARAGRAPH * 2, PARAGRAPH * 3 + "\nThe Problems of harvesting are many.\n" + PARAGRAPH]
    doc = extract_pdf(make_wrapped_pdf(pages, [[1, "Intro", 1], [1, "Problems", 2]]))
    intro = list_chapters(doc)[0]
    assert (intro.end_page, intro.end_shared) == (1, False)


def test_heading_near_the_top_of_its_page_is_not_mid_page():
    pages = [PARAGRAPH * 4, "42\nCRYSTALS\n4.9\nCRYSTAL DUST\n" + PARAGRAPH * 3]
    doc = extract_pdf(make_wrapped_pdf(pages, [[1, "4.8 Crystal Light", 1], [1, "4.9 Crystal Dust", 2]]))
    light = list_chapters(doc)[0]
    assert (light.end_page, light.end_shared) == (1, False)


def test_selection_carries_the_overlapping_sections_with_their_real_ranges():
    doc = section_doc()
    sel = select_page_range(doc, 2, 3)
    assert [(s.title, s.start_page, s.end_page) for s in sel.sections] == [
        ("CHAPTER 4: Crystals of Orvane", 1, 3), ("4.2 Crystal Growth", 1, 3), ("4.3 Crystal Harvesting", 3, 3)]
    assert [p.page_number for p in sel.pages] == [2, 3]          # physical numbering is unchanged
    assert [p.label for p in sel.pages] == ["42", "43"]           # printed labels stay separate


def test_selection_without_a_toc_has_no_sections():
    sel = select_page_range(extract_pdf(make_pdf([BODY] * 3)), 1, 3)
    assert sel.sections == []


def test_chapter_selection_covers_the_shared_page():
    doc = section_doc()
    growth = next(c for c in list_chapters(doc) if c.title == "4.2 Crystal Growth")
    sel = select_chapter(doc, growth)
    assert [p.page_number for p in sel.pages] == [1, 2, 3]
    assert "4.2 Crystal Growth" in [s.title for s in sel.sections]


# ---------------------------------------------------------------- page range selection

def test_select_page_range():
    texts = [f"Page marker {n}. {BODY}" for n in range(1, 6)]
    doc = extract_pdf(make_pdf(texts))
    sel = select_page_range(doc, 2, 4)
    assert sel.source == "page_range"
    assert sel.label == "Pages 2-4"
    assert [p.page_number for p in sel.pages] == [2, 3, 4]
    assert sel.warnings == []
    assert select_page_range(doc, 5, 5).label == "Page 5"


@pytest.mark.parametrize("start,end", [(0, 2), (3, 2), (1, 6), (-1, 1), (1.5, 2), ("1", 2), (True, 2)])
def test_invalid_page_ranges_are_rejected(start, end):
    doc = extract_pdf(make_pdf([BODY] * 5))
    with pytest.raises(PageRangeError):
        select_page_range(doc, start, end)


def test_selection_warns_about_pages_without_text():
    doc = extract_pdf(make_pdf([BODY, None, BODY, None], scanned_pages={4}))
    sel = select_page_range(doc, 1, 4)
    assert [p.page_number for p in sel.text_pages] == [1, 3]
    assert sel.warnings == ["2 of 4 selected pages have no usable text (pages 2, 4) and will be skipped."]
    empty = select_page_range(doc, 2, 2)
    assert not empty.has_usable_text
    assert empty.warnings == ["The selected pages contain no extractable text."]


def test_format_page_list():
    assert format_page_list([7, 1, 2, 3, 5, 6]) == "1-3, 5-7"
    assert format_page_list([4]) == "4"
    assert format_page_list([]) == ""


# ---------------------------------------------------------------- invalid input

def test_encrypted_pdf_is_rejected_gracefully():
    with pytest.raises(PDFExtractionError, match="password protected"):
        extract_pdf(make_pdf([BODY], encrypt=True))


@pytest.mark.parametrize("data,message", [
    (b"", "empty"),
    (b"This is just a text file, not a PDF.", "not a PDF"),
    (b"\x89PNG\r\n\x1a\n" + b"\x00" * 100, "not a PDF"),
    (b"%PDF-1.7\n" + b"\x00garbage" * 50, "could not be opened|no pages"),
])
def test_invalid_files_are_rejected_gracefully(data, message):
    with pytest.raises(PDFExtractionError, match=message):
        extract_pdf(data)


def test_truncated_pdf_either_recovers_or_fails_cleanly():
    data = make_pdf([BODY] * 3)
    try:
        doc = extract_pdf(data[: len(data) // 2])
    except PDFExtractionError:
        return
    assert all(p.page_number == i for i, p in enumerate(doc.pages, start=1))


def test_oversized_file_is_rejected():
    with pytest.raises(PDFExtractionError, match="too large"):
        extract_pdf(make_pdf([BODY]), max_bytes=100)


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(PDFExtractionError, match="Could not read"):
        extract_pdf(tmp_path / "missing.pdf")
