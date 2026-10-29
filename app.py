"""CourseForge AI (Project Axiora): Streamlit interface.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from courseforge import ui_helpers as ui
from courseforge.pdf_extract import (
    PageRangeError, PDFExtractionError, extract_pdf, format_page_list, list_chapters, select_chapter,
    select_page_range,
)
from courseforge.sample import SAMPLE_FILENAME, sample_pdf_bytes
from courseforge.validate import TRACEABILITY_NOTE, CitationStatus

st.set_page_config(page_title="CourseForge AI", page_icon="\U0001F4D8", layout="wide")

st.markdown("""
<style>
:root { --navy: #0B1F3A; --emerald: #047857; --mint: #E6F6EF; --line: #D5DCE6; --muted: #5B677A; }
.block-container { padding-top: 2rem; max-width: 1200px; }
.cf-hero { background: var(--navy); color: #fff; border-radius: 14px; padding: 1.6rem 2rem; margin-bottom: 1.2rem; }
.cf-hero .kicker { color: #34D399; font-weight: 700; letter-spacing: .08em; font-size: .8rem; }
.cf-hero h1 { color: #fff; margin: .2rem 0 0; font-size: 2.2rem; padding: 0; }
.cf-hero .sub { color: #C9D6EA; font-size: 1.05rem; margin-top: .2rem; }
.cf-step { color: var(--emerald); font-weight: 700; font-size: .8rem; letter-spacing: .06em; margin: 1.4rem 0 0; }
.cf-badge { display: inline-block; padding: .2rem .7rem; border-radius: 999px; font-size: .85rem;
            font-weight: 600; margin: 0 .4rem .4rem 0; border: 1px solid var(--line); }
.cf-ai { background: var(--navy); color: #fff; border-color: var(--navy); }
.cf-demo { background: #F2F5F9; color: var(--navy); }
.cf-draft { background: #FFF7E6; color: #8A5300; border-color: #F2D9A6; }
.cf-approved { background: var(--mint); color: var(--emerald); border-color: #9FDCC3; }
.cf-ok { background: var(--mint); color: var(--emerald); border-color: #9FDCC3; }
.cf-warn { background: #FDECEC; color: #9B1C1C; border-color: #F5C2C2; }
</style>
""", unsafe_allow_html=True)

DEFAULTS = {
    "doc": None, "doc_id": None, "doc_error": None, "last_upload_id": None,
    "course_id": 0, "course_data": None, "gen_info": None, "gen_selection": None,
    "gen_error": None, "pending_mode": None, "generating": False, "gen_notice": None,
    "approved_fp": None, "mode_choice": "auto", "uploader_nonce": 0,
}
for name, value in DEFAULTS.items():
    st.session_state.setdefault(name, value)
state = st.session_state


def badge(text: str, kind: str) -> str:
    return f'<span class="cf-badge cf-{kind}">{text}</span>'


def step(label: str) -> None:
    st.markdown(f'<div class="cf-step">{label}</div>', unsafe_allow_html=True)


def reset_course() -> None:
    for name in ("course_data", "gen_info", "gen_selection", "gen_error", "approved_fp", "gen_notice"):
        state[name] = None


def load_document(data: bytes, filename: str, doc_id: str) -> None:
    reset_course()
    try:
        state.doc = extract_pdf(data, filename)
        state.doc_id, state.doc_error = doc_id, None
    except PDFExtractionError as exc:
        state.doc, state.doc_id, state.doc_error = None, None, str(exc)


def request_generation(mode: str) -> None:
    """Button callback. Ignores repeat clicks and never silently replaces an existing course."""
    if state.generating:
        return
    if state.course_data is not None and not state.get("confirm_regenerate"):
        state.gen_notice = ("A course already exists. Tick the box to confirm that you want to "
                            "replace it (your edits will be lost), then generate again.")
        return
    state.pending_mode = mode
    state.gen_notice = None


def switch_to_demo() -> None:
    state.mode_choice = "demo"
    state.gen_error = None
    request_generation("demo")


def approve(fp: str) -> None:
    state.approved_fp = fp


def revoke() -> None:
    state.approved_fp = None


# ---------------------------------------------------------------- header and sidebar

st.markdown("""
<div class="cf-hero">
  <div class="kicker">PROJECT AXIORA</div>
  <h1>CourseForge AI</h1>
  <div class="sub">From Textbooks to Structured Learning Experiences</div>
</div>
""", unsafe_allow_html=True)
st.write("Turn one textbook chapter into a five-lesson draft course with objectives, explanations, "
         "examples, practice and quizzes, each linked to the textbook pages it came from. "
         "**Teachers remain responsible for reviewing and approving everything before use.**")

ai_ready, ai_message = ui.ai_status()
with st.sidebar:
    st.subheader("Workflow")
    done = [state.doc is not None, state.doc is not None, state.course_data is not None,
            state.course_data is not None and state.approved_fp is not None]
    for flag, label in zip(done, ["1. Upload textbook", "2. Select chapter", "3. Generate draft",
                                  "4. Review and approve"]):
        st.write(("✅ " if flag else "○ ") + label)
    st.divider()
    st.subheader("AI configuration")
    (st.success if ai_ready else st.info)(ai_message)
    st.caption("Settings come from environment variables or a local .env file. The API key is never shown.")
    st.caption(ui.FREE_TIER_NOTICE)
    st.divider()
    if st.button("Start over", key="start_over", width="stretch"):
        nonce = state.uploader_nonce + 1  # a new key also clears the uploaded file
        for name in list(state.keys()):
            del state[name]
        state.uploader_nonce = nonce
        st.rerun()

# ---------------------------------------------------------------- 1. upload

step("STEP 1  ·  UPLOAD TEXTBOOK")
col_upload, col_sample = st.columns([3, 1])
with col_upload:
    uploaded = st.file_uploader("Textbook PDF", type=["pdf"], key=f"uploader|{state.uploader_nonce}",
                                help="A PDF with selectable text. Scanned pages are not OCR-processed.")
with col_sample:
    st.write("")
    st.write("")
    if st.button("Use the fictional sample textbook", key="load_sample", width="stretch"):
        load_document(sample_pdf_bytes(), SAMPLE_FILENAME, "sample")
if uploaded is not None and uploaded.file_id != state.last_upload_id:
    state.last_upload_id = uploaded.file_id
    load_document(uploaded.getvalue(), uploaded.name, uploaded.file_id)

if state.doc_error:
    st.error(f"This PDF could not be used: {state.doc_error}")

doc = state.doc
if doc is None:
    st.info("Upload a textbook PDF, or try the fictional sample textbook, to begin.")
    st.stop()

meta_cols = st.columns(4)
meta_cols[0].metric("PDF pages", doc.page_count)
meta_cols[1].metric("Pages with text", doc.text_page_count)
meta_cols[2].metric("Contents entries", len(doc.toc))
meta_cols[3].metric("File", doc.filename if len(doc.filename) <= 24 else doc.filename[:21] + "...")
details = ", ".join(f"{k}: {doc.metadata[k]}" for k in ("title", "author") if k in doc.metadata)
if details:
    st.caption(details)
if doc.warnings:
    with st.expander(f"{len(doc.warnings)} extraction warning(s)", expanded=True):
        for warning in doc.warnings:
            st.warning(warning)

# ---------------------------------------------------------------- 2. select chapter

step("STEP 2  ·  SELECT A CHAPTER")
chapters = list_chapters(doc)
methods = (["toc", "range"] if chapters else ["range"])
method = st.radio("Select by", methods, horizontal=True, key=f"method|{state.doc_id}",
                  format_func={"toc": "Table of contents", "range": "Page range"}.get)
if not chapters:
    st.caption("This PDF has no usable table of contents, so choose the physical PDF pages yourself.")

selection = None
if method == "toc":
    index = st.selectbox("Chapter (from the PDF's own table of contents)", range(len(chapters)),
                         format_func=lambda i: ui.chapter_label(chapters[i]), key=f"chapter|{state.doc_id}")
    selection = select_chapter(doc, chapters[index])
else:
    c1, c2 = st.columns(2)
    start = c1.number_input("First PDF page", min_value=1, max_value=doc.page_count, value=1, step=1,
                            key=f"start|{state.doc_id}")
    end = c2.number_input("Last PDF page", min_value=1, max_value=doc.page_count, value=doc.page_count,
                          step=1, key=f"end|{state.doc_id}")
    try:
        selection = select_page_range(doc, int(start), int(end))
    except PageRangeError as exc:
        st.error(str(exc))

if selection is not None:
    st.markdown(f"**Selected:** {selection.label}, physical PDF pages "
                f"{selection.start_page}-{selection.end_page} "
                f"({len(selection.text_pages)} of {len(selection.pages)} pages with usable text)")
    for warning in selection.warnings:
        st.warning(warning)
st.caption("Page numbers are physical PDF pages (page 1 is the first page of the file). "
           "Scanned pages are not OCR-processed and are skipped.")

# ---------------------------------------------------------------- 3. generation mode

step("STEP 3  ·  CHOOSE HOW TO GENERATE")
mode = st.radio("Generation mode", list(ui.MODES), format_func=ui.MODES.get, key="mode_choice")
if mode == "demo":
    st.info("Demo mode organizes the textbook's own sentences with fixed templates. It is useful for "
            "trying the workflow, but it does not write explanations like AI mode does.")
elif not ai_ready:
    st.warning(f"AI is unavailable, so this mode cannot generate a course: {ai_message} "
               "Choose Demo mode if you want an offline draft instead.")
else:
    st.caption(ui.FREE_TIER_NOTICE)

# ---------------------------------------------------------------- 4. generate

step("STEP 4  ·  GENERATE THE DRAFT COURSE")
if state.pending_mode:
    run_mode, state.pending_mode = state.pending_mode, None
    gen_selection = selection
    if gen_selection is None:
        state.gen_error = {"title": "No pages selected", "message": "Select a valid chapter or page range.",
                           "details": [], "offer_demo": False}
    else:
        state.generating = True
        try:
            spinner = ("Generating with Google Gemini. This can take a few minutes..." if run_mode != "demo"
                       else "Building the demo course...")
            with st.spinner(spinner):
                outcome = ui.run_generation(gen_selection, doc, run_mode)
        finally:
            state.generating = False
        if outcome.ok:
            state.course_id += 1
            state.course_data = outcome.result.course.model_dump(mode="json")
            state.gen_info = ui.generation_info(outcome.result)
            state.gen_selection = gen_selection
            state.gen_error = None
            state.approved_fp = None
            state.confirm_regenerate = False
        else:
            state.gen_error = {"title": outcome.title, "message": outcome.message,
                               "details": outcome.details, "offer_demo": outcome.offer_demo}

if state.course_data is not None:
    st.checkbox("Replace the current course (discards its edits)", key="confirm_regenerate")
st.button("Generate course" if state.course_data is None else "Regenerate course",
          type="primary", key="generate", disabled=selection is None or state.generating,
          on_click=request_generation, args=(mode,))
if state.gen_notice:
    st.info(state.gen_notice)

if state.gen_error:
    err = state.gen_error
    st.error(f"**{err['title']}.** {err['message']}")
    if err["details"]:
        with st.expander("Problems reported by the validator"):
            for problem in err["details"]:
                st.write(f"- {problem}")
    if err["offer_demo"]:
        st.write("No course was generated. You can try again later, or build a demo course instead "
                 "(offline, no AI; it organizes source sentences rather than explaining them).")
        st.button("Generate in demo mode instead", key="switch_demo", on_click=switch_to_demo)

if state.course_data is None:
    st.stop()

# ---------------------------------------------------------------- 5. review

step("STEP 5  ·  REVIEW, EDIT AND APPROVE")
cid = state.course_id
ui.seed_edit_state(state, cid, state.course_data)
edited = ui.collect_edits(state, cid, state.course_data)
state.course_data = edited  # keep edits even if a widget's state is cleared later
evaluation = ui.evaluate_course(edited, state.gen_selection, doc.page_count)
course, report = evaluation.course, evaluation.report
info = state.gen_info

badges = [badge(f"AI-generated · Gemini · {info['model']}", "ai") if info["mode"] == "llm"
          else badge("Offline demo · no AI used", "demo")]
approved = course is not None and state.approved_fp == ui.fingerprint(course)
badges.append(badge("Approved by teacher", "approved") if approved else badge("Draft · not approved", "draft"))
if report is not None:
    counts = ui.citation_counts(report)
    badges.append(badge(f"{counts['verified']} of {len(report.checks)} citations verified",
                        "ok" if report.all_verified else "warn"))
st.markdown(" ".join(badges), unsafe_allow_html=True)
st.caption(TRACEABILITY_NOTE)

sel = state.gen_selection
if selection is not None and (selection.start_page, selection.end_page) != (sel.start_page, sel.end_page):
    st.info(f"This course was generated from {sel.label} (pages {sel.start_page}-{sel.end_page}). "
            "Your current selection is different; regenerate to use it.")
if info["notices"]:
    with st.expander(f"{len(info['notices'])} generation note(s)"):
        for notice in info["notices"]:
            st.write(f"- {notice}")
if evaluation.errors:
    st.error("Some edits are not valid yet, so citations, approval and export are paused until they are fixed:")
    for error in evaluation.errors:
        st.write(f"- {error}")


def status_line(path: tuple) -> None:
    check = evaluation.checks_by_path.get(path)
    if check is None:
        return
    text = ui.citation_label(check)
    if check.verified:
        st.caption(f"✅ {text}")
    elif check.status == CitationStatus.PAGE_HAS_NO_TEXT:
        st.caption(f"⚠️ {text}")
    else:
        st.caption(f"❌ {text}: {check.message}")


def source_editor(path: tuple, sources: list) -> None:
    if not sources:
        st.caption("❌ No source reference.")
    for si in range(len(sources)):
        sp = path + ("sources", si)
        c_page, c_quote = st.columns([1, 5])
        c_page.number_input("PDF page", min_value=1, step=1, key=ui.widget_key(cid, sp + ("page",)))
        c_quote.text_area("Source quotation", key=ui.widget_key(cid, sp + ("quote",)), height=80)
        status_line(sp)


def text(label: str, path: tuple, area: bool = False, height: int = 120) -> None:
    key = ui.widget_key(cid, path)
    if area:
        st.text_area(label, key=key, height=height)
    else:
        st.text_input(label, key=key)


lesson_tabs = st.tabs([f"Lesson {l['number']}" for l in edited["lessons"]] + ["Citations", "Approve & export"])
for li, lesson in enumerate(edited["lessons"]):
    base = ("lessons", li)
    with lesson_tabs[li]:
        text("Lesson title", base + ("title",))
        st.markdown("##### Learning objectives")
        for oi in range(len(lesson["objectives"])):
            text(f"Objective {oi + 1}", base + ("objectives", oi, "text"))
        st.markdown("##### Explanation")
        text("Explanation", base + ("explanation",), area=True, height=260)
        source_editor(base, lesson["sources"])
        st.markdown("##### Examples")
        for ei, example in enumerate(lesson["examples"]):
            with st.container(border=True):
                text("Example title", base + ("examples", ei, "title"))
                text("Example", base + ("examples", ei, "explanation"), area=True)
                source_editor(base + ("examples", ei), example["sources"])
        st.markdown("##### Practice")
        for pi, activity in enumerate(lesson["practice"]):
            with st.container(border=True):
                text(f"Practice activity {pi + 1}", base + ("practice", pi, "prompt"), area=True, height=90)
                text("Suggested answer", base + ("practice", pi, "suggested_answer"), area=True, height=90)
                source_editor(base + ("practice", pi), activity["sources"])
        st.markdown("##### Quiz")
        for qi, question in enumerate(lesson["quiz"]):
            qp = base + ("quiz", qi)
            with st.container(border=True):
                text(f"Question {qi + 1}", qp + ("question",), area=True, height=90)
                for oi, option in enumerate(question["options"]):
                    text(f"Option {option['id']}", qp + ("options", oi, "text"))
                st.selectbox("Correct answer", [o["id"] for o in question["options"]],
                             key=ui.widget_key(cid, qp + ("correct_option",)))
                text("Answer explanation", qp + ("explanation",), area=True, height=90)
                source_editor(qp, question["sources"])

with lesson_tabs[-2]:
    st.markdown("#### Citation verification")
    st.write(TRACEABILITY_NOTE)
    if report is None:
        st.warning("Citations cannot be checked until the edits listed above are valid.")
    else:
        counts = ui.citation_counts(report)
        m = st.columns(4)
        m[0].metric("Verified", counts["verified"])
        m[1].metric("Not verified", counts["not_verified"])
        m[2].metric("Cannot check (no text)", counts["cannot_check"])
        m[3].metric("Missing citations", counts["missing"])
        for error in report.course_errors:
            st.error(error)
        st.dataframe(ui.citation_rows(report), width="stretch", hide_index=True)

with lesson_tabs[-1]:
    st.markdown("#### Approve and export")
    if course is None:
        st.warning("Fix the invalid edits listed above before approving or exporting.")
    else:
        fp = ui.fingerprint(course)
        if state.approved_fp and state.approved_fp != fp:
            st.info("The course was edited after approval, so it is back in draft. Approve it again when ready.")
        if approved:
            st.success("Approved by the teacher. Exports are marked as approved.")
            st.button("Return to draft", key="revoke", on_click=revoke)
        else:
            needs_ack = not report.all_verified
            if needs_ack:
                st.warning("Some citations are not verified or are missing. Review them in the Citations tab first.")
            ack = st.checkbox("I have reviewed the unverified citations", key=f"ack|{cid}") if needs_ack else True
            reviewed = st.checkbox("I have reviewed this course and take responsibility for its use",
                                   key=f"reviewed|{cid}")
            st.button("Approve course", key="approve", type="primary", disabled=not (ack and reviewed),
                      on_click=approve, args=(fp,))
        export = ui.with_review_status(course, state.approved_fp)
        stem = (export.metadata.chapter_label or "course").lower()
        stem = "".join(ch if ch.isalnum() else "-" for ch in stem).strip("-")[:50] or "course"
        status = export.metadata.review_status
        st.caption(f"Exports include your latest edits and are marked **{status}**.")
        c1, c2 = st.columns(2)
        c1.download_button("Download JSON", ui.course_to_json(export, report), f"{stem}-{status}.json",
                           "application/json", key="dl_json", width="stretch")
        c2.download_button("Download Markdown", ui.course_to_markdown(export, report), f"{stem}-{status}.md",
                           "text/markdown", key="dl_md", width="stretch")
