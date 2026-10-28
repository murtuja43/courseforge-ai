"""Tests for courseforge.schema and the readable error reporting in courseforge.validate."""

import copy
import json

import pytest
from pydantic import ValidationError

from courseforge.schema import Course, QuizQuestion
from courseforge.validate import parse_course

QUOTE = "converts light energy into chemical energy"


def ref(page=1, quote=QUOTE):
    return {"page": page, "quote": quote}


def quiz_question():
    return {
        "question": "What does photosynthesis produce?",
        "options": [{"id": "A", "text": "Glucose"}, {"id": "B", "text": "Salt"},
                    {"id": "C", "text": "Iron"}, {"id": "D", "text": "Sand"}],
        "correct_option": "A",
        "explanation": "Light energy is stored as chemical energy in glucose.",
        "sources": [ref()],
    }


def lesson(n):
    return {
        "number": n,
        "title": f"Lesson title {n}",
        "objectives": [{"text": "Describe the inputs and outputs of photosynthesis."}],
        "explanation": "Plants capture light and store its energy in sugar molecules.",
        "examples": [{"title": "A leaf in sunlight", "explanation": "Leaves make glucose in light.",
                      "sources": [ref()]}],
        "practice": [{"prompt": "Name the energy source.", "suggested_answer": "Light.", "sources": []}],
        "quiz": [quiz_question()],
        "sources": [ref()],
    }


def course_dict():
    return {
        "metadata": {"title": "Photosynthesis", "source_filename": "biology.pdf",
                     "chapter_label": "Chapter 3", "start_page": 1, "end_page": 3,
                     "generation_mode": "demo"},
        "lessons": [lesson(n) for n in range(1, 6)],
    }


def errors_for(mutate):
    data = course_dict()
    mutate(data)
    course, errors = parse_course(data)
    assert course is None
    return errors


# ---------------------------------------------------------------- valid data

def test_valid_course_parses():
    course, errors = parse_course(course_dict())
    assert errors == []
    assert len(course.lessons) == 5
    assert course.metadata.review_status == "draft"
    assert course.lessons[0].quiz[0].correct_answer.text == "Glucose"


def test_valid_course_round_trips_through_json():
    course, _ = parse_course(course_dict())
    again, errors = parse_course(course.model_dump_json())
    assert errors == [] and again == course


def test_text_is_stripped_and_option_ids_uppercased():
    data = course_dict()
    data["lessons"][0]["title"] = "  Padded title  "
    data["lessons"][0]["quiz"][0]["correct_option"] = "a"
    course, errors = parse_course(data)
    assert errors == []
    assert course.lessons[0].title == "Padded title"
    assert course.lessons[0].quiz[0].correct_option == "A"


def test_lesson_can_cite_multiple_pages():
    data = course_dict()
    data["lessons"][0]["sources"] = [ref(1), ref(2, "the green pigment that absorbs light")]
    course, errors = parse_course(data)
    assert errors == [] and [r.page for r in course.lessons[0].sources] == [1, 2]


# ---------------------------------------------------------------- missing or malformed fields

def test_missing_required_field_is_reported_with_its_location():
    errors = errors_for(lambda d: d["lessons"][1].pop("explanation"))
    assert errors == ["lessons 2 > explanation: Field required"]


def test_unknown_field_is_rejected():
    errors = errors_for(lambda d: d["lessons"][0].update(confidence=0.9))
    assert errors == ["lessons 1 > confidence: Extra inputs are not permitted"]


def test_wrong_type_is_rejected():
    errors = errors_for(lambda d: d["lessons"][0]["sources"][0].update(page="first"))
    assert len(errors) == 1 and errors[0].startswith("lessons 1 > sources 1 > page:")


@pytest.mark.parametrize("field", ["title", "explanation"])
def test_empty_or_blank_lesson_content_is_rejected(field):
    errors = errors_for(lambda d: d["lessons"][2].update({field: "   "}))
    assert len(errors) == 1 and errors[0].startswith(f"lessons 3 > {field}:")


@pytest.mark.parametrize("field", ["objectives", "examples", "practice", "quiz", "sources"])
def test_empty_lesson_lists_are_rejected(field):
    errors = errors_for(lambda d: d["lessons"][0].update({field: []}))
    assert len(errors) == 1 and errors[0].startswith(f"lessons 1 > {field}:")


def test_invalid_json_text_is_reported():
    course, errors = parse_course("{not json")
    assert course is None and errors[0].startswith("Not valid JSON:")


def test_several_errors_are_all_reported():
    def mutate(d):
        d["lessons"][0]["title"] = ""
        d["lessons"][4]["quiz"][0]["correct_option"] = "Z"
    assert len(errors_for(mutate)) == 2


# ---------------------------------------------------------------- source references

@pytest.mark.parametrize("bad_ref,fragment", [
    ({"page": 0, "quote": QUOTE}, "greater than or equal to 1"),
    ({"page": -2, "quote": QUOTE}, "greater than or equal to 1"),
    ({"page": 1, "quote": "too short"}, "at least 15 characters"),
    ({"page": 1, "quote": "x" * 301}, "at most 300 characters"),
    ({"page": 1}, "Field required"),
])
def test_malformed_source_references_are_rejected(bad_ref, fragment):
    errors = errors_for(lambda d: d["lessons"][0].update(sources=[bad_ref]))
    assert len(errors) == 1 and errors[0].startswith("lessons 1 > sources 1") and fragment in errors[0]


def test_quiz_question_needs_a_source():
    errors = errors_for(lambda d: d["lessons"][0]["quiz"][0].update(sources=[]))
    assert errors[0].startswith("lessons 1 > quiz 1 > sources:")


# ---------------------------------------------------------------- quiz options and answers

def test_correct_option_must_be_one_of_the_options():
    errors = errors_for(lambda d: d["lessons"][0]["quiz"][0].update(correct_option="E"))
    assert errors == ["lessons 1 > quiz 1: correct_option 'E' is not one of the options "
                      "['A', 'B', 'C', 'D']"]


def test_invalid_option_id_is_rejected():
    errors = errors_for(lambda d: d["lessons"][0]["quiz"][0]["options"][1].update(id="7"))
    assert errors[0].startswith("lessons 1 > quiz 1 > options 2 > id:")


def test_duplicate_option_ids_are_rejected():
    errors = errors_for(lambda d: d["lessons"][0]["quiz"][0]["options"][1].update(id="A"))
    assert "option ids must be unique" in errors[0]


def test_duplicate_option_texts_are_rejected():
    errors = errors_for(lambda d: d["lessons"][0]["quiz"][0]["options"][1].update(text="glucose"))
    assert "option texts must be different" in errors[0]


@pytest.mark.parametrize("count", [1, 7])
def test_option_count_limits(count):
    options = [{"id": "ABCDEFG"[i], "text": f"Option {i}"} for i in range(count)]
    errors = errors_for(lambda d: d["lessons"][0]["quiz"][0].update(options=options))
    assert errors[0].startswith("lessons 1 > quiz 1 > options:")


def test_question_with_two_options_is_allowed():
    q = quiz_question()
    q["options"] = q["options"][:2]
    assert QuizQuestion.model_validate(q).correct_answer.id == "A"


# ---------------------------------------------------------------- course structure

@pytest.mark.parametrize("count", [4, 6])
def test_course_must_have_five_lessons(count):
    def mutate(d):
        d["lessons"] = [lesson(n) for n in range(1, count + 1)] if count < 6 else \
            d["lessons"] + [copy.deepcopy(d["lessons"][0])]
    errors = errors_for(mutate)
    assert errors[0].startswith("lessons:")


def test_lessons_must_be_numbered_in_order():
    def mutate(d):
        d["lessons"][0]["number"], d["lessons"][1]["number"] = 2, 1
    assert "numbered 1 to 5 in order" in errors_for(mutate)[0]


def test_lesson_titles_must_be_distinct():
    errors = errors_for(lambda d: d["lessons"][1].update(title=d["lessons"][0]["title"].upper()))
    assert "lesson titles must be different" in errors[0]


def test_metadata_page_range_and_mode_are_checked():
    errors = errors_for(lambda d: d["metadata"].update(start_page=5, end_page=2, generation_mode="magic"))
    assert any(e.startswith("metadata > generation_mode:") for e in errors)
    data = course_dict()
    data["metadata"].update(start_page=5, end_page=2)
    _, errors = parse_course(data)
    assert errors == ["metadata: start_page must not be after end_page"]


def test_schema_error_is_raised_directly_by_the_model():
    with pytest.raises(ValidationError):
        Course.model_validate(json.loads('{"metadata": {}, "lessons": []}'))


def test_uncited_examples_and_practice_are_listed():
    course, _ = parse_course(course_dict())
    uncited = course.uncited_items()
    assert len(uncited) == 5 and uncited[0] == "Lesson 1, practice activity 1"
