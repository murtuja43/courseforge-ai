# CourseForge AI (Project Axiora)

From Textbooks to Structured Learning Experiences.

Team KayCode · Developer: Kayes Md Golam Murtuja · Track: AI for Education & Learning · SDG 4: Quality Education

## Status

Early development. The environment is set up; the application itself is **not yet implemented, tested, or deployed**.

## What it will do (MVP scope)

A teacher uploads a textbook PDF, selects one chapter, and receives a draft mini-course:

- Five structured lessons with learning objectives, explanations and examples
- Practice questions and quizzes with answers and explanations
- References to the original textbook pages, checked against the extracted page text
- Teacher review and editing before any use

Without an API key, the app runs in a clearly labeled **demo mode** that builds the course
deterministically from the extracted text and uses no AI.

## Setup (Windows, PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # then add your API key and model id to .env
```

Requires Python 3.14 (tested for installation only) or 3.12.

## Configuration

All settings come from environment variables (or a local `.env` file); see `.env.example`.

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | LLM provider (`anthropic`); empty means demo mode |
| `ANTHROPIC_API_KEY` | Provider API key, never committed |
| `COURSEFORGE_MODEL` | Model identifier; verified against the provider before use |
| `COURSEFORGE_EFFORT` | Optional effort level (`low` to `max`); empty uses the model default |

AI-generated courses are checked automatically: every source quotation must appear on its
cited page. This confirms traceability to the textbook, not that every explanation is
correct, so each course stays a draft until a teacher reviews it.

## Repository notes

`CourseForge-AI-Hackathon.pptx` and `CourseForge-AI-Hackathon.pdf` are the submitted hackathon
presentation and must not be modified.
