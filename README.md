# CourseForge AI (Project Axiora)

From Textbooks to Structured Learning Experiences.

Team KayCode · Developer: Kayes Md Golam Murtuja · Track: AI for Education & Learning · SDG 4: Quality Education

## Status

MVP implemented and covered by automated tests (all offline; AI responses are mocked).
The full workflow runs in offline demo mode. **The live Gemini path has not yet been
tested against the real API**, and the app has not been deployed.

## What it does

A teacher uploads a textbook PDF, selects one chapter, and receives a draft mini-course:

- Five structured lessons with learning objectives, explanations and examples
- Practice questions and quizzes with answers and explanations
- References to the original textbook pages, checked against the extracted page text
- Teacher review and editing before any use

AI generation uses the **Google Gemini API free tier**. **Demo mode** is a separate,
explicitly selected option that builds a course offline from the extracted text by fixed
rules and uses no AI. A failed or unavailable AI request is never replaced by demo output.

## Setup (Windows, PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Developed on Python 3.14; Python 3.12 is the fallback if a dependency will not install.

### Get a free Gemini API key

1. Sign in at https://aistudio.google.com and open **Get API key**.
2. Click **Create API key**. New keys are created as auth keys; older "Standard" keys are
   no longer accepted by the Gemini API.
3. Create the key in a Google Cloud project **without a billing account**. The tier is set
   per project: a project with linked billing is on a paid tier and is charged. CourseForge
   cannot detect billing, so keep billing off for this project.

### Configure the key

Either create a `.env` file in the project folder (it is git-ignored):

```powershell
Copy-Item .env.example .env
notepad .env        # set GEMINI_API_KEY=your-key and keep LLM_PROVIDER=gemini
```

or set the variables for the current PowerShell session only:

```powershell
$env:LLM_PROVIDER = "gemini"
$env:GEMINI_API_KEY = "your-key"
```

Never commit the key or paste it into screenshots or exported files.

## Run the app

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Then open http://localhost:8501. No API key is needed to try it: click **Use the fictional
sample textbook** and choose **Demo mode**. The sample textbook is generated in memory and
its content is entirely invented.

## Run the tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

The tests mock every Gemini response and make no external requests.

## Configuration

All settings come from environment variables (or a local `.env` file); see `.env.example`.

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | Must be `gemini`; empty turns AI generation off |
| `GEMINI_API_KEY` | Gemini API key from Google AI Studio; never committed |
| `COURSEFORGE_MODEL` | Optional; must be an approved free-tier model (default `gemini-3.8-flash`) |

## Gemini free tier: what to know

- **Google controls the free-tier quotas** (requests per minute and per day, tokens per
  minute). They vary by model and can change; check your limits in Google AI Studio.
  Daily limits reset at midnight Pacific time.
- **Free-tier content may be used by Google to improve its products.** That includes the
  textbook pages you send. Do not send material you are not allowed to share.
- **No paid fallback.** CourseForge calls only the approved free-tier model, never enables
  billing and never switches to another provider. When a quota or rate limit is reached it
  stops and explains why; it does not retry automatically.
- **Bounded calls.** Each course uses at most three generation calls (the first attempt plus
  up to two corrective retries), plus one model check that generates no text. The SDK's own
  automatic retries are switched off.
- Requests use the stateless `generateContent` endpoint, so no conversation is stored
  server-side by CourseForge.

AI-generated courses are checked automatically: every source quotation must appear on its
cited page. This confirms traceability to the textbook, not that every explanation is
correct, so each course stays a draft until a teacher reviews it.

## Repository notes

`CourseForge-AI-Hackathon.pptx` and `CourseForge-AI-Hackathon.pdf` are the submitted hackathon
presentation and must not be modified.
