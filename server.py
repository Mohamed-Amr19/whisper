import os
import re
import shutil
import uuid
from pathlib import Path

import torch
import yt_dlp
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from pydantic import BaseModel, Field
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMP_DIR = BASE_DIR / "tmp"
TEMP_DIR.mkdir(exist_ok=True)

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
SUMMARIZER_MODEL_NAME = os.getenv("SUMMARIZER_MODEL_NAME", "google/flan-t5-small")
SUMMARY_INPUT_TOKENS = int(os.getenv("SUMMARY_INPUT_TOKENS", "384"))
SUMMARY_MAX_NEW_TOKENS = int(os.getenv("SUMMARY_MAX_NEW_TOKENS", "96"))
YTDLP_COOKIES_FILE = os.getenv("YTDLP_COOKIES_FILE", "").strip()
YTDLP_REMOTE_COMPONENTS = {
    component.strip()
    for component in os.getenv("YTDLP_REMOTE_COMPONENTS", "ejs:github").split(",")
    if component.strip()
}
YTDLP_FORMAT_CANDIDATES = [
    "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best",
    "bestaudio/best",
    "best",
]

app = FastAPI(title="Whisper YouTube Summarizer")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

print(f"Loading Whisper model '{WHISPER_MODEL_SIZE}' on CPU...")
transcription_model = WhisperModel(
    WHISPER_MODEL_SIZE,
    device="cpu",
    compute_type=WHISPER_COMPUTE_TYPE,
)
print("Whisper model loaded successfully!")

summary_tokenizer = None
summary_model = None


class UrlRequest(BaseModel):
    url: str = Field(..., min_length=5)


class SummaryRequest(BaseModel):
    text: str = Field(..., min_length=20)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def cleanup_path(path: Path | None) -> None:
    if path and path.exists():
        path.unlink()


def cleanup_with_stem(stem: str) -> None:
    for item in TEMP_DIR.glob(f"{stem}.*"):
        if item.is_file():
            item.unlink()


def get_ytdlp_options(
    output_template: str,
    cookie_file: Path | None = None,
    format_selector: str | None = None,
) -> dict:
    ydl_options = {
        "format": format_selector or YTDLP_FORMAT_CANDIDATES[0],
        "outtmpl": output_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }

    if cookie_file is not None:
        ydl_options["cookiefile"] = str(cookie_file)
    if YTDLP_REMOTE_COMPONENTS:
        ydl_options["remote_components"] = set(YTDLP_REMOTE_COMPONENTS)

    return ydl_options


def prepare_ytdlp_cookie_copy(file_stem: str) -> Path | None:
    if not YTDLP_COOKIES_FILE:
        return None

    source_cookie_path = Path(YTDLP_COOKIES_FILE)
    if not source_cookie_path.exists():
        raise ValueError(
            f"Configured cookie file was not found: {source_cookie_path}. "
            "Update YTDLP_COOKIES_FILE or mount the file into the container."
        )

    cookie_copy_path = TEMP_DIR / f"{file_stem}.cookies.txt"
    shutil.copy2(source_cookie_path, cookie_copy_path)
    return cookie_copy_path


def is_unavailable_format_error(message: str) -> bool:
    return "Requested format is not available" in message


def get_summary_components():
    global summary_model, summary_tokenizer

    if summary_model is None or summary_tokenizer is None:
        print(f"Loading summarizer '{SUMMARIZER_MODEL_NAME}' on CPU...")
        summary_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL_NAME)
        summary_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL_NAME)
        summary_model.eval()
        print("Summarizer model loaded successfully!")

    return summary_tokenizer, summary_model


def build_summary_prompt(text: str) -> str:
    return (
        "Summarize the following transcript into a concise, readable summary. "
        "Include the main ideas, notable details, and action items when present.\n\n"
        f"{text}"
    )


def prompt_token_length(tokenizer, text: str) -> int:
    prompt = build_summary_prompt(text)
    return len(tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"])


def split_long_sentence(sentence: str, tokenizer) -> list[str]:
    words = sentence.split()
    if not words:
        return []

    chunks = []
    current_words = []

    for word in words:
        candidate = " ".join(current_words + [word]).strip()
        if prompt_token_length(tokenizer, candidate) <= SUMMARY_INPUT_TOKENS:
            current_words.append(word)
            continue

        if current_words:
            chunks.append(" ".join(current_words).strip())
        current_words = [word]

    if current_words:
        chunks.append(" ".join(current_words).strip())

    return chunks


def split_text_into_chunks(text: str, tokenizer) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    chunks = []
    current_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        candidate = " ".join(current_sentences + [sentence]).strip()
        if candidate and prompt_token_length(tokenizer, candidate) <= SUMMARY_INPUT_TOKENS:
            current_sentences.append(sentence)
            continue

        if current_sentences:
            chunks.append(" ".join(current_sentences).strip())
            current_sentences = []

        if prompt_token_length(tokenizer, sentence) <= SUMMARY_INPUT_TOKENS:
            current_sentences = [sentence]
            continue

        long_sentence_chunks = split_long_sentence(sentence, tokenizer)
        if long_sentence_chunks:
            chunks.extend(long_sentence_chunks[:-1])
            current_sentences = [long_sentence_chunks[-1]]

    if current_sentences:
        chunks.append(" ".join(current_sentences).strip())

    return [chunk for chunk in chunks if chunk]


def summarize_chunk(text: str, tokenizer, model) -> str:
    prompt = build_summary_prompt(text)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=SUMMARY_INPUT_TOKENS,
    )

    with torch.inference_mode():
        summary_tokens = model.generate(
            **encoded,
            max_new_tokens=SUMMARY_MAX_NEW_TOKENS,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    return normalize_text(tokenizer.decode(summary_tokens[0], skip_special_tokens=True))


def summarize_text(text: str) -> dict:
    cleaned_text = normalize_text(text)
    if not cleaned_text:
        raise ValueError("Nothing to summarize.")

    tokenizer, model = get_summary_components()
    original_chunks = split_text_into_chunks(cleaned_text, tokenizer)
    if not original_chunks:
        raise ValueError("Nothing to summarize.")

    if len(original_chunks) == 1:
        return {
            "summary": summarize_chunk(original_chunks[0], tokenizer, model),
            "chunk_count": 1,
            "model": SUMMARIZER_MODEL_NAME,
        }

    round_inputs = original_chunks
    combined = ""

    for _ in range(4):
        round_summaries = [summarize_chunk(chunk, tokenizer, model) for chunk in round_inputs]
        combined = normalize_text(" ".join(round_summaries))
        next_round = split_text_into_chunks(combined, tokenizer)

        if len(next_round) <= 1:
            final_input = next_round[0] if next_round else combined
            return {
                "summary": summarize_chunk(final_input, tokenizer, model),
                "chunk_count": len(original_chunks),
                "model": SUMMARIZER_MODEL_NAME,
            }

        round_inputs = next_round

    return {
        "summary": combined,
        "chunk_count": len(original_chunks),
        "model": SUMMARIZER_MODEL_NAME,
    }


def transcribe_audio_file(audio_path: Path) -> dict:
    segments, info = transcription_model.transcribe(str(audio_path), beam_size=5)

    transcript = []
    full_text_parts = []
    for segment in segments:
        cleaned = segment.text.strip()
        transcript.append(
            {
                "start": segment.start,
                "end": segment.end,
                "text": cleaned,
            }
        )
        if cleaned:
            full_text_parts.append(cleaned)

    full_text = normalize_text(" ".join(full_text_parts))
    if not full_text:
        raise ValueError("The audio was processed, but no speech was detected.")

    return {
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "segments": transcript,
        "full_text": full_text,
        "word_count": len(full_text.split()),
    }


def download_youtube_audio(url: str) -> tuple[Path, dict, str]:
    normalized_url = url.strip()
    if not normalized_url:
        raise ValueError("A YouTube URL is required.")

    if "://" not in normalized_url:
        normalized_url = f"https://{normalized_url}"

    file_stem = uuid.uuid4().hex
    output_template = str(TEMP_DIR / f"{file_stem}.%(ext)s")
    cookie_copy_path = prepare_ytdlp_cookie_copy(file_stem)
    info = None
    last_error_message = ""

    try:
        for format_selector in YTDLP_FORMAT_CANDIDATES:
            ydl_options = get_ytdlp_options(
                output_template,
                cookie_copy_path,
                format_selector,
            )
            try:
                with yt_dlp.YoutubeDL(ydl_options) as downloader:
                    info = downloader.extract_info(normalized_url, download=True)
                break
            except yt_dlp.utils.DownloadError as exc:
                message = str(exc)
                last_error_message = message

                if "Sign in to confirm you're not a bot" in message:
                    if YTDLP_COOKIES_FILE:
                        raise ValueError(
                            "YouTube blocked the request even with the configured cookies file. "
                            "Refresh the exported YouTube cookies on the VPS and try again."
                        ) from exc
                    raise ValueError(
                        "YouTube is asking for a bot-check. Export your YouTube cookies to a "
                        "cookies.txt file, mount it into the container, and set YTDLP_COOKIES_FILE "
                        "to that path."
                    ) from exc

                if is_unavailable_format_error(message):
                    continue

                raise ValueError(f"yt-dlp failed to download the video: {message}") from exc

        if info is None:
            raise ValueError(
                "yt-dlp could not find a downloadable format for this video. "
                f"Last error: {last_error_message}"
            )
    finally:
        cleanup_path(cookie_copy_path)

    audio_path = (TEMP_DIR / file_stem).with_suffix(".mp3")
    if not audio_path.exists():
        matches = list(TEMP_DIR.glob(f"{file_stem}.*"))
        if not matches:
            raise FileNotFoundError("Unable to find the downloaded audio file.")
        audio_path = matches[0]

    metadata = {
        "title": info.get("title"),
        "channel": info.get("channel") or info.get("uploader"),
        "webpage_url": info.get("webpage_url") or normalized_url,
        "thumbnail": info.get("thumbnail"),
        "duration": info.get("duration"),
    }
    return audio_path, metadata, file_stem


@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/transcribe")
async def transcribe_uploaded_audio(file: UploadFile = File(...)):
    suffix = Path(file.filename or "audio.bin").suffix or ".bin"
    temp_path = TEMP_DIR / f"{uuid.uuid4().hex}{suffix}"

    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return transcribe_audio_file(temp_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        cleanup_path(temp_path)


@app.post("/transcribe-url")
async def transcribe_youtube_url(request: UrlRequest):
    audio_path = None
    file_stem = None

    try:
        audio_path, metadata, file_stem = download_youtube_audio(request.url)
        result = transcribe_audio_file(audio_path)
        result["source"] = {"type": "youtube", "url": request.url, **metadata}
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        cleanup_path(audio_path)
        if file_stem:
            cleanup_with_stem(file_stem)


@app.post("/summarize")
async def summarize_transcript(request: SummaryRequest):
    try:
        summary = summarize_text(request.text)
        summary["source_word_count"] = len(normalize_text(request.text).split())
        return summary
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/transcribe-url-and-summarize")
async def transcribe_and_summarize_youtube_url(request: UrlRequest):
    audio_path = None
    file_stem = None

    try:
        audio_path, metadata, file_stem = download_youtube_audio(request.url)
        result = transcribe_audio_file(audio_path)
        result["summary"] = summarize_text(result["full_text"])
        result["source"] = {"type": "youtube", "url": request.url, **metadata}
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        cleanup_path(audio_path)
        if file_stem:
            cleanup_with_stem(file_stem)


@app.get("/health")
def health_check():
    return {
        "status": "active",
        "transcription_model": WHISPER_MODEL_SIZE,
        "summarizer_model": SUMMARIZER_MODEL_NAME,
        "summarizer_loaded": summary_model is not None,
        "youtube_cookies_configured": bool(YTDLP_COOKIES_FILE),
        "youtube_remote_components": sorted(YTDLP_REMOTE_COMPONENTS),
    }
