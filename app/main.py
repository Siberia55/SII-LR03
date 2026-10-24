import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import av
from fastapi import FastAPI, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.config import load_config
from app.keyword_search import find_keywords
from app.schemas import HealthResponse, TranscribeResponse
from app.transcriber import Transcriber


MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")
MAX_AUDIO_SECONDS = 60
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

SUPPORTED_FORMATS = {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".aac", ".opus"
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Конфигурация и модель загружаются один раз при старте.
    app.state.config = load_config()
    app.state.transcriber = Transcriber(model_size=MODEL_SIZE)
    yield


app = FastAPI(
    title="SII LR03 — Транскрипция аудио",
    description="Распознавание русской речи и поиск ключевых слов",
    version="1.0.0",
    lifespan=lifespan,
)


def get_audio_duration(path: str) -> float:
    """Проверяет аудиофайл и возвращает длительность в секундах."""
    try:
        with av.open(path) as container:
            audio_streams = [
                stream for stream in container.streams
                if stream.type == "audio"
            ]

            if not audio_streams:
                raise HTTPException(
                    status_code=400,
                    detail="Файл не содержит аудиодорожку",
                )

            duration = container.duration

            if duration is not None:
                duration_seconds = duration / av.time_base
            else:
                stream = audio_streams[0]
                if stream.duration is None or stream.time_base is None:
                    raise HTTPException(
                        status_code=400,
                        detail="Не удалось определить длительность аудио",
                    )
                duration_seconds = float(stream.duration * stream.time_base)

            if duration_seconds <= 0:
                raise HTTPException(
                    status_code=400,
                    detail="Аудиофайл пустой или имеет неверную длительность",
                )

            return duration_seconds

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Не удалось декодировать аудио. Проверьте формат файла",
        ) from exc


@app.get("/health", response_model=HealthResponse)
def health():
    config = getattr(app.state, "config", None)
    transcriber = getattr(app.state, "transcriber", None)

    return {
        "status": "ok" if config is not None and transcriber is not None else "starting",
        "model_loaded": transcriber is not None,
        "scenario_id": config["scenario_id"] if config else "unknown",
    }


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()

    if not filename:
        raise HTTPException(status_code=400, detail="Не указано имя файла")

    if suffix not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=415,
            detail=f"Неподдерживаемый формат. Допустимые расширения: {', '.join(sorted(SUPPORTED_FORMATS))}",
        )

    temp_path = None

    try:
        # Сохраняем загруженный файл во временный файл с лимитом размера.
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=suffix, delete=False
        ) as temp_file:
            temp_path = temp_file.name
            total_size = 0

            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Размер аудиофайла превышает лимит 20 МБ",
                    )

                temp_file.write(chunk)

        if total_size == 0:
            raise HTTPException(
                status_code=400,
                detail="Загружен пустой файл",
            )

        duration = get_audio_duration(temp_path)

        if duration > MAX_AUDIO_SECONDS:
            raise HTTPException(
                status_code=413,
                detail="Длительность аудио превышает лимит 60 секунд",
            )

        start_time = time.perf_counter()

        result = await run_in_threadpool(
            app.state.transcriber.transcribe,
            temp_path,
        )

        latency_ms = round((time.perf_counter() - start_time) * 1000)
        config = app.state.config
        keywords_found = find_keywords(
            result["text"],
            config["keywords"],
        )

        return {
            "scenario_id": config["scenario_id"],
            "scenario_name": config["scenario_name"],
            "text": result["text"],
            "language": "ru",
            "segments": result["segments"],
            "keywords_found": keywords_found,
            "model": f"whisper-{MODEL_SIZE}",
            "model_version": "v1.0",
            "latency_ms": latency_ms,
        }

    finally:
        await file.close()
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
