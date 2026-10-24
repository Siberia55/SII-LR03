import io
import wave

from fastapi.testclient import TestClient

import app.main as main


class FakeTranscriber:
    """Заменяет Whisper во время тестов."""

    def __init__(self, model_size="base"):
        self.model_size = model_size

    def transcribe(self, audio_path):
        return {
            "text": "Я забронировал отель и купил билет на тур. Экскурсия включена.",
            "language": "ru",
            "segments": [
                {
                    "start": 0.0,
                    "end": 3.0,
                    "text": "Я забронировал отель и купил билет на тур. Экскурсия включена.",
                }
            ],
        }


def make_wav(duration_seconds=1):
    """Создаёт короткий WAV-файл для тестирования."""
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(
            b"\x00\x00" * 16000 * duration_seconds
        )

    return buffer.getvalue()


def test_health(monkeypatch):
    monkeypatch.setattr(main, "Transcriber", FakeTranscriber)

    with TestClient(main.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True
    assert response.json()["scenario_id"] == "tourism"


def test_transcribe_success(monkeypatch):
    monkeypatch.setattr(main, "Transcriber", FakeTranscriber)

    with TestClient(main.app) as client:
        response = client.post(
            "/transcribe",
            files={
                "file": (
                    "sample.wav",
                    make_wav(),
                    "audio/wav",
                )
            },
        )

    assert response.status_code == 200

    data = response.json()
    assert data["scenario_id"] == "tourism"
    assert data["language"] == "ru"
    assert data["model"] == "whisper-base"
    assert data["text"]
    assert data["latency_ms"] >= 0

    found = {item["keyword"]: item["count"]
             for item in data["keywords_found"]}

    assert found["отель"] == 1
    assert found["билет"] == 1
    assert found["экскурсия"] == 1


def test_empty_file(monkeypatch):
    monkeypatch.setattr(main, "Transcriber", FakeTranscriber)

    with TestClient(main.app) as client:
        response = client.post(
            "/transcribe",
            files={"file": ("empty.wav", b"", "audio/wav")},
        )

    assert response.status_code == 400


def test_unsupported_format(monkeypatch):
    monkeypatch.setattr(main, "Transcriber", FakeTranscriber)

    with TestClient(main.app) as client:
        response = client.post(
            "/transcribe",
            files={"file": ("document.txt", b"hello", "text/plain")},
        )

    assert response.status_code == 415


def test_audio_too_long(monkeypatch):
    monkeypatch.setattr(main, "Transcriber", FakeTranscriber)

    with TestClient(main.app) as client:
        response = client.post(
            "/transcribe",
            files={
                "file": (
                    "long.wav",
                    make_wav(61),
                    "audio/wav",
                )
            },
        )

    assert response.status_code == 413
