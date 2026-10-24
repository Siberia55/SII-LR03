from faster_whisper import WhisperModel


class Transcriber:
    """Обёртка над faster-whisper для распознавания русской речи."""

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
        )

    def transcribe(self, audio_path: str) -> dict:
        segments, info = self.model.transcribe(
            audio_path,
            language="ru",
            beam_size=1,
            vad_filter=True,
        )

        result_segments = []
        text_parts = []

        for segment in segments:
            segment_text = segment.text.strip()

            result_segments.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment_text,
            })

            if segment_text:
                text_parts.append(segment_text)

        return {
            "text": " ".join(text_parts),
            "language": info.language,
            "segments": result_segments,
        }
