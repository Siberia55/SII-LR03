from pydantic import BaseModel


class SegmentResponse(BaseModel):
    start: float
    end: float
    text: str


class KeywordResponse(BaseModel):
    keyword: str
    count: int


class TranscribeResponse(BaseModel):
    scenario_id: str
    scenario_name: str
    text: str
    language: str
    segments: list[SegmentResponse]
    keywords_found: list[KeywordResponse]
    model: str
    model_version: str
    latency_ms: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    scenario_id: str
