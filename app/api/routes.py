from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, Any, List, Optional, Literal
from app.services.email_topic_inference import EmailTopicInferenceService
from app.dataclasses import Email

from pathlib import Path
import json
import uuid
import time

# For similarity-based classification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

router = APIRouter()

# ---------- File paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # project root
DATA_DIR = BASE_DIR / "data"
TOPICS_FILE = DATA_DIR / "topic_keyword.json"
EMAILS_FILE = DATA_DIR / "emails.json"

# ---------- Pydantic Models ----------
class EmailRequest(BaseModel):
    subject: str
    body: str

class EmailWithTopicRequest(BaseModel):
    subject: str
    body: str
    topic: Optional[str] = None   # ground truth is optional

class EmailClassificationResponse(BaseModel):
    predicted_topic: str
    topic_scores: Dict[str, float]
    features: Dict[str, Any]
    available_topics: List[str]

class EmailAddResponse(BaseModel):
    message: str
    email_id: str
    ground_truth: Optional[str] = None

class TopicIn(BaseModel):
    topic: str

# ---------- Helpers ----------
def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def write_json(path: Path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

# ---------- Existing classifier endpoint ----------
@router.post("/emails/classify", response_model=EmailClassificationResponse)
async def classify_email(
    request: EmailRequest,
    mode: Literal["topic", "similarity", "auto"] = Query("topic")
):
    """
    mode=topic       -> use pipeline classifier (default, existing)
    mode=similarity  -> find most similar stored email with ground truth
    mode=auto        -> try similarity first, else pipeline
    """
    try:
        inference_service = EmailTopicInferenceService()
        email = Email(subject=request.subject, body=request.body)

        topics = read_json(TOPICS_FILE, [])
        emails = read_json(EMAILS_FILE, [])

        # --- similarity classifier ---
        def similarity_label() -> Optional[str]:
            labeled = [e for e in emails if e.get("topic")]
            if not labeled:
                return None
            corpus = [e["body"] for e in labeled]
            vect = TfidfVectorizer(min_df=1, stop_words="english")
            X = vect.fit_transform(corpus + [request.body])
            sims = cosine_similarity(X[-1], X[:-1]).flatten()
            idx = sims.argmax()
            return labeled[idx]["topic"]

        label, chosen_method = None, None
        result = None

        if mode == "similarity":
            label = similarity_label()
            chosen_method = "similarity"
        elif mode == "topic":
            result = inference_service.classify_email(email)
            label = result["predicted_topic"]
            chosen_method = "topic"
        else:  # auto
            label = similarity_label()
            if label:
                chosen_method = "similarity"
            else:
                result = inference_service.classify_email(email)
                label = result["predicted_topic"]
                chosen_method = "topic"

        if not label:
            raise HTTPException(status_code=400, detail="No data available to classify")

        # If we used pipeline, return its rich result
        if chosen_method == "topic" and result:
            return EmailClassificationResponse(
                predicted_topic=result["predicted_topic"],
                topic_scores=result["topic_scores"],
                features=result["features"],
                available_topics=result["available_topics"]
            )

        # If we used similarity, return a simpler response
        return {
            "predicted_topic": label,
            "topic_scores": {},
            "features": {},
            "available_topics": topics
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- Topics Endpoints ----------
@router.post("/topics")
async def add_topic(payload: TopicIn):
    topics = read_json(TOPICS_FILE, [])
    if payload.topic in topics:
        return {"ok": True, "message": "Topic already exists", "topic": payload.topic}
    topics.append(payload.topic)
    write_json(TOPICS_FILE, topics)
    return {"ok": True, "message": "Topic added", "topic": payload.topic, "count": len(topics)}

@router.get("/topics")
async def topics():
    inference_service = EmailTopicInferenceService()
    info = inference_service.get_pipeline_info()
    return {"topics": info["available_topics"]}

# ---------- Emails Endpoints ----------
@router.post("/emails", response_model=EmailAddResponse)
async def store_email(payload: EmailWithTopicRequest):
    emails = read_json(EMAILS_FILE, [])
    record = {
        "id": str(uuid.uuid4()),
        "subject": payload.subject,
        "body": payload.body,
        "topic": payload.topic,
        "ts": int(time.time())
    }
    emails.append(record)
    write_json(EMAILS_FILE, emails)
    return EmailAddResponse(message="Email stored", email_id=record["id"], ground_truth=record["topic"])

@router.get("/emails")
async def list_emails(limit: int = 50):
    emails = read_json(EMAILS_FILE, [])
    return {"ok": True, "count": len(emails), "items": emails[-limit:]}
