# Ensure NLTK data is available and force re-download if needed
import nltk
import os
NLTK_DATA_PATH = '/workspace/fastapi_project/nltk_data'
os.makedirs(NLTK_DATA_PATH, exist_ok=True)
nltk.data.path = [NLTK_DATA_PATH]
nltk.download('punkt', download_dir=NLTK_DATA_PATH, force=True)
nltk.download('wordnet', download_dir=NLTK_DATA_PATH, force=True)

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi_users import FastAPIUsers, schemas
from fastapi_users.db import SQLAlchemyUserDatabase, SQLAlchemyBaseUserTableUUID
from fastapi_users.manager import BaseUserManager
from fastapi_users.authentication import AuthenticationBackend, JWTStrategy, BearerTransport
from sqlalchemy import Column, Integer, JSON, String, Boolean, DateTime
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import uuid
from typing import List, Optional
import whisper
import openai
import re
import os
import json
from datetime import datetime
import logging
import sys
import time
from dotenv import load_dotenv
import asyncio
from concurrent.futures import ProcessPoolExecutor
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
load_dotenv(dotenv_path="/workspace/fastapi_project/frontend/env")
print("OPENAI_API_KEY loaded:", os.getenv("OPENAI_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")

# ---- LOGGING SETUP ----
logging.basicConfig(
    level=logging.DEBUG,  # Change to INFO or WARNING in production
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("coldcall")

DATABASE_URL = "sqlite+aiosqlite:///./database.db"
engine = create_async_engine(DATABASE_URL)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

COMPANY = os.getenv('COMPANY', 'Greener Living')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# --- Known categories for call categorization ---
KNOWN_CATEGORIES = [
    "Voicemail",
    "Estimate not given",
    "Gave estimate and rejected",
    "Gave estimate and set a follow-up",
    "Gave estimate and sold"
]

# --- Fuzzy match rep name helper (robust import for multiprocessing) ---
def fuzzy_match_rep_name(extracted_name, current_reps):
    try:
        from rapidfuzz import process as rapidfuzz_process, fuzz
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "rapidfuzz"])
        from rapidfuzz import process as rapidfuzz_process, fuzz
    if not extracted_name or not current_reps:
        return "Unknown Rep"
    # Normalize case and whitespace
    norm = lambda s: " ".join(s.strip().lower().split())
    extracted_name_norm = norm(extracted_name)
    current_reps_norm = [norm(rep) for rep in current_reps]
    match, score, idx = rapidfuzz_process.extractOne(
        extracted_name_norm, current_reps_norm, scorer=fuzz.ratio, score_cutoff=80
    ) or (None, 0, None)
    if match and idx is not None:
        # Return the original current_reps entry (preserve original capitalization)
        return current_reps[idx]
    return "Unknown Rep"

# --- Fuzzy match call category helper (robust import for multiprocessing) ---
def fuzzy_match_category(extracted_category, known_categories=None):
    try:
        from rapidfuzz import process as rapidfuzz_process, fuzz
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "rapidfuzz"])
        from rapidfuzz import process as rapidfuzz_process, fuzz
    if not extracted_category or not known_categories:
        return "Unknown category"
    match, score, idx = rapidfuzz_process.extractOne(
        extracted_category, known_categories, scorer=fuzz.ratio, score_cutoff=80
    ) or (None, 0, None)
    return match if match else "Unknown category"

# --- Robust Red Flag Detection Helper ---
def robust_red_flag_detection(transcript, custom_red_flags, score_cutoff=85):
    try:
        from rapidfuzz import process as rapidfuzz_process, fuzz
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "rapidfuzz"])
        from rapidfuzz import process as rapidfuzz_process, fuzz
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    if not transcript or not custom_red_flags:
        return []

    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(transcript.lower())
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]

    flag_hits = []
    for flag in custom_red_flags:
        match, score, idx = rapidfuzz_process.extractOne(
            flag.lower(), lemmas, scorer=fuzz.ratio, score_cutoff=score_cutoff
        ) or (None, 0, None)
        if match:
            flag_hits.append(flag)
    return flag_hits

class User(SQLAlchemyBaseUserTableUUID, Base):
    email = Column(String, nullable=False, unique=True, index=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    minutes = Column(Integer, default=10000, nullable=False)
    custom_red_flags = Column(JSON, default=[])

class Job(Base):
    __tablename__ = "jobs"
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, index=True)
    filename = Column(String)
    status = Column(String, default="pending")  # pending, processing, done, error
    result = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

async def get_user_db():
    async with async_session() as session:
        yield SQLAlchemyUserDatabase(session, User)

class UserRead(schemas.BaseUser[uuid.UUID]):
    minutes: int = 10000
    custom_red_flags: List[str] = []

class UserCreate(schemas.BaseUserCreate):
    email: str
    password: str
    minutes: Optional[int] = 10000
    custom_red_flags: Optional[List[str]] = []

class UserUpdate(schemas.BaseUserUpdate):
    minutes: Optional[int] = 10000
    custom_red_flags: Optional[List[str]] = []

SECRET = "CHANGE_ME_TO_A_SECURE_RANDOM_STRING"

class UserManager(BaseUserManager[User, uuid.UUID]):
    user_db_model = User
    reset_password_token_secret = SECRET
    verification_token_secret = SECRET

    async def validate_password(self, password: str, user: Optional[UserCreate] = None) -> None:
        if len(password) < 3:
            raise HTTPException(status_code=400, detail="Password too short")

    async def create(
        self,
        user_create: UserCreate,
        safe: bool = False,
        request: Optional[Request] = None,
    ) -> User:
        if not user_create.email:
            raise HTTPException(status_code=400, detail="Email is required")
        if not user_create.password:
            raise HTTPException(status_code=400, detail="Password is required")
        return await super().create(user_create, safe=safe, request=request)

    async def on_after_register(self, user: User, request: Optional[Request] = None):
        logger.info(f"User {user.id} has registered.")

    # --- REQUIRED FOR UUID PRIMARY KEYS ---
    def parse_id(self, value: str) -> uuid.UUID:
        return uuid.UUID(value)

async def get_user_manager(user_db=Depends(get_user_db)):
    yield UserManager(user_db)

bearer_transport = BearerTransport(tokenUrl="auth/jwt/login")

def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(secret=SECRET, lifetime_seconds=3600)

auth_backend = AuthenticationBackend(
    name="jwt",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
)

fastapi_users = FastAPIUsers(
    get_user_manager,
    [auth_backend],
)

current_user = fastapi_users.current_user()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def on_startup():
    logger.info("Starting FastAPI application and initializing database...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created.")

app.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/auth/jwt",
    tags=["auth"],
)

app.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/auth",
    tags=["auth"],
)

app.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate),
    prefix="/users",
    tags=["users"],
)

@app.get("/")
def root():
    logger.info("Received request to root endpoint.")
    return {"message": "Cold Call Analyzer API is running!"}

# --- Whisper/OpenAI Logic Below ---

import time
from multiprocessing import Pool
import whisper

def init_worker():
    import torch
    import whisper
    global model
    model = whisper.load_model("small")
    device = next(model.parameters()).device
    import logging
    logging.info(f"Whisper model loaded on device: {device}")

def sync_transcribe(chunk_path):
    global model
    chunk_start = time.perf_counter()
    result = model.transcribe(chunk_path)
    print(f"[Worker] Transcribed {chunk_path} in {time.perf_counter() - chunk_start:.2f}s")
    return result.get('text', '').strip()

EVAL_PROMPT_TEMPLATE = """
You are a cold call analyzer for {COMPANY}, a lawn care company. Evaluate the following sales call transcript thoroughly, providing accurate feedback using direct quotes from the transcript. Follow the format below exactly.

Overall Performance: X/10
Rep Name: [You MUST select exactly one of the following names: {CURRENT_REPS}. If no name matches, return 'Unknown Rep'. Do not invent or alter names. Only use a name from the provided list or 'Unknown Rep'.]
Call Categorization: [Choose from: {KNOWN_CATEGORIES}]
Talk-to-listen percentage: Rep: <rep_word_count>, Customer: <customer_word_count>, Calculation: (<rep_word_count>/(<rep_word_count>+<customer_word_count>))*100 = <percentage>%
No's before accept: <count>
Red Flags: Yes or No

---

Custom Red Flags Section:
{CUSTOM_RED_FLAGS_LIST}

For each custom red flag above:
- Search the transcript for any direct or paraphrased matches.
- If found, provide the direct quote(s) and brief context.
- If not found, say "Not found."

Only include flags from the provided list that were found in the transcript.

At the end of this section, output:
CustomRedFlagsFound = ["flag1", "flag2", ...]

---

System Red Flags Section:
Reason: <reason for system red flag>
- Direct quote: <direct quote if available>
- Context: <brief context>

At the end of this section, output:
SystemRedFlagsFound = ["flag1", "flag2", ...]

---

Instructions
- For "No's before accept": Count the number of times the customer says "no" (including variations like "no, thank you", "not interested", "I'm not sure", "better off talking to my husband", "I would have to discuss this with...", etc.) **before** the customer accepts the offer. Only count clear rejections or hesitations before the first acceptance.
- Only count “no” or hesitation responses that are in direct reply to the representative’s offer, pitch, or attempt to move the sale forward. Do NOT count statements about past history or unrelated topics.
  Example phrases to count as a 'no':
    - "I'm not sure"
    - "I would have to discuss with my husband"
    - "You're better off talking to him"
    - "No, thank you"
    - "Not interested"
    - "I don't want to step on his toes"
  Example NOT to count:
    - "No, I only take care of it myself." (when asked about past service, not the current offer)
- For "Talk-to-listen percentage":
    1. Count the number of words spoken by the representative and by the customer in the transcript.
    2. Show the word counts for each.
    3. Calculate the talk-to-listen percentage as (rep_words / (rep_words + customer_words)) * 100.
    4. Show your calculation and then give the final percentage (rounded to the nearest whole number).
    5. Use only the transcript lines attributed to the representative and the customer for your counts.
    6. Do *NOT* show your work for this, only return the correct result. 

---

# --- BEGIN DETAILED EVALUATION (DO NOT REMOVE) ---

Coaching Plan:
In 2-3 sentences, provide a concise, actionable coaching strategy for this rep based on this call. Focus on the most important area(s) for improvement and give specific, practical advice.

Detailed Feedback:

Persistence (X/10)
Criteria: Persistence includes overcoming initial resistance, exploring alternative angles to engage the customer, and not ending the call prematurely. It also includes being persistent in securing the sale and not giving up after initial objections.
Positive examples: Include direct quotes demonstrating effective persistence.
Missed opportunities: Highlight direct quotes or scenarios where persistence could have been improved.
Tip: Provide a clear recommendation for enhancing persistence.
Example: Offer a specific example of how the rep could have demonstrated better persistence.

Problem Exploration (X/10)
Positive examples: Quote directly from the transcript showing effective problem exploration.
Missed opportunities: Identify direct quotes where problem exploration questions were not asked or insufficiently explored.
Tip: Suggest specific open-ended questions the rep could ask.
Example: Provide a concrete example of a better question the rep could have used.

Customer History Inquiry (X/10)
Positive examples: Include direct quotes showing effective inquiry into customer history.
Missed opportunities: Indicate direct quotes where customer history was neglected or poorly addressed.
Tip: Suggest specific questions to uncover customer history effectively.
Example: Illustrate exactly what the rep could have asked about past lawn care practices.

Solution-Oriented Responses (X/10)
Positive examples: Use direct quotes demonstrating effective, solution-focused responses.
Missed opportunities: Identify direct quotes where specific solutions were not clearly articulated.
Tip: Recommend strategies for clearly linking services to customer needs.
Example: Provide a specific example of how the rep could present a clearer solution.

Value Building (X/10)
Positive examples: Quote directly from the transcript demonstrating effective value building.
Missed opportunities: Include direct quotes where value building opportunities were missed.
Tip: Suggest ways to clearly articulate {COMPANY}'s unique benefits.
Example: Offer a specific example of how to better communicate the company's value.

Objection Handling (X/10)
Criteria: Handling common objections and providing relevant information to alleviate customer concerns.
Positive examples: Quote effective handling of objections.
Missed opportunities: Identify direct quotes where objections were not addressed or poorly handled.
Tip: Recommend specific responses to common objections.
Example: Provide an exact example of a better response to a common objection.
If there are no objections, mark as 'No objections found'.

Additional Services Offered (X/10)
Positive examples: Include direct quotes showing successful mention of additional services.
Missed opportunities: Highlight direct quotes or moments when additional services could have been mentioned but were not.
Tip: Suggest specific additional services that could have been recommended.
Example: Provide an example sentence introducing additional services effectively.

Sales Closure Effectiveness (X/10)
Criteria: Attempting to close the sale, secure a commitment, or scheduling a follow-up.
Positive examples: Include direct quotes of effective attempts to close or set follow-ups.
Missed opportunities: Identify direct quotes or instances where closure or follow-up attempts were not made.
Tip: Recommend clear closing statements or questions.
Example: Offer a specific example of an effective closing statement.

Buying Signals:
Direct quotes indicating interest:
Missed buying signals:

Final Thoughts:
Areas for improvement:
Action steps of how to improve:
Encouragement:

Important Clarifications to Ensure Accuracy:
Carefully review the transcript for missed opportunities.
Quote directly from the transcript for accuracy.
Clearly distinguish between genuine lack of opportunity and overlooked actions.
Highlight specific, actionable areas of improvement.

Transcript for analysis: {transcript}
"""

# --- ASYNC/BACKGROUND WORKFLOW ---

import subprocess
import glob
import math
import wave

async def split_audio(filepath, chunk_length_sec=120):
    """
    Split audio file into chunks of chunk_length_sec seconds using ffmpeg.
    Returns a list of chunk file paths.
    """
    output_pattern = filepath.replace('.wav', '_chunk_%03d.wav')
    cmd = [
        'ffmpeg', '-y', '-i', filepath,
        '-f', 'segment',
        '-segment_time', str(chunk_length_sec),
        '-c', 'copy',
        output_pattern
    ]
    subprocess.run(cmd, check=True)
    return sorted(glob.glob(filepath.replace('.wav', '_chunk_*.wav')))

async def get_audio_duration(filepath):
    """Get duration of wav file in seconds."""
    with wave.open(filepath, 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)

async def transcribe_audio_chunk(chunk_path):
    """
    Transcribe audio chunk using the loaded Whisper model.
    """
    result = sync_transcribe(chunk_path)
    return result

try:
    from rapidfuzz import fuzz, process as rapidfuzz_process
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "rapidfuzz"])
    from rapidfuzz import fuzz, process as rapidfuzz_process

# --- Helper: Count and extract red flags for rep stats ---
# --- Robust Red Flag Count Helper ---
def count_and_extract_red_flags(job_results):
    """
    Count robust red flags from job results (including string or list-based detection).
    Returns a dict with total_red_flags.
    """
    total_red_flags = 0
    for result in job_results:
        if not result:
            continue
        rf = result.get('red_flags')
        # Count if red_flags is 'Yes' (case-insensitive string)
        if isinstance(rf, str) and rf.strip().lower() == 'yes':
            total_red_flags += 1
        # Optionally, count custom/fuzzy flags here if needed
    return {'total_red_flags': total_red_flags}

# --- Robust Red Flag Stats Endpoint ---
@app.post("/api/rep-red-flag-stats")
async def rep_red_flag_stats(request: Request):
    """
    Accepts a JSON body with a list of job.result dicts.
    Returns robust red flag stats using count_and_extract_red_flags.
    """
    try:
        data = await request.json()
        job_results = data.get("job_results") or data
        stats = count_and_extract_red_flags(job_results)
        return stats
    except Exception as e:
        import logging
        logging.error(f"Error in /api/rep-red-flag-stats: {e}")
        return {"error": str(e)}

@app.post("/api/simple-red-flag-count")
async def simple_red_flag_count(request: Request):
    """
    Accepts a JSON body with a list of job.result dicts.
    Returns the number of calls with red_flags == 'Yes'.
    """
    try:
        data = await request.json()
        job_results = data.get("job_results") or data
        count = sum(1 for result in job_results if result and str(result.get("red_flags", "")).strip().lower() == "yes")
        return {"red_flag_count": count}
    except Exception as e:
        import logging
        logging.error(f"Error in /api/simple-red-flag-count: {e}")
        return {"error": str(e)}

@app.get("/job-status/{job_id}")
async def job_status(job_id: str, user: User = Depends(current_user)):
    logger.info(f"User {user.id} checking job status for job {job_id}")
    async with async_session() as session:
        job = await session.get(Job, job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            raise HTTPException(404, "Job not found")
        logger.info(f"Job {job_id} is owned by user {job.user_id}")
        if job.user_id != str(user.id):
            logger.error(f"User {user.id} not authorized for job {job_id} (job owned by {job.user_id})")
            raise HTTPException(403, "Not authorized")
        logger.info(f"Job {job_id} status: {job.status}")
        if job.status in ("pending", "processing"):
            return {
                "status": job.status,
                "message": "Still processing. Large files may take several minutes to analyze."
            }
        return {"status": job.status, "result": job.result}

@app.post("/update-red-flags/")
async def update_red_flags(red_flags: List[str], user: User = Depends(current_user)):
    logger.info(f"User {user.id} updating red flags")
    user.custom_red_flags = red_flags
    async with async_session() as session:
        await session.merge(user)
        await session.commit()
    logger.info(f"User {user.id} red flags updated")
    return {"custom_red_flags": user.custom_red_flags}

@app.get("/red-flags/")
async def get_red_flags(user: User = Depends(current_user)):
    logger.info(f"User {user.id} retrieving red flags")
    logger.info(f"User {user.id} red flags: {user.custom_red_flags}")
    return {"custom_red_flags": user.custom_red_flags}

@app.get("/users/me")
async def get_me(user: User = Depends(current_user)):
    logger.info(f"User {user.id} retrieving user info")
    logger.info(f"User {user.id} info: {user.email}, {user.minutes}, {user.custom_red_flags}, {user.is_active}, {user.is_superuser}, {user.is_verified}")
    return {
        "id": str(user.id),
        "email": user.email,
        "minutes": user.minutes if user.minutes is not None else 10000,
        "custom_red_flags": user.custom_red_flags,
        "is_active": user.is_active,
        "is_superuser": user.is_superuser,
        "is_verified": user.is_verified,
    }

@app.post("/buy-minutes/")
async def buy_minutes(amount: int = Form(...), user: User = Depends(current_user)):
    logger.info(f"User {user.id} buying minutes")
    if amount <= 0:
        logger.error(f"Invalid amount: {amount}")
        raise HTTPException(400, "Amount must be positive")
    user.minutes = (user.minutes or 0) + amount
    async with async_session() as session:
        await session.merge(user)
        await session.commit()
    logger.info(f"User {user.id} minutes updated: {user.minutes}")
    return {"minutes": user.minutes}

@app.post("/api/generate-coaching-plan")
async def generate_coaching_plan(request: Request):
    data = await request.json()
    prompt = data.get("prompt")
    if not prompt:
        return JSONResponse(status_code=400, content={"result": "No prompt provided."})
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=6144
        )
        # Log usage and cost
        if 'usage' in response:
            prompt_tokens = response['usage'].get('prompt_tokens', 0)
            completion_tokens = response['usage'].get('completion_tokens', 0)
            input_cost = prompt_tokens * 0.000005  # $5 per 1M tokens
            output_cost = completion_tokens * 0.000015  # $15 per 1M tokens
            total_cost = input_cost + output_cost
            logger.info(f"OpenAI usage: prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}, total_cost=${total_cost:.6f}")
        coaching_plan = response.choices[0].message.content.strip()
        return {"result": coaching_plan, "usage": response.get('usage', {}), "cost": total_cost if 'usage' in response else None}
    except Exception as e:
        logger.error(f"Coaching plan generation failed: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"result": "Error generating coaching plan.", "error": str(e)})

@app.post("/upload-audio/")
async def upload_audio(
    file: UploadFile = File(...),
    custom_red_flags: str = Form('[]'),
    params: str = Form('{}'),
    current_reps: str = Form('[]'),  # Accept as a JSON string
    user: User = Depends(current_user),
):
    start_time = time.perf_counter()
    logger.info(f"User {user.id} uploading file: {file.filename}")
    if getattr(user, 'minutes', 0) is None or user.minutes <= 0:
        logger.warning(f"User {user.id} has insufficient minutes.")
        raise HTTPException(402, "Insufficient minutes")
    # Parse custom_red_flags from form field
    try:
        custom_red_flags_parsed = json.loads(custom_red_flags)
    except Exception:
        custom_red_flags_parsed = []
    try:
        reps = json.loads(current_reps) if current_reps else []
    except Exception:
        reps = []
    logger.debug(f"Custom red flags for job: {custom_red_flags_parsed}")
    job_id = str(uuid.uuid4())
    os.makedirs("./uploads", exist_ok=True)
    filepath = f"./uploads/{job_id}_{file.filename}"
    with open(filepath, "wb") as audio_file:
        audio_file.write(await file.read())
    logger.info(f"File saved to {filepath}")
    async with async_session() as session:
        job = Job(
            id=job_id,
            user_id=str(user.id),
            filename=file.filename,
            status="pending",
            result=None
        )
        session.add(job)
        await session.commit()
        logger.info(f"Job {job_id} created for user {user.id}")
    # Import here to avoid circular import
    from celery_worker import process_audio_job_task
    # Send job to Celery for background processing
    process_audio_job_task.delay(job_id, filepath, params, str(user.id), custom_red_flags_parsed, reps)
    logger.info(f"Celery background task started for job {job_id}")
    elapsed = time.perf_counter() - start_time
    logger.info(f"upload_audio process for job {job_id} took {elapsed:.2f} seconds")
    return {"job_id": job_id, "status": "pending"}

import traceback

async def process_audio_job(job_id, filepath, params, user_id, custom_red_flags, current_reps=None):
    pipeline_start = time.perf_counter()
    logger.info(f"Processing audio job {job_id} for user {user_id}")
    if current_reps is None:
        current_reps = []
    try:
        # Always define filename early for use in both success and error paths
        try:
            original_file_name = job.filename if hasattr(job, 'filename') else os.path.basename(filepath)
        except Exception:
            original_file_name = os.path.basename(filepath)
        filename = original_file_name

        # Set job to processing
        async with async_session() as session:
            job = await session.get(Job, job_id)
            job.status = "processing"
            await session.commit()

        # --- Check duration ---
        duration_start = time.perf_counter()
        duration = await get_audio_duration(filepath)
        duration_time = time.perf_counter() - duration_start
        logger.info(f"Audio file {filepath} duration: {duration:.2f} seconds (duration check took {duration_time:.2f}s)")
        chunking_start = time.perf_counter()
        chunk_length_sec = 120
        chunks = await split_audio(filepath, chunk_length_sec=chunk_length_sec)
        chunking_time = time.perf_counter() - chunking_start
        logger.info(f"Chunking took {chunking_time:.2f}s, number of chunks: {len(chunks)}")

        # --- Transcribe/analyze each chunk (Persistent Pool Parallelized) ---
        pool_start = time.perf_counter()
        max_gpu_workers = 2
        with multiprocessing.get_context("spawn").Pool(
            processes=min(max_gpu_workers, len(chunks)),
            initializer=init_worker,
        ) as pool:
            transcripts = pool.map(sync_transcribe, chunks)
        pool_time = time.perf_counter() - pool_start
        logger.info(f"All chunks transcribed in {pool_time:.2f}s")

        # --- Combine results ---
        combine_start = time.perf_counter()
        full_transcript = '\n'.join(transcripts)
        combine_time = time.perf_counter() - combine_start
        logger.info(f"Combined transcript length: {len(full_transcript)} characters (combine took {combine_time:.2f}s)")

        # --- Get formatted transcript only (MUST come first) ---
        transcript_prompt = (
            "Return only the formatted transcript of the following sales call, "
            "clearly attributing each line to either the representative or the customer. "
            "Do not include any analysis or extra commentary.\n\n"
            "For security, if any credit card numbers appear in the transcript, format them as xxxx xxxx xxxx xxxx.\n\n"
            f"Transcript:\n\"\"\"\n{full_transcript}\n\"\"\""
        )
        transcript_response = openai.ChatCompletion.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": transcript_prompt}],
            temperature=0.0,
            max_tokens=6144
        )
        formatted_transcript = transcript_response.choices[0].message.content.strip()

        # --- Run analysis pipeline (OpenAI GPT, red flag extraction, etc.) ---
        format_start = time.perf_counter()
        # Use the EVAL_PROMPT_TEMPLATE for detailed evaluation
        custom_flags_str = ", ".join(custom_red_flags) if custom_red_flags else "None"
        robust_red_flag_instructions = f"""
Custom Red Flags: [{custom_flags_str}]
- For each custom red flag, check for any mention, synonym, or related phrase in the transcript, even if phrased differently.
- If a custom red flag is triggered, provide the exact quote and a brief explanation of the context.
- If none are present, return 'None'.
"""
        eval_prompt = EVAL_PROMPT_TEMPLATE.format(
            COMPANY=COMPANY,
            transcript='''{}'''.format(full_transcript),
            CURRENT_REPS=", ".join(current_reps) if current_reps else "",
            CUSTOM_RED_FLAGS_LIST=", ".join(custom_red_flags) if custom_red_flags else "None",
            KNOWN_CATEGORIES=", ".join(KNOWN_CATEGORIES)
        )
        openai_start = time.perf_counter()
        analysis_response = openai.ChatCompletion.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0.7,
            max_tokens=6144
        )
        openai_time = time.perf_counter() - openai_start
        logger.info(f"OpenAI API call (analysis) took {openai_time:.2f}s")
        analysis_content = analysis_response.choices[0].message.content.strip()

        # --- Extract rating from analysis_content ---
        rating_match = re.search(r"(?:Overall Performance|Performance Rating):\s*([^\n\r]+)", analysis_content, re.IGNORECASE)
        if rating_match:
            rating = rating_match.group(1).strip()
        else:
            rating = "N/A"

        # --- Rep Name Extraction (original, simple robust version) ---
        name_extraction_prompt = f"""
Extract the name of the salesperson. Only return the name and nothing else. They are the person who said they are with {COMPANY}
""" + formatted_transcript
        try:
            name_extraction_response = openai.ChatCompletion.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": name_extraction_prompt}],
                temperature=0.0,
                max_tokens=6144
            )
            rep_name_clean = name_extraction_response.choices[0].message.content.strip()
            rep_name_clean = rep_name_clean.replace('"', '').replace("'", '').strip()
        except Exception as e:
            logger.error(f"Error extracting rep name: {e}")
            rep_name_clean = "Unknown"

        if rep_name_clean.lower() in ["customer", "client", "unknown", ""]:
            rep_name_clean = "Unknown"
        rep_name_sanitized = re.sub(r"\s+", "_", rep_name_clean)

        # --- Call Category Extraction (original version) ---
        def extract_call_category(text, known_categories):
            import re
            patterns = [
                r"Call Categorization\s*[:\-—=\u2013]?\s*(.+)",
                r"Call Category\s*[:\-—=\u2013]?\s*(.+)",
                r"Categorization\s*[:\-—=\u2013]?\s*(.+)"
            ]
            for pat in patterns:
                match = re.search(pat, text, re.I)
                if match:
                    candidate = match.group(1).strip()
                    candidate_clean = re.split(r'[.\n\r;,-]', candidate)[0].strip().lower()
                    fuzzy_result = fuzzy_match_category(candidate_clean, known_categories)
                    # If fuzzy_result is UnknownType or not a close match, use cleaned candidate
                    if fuzzy_result.lower() in ["unknowntype", "unknown", "unknown_type"] or not fuzzy_result or fuzzy_result == candidate_clean:
                        return candidate.strip()
                    return fuzzy_result
            # Fallback: scan all lines for 'ategorization'
            for line in text.splitlines():
                if "ategorization" in line.lower():
                    candidate = line.split(":", 1)[-1].strip()
                    candidate_clean = re.split(r'[.\n\r;,-]', candidate)[0].strip().lower()
                    fuzzy_result = fuzzy_match_category(candidate_clean, known_categories)
                    if fuzzy_result.lower() in ["unknowntype", "unknown", "unknown_type"] or not fuzzy_result or fuzzy_result == candidate_clean:
                        return candidate.strip()
                    return fuzzy_result
            return "UnknownType"

        call_type = extract_call_category(analysis_response.choices[0].message.content.strip(), KNOWN_CATEGORIES)
        call_type_sanitized = call_type.replace(" ", "_")

        red_flags = re.search(r"Red Flags:\s*(Yes|No)", analysis_response.choices[0].message.content.strip(), re.I)
        red_flags = red_flags.group(1).capitalize() if red_flags else "No"
        red_flag_reason = re.search(r"Reason for Red Flag:\s*(.+)", analysis_response.choices[0].message.content.strip())
        red_flag_reason = red_flag_reason.group(1).strip() if red_flag_reason else "None"
        red_flag_quotes = re.search(r'Red Flag Direct Quotes:\s*\["(.+)"\]', analysis_response.choices[0].message.content.strip())
        red_flag_quotes = red_flag_quotes.group(1).strip() if red_flag_quotes else "None"

        # --- Run fuzzy matching as a second pass ---
        fuzzy_flag_hits = robust_red_flag_detection(full_transcript, custom_red_flags)

        # --- Parse CustomRedFlagsFound and SystemRedFlagsFound arrays ---
        import ast
        custom_red_flags_found = []
        system_red_flags_found = []
        # CustomRedFlagsFound
        match_custom = re.search(r"CustomRedFlagsFound\s*=\s*(\[.*?\])", analysis_content, re.S)
        if match_custom:
            try:
                custom_red_flags_found = ast.literal_eval(match_custom.group(1))
            except Exception:
                custom_red_flags_found = []
        # SystemRedFlagsFound
        match_system = re.search(r"SystemRedFlagsFound\s*=\s*(\[.*?\])", analysis_content, re.S)
        if match_system:
            try:
                system_red_flags_found = ast.literal_eval(match_system.group(1))
            except Exception:
                system_red_flags_found = []

        # --- VALIDATION: Ensure system and custom red flags are not mixed ---
        # Remove any custom red flags from system reason/quotes
        if custom_red_flags:
            for flag in custom_red_flags:
                if flag.lower() in (red_flag_reason or '').lower():
                    logger.warning(f"Custom red flag '{flag}' found in system red flag reason. Removing from system reason.")
                    red_flag_reason = red_flag_reason.replace(flag, '').strip()
                if flag.lower() in (red_flag_quotes or '').lower():
                    logger.warning(f"Custom red flag '{flag}' found in system red flag quotes. Removing from system quotes.")
                    red_flag_quotes = red_flag_quotes.replace(flag, '').strip()
        # Ensure custom_red_flags_found only contains user custom flags
        if custom_red_flags_found:
            validated_custom = [flag for flag in custom_red_flags_found if any(flag.lower() == crf.lower() for crf in custom_red_flags)]
            if len(validated_custom) != len(custom_red_flags_found):
                logger.warning(f"Found non-custom flags in custom_red_flags_found: {set(custom_red_flags_found) - set(validated_custom)}. Removing.")
            custom_red_flags_found = validated_custom
        # Ensure system_red_flags_found only contains non-custom flags
        if system_red_flags_found:
            validated_system = [flag for flag in system_red_flags_found if not any(flag.lower() == crf.lower() for crf in custom_red_flags)]
            if len(validated_system) != len(system_red_flags_found):
                logger.warning(f"Found custom flags in system_red_flags_found: {set(system_red_flags_found) - set(validated_system)}. Removing.")
            system_red_flags_found = validated_system

        # --- Extract metrics from analysis_content ---
        metrics = {}
        # Extract talk-to-listen percentage (e.g., 74%)
        match_ttlr_pct = re.search(r"Talk[- ]to[- ]listen percentage:[^=]*=\s*([\d\.]+)%", analysis_content, re.IGNORECASE)
        if match_ttlr_pct:
            metrics["talk_to_listen_percentage_str"] = f"{match_ttlr_pct.group(1)}%"
            metrics["talk_to_listen_ratio"] = float(match_ttlr_pct.group(1)) / 100.0
        else:
            metrics["talk_to_listen_percentage_str"] = None
            metrics["talk_to_listen_ratio"] = None
        # Extract nos before accept (e.g., No’s before accept: 2)
        match_nos = re.search(r"No[’'`s]{0,2} before accept[:=]?\s*(\d+)", analysis_content, re.IGNORECASE)
        if match_nos:
            metrics["nos_before_accept"] = int(match_nos.group(1))
        else:
            metrics["nos_before_accept"] = None

        # --- Ensure red_flags is always a string: 'Yes' or 'No' ---
        if not isinstance(red_flags, str):
            if red_flags:
                red_flags = "Yes"
            else:
                red_flags = "No"
        red_flags = red_flags if red_flags in ["Yes", "No"] else "No"

        # --- Clean original_file_name to remove job ID/UUID prefix ---
        if '_' in original_file_name:
            original_file_name_clean = original_file_name.split('_', 1)[1]
        else:
            original_file_name_clean = original_file_name
        # --- Restore the descriptive filename format BEFORE storing result
        filename = f"{rating}_{rep_name_sanitized}_{call_type_sanitized}_Red_Flags_{red_flags}_{original_file_name_clean}.html"
        filename = filename.replace("__", "_").replace(" ", "_")

        # --- Store result in job ---
        db_start = time.perf_counter()
        full_analysis_content = (
            analysis_response.choices[0].message.content.strip()
            + "\n\n"
            + "="*60 + "\nFORMATTED TRANSCRIPT\n" + "="*60 + "\n\n"
            + formatted_transcript
            + "\n\nGenerated automatically by " + COMPANY + " AI Analysis.\n"
        )

        async with async_session() as session:
            job = await session.get(Job, job_id)
            job.status = "done"
            job.result = {
                "overall_perf": rating,
                "rep_name": rep_name_sanitized,
                "categorization": call_type,  
                "system_red_flags": red_flags,  
                "system_red_flag_reason": red_flag_reason,
                "system_red_flag_quotes": red_flag_quotes,
                "system_red_flags_count": len(system_red_flags_found),
                "system_red_flags_found": system_red_flags_found,
                "custom_red_flags_found": custom_red_flags_found,  
                "custom_red_flags_count": len(custom_red_flags_found),
                "fileName": filename,  
                "full_analysis_content": full_analysis_content,
                "title": filename,
                "talk_to_listen_percentage_str": metrics["talk_to_listen_percentage_str"],
                "talk_to_listen_ratio": metrics["talk_to_listen_ratio"],
                "nos_before_accept": metrics["nos_before_accept"],
                "fuzzy_flag_hits": fuzzy_flag_hits,
            }
            await session.commit()
        db_time = time.perf_counter() - db_start
        logger.info(f"DB write took {db_time:.2f}s")
        logger.info(f"Job {job_id} completed and result stored.")
    except Exception as e:
        logger.error(f"Audio job {job_id} failed: {e}", exc_info=True)
        print(f"ERROR processing job {job_id}: {e}")
        traceback.print_exc()  
        async with async_session() as session:
            job = await session.get(Job, job_id)
            job.status = "error"
            job.result = {
                "error": str(e),
                "fileName": filename,
                "full_analysis_content": f"Analysis failed: {str(e)}\nPlease try again or contact support."
            }
            await session.commit()
    total_pipeline_time = time.perf_counter() - pipeline_start
    logger.info(f"Total pipeline time for job {job_id}: {total_pipeline_time:.2f}s")