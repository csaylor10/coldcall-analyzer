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

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Request, BackgroundTasks
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

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

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
    model = whisper.load_model("medium")
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
Rep Name: [Explicitly extract the representative's name from the transcript by identifying who introduces themselves as being 'with {COMPANY}'. If unclear, mark as 'Unknown']
Call Categorization: [Voicemail | Estimate not given | Gave estimate and rejected | Gave estimate and set a follow-up | Gave estimate and sold]
Red Flags: (Yes or No)
Reason for Red Flag: [If Yes, briefly explain why the call contains red flags]
Red Flag Direct Quotes: ["Direct quote from transcript showing the red flag"] (leave empty bracket [] if none)

[SYSTEM_RED_FLAGS_STRUCTURED_OUTPUT_BEGIN]
SystemRedFlagsFound = []
[SYSTEM_RED_FLAGS_STRUCTURED_OUTPUT_END]

Coaching Plan: In 2-3 sentences, provide a concise, actionable coaching strategy for this rep based on this call. Focus on the most important area(s) for improvement and give specific, practical advice.

Custom Red Flags to check for in this call:
{CUSTOM_RED_FLAGS_SECTION}
For each custom red flag above:
  - List the Custom Red Flags in 'Custom Red Flags'
  - Search the transcript for both exact matches and close paraphrases.
  - If found, output:
      - The red flag text (from the list)
      - The direct quote(s) from the transcript
      - A brief context/explanation
      - Mark as "Found"
  - If not found, output:
      - The red flag text (from the list)
      - Mark as "Not found"

[CUSTOM_RED_FLAGS_STRUCTURED_OUTPUT_BEGIN]
CustomRedFlagsFound = []
CustomRedFlagsExplanations = {{}}
[CUSTOM_RED_FLAGS_STRUCTURED_OUTPUT_END]

[Metrics]
Talk-to-listen percentage: <percent>%
No's before accept: <count>
[/Metrics]

Detailed Feedback:

[Metrics Instructions]
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
    6. IMPORTANT: On a separate line, output the result in this exact format so it can be extracted by code:
        Talk-to-listen percentage: <number>%
    7. Example:
        Talk-to-listen percentage: 72%
[/Metrics Instructions]

Custom Red Flags:
(List the custom red flags here))

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

@app.post("/upload/")
async def upload_audio(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    params: str = Form('{}'),
    user: User = Depends(current_user),
):
    start_time = time.perf_counter()
    logger.info(f"User {user.id} uploading {len(files)} files")
    if getattr(user, 'minutes', 0) is None or user.minutes <= 0:
        logger.warning(f"User {user.id} has insufficient minutes.")
        raise HTTPException(402, "Insufficient minutes")
    params_dict = json.loads(params)
    custom_red_flags: List[str] = params_dict.get("custom_red_flags", user.custom_red_flags)
    current_reps: list = params_dict.get("current_reps", [])
    os.makedirs("./uploads", exist_ok=True)
    job_ids = []

    async with async_session() as session:
        for upload in files:
            job_id = str(uuid.uuid4())
            filepath = f"./uploads/{job_id}_{upload.filename}"
            with open(filepath, "wb") as audio_file:
                audio_file.write(await upload.read())
            job = Job(
                id=job_id,
                user_id=str(user.id),
                filename=upload.filename,
                status="pending",
                result=None
            )
            session.add(job)
            background_tasks.add_task(process_audio_job, job_id, filepath, params, str(user.id), custom_red_flags, current_reps)
            job_ids.append(job_id)
        await session.commit()
    elapsed = time.perf_counter() - start_time
    logger.info(f"upload_audio processed {len(job_ids)} files in {elapsed:.2f} seconds")
    return {"job_ids": job_ids, "status": "pending"}

async def process_audio_job(job_id, filepath, params, user_id, custom_red_flags, current_reps=None):
    pipeline_start = time.perf_counter()
    logger.info(f"Processing audio job {job_id} for user {user_id}")
    try:
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

        # --- Run analysis pipeline (OpenAI GPT, red flag extraction, etc.) ---
        format_start = time.perf_counter()
        # Use the EVAL_PROMPT_TEMPLATE for detailed evaluation
        # Insert custom red flags into the prompt
        custom_red_flags_section = f"Custom Red Flags Section:\n{chr(10).join(custom_red_flags) if custom_red_flags else 'None'}\n\n"
        eval_prompt = EVAL_PROMPT_TEMPLATE.format(
            COMPANY=COMPANY,
            CUSTOM_RED_FLAGS_SECTION=custom_red_flags_section,
            transcript=full_transcript
        )
        openai_start = time.perf_counter()
        analysis_response = openai.ChatCompletion.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0.5,  # Lowered for more deterministic extraction
            max_tokens=6144
        )
        openai_time = time.perf_counter() - openai_start
        logger.info(f"OpenAI API call (analysis) took {openai_time:.2f}s")
        analysis_content = analysis_response.choices[0].message.content.strip()

        # --- Extract metrics directly from GPT response ---
        import re
        metrics = None
        def extract_metrics_from_response(response_text):
            import logging
            match = re.search(r"\[Metrics\](.*?)\[/Metrics\]", response_text, re.DOTALL | re.IGNORECASE)
            if match:
                metrics = match.group(1)
                logging.info(f"[Metrics Block Extracted]: {metrics}")
                # Try original single-line match
                ratio_match = re.search(r"talk[- ]?to[- ]?listen percentage\s*:\s*(\d+(?:\.\d+)?)%", metrics, re.IGNORECASE)
                nos_match = re.search(r"no['’`s ]*before accept\s*:\s*(\d+)", metrics, re.IGNORECASE)
                ratio_percent = None
                if ratio_match:
                    ratio_percent = float(ratio_match.group(1))
                else:
                    # Try to find 'Final percentage: <number>%' or any '<number>%' in the block
                    percent_matches = re.findall(r"(\d+(?:\.\d+)?)%", metrics)
                    if percent_matches:
                        try:
                            ratio_percent = float(percent_matches[-1])
                        except Exception:
                            ratio_percent = None
                nos = int(nos_match.group(1)) if nos_match else None
                return {
                    "talk_to_listen_ratio": ratio_percent,  # as a percentage
                    "nos_before_accept": nos
                }
            logging.warning("[Metrics Extraction Failed]: No [Metrics] block found or regex did not match.")
            return {"talk_to_listen_ratio": None, "nos_before_accept": None}
        metrics_result = extract_metrics_from_response(analysis_content)

        # --- Extremely robust extraction that handles all observed LLM output formats ---
        def extract_section(text, start_marker, end_marker):
            pattern = rf"{re.escape(start_marker)}(.*?){re.escape(end_marker)}"
            match = re.search(pattern, text, re.DOTALL)
            return match.group(1) if match else ""

        # Extract custom red flag section
        custom_section = extract_section(
            analysis_content,
            "[CUSTOM_RED_FLAGS_STRUCTURED_OUTPUT_BEGIN]",
            "[CUSTOM_RED_FLAGS_STRUCTURED_OUTPUT_END]"
        )
        
        custom_flags_found = []
        custom_flag_explanations = {}
        
        # Try to get directly from CustomRedFlagsFound
        flags_match = re.search(r'CustomRedFlagsFound\s*=\s*\[(.*?)\]', custom_section, re.DOTALL)
        if flags_match:
            for item in re.findall(r'"([^"]+)"', flags_match.group(1)):
                flag = item.strip()
                if flag and flag not in custom_flags_found:
                    custom_flags_found.append(flag)
        
        # Process explanations in multiple potential formats
        explanations_match = re.search(r'CustomRedFlagsExplanations\s*=\s*\{(.*?)\}', custom_section, re.DOTALL)
        if explanations_match:
            explanations_content = explanations_match.group(1)
            
            # Process each flag in explanations
            for flag_match in re.finditer(r'"([^"]+)"\s*:', explanations_content):
                flag = flag_match.group(1).strip()
                
                # Pattern 1: Check for explicit "Not found" status
                not_found_pattern = rf'"{re.escape(flag)}"\s*:.*?"Found"\s*:\s*"Not found"'
                if re.search(not_found_pattern, explanations_content, re.DOTALL):
                    continue
                
                # Pattern 2: Check for empty/missing Direct Quotes
                empty_quotes_pattern = rf'"{re.escape(flag)}"\s*:.*?"Direct Quotes"\s*:\s*\[\s*\]'
                if re.search(empty_quotes_pattern, explanations_content, re.DOTALL):
                    continue
                    
                # If we've reached here, assume this flag is found unless proven otherwise
                flag_found = True
                
                # Add to found flags if not already there
                if flag not in custom_flags_found:
                    custom_flags_found.append(flag)
                
                # Try to extract explanation from "Context" field (highest priority)
                context_match = re.search(rf'"{re.escape(flag)}"\s*:.*?"Context"\s*:\s*"([^"]+)"', 
                                         explanations_content, re.DOTALL)
                if context_match:
                    custom_flag_explanations[flag] = context_match.group(1)
                    continue
                    
                # If Context has a different format (without quotes)
                context_alt_match = re.search(rf'"{re.escape(flag)}"\s*:.*?"Context"\s*:\s*([^\}}]+)(?:,|\}})', 
                                             explanations_content, re.DOTALL)  
                if context_alt_match:
                    context_text = context_alt_match.group(1).strip()
                    if context_text and context_text != "" and not context_text.startswith('"'):
                        custom_flag_explanations[flag] = context_text
                        continue
                
                # Try getting the custom flag explanation via the specific format in recent examples
                specific_format_match = re.search(rf'"{re.escape(flag)}"\s*:\s*\{{\s*"Direct Quotes"\s*:\s*\[(.*?)\],\s*"Context"\s*:\s*"([^"]+)"', explanations_content, re.DOTALL)
                if specific_format_match and specific_format_match.group(2).strip():
                    custom_flag_explanations[flag] = specific_format_match.group(2).strip()
                    continue
                
                # Try to extract from Direct Quotes if Context failed
                quotes_match = re.search(rf'"{re.escape(flag)}"\s*:.*?"Direct Quotes"\s*:\s*\[(.*?)\]', 
                                       explanations_content, re.DOTALL)
                if quotes_match and quotes_match.group(1).strip():
                    custom_flag_explanations[flag] = f"Found in: {quotes_match.group(1).strip()}"
                    continue
                
                # Fallback if nothing else worked
                custom_flag_explanations[flag] = "Flag identified but no explanation provided."

        # --- Filter out all metadata fields being mistakenly counted as flags ---
        metadata_fields = ["direct quotes", "context", "context/explanation", "found", "not found", 
                           "quotes", "explanation", "context/explanation"]
        custom_flags_found = [flag for flag in custom_flags_found if flag.lower() not in metadata_fields]
        
        # Also filter out any empty flags
        custom_flags_found = [flag for flag in custom_flags_found if flag.strip()]
        
        # --- Strictly extract system red flags from the dedicated section ---
        system_section = extract_section(
            analysis_content,
            "[SYSTEM_RED_FLAGS_STRUCTURED_OUTPUT_BEGIN]",
            "[SYSTEM_RED_FLAGS_STRUCTURED_OUTPUT_END]"
        )
        system_flags_found = []
        system_flags_match = re.search(r'SystemRedFlagsFound\s*=\s*\[(.*?)\]', system_section, re.DOTALL)
        if system_flags_match:
            for line in re.findall(r'"([^"]+)"', system_flags_match.group(1)):
                flag = line.strip()
                if flag:
                    system_flags_found.append(flag)

        # Remove any custom flags from system flags
        system_flags_found = [flag for flag in system_flags_found if flag not in custom_flags_found]

        # --- Get formatted transcript only ---
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

        # --- Remove old Python logic for talk_to_listen_ratio and nos_before_accept ---
        # (No calculation of rep_words/customer_words or nos_before_accept here)

        # Ultra-robust extraction for overall performance rating
        # Matches: 'Overall Performance: 8/10', 'Performance Score - 7 / 10', 'Result = 7/10', etc.
        rating = re.search(
            r"""(?ixum)           # Ignore case, verbose, unicode, multiline
            (?:overall\s*)?       # Optional 'overall'
            (?:call\s*)?          # Optional 'call'
            (?:performance|rating|score|result|perfomance|performnce)  # Synonyms and typos
            [\s\u00A0:：\-–—.=|]*  # Flexible separator(s)
            (\d{1,2})             # 1-2 digit rating
            \s*[/\\]\s*           # Slash or backslash with optional spaces
            10                    # Out of 10
            """,
            analysis_response.choices[0].message.content.strip()
        )
        rating = rating.group(1).strip() if rating else "UnknownRating"
        # --- Robust Call Categorization Extraction ---
        call_type_match = re.search(r"Call Categorization:\\s*(.+)", analysis_response.choices[0].message.content.strip())
        if call_type_match:
            call_type = call_type_match.group(1).strip()
        else:
            # Fallback: look for any line containing 'ategorization'
            call_type = None
            for line in analysis_response.choices[0].message.content.strip().splitlines():
                if "ategorization" in line:
                    call_type = line.split(":", 1)[-1].strip()
                    break
            if not call_type:
                call_type = "UnknownType"
        call_type_sanitized = call_type.replace(" ", "_")
        red_flags = re.search(r"Red Flags:\\s*(Yes|No)", analysis_response.choices[0].message.content.strip(), re.I)
        red_flags = red_flags.group(1).capitalize() if red_flags else "No"
        red_flag_reason = re.search(r"Reason for Red Flag:\\s*(.+)", analysis_response.choices[0].message.content.strip())
        red_flag_reason = red_flag_reason.group(1).strip() if red_flag_reason else "None"
        red_flag_quotes = re.search(r'Red Flag Direct Quotes:\\s*\["(.+)"\]', analysis_response.choices[0].message.content.strip())
        red_flag_quotes = red_flag_quotes.group(1).strip() if red_flag_quotes else "None"

        # --- EXTREMELY ROBUST Rep Name Extraction ---
        # Use current_reps for name extraction prompt
        name_list_str = '\n'.join(current_reps) if current_reps else '[No reps provided]'
        name_extraction_prompt = f'''
Extract the name of the salesperson from the following transcript. Only return the name and nothing else.\n\nChoose the name ONLY from this list of possible representatives:\n{name_list_str}\n\nIf you cannot confidently match a name from the list, return "Unknown Rep".\n\nTranscript:\n{formatted_transcript}
'''
        try:
            name_extraction_response = openai.ChatCompletion.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": name_extraction_prompt}],
                temperature=0.0,
                max_tokens=6144
            )
            rep_name = name_extraction_response.choices[0].message.content.strip()
            rep_name_clean = rep_name.replace('"', '').replace("'", '').strip()
        except Exception as e:
            logger.error(f"Error extracting rep name: {e}")
            rep_name_clean = "Unknown Rep"

        # Fuzzy match to current_reps for extra robustness
        rep_name_sanitized = fuzzy_match_rep_name(rep_name_clean, current_reps)

        if rep_name_sanitized.lower() in ["customer", "client", "unknown", "unknown rep", "none", ""]:
            rep_name_sanitized = "Unknown Rep"
        rep_name_sanitized = re.sub(r"\s+", "_", rep_name_sanitized)

        # --- Store result in job ---
        db_start = time.perf_counter()
        # Remove job_id from the downloadable filename (use original filename only)
        original_filename = files[0].filename if 'files' in locals() else os.path.basename(filepath).split('_', 1)[-1]
        title = f"{rating} {rep_name_sanitized} {call_type_sanitized} {original_filename}.html"
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
                "categorization": call_type,  # <-- Ensure this is stored
                "red_flags": red_flags,
                "red_flag_reason": red_flag_reason,
                "red_flag_quotes": red_flag_quotes,
                "custom_red_flags_found": custom_flags_found,
                "custom_red_flags_explanations": custom_flag_explanations,
                "system_red_flags_found": system_flags_found,
                "full_analysis_content": full_analysis_content,
                "title": title,
                "fileName": title,  # <-- Add robust filename for frontend download
                "talk_to_listen_ratio": metrics_result["talk_to_listen_ratio"],
                "nos_before_accept": metrics_result["nos_before_accept"]
            }
            await session.commit()
        db_time = time.perf_counter() - db_start
        logger.info(f"DB write took {db_time:.2f}s")
        logger.info(f"Job {job_id} completed and result stored.")
    except Exception as e:
        logger.error(f"Audio job {job_id} failed: {e}", exc_info=True)
        async with async_session() as session:
            job = await session.get(Job, job_id)
            job.status = "error"
            job.result = {"error": str(e)}
            await session.commit()
    total_pipeline_time = time.perf_counter() - pipeline_start
    logger.info(f"Total pipeline time for job {job_id}: {total_pipeline_time:.2f}s")

@app.get("/job-status/{job_id}")
async def job_status(job_id: str, user: User = Depends(current_user)):
    logger.info(f"User {user.id} checking job status for job {job_id}")
    async with async_session() as session:
        job = await session.get(Job, job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            raise HTTPException(404, "Job not found")
        if job.user_id != str(user.id):
            logger.error(f"User {user.id} not authorized for job {job_id}")
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
            model="gpt-4.1-mini",
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