import os
import uuid
import re
import json
from datetime import datetime
from typing import List, Optional, Any
from collections import defaultdict, OrderedDict, Counter
import pathlib
import builtins

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi_users import FastAPIUsers, schemas
from fastapi_users.db import SQLAlchemyUserDatabase, SQLAlchemyBaseUserTableUUID
from fastapi_users.manager import BaseUserManager
from fastapi_users.authentication import AuthenticationBackend, JWTStrategy, BearerTransport
from sqlalchemy import Column, Integer, JSON as SAJSON, String, Boolean, DateTime, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base

import openai
import torch
import whisper  # Using regular whisper instead of whisperx
from pydub import AudioSegment

# Database and basic setup
DATABASE_URL = "sqlite+aiosqlite:///./database.db"
engine = create_async_engine(DATABASE_URL)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

COMPANY = os.environ.get("COMPANY", "Greener Living")
OPENAI_API_KEY = os.environ.get("CHATGPT_API_KEY", "")

# Initialize OpenAI client and Whisper model
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
whisper_model = whisper.load_model("small")

class User(SQLAlchemyBaseUserTableUUID, Base):
    email = Column(String, nullable=False, unique=True, index=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    minutes = Column(Integer, default=600)
    custom_red_flags = Column(SAJSON, default=[])

class Job(Base):
    __tablename__ = "jobs"
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, index=True)
    filename = Column(String)
    status = Column(String, default="pending")
    result = Column(SAJSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

async def get_user_db():
    async with async_session() as session:
        yield SQLAlchemyUserDatabase(session, User)

class UserRead(schemas.BaseUser[uuid.UUID]):
    minutes: int
    custom_red_flags: List[str]

class UserCreate(schemas.BaseUserCreate):
    email: str
    password: str
    minutes: Optional[int] = 600
    custom_red_flags: Optional[List[str]] = []

class UserUpdate(schemas.BaseUserUpdate):
    minutes: Optional[int]
    custom_red_flags: Optional[List[str]]

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
        print(f"User {user.id} has registered.")

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
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

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
    return {"message": f"{COMPANY} Cold Call Analyzer API is running!"}
    EVAL_PROMPT_TEMPLATE = f""" SYSTEM
You are a cold call analyzer for {COMPANY}, a lawn care company. Evaluate the following sales call transcript thoroughly, providing accurate feedback using direct quotes from the transcript. Follow the format below exactly.

Overall Performance: X/10
Rep Name: [Explicitly extract the representative's name from the transcript by identifying who introduces themselves as being "with {COMPANY}". If unclear, mark as "Unknown"]
Call Categorization: [Voicemail | Estimate not given | Gave estimate and rejected | Gave estimate and set a follow-up | Gave estimate and sold]
Main Objection: [Customer's exact words or "None"]
Red Flags: [Yes or No]
Reason for Red Flag: [Briefly state reason or "None"]
Red Flag Direct Quotes: ["Exact quote(s)" or "None"]

Detailed Feedback:
Persistence (X/10)
Criteria: Persistence includes overcoming initial resistance, exploring alternative angles to engage the customer, and not ending the call prematurely.
Positive examples: Include direct quotes demonstrating effective persistence.
Missed opportunities: Highlight direct quotes or scenarios where persistence could have been improved.

Problem Exploration (X/10)
Positive examples: Quote directly from the transcript showing effective problem exploration.
Missed opportunities: Identify direct quotes where problem exploration questions were not asked.

Solution-Oriented Responses (X/10)
Positive examples: Use direct quotes demonstrating effective, solution-focused responses.
Missed opportunities: Identify direct quotes where specific solutions were not clearly articulated.

Value Building (X/10)
Positive examples: Quote directly from the transcript demonstrating effective value building.
Missed opportunities: Include direct quotes where value building opportunities were missed.

Objection Handling (X/10)
Positive examples: Quote effective handling of objections.
Missed opportunities: Identify direct quotes where objections were not addressed.

Sales Closure Effectiveness (X/10)
Positive examples: Include direct quotes of effective attempts to close.
Missed opportunities: Identify instances where closure attempts were not made.

Transcript for analysis:
\"\"\"
{transcript}
\"\"\" """

def extract_details(analysis, transcript):
    rating = re.search(r"Overall Performance:\s*(\d+)/10", analysis)
    rating = rating.group(1).strip() if rating else "UnknownRating"
    call_type = re.search(r"Call Categorization:\s*(.+)", analysis)
    call_type = call_type.group(1).strip() if call_type else "UnknownType"
    call_type_sanitized = call_type.replace(" ", "_")
    red_flags = re.search(r"Red Flags:\s*(Yes|No)", analysis, re.I)
    red_flags = red_flags.group(1).capitalize() if red_flags else "No"
    red_flag_reason = re.search(r"Reason for Red Flag:\s*(.+)", analysis)
    red_flag_reason = red_flag_reason.group(1).strip() if red_flag_reason else "None"
    red_flag_quotes = re.search(r'Red Flag Direct Quotes:\s*\["(.+)"\]', analysis)
    red_flag_quotes = red_flag_quotes.group(1).strip() if red_flag_quotes else "None"

    name_extraction_prompt = f"""
Extract the name of the salesperson. Only return the name and nothing else. They are the person who said they are with {COMPANY}"
\"\"\"
{transcript}
\"\"\"
""" 
    try:
        name_extraction_response = openai_client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": name_extraction_prompt}],
            temperature=0.0
        )
        rep_name = name_extraction_response.choices[0].message.content.strip()
        rep_name_clean = rep_name.replace('"', '').replace("'", '').strip()
    except Exception:
        rep_name_clean = "Unknown"

    if rep_name_clean.lower() == "unknown" or not rep_name_clean:
        patterns = [
            f'this is (\\w+) with {COMPANY}',
            f'this is (\\w+) from {COMPANY}',
            f'hey.*?this is (\\w+).*?{COMPANY}',
            f'hello.*?this is (\\w+).*?{COMPANY}',
            f'my name is (\\w+) with {COMPANY}',
            f'my name is (\\w+) from {COMPANY}',
            f'{COMPANY}.*?this is (\\w+)',
            f'{COMPANY}.*?my name is (\\w+)',
            f'it[\'s]{{0,2}} (\\w+) with {COMPANY}',
            f'you[\'re]{{0,3}} speaking with (\\w+)',
            f'speaking with (\\w+) from {COMPANY}',
            f'speaking with (\\w+) with {COMPANY}',
            f'(\\w+) from {COMPANY}',
            f'(\\w+) with {COMPANY}',
        ]
        found = None
        for pat in patterns:
            match = re.search(pat, transcript, re.I)
            if match:
                found = match.group(1)
                break
        rep_name_clean = found if found else "Unknown"

    if rep_name_clean.lower() in ["customer", "client"]:
        rep_name_clean = "Unknown"
    rep_name_sanitized = re.sub(r"\s+", "_", rep_name_clean)
    return {
        "rating": rating,
        "call_type": call_type_sanitized,
        "red_flags": red_flags,
        "red_flag_reason": red_flag_reason,
        "red_flag_quotes": red_flag_quotes,
        "rep_name": rep_name_sanitized
    }

def get_audio_duration_minutes(filepath):
    audio = AudioSegment.from_file(filepath)
    duration_sec = len(audio) / 1000
    return max(1, int(round(duration_sec / 60.0)))

@app.post("/upload/")
async def upload_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    params: str = Form('{}'),
    user: User = Depends(current_user),
):
    params_dict = json.loads(params)
    custom_red_flags: List[str] = params_dict.get("custom_red_flags", user.custom_red_flags)
    job_id = str(uuid.uuid4())
    os.makedirs("./uploads", exist_ok=True)
    filepath = f"./uploads/{job_id}_{file.filename}"
    with open(filepath, "wb") as audio_file:
        audio_file.write(await file.read())

    minutes = get_audio_duration_minutes(filepath)
    if user.minutes < minutes:
        os.remove(filepath)
        raise HTTPException(402, f"Insufficient minutes. This file requires {minutes} minute(s).")
    user.minutes -= minutes
    async with async_session() as session:
        session.add(Job(
            id=job_id,
            user_id=str(user.id),
            filename=file.filename,
            status="pending",
            result=None
        ))
        await session.merge(user)
        await session.commit()
    background_tasks.add_task(process_audio_job, job_id, filepath, params, str(user.id), custom_red_flags)
    return {"job_id": job_id, "status": "pending"}

async def process_audio_job(job_id, filepath, params, user_id, custom_red_flags):
    try:
        async with async_session() as session:
            job = await session.get(Job, job_id)
            job.status = "processing"
            await session.commit()

        # Load and preprocess audio file with detailed error handling
        try:
            print(f"Processing file: {filepath}")  # Debug log
            
            # First, verify the file exists
            if not os.path.exists(filepath):
                raise Exception(f"File not found: {filepath}")

            # Load audio with explicit format
            try:
                audio = AudioSegment.from_file(filepath)
            except Exception as e:
                raise Exception(f"Failed to load audio file: {str(e)}")

            # Convert to WAV with specific parameters
            wav_path = filepath + ".wav"
            try:
                audio = audio.set_channels(1)  # Convert to mono
                audio = audio.set_frame_rate(16000)  # Set to 16kHz
                audio.export(wav_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
            except Exception as e:
                raise Exception(f"Failed to convert audio: {str(e)}")

            # Verify the WAV file was created
            if not os.path.exists(wav_path):
                raise Exception("WAV conversion failed")

            print(f"Successfully converted to WAV: {wav_path}")  # Debug log

            # Use whisper with explicit device placement
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"Using device: {device}")  # Debug log
                
                transcription_result = whisper_model.transcribe(
                    wav_path,
                    language="en",
                    task="transcribe",
                )
                
                if not transcription_result or "text" not in transcription_result:
                    raise Exception("Transcription failed to return valid result")
                
                transcript = transcription_result["text"].strip()
            except Exception as e:
                raise Exception(f"Transcription failed: {str(e)}")

            # Clean up temporary wav file
            try:
                os.remove(wav_path)
            except Exception as e:
                print(f"Warning: Failed to remove temporary WAV file: {str(e)}")  # Non-critical error

        except Exception as e:
            raise Exception(f"Audio processing failed: {str(e)}")

        if not transcript:
            raise Exception("Transcript is empty. Please provide a valid audio file.")

        # Rest of the function remains the same...


@app.get("/job-status/{job_id}")
async def job_status(job_id: str, user: User = Depends(current_user)):
    async with async_session() as session:
        job = await session.get(Job, job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        if job.user_id != str(user.id):
            raise HTTPException(403, "Not authorized")
        return {"status": job.status, "result": job.result}

@app.post("/update-red-flags/")
async def update_red_flags(red_flags: List[str], user: User = Depends(current_user)):
    user.custom_red_flags = red_flags
    async with async_session() as session:
        await session.merge(user)
        await session.commit()
    return {"custom_red_flags": user.custom_red_flags}

@app.get("/red-flags/")
async def get_red_flags(user: User = Depends(current_user)):
    return {"custom_red_flags": user.custom_red_flags}

@app.get("/users/me")
async def get_me(user: User = Depends(current_user)):
    return {
        "id": str(user.id),
        "email": user.email,
        "minutes": user.minutes,
        "custom_red_flags": user.custom_red_flags,
        "is_active": user.is_active,
        "is_superuser": user.is_superuser,
        "is_verified": user.is_verified,
    }

@app.post("/buy-minutes/")
async def buy_minutes(amount: int = Form(...), user: User = Depends(current_user)):
    if amount <= 0:
        raise HTTPException(400, "Amount must be positive")
    user.minutes += amount
    async with async_session() as session:
        await session.merge(user)
        await session.commit()
    return {"minutes": user.minutes}

@app.post("/coaching-plan/{rep_name}")
async def coaching_plan(rep_name: str, user: User = Depends(current_user)):
    async with async_session() as session:
        jobs = await session.execute(
            text("SELECT result FROM jobs WHERE user_id = :uid"),
            {"uid": str(user.id)}
        )
        jobs = jobs.fetchall()
    rep_transcripts = []
    for j in jobs:
        if not j[0]:
            continue
        res = j[0]
        if isinstance(res, str):
            try:
                res = json.loads(res)
            except Exception:
                continue
        if res is None:
            continue
        if res.get("rep_name", "").lower() == rep_name.lower():
            rep_transcripts.append(res.get("transcript", ""))
    if not rep_transcripts:
        raise HTTPException(404, "No transcripts found for this rep.")
    prompt = f"""You are a sales coach for {COMPANY}. Analyze the following call transcripts for rep '{rep_name}'. Identify the most impactful needs for improvement and provide actionable coaching strategies for each. Be specific and concise. List the top 3-5 areas and a strategy for each.

Transcripts:
\"\"\"
{chr(10).join(rep_transcripts)}
\"\"\"
"""
    response = openai_client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return {"coaching_plan": response.choices[0].message.content.strip()}