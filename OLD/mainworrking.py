# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded .env file with dotenv.")
except ImportError:
    print("python-dotenv not installed; skipping .env loading.")

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
import whisper
from pydub import AudioSegment
from pyannote.audio import Pipeline

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
print("OPENAI_API_KEY from env (first 10 chars):", repr(OPENAI_API_KEY[:10]))
openai.api_key = OPENAI_API_KEY

# --- END: Extremely Robust PyTorch 2.6+ checkpoint compatibility fix ---

DATABASE_URL = "sqlite+aiosqlite:///./database.db"
engine = create_async_engine(DATABASE_URL)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

COMPANY = os.environ.get("COMPANY", "Greener Living")

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

@app.options("/upload/")
async def upload_options():
    return JSONResponse(
        content={"message": "CORS preflight OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Authorization,Content-Type",
        }
    )

EVAL_PROMPT_TEMPLATE = f""" SYSTEM
You are a cold call analyzer for {COMPANY}, a lawn care company. Evaluate the following sales call transcript thoroughly, providing accurate feedback using direct quotes from the transcript. Follow the format below exactly.

Overall Performance: X/10
Rep Name: [Explicitly extract the representative's name from the transcript by identifying who introduces themselves as being "with {COMPANY}". If unclear, mark as "Unknown"]
Call Categorization: [Voicemail | Estimate not given | Gave estimate and rejected | Gave estimate and set a follow-up | Gave estimate and sold]
Main Objection: [Customer's exact words or "None"]
Red Flags: [Yes or No]
Reason for Red Flag: [Briefly state reason or "None"]
Red Flag Direct Quotes: ["Exact quote(s)" or "None"]

... (rest of your template unchanged, just replace Greener Living with {COMPANY}) ...
"""

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

    # Rep name extraction with {COMPANY}
    name_extraction_prompt = f"""
Extract the name of the salesperson. Only return the name and nothing else. They are the person who said they are with {COMPANY}"
\"\"\"
{transcript}
\"\"\"
""" 
    try:
        name_extraction_response = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": name_extraction_prompt}],
            temperature=0.0,
            max_tokens=10
        )
        rep_name = name_extraction_response["choices"][0]["message"]["content"].strip()
        rep_name_clean = rep_name.replace('"', '').replace("'", '').strip()
    except Exception:
        rep_name_clean = "Unknown"

    if rep_name_clean.lower() == "unknown" or not rep_name_clean:
        patterns = [
            rf'this is (\w+) with {COMPANY}',
            rf'this is (\w+) from {COMPANY}',
            rf'hey.*?this is (\w+).*?{COMPANY}',
            rf'hello.*?this is (\w+).*?{COMPANY}',
            rf'my name is (\w+) with {COMPANY}',
            rf'my name is (\w+) from {COMPANY}',
            rf'{COMPANY}.*?this is (\w+)',
            rf'{COMPANY}.*?my name is (\w+)',
            rf'it[’\'s]{{0,2}} (\w+) with {COMPANY}',
            rf'you[’\'re]{{0,3}} speaking with (\w+)',
            rf'speaking with (\w+) from {COMPANY}',
            rf'speaking with (\w+) with {COMPANY}',
            rf'(\w+) from {COMPANY}',
            rf'(\w+) with {COMPANY}',
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

        # Import all necessary libraries at the start
        import sys
        import os
        import inspect
        import traceback
        import soundfile as sf
        import numpy as np
        import torch
        from scipy import signal
        import whisper
        
        device = "cpu"
        
        print("\n--- AUDIO ANALYSIS DEBUG LOG ---")
        print("filepath:", filepath)
        print("File exists:", os.path.exists(filepath))
        print("File size:", os.path.getsize(filepath) if os.path.exists(filepath) else "N/A")
        print("HUGGINGFACE_TOKEN:", os.environ.get("HUGGINGFACE_TOKEN"))
        print("device:", device)
        print("Python version:", sys.version)
        print("torch version:", torch.__version__)
        print("whisper version:", getattr(whisper, '__version__', 'unknown'))

        try:
            # Load audio using soundfile
            audio, sample_rate = sf.read(filepath)
        except Exception as e:
            raise Exception(f"Failed to load audio file: {str(e)}")
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            audio = signal.resample(audio, int(len(audio) * 16000 / sample_rate))
        
        # Convert to float32 and normalize if needed
        audio = audio.astype(np.float32)
        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()

        print("Loaded audio shape:", audio.shape)
        print("Loaded audio dtype:", audio.dtype)
        print("Loaded audio min/max:", np.min(audio), np.max(audio))

        # Input validation
        if audio is None or not isinstance(audio, np.ndarray):
            print("Audio validation failed: not a numpy array or None")
            raise ValueError("Audio data is invalid or could not be loaded.")
        if audio.size == 0:
            print("Audio validation failed: empty array")
            raise ValueError("Audio file appears empty or corrupt.")
        if len(audio.shape) != 1:
            print(f"Audio validation failed: expected mono, got shape {audio.shape}")
            raise ValueError(f"Expected mono audio, got shape {audio.shape}.")

        # Defensive tensor conversion and device transfer
        try:
            audio_tensor = torch.from_numpy(audio)
        except Exception as e:
            print("Tensor conversion failed:", traceback.format_exc())
            raise ValueError(f"Failed to convert audio to tensor: {str(e)}")
        if audio_tensor is None or not isinstance(audio_tensor, torch.Tensor):
            print("Audio tensor is invalid or None before diarization.")
            raise ValueError("Audio tensor is invalid or None before diarization.")

        # Additional logging and tensor checks
        print("audio_tensor:", audio_tensor)
        print("audio_tensor dtype:", audio_tensor.dtype)
        print("audio_tensor shape:", getattr(audio_tensor, 'shape', None))
        print("audio_tensor device:", audio_tensor.device)
        print("audio_tensor min/max:", audio_tensor.min().item(), audio_tensor.max().item())
        if audio_tensor.numel() == 0:
            print("Audio tensor is empty.")
            raise ValueError("Audio tensor is empty.")
        if torch.all(audio_tensor == 0):
            print("Audio tensor is all zeros.")
            raise ValueError("Audio tensor is all zeros.")

        # 1. TRANSCRIBE AUDIO (OpenAI Whisper)
        print("--- BEGIN TRANSCRIPTION STEP ---")
        print("DEBUG: sys.path:", sys.path)
        try:
            import whisper
            print("DEBUG: whisper version:", getattr(whisper, '__version__', 'unknown'))
        except Exception as e:
            print("DEBUG: whisper import error:", e)
            raise
        print("DEBUG: OPENAI_API_KEY:", os.environ.get("OPENAI_API_KEY"))
        print("DEBUG: HUGGINGFACE_TOKEN:", os.environ.get("HUGGINGFACE_TOKEN"))
        # Advanced transcription options for better accuracy
        asr_model = whisper.load_model("medium", device="cpu")
        try:
            asr_result = asr_model.transcribe(
                filepath,
                language="en",        # Explicitly set language
                beam_size=5,           # Increase for better accuracy
                best_of=5,             # Increase for better accuracy
                temperature=0.0        # Lower for more deterministic output
            )
            print("DEBUG: asr_result keys:", list(asr_result.keys()) if hasattr(asr_result, 'keys') else type(asr_result))
            print("DEBUG: transcription text:", asr_result.get("text", "<no text>"))
        except Exception as e:
            print("ERROR during transcription:", e)
            traceback.print_exc()
            raise

        # TIP: For even better results, consider preprocessing audio to denoise, normalize, and remove silences.

        # 2. DIARIZATION (pyannote.audio)
        try:
            from pyannote.audio import Pipeline
            diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization@2.1",
                use_auth_token=os.environ.get("HUGGINGFACE_TOKEN")
            )
            diarization = diarization_pipeline(filepath)
            print("DEBUG: diarization:", diarization)
        except Exception as e:
            print("ERROR during diarization:", e)
            traceback.print_exc()
            raise

        # 3. ALIGN TRANSCRIPT SEGMENTS WITH SPEAKERS
        # This is a simple alignment based on overlapping timestamps.
        def align_segments_with_speakers(segments, diarization):
            aligned = []
            for seg in segments:
                seg_start = seg.get('start', 0)
                seg_end = seg.get('end', 0)
                speaker = None
                for turn, _, speaker_label in diarization.itertracks(yield_label=True):
                    if turn.start <= seg_start < turn.end or turn.start < seg_end <= turn.end:
                        speaker = speaker_label
                        break
                aligned.append({
                    'start': seg_start,
                    'end': seg_end,
                    'text': seg.get('text', ''),
                    'speaker': speaker
                })
            return aligned

        whisper_segments = asr_result.get('segments', [])
        aligned_segments = align_segments_with_speakers(whisper_segments, diarization)
        print("DEBUG: aligned_segments:", aligned_segments)

        # 4. BUILD TRANSCRIPT
        lines = []
        for seg in aligned_segments:
            speaker = seg.get("speaker", "Unknown")
            text = seg.get("text", "").strip()
            if text:
                lines.append(f"{speaker}: {text}")
        transcript = "\n".join(lines)
        print("DEBUG: transcript content preview:", repr(transcript[:500]))
        print("DEBUG: transcript length:", len(transcript))
        if not transcript:
            raise Exception("Transcript is empty. Please provide a valid audio file.")

        eval_prompt_dynamic = EVAL_PROMPT_TEMPLATE
        if custom_red_flags:
            custom_flags_text = "\n- " + "\n- ".join(custom_red_flags)
            eval_prompt_dynamic = EVAL_PROMPT_TEMPLATE.replace(
                "Check ***ONLY*** these explicit red flags:",
                f"Check ***ONLY*** these explicit red flags:{custom_flags_text}"
            )
        analysis_prompt = eval_prompt_dynamic.format(transcript=transcript)
        analysis_response = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": analysis_prompt}],
            temperature=0,
            max_tokens=1200,
        )
        analysis_text = analysis_response["choices"][0]["message"]["content"].strip()
        analysis_lines = analysis_text.splitlines()
        if analysis_lines and analysis_lines[0].startswith("📊 Analysis Report:"):
            analysis_text = "\n".join(analysis_lines[1:]).lstrip()
        
        extracted = extract_details(analysis_text, transcript)
        title = f"{extracted['rating']} {extracted['rep_name']} {extracted['call_type']} {os.path.basename(filepath)}.html"
        full_analysis_content = (
            analysis_text
            + "\n\n"
            + "="*60 + "\nFORMATTED TRANSCRIPT\n" + "="*60 + "\n\n"
            + transcript
            + "\n\nGenerated automatically by "
            + COMPANY + " AI Analysis.\n"
        )
        
        async with async_session() as session:
            job = await session.get(Job, job_id)
            job.status = "done"
            job.result = {
                "full_analysis_content": full_analysis_content,
                "overall_perf": extracted.get("rating", "N/A"),
                "rep_name": extracted.get("rep_name", "N/A").replace("_", " "),
                "categorization": extracted.get("call_type", "N/A").replace("_", " "),
                "red_flags": extracted.get("red_flags", "N/A"),
                "red_flag_reason": extracted.get("red_flag_reason", "N/A"),
                "red_flag_quotes": extracted.get("red_flag_quotes", "N/A"),
                "original_filename": os.path.basename(filepath),
                "transcript": transcript,
                "title": title
            }
            print("DEBUG: Job result:", job.result)
            await session.commit()
        print("--- END AUDIO ANALYSIS DEBUG LOG ---\n")
    except Exception as e:
        async with async_session() as session:
            job = await session.get(Job, job_id)
            job.status = "error"
            job.result = {"error": str(e)}
            await session.commit()
    finally:
        try:
            os.remove(filepath)
        except Exception:
            pass

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
    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1200,
    )
    return {"coaching_plan": response["choices"][0]["message"]["content"].strip()}

@app.get("/rep-stats/")
async def rep_stats(user: User = Depends(current_user)):
    async with async_session() as session:
        jobs = await session.execute(
            text("SELECT result FROM jobs WHERE user_id = :uid"),
            {"uid": str(user.id)}
        )
        jobs = jobs.fetchall()
    rep_data = {}
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
        rep = res.get("rep_name", "Unknown")
        if rep not in rep_data:
            rep_data[rep] = {
                "scores": [],
                "talk_seconds": 0,
                "listen_seconds": 0,
                "no_count": 0,
                "calls": 0,
            }
        try:
            score = int(res.get("overall_perf", "0"))
            rep_data[rep]["scores"].append(score)
        except Exception:
            pass
        transcript = res.get("transcript", "")
        rep_data[rep]["calls"] += 1
        rep_data[rep]["no_count"] += transcript.lower().count(" no ")
        rep_lines = [l for l in transcript.splitlines() if l.strip().startswith("Rep:")]
        client_lines = [l for l in transcript.splitlines() if l.strip().startswith("Client:")]
        rep_data[rep]["talk_seconds"] += len(rep_lines) * 3
        rep_data[rep]["listen_seconds"] += len(client_lines) * 3
    stats = []
    for rep, d in rep_data.items():
        avg_score = round(sum(d["scores"]) / len(d["scores"]), 2) if d["scores"] else 0
        talk_listen = round(d["talk_seconds"] / d["listen_seconds"], 2) if d["listen_seconds"] > 0 else None
        avg_no = round(d["no_count"] / d["calls"], 2) if d["calls"] else 0
        stats.append({
            "rep_name": rep,
            "average_performance": avg_score,
            "talk_to_listen_ratio": talk_listen,
            "average_nos_before_accept": avg_no,
            "calls": d["calls"],
        })
    return {"rep_stats": stats}