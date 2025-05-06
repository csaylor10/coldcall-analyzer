from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Request, BackgroundTasks
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

DATABASE_URL = "sqlite+aiosqlite:///./database.db"
engine = create_async_engine(DATABASE_URL)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

class User(SQLAlchemyBaseUserTableUUID, Base):
    email = Column(String, nullable=False, unique=True, index=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    credits = Column(Integer, default=10000)
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
    credits: int
    custom_red_flags: List[str]

class UserCreate(schemas.BaseUserCreate):
    email: str
    password: str
    credits: Optional[int] = 10000
    custom_red_flags: Optional[List[str]] = []

class UserUpdate(schemas.BaseUserUpdate):
    credits: Optional[int]
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

# --- CORS MIDDLEWARE: MUST COME FIRST ---
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
    return {"message": "Cold Call Analyzer API is running!"}

# --- Explicit CORS Preflight Handler for /upload/ ---
from fastapi.responses import JSONResponse

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

# --- Whisper/OpenAI Logic Below ---

openai.api_key = "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"
whisper_model = whisper.load_model("small")

EVAL_PROMPT_TEMPLATE = """ SYSTEM
You are a cold call analyzer for Greener Living, a lawn care company. Evaluate the following sales call transcript thoroughly, providing accurate feedback using direct quotes from the transcript. Follow the format below exactly.

Overall Performance: X/10
Rep Name: [Explicitly extract the representative's name from the transcript by identifying who introduces themselves as being "with Greener Living". If unclear, mark as "Unknown"]
Call Categorization: [Voicemail | Estimate not given | Gave estimate and rejected | Gave estimate and set a follow-up | Gave estimate and sold]
Main Objection: [Customer's exact words or "None"]
Red Flags: [Yes or No]
Reason for Red Flag: [Briefly state reason or "None"]
Red Flag Direct Quotes: ["Exact quote(s)" or "None"]

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
Tip: Suggest ways to clearly articulate Greener Living's unique benefits.
Example: Offer a specific example of how to better communicate the company's value.

Objection Handling (X/10)
Criteria: Handling common objections and providing relevant information to alleviate customer concerns.
Positive examples: Quote effective handling of objections.
Missed opportunities: Identify direct quotes where objections were not addressed or poorly handled.
Tip: Recommend specific responses to common objections.
Example: Provide an exact example of a better response to a common objection.
If there are no objections, mark as "No objections found".

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

    # --- EXTREMELY ROBUST Rep Name Extraction ---
    name_extraction_prompt = f"""
Extract the name of the salesperson. Only return the name and nothing else. They are the person who said they are with Greener Living"
            
\"\"\"
{transcript}
\"\"\"
""" 

    try:
        name_extraction_response = openai.ChatCompletion.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": name_extraction_prompt}],
            temperature=0.0
        )
        rep_name = name_extraction_response.choices[0].message['content'].strip()
        rep_name_clean = rep_name.replace('"', '').replace("'", '').strip()
    except Exception as e:
        rep_name_clean = "Unknown"

    if rep_name_clean.lower() == "unknown" or not rep_name_clean:
        patterns = [
            r'this is (\w+) with Greener Living',
            r'this is (\w+) from Greener Living',
            r'hey.*?this is (\w+).*?Greener Living',
            r'hello.*?this is (\w+).*?Greener Living',
            r'my name is (\w+) with Greener Living',
            r'my name is (\w+) from Greener Living',
            r'Greener Living.*?this is (\w+)',
            r'Greener Living.*?my name is (\w+)',
            r'it[’\'s]{0,2} (\w+) with Greener Living',
            r'you[’\'re]{0,3} speaking with (\w+)',
            r'speaking with (\w+) from Greener Living',
            r'speaking with (\w+) with Greener Living',
            r'(\w+) from Greener Living',
            r'(\w+) with Greener Living',
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

# --- ASYNC/BACKGROUND WORKFLOW ---

@app.post("/upload/")
async def upload_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    params: str = Form('{}'),
    user: User = Depends(current_user),
):
    if user.credits <= 0:
        raise HTTPException(402, "Insufficient credits")

    params_dict = json.loads(params)
    custom_red_flags: List[str] = params_dict.get("custom_red_flags", user.custom_red_flags)

    job_id = str(uuid.uuid4())
    os.makedirs("./uploads", exist_ok=True)
    filepath = f"./uploads/{job_id}_{file.filename}"
    with open(filepath, "wb") as audio_file:
        audio_file.write(await file.read())

    # Create job entry in DB
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

    # Start background processing
    background_tasks.add_task(process_audio_job, job_id, filepath, params, str(user.id), custom_red_flags)

    return {"job_id": job_id, "status": "pending"}

async def process_audio_job(job_id, filepath, params, user_id, custom_red_flags):
    try:
        # Update job to processing
        async with async_session() as session:
            job = await session.get(Job, job_id)
            job.status = "processing"
            await session.commit()

        # --- Insert your existing audio processing/transcription logic here ---
        transcription_result = whisper_model.transcribe(filepath)
        transcript = transcription_result.get('text', '').strip()

        if not transcript:
            raise Exception("Transcript is empty. Please provide a valid audio file.")

        format_prompt = f'''
Organize the following transcript into a structured conversation. Label the lawn care representative as "Rep" and the customer as "Client".

Transcript:
"""
{transcript}
"""
'''

        formatted_response = openai.ChatCompletion.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": format_prompt}],
            temperature=0
        )

        formatted_transcript = formatted_response.choices[0].message['content'].strip()
        cleaned_transcript = re.sub(r'^(Rep|Client):\s*', '', formatted_transcript, flags=re.MULTILINE)

        # --- FIXED custom red flags block ---
        if custom_red_flags:
            custom_flags_text = "\n- " + "\n- ".join(custom_red_flags)
            eval_prompt_dynamic = EVAL_PROMPT_TEMPLATE.replace(
                "Check ***ONLY*** these explicit red flags:",
                f"Check ***ONLY*** these explicit red flags:{custom_flags_text}"
            )
        else:
            eval_prompt_dynamic = EVAL_PROMPT_TEMPLATE

        analysis_prompt = eval_prompt_dynamic.format(transcript=formatted_transcript)

        analysis_response = openai.ChatCompletion.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": analysis_prompt}],
            temperature=0
        )

        analysis_text = analysis_response.choices[0].message['content'].strip()
        analysis_lines = analysis_text.splitlines()
        if analysis_lines and analysis_lines[0].startswith("📊 Analysis Report:"):
            analysis_text = "\n".join(analysis_lines[1:]).lstrip()

        extracted = extract_details(analysis_text, formatted_transcript)
        title = f"{extracted['rating']} {extracted['rep_name']} {extracted['call_type']} {os.path.basename(filepath)}.html"

        full_analysis_content = (
            analysis_text
            + "\n\n"
            + "="*60 + "\nFORMATTED TRANSCRIPT\n" + "="*60 + "\n\n"
            + cleaned_transcript
            + "\n📜 Call Transcript\n\n"
            + cleaned_transcript
            + "\n\nGenerated automatically by Greener Living AI Analysis.\n"
        )

        # Update job to done
        async with async_session() as session:
            job = await session.get(Job, job_id)
            job.status = "done"
            job.result = {
                "full_analysis_content": full_analysis_content,
                "overall_perf": extracted["rating"],
                "rep_name": extracted["rep_name"].replace("_", " "),
                "categorization": extracted["call_type"].replace("_", " "),
                "red_flags": extracted["red_flags"],
                "red_flag_reason": extracted["red_flag_reason"],
                "red_flag_quotes": extracted["red_flag_quotes"],
                "original_filename": os.path.basename(filepath),
                "transcript": cleaned_transcript,
                "title": title
            }
            await session.commit()
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

@app.get("/job-status/{job_id}")
async def job_status(job_id: str, user: User = Depends(current_user)):
    # This endpoint is always fast: DB read only, no processing
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
        "credits": user.credits,
        "custom_red_flags": user.custom_red_flags,
        "is_active": user.is_active,
        "is_superuser": user.is_superuser,
        "is_verified": user.is_verified,
    }

@app.post("/buy-credits/")
async def buy_credits(amount: int = Form(...), user: User = Depends(current_user)):
    if amount <= 0:
        raise HTTPException(400, "Amount must be positive")
    user.credits += amount
    async with async_session() as session:
        await session.merge(user)
        await session.commit()
    return {"credits": user.credits}