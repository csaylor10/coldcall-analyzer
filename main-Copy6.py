from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi_users import FastAPIUsers, schemas
from fastapi_users.db import SQLAlchemyUserDatabase, SQLAlchemyBaseUserTableUUID
from fastapi_users.manager import BaseUserManager
from fastapi_users.authentication import AuthenticationBackend, JWTStrategy, BearerTransport
from sqlalchemy import Column, Integer, JSON, String, Boolean
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import uuid
from typing import List, Optional
import whisper
import openai
import re
import os
import json

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
    return {"message": "Cold Call Analyzer API is running!"}

# --- Whisper/OpenAI Logic Below ---

openai_client = openai.OpenAI(api_key="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")
whisper_model = whisper.load_model("medium")

EVAL_PROMPT_TEMPLATE = """ ... (your long prompt here, unchanged) ... """

def extract_details(analysis, transcript):
    # ... (your extract_details function, unchanged) ...
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
    rep_name = "Unknown"
    patterns = [
        r'this is (\w+) with Greener Living',
        r'this is (\w+) from Greener Living',
        r'hey.*?this is (\w+).*?Greener Living',
        r'hello.*?this is (\w+).*?Greener Living',
        r'my name is (\w+) with Greener Living',
        r'Greener Living.*?this is (\w+)',
        r'Greener Living.*?my name is (\w+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, transcript, re.I)
        if match:
            rep_name = match.group(1).strip()
            break
    if rep_name.lower() in ["customer", "client"]:
        rep_name = "Unknown"
    rep_name_sanitized = re.sub(r"\s+", "_", rep_name)
    return {
        "rating": rating,
        "call_type": call_type_sanitized,
        "red_flags": red_flags,
        "red_flag_reason": red_flag_reason,
        "red_flag_quotes": red_flag_quotes,
        "rep_name": rep_name_sanitized
    }

@app.post("/upload/")
async def upload_audio(
    file: UploadFile = File(...),
    params: str = Form('{}'),
    user: User = Depends(current_user),
):
    if user.credits <= 0:
        raise HTTPException(402, "Insufficient credits")

    params_dict = json.loads(params)
    custom_red_flags: List[str] = params_dict.get("custom_red_flags", user.custom_red_flags)

    filepath = f"./{file.filename}"
    with open(filepath, "wb") as audio_file:
        audio_file.write(await file.read())

    transcription_result = whisper_model.transcribe(filepath)
    transcript = transcription_result.get('text', '').strip()

    format_prompt = f'''
Organize the following transcript into a structured conversation. Label the lawn care representative as "Rep" and the customer as "Client".

Transcript:
"""
{transcript}
"""
'''

    formatted_response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": format_prompt}],
        temperature=0
    )

    formatted_transcript = formatted_response.choices[0].message.content.strip()
    cleaned_transcript = re.sub(r'^(Rep|Client):\s*', '', formatted_transcript, flags=re.MULTILINE)

    custom_flags_text = '\n- '.join(custom_red_flags)
    eval_prompt_dynamic = EVAL_PROMPT_TEMPLATE.replace(
        "Check ***ONLY*** these explicit red flags:",
        f"Check ***ONLY*** these explicit red flags:\n- {custom_flags_text}" if custom_red_flags else "Check ***ONLY*** these explicit red flags:"
    )

    analysis_prompt = eval_prompt_dynamic.format(transcript=formatted_transcript)

    analysis_response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": analysis_prompt}],
        temperature=0
    )

    analysis_text = analysis_response.choices[0].message.content.strip()
    extracted = extract_details(analysis_text, formatted_transcript)

    full_analysis_content = (
        analysis_text + "\n\n" + "="*60 + "\nFORMATTED TRANSCRIPT\n" + "="*60 + "\n\n" + cleaned_transcript + "\n"
    )

    os.remove(filepath)

    user.credits -= 1
    async with async_session() as session:
        session.add(user)
        await session.commit()

    return {
        "full_analysis_content": full_analysis_content,
        "overall_perf": extracted["rating"],
        "rep_name": extracted["rep_name"].replace("_", " "),
        "categorization": extracted["call_type"].replace("_", " "),
        "red_flags": extracted["red_flags"],
        "red_flag_reason": extracted["red_flag_reason"],
        "red_flag_quotes": extracted["red_flag_quotes"],
        "original_filename": file.filename,
        "transcript": cleaned_transcript,
        "credits_left": user.credits
    }

@app.post("/update-red-flags/")
async def update_red_flags(red_flags: List[str], user: User = Depends(current_user)):
    user.custom_red_flags = red_flags
    async with async_session() as session:
        session.add(user)
        await session.commit()
    return {"custom_red_flags": user.custom_red_flags}

@app.get("/red-flags/")
async def get_red_flags(user: User = Depends(current_user)):
    return {"custom_red_flags": user.custom_red_flags}