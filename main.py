# Ensure NLTK data is available and force re-download if needed
from fastapi.responses import JSONResponse, HTMLResponse
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from audio_utils import process_audio_job, sync_transcribe, split_audio, set_async_session, get_audio_duration, transcribe_audio_chunk, fuzzy_match_rep_name
from models import User, Job, Base
from tenacity import retry, stop_after_attempt, wait_exponential
import jwt
from celery.backends.redis import RedisBackend
from celery.result import AsyncResult
from dotenv import load_dotenv
import time
import sys
import logging
from datetime import datetime
import json
import re
import openai
from typing import List, Optional
import uuid
from sqlalchemy import text
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi_users.authentication import AuthenticationBackend, JWTStrategy, BearerTransport
from fastapi_users.manager import BaseUserManager
from fastapi_users.db import SQLAlchemyUserDatabase
from fastapi_users import FastAPIUsers, schemas
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Request, BackgroundTasks, Security
import multiprocessing
import nltk
import os
# NLTK_DATA_PATH = 'workspace/fastapi_project/nltk_data'
NLTK_DATA_PATH = 'nltk_data'
os.makedirs(NLTK_DATA_PATH, exist_ok=True)
nltk.data.path = [NLTK_DATA_PATH]
nltk.download('punkt', download_dir=NLTK_DATA_PATH, force=True)
nltk.download('wordnet', download_dir=NLTK_DATA_PATH, force=True)

multiprocessing.set_start_method("spawn", force=True)


# Import models from models.py

# Import from audio_utils

load_dotenv()
print("OPENAI_API_KEY loaded:", os.getenv("OPENAI_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")

# ---- LOGGING SETUP ----
# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Create a custom formatter
formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(name)s: %(message)s')

# Apply the formatter to the root logger's handlers
for handler in logging.getLogger().handlers:
    handler.setFormatter(formatter)

# Set logging level for aiosqlite to INFO to suppress debug messages
logging.getLogger("aiosqlite").setLevel(logging.INFO)

logger = logging.getLogger("coldcall")

DATABASE_URL = "sqlite+aiosqlite:///./database.db"
engine = create_async_engine(DATABASE_URL)
async_session = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False)

# Set the async_session in audio_utils
set_async_session(async_session)

COMPANY = os.getenv('COMPANY', 'Greener Living')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


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

    def parse_id(self, value: str) -> str:
        # Preserve the hyphens in the UUID string
        try:
            # Validate that the value is a valid UUID
            uuid_obj = uuid.UUID(value)
            # Return the string representation with hyphens
            return str(uuid_obj)
        except ValueError as e:
            logger.error(
                f"Invalid UUID format for user ID: {value}, error: {str(e)}")
            raise HTTPException(
                status_code=400, detail=f"Invalid user ID format: {str(e)}")


async def get_user_manager(user_db=Depends(get_user_db)):
    yield UserManager(user_db)

bearer_transport = BearerTransport(tokenUrl="auth/jwt/login")


class CustomJWTStrategy(JWTStrategy):
    async def read_token(self, token: Optional[str], user_manager) -> Optional[User]:
        if not token:
            logger.error("No token provided in Authorization header")
            raise HTTPException(status_code=401, detail="No token provided")
        try:
            payload = jwt.decode(
                token,
                self.secret,
                algorithms=[self.algorithm],
                audience=["fastapi-users:auth"],
                issuer=None,
                options={"verify_exp": True},
            )
            logger.debug(f"Token payload: {payload}")
            user_id = payload.get("sub")
            if user_id is None:
                logger.error("Token payload does not contain 'sub' claim")
                raise HTTPException(
                    status_code=401, detail="Invalid token: no user ID")
            user = await user_manager.get(user_manager.parse_id(user_id))
            if user is None:
                logger.error(f"User not found for ID: {user_id}")
                raise HTTPException(status_code=401, detail="User not found")
            if not user.is_active:
                logger.error(f"User {user_id} is not active")
                raise HTTPException(
                    status_code=401, detail="User is not active")
            return user
        except jwt.ExpiredSignatureError as e:
            logger.error(f"Token expired: {str(e)}")
            raise HTTPException(
                status_code=401, detail=f"Token expired: {str(e)}")
        except jwt.InvalidTokenError as e:
            logger.error(f"Invalid token: {str(e)}")
            raise HTTPException(
                status_code=401, detail=f"Invalid token: {str(e)}")
        except Exception as e:
            logger.error(
                f"Unexpected error during token validation: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=401, detail=f"Token validation error: {str(e)}")


def get_jwt_strategy() -> CustomJWTStrategy:
    return CustomJWTStrategy(secret=SECRET, lifetime_seconds=3600)


auth_backend = AuthenticationBackend(
    name="jwt",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
)

fastapi_users = FastAPIUsers(
    get_user_manager,
    [auth_backend],
)

# Custom dependency to log authentication failures
security = HTTPBearer()


async def current_user_with_logging(
    credentials: HTTPAuthorizationCredentials = Security(security),
    user=Depends(fastapi_users.current_user(active=True))
):
    try:
        # Decode the token to log its details
        token = credentials.credentials
        try:
            decoded_token = jwt.decode(token, SECRET, algorithms=[
                                       "HS256"], audience=["fastapi-users:auth"])
            logger.debug(f"Token details: {decoded_token}")
        except jwt.ExpiredSignatureError as e:
            logger.error(f"Token expired: {str(e)}")
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError as e:
            logger.error(f"Invalid token: {str(e)}")
            raise HTTPException(
                status_code=401, detail=f"Invalid token: {str(e)}")

        if user is None:
            logger.error("User not found or inactive")
            raise HTTPException(
                status_code=401, detail="User not found or inactive")
        return user
    except HTTPException as e:
        logger.error(f"Authentication failed: {str(e.detail)}")
        raise e
    except Exception as e:
        logger.error(
            f"Unexpected error during authentication: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=401, detail=f"Authentication error: {str(e)}")

app = FastAPI()

origins = [
    "https://pmcdnsk3jzt4-3000.proxy.runpod.net",  # Frontend origin
    "http://localhost:3000",  # For local testing (optional)
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_frontend() -> HTMLResponse:
    """
    Serve the Frontend
    """
    index_path = os.path.join("static", "index.html")
    with open(index_path) as f:
        return HTMLResponse(content=f.read())


# Custom middleware to log authentication failures (additional layer)
@app.middleware("http")
async def log_authentication_failures(request: Request, call_next):
    response = await call_next(request)
    if response.status_code == 401:
        auth_header = request.headers.get(
            "Authorization", "No Authorization header")
        logger.error(
            f"Authentication failed for request {request.url}: status={response.status_code}, Authorization={auth_header}")
    return response


@app.on_event("startup")
async def on_startup():
    logger.info("Starting FastAPI application and initializing database...")

    # Check database connection
    try:
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("Database connection successful.")
    except Exception as e:
        logger.error(f"Failed to connect to the database: {str(e)}")
        raise Exception(f"Database connection failed: {str(e)}")

    # Create database tables
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created.")
    except Exception as e:
        logger.error(f"Failed to create database tables: {str(e)}")
        raise Exception(f"Failed to create database tables: {str(e)}")


app.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/auth/jwt",
    tags=["auth"],
)

# app.include_router(
#     fastapi_users.get_register_router(UserRead, UserCreate),
#     prefix="/auth",
#     tags=["auth"],
# )

app.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate),
    prefix="/users",
    tags=["users"],
)


@app.post("/auth/register")
async def register_user(user_create: UserCreate, user_manager: UserManager = Depends(get_user_manager)):
    try:
        user = await user_manager.create(user_create)
        return {"message": "User registered successfully", "email": user.email}
    except Exception as e:
        logger.error(f"User registration failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API server is running and can connect to dependencies.
    """
    try:
        # Check database connection
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))

        # Check Redis connection via Celery backend
        task_result = AsyncResult("health-check-dummy-task")
        backend = task_result.backend
        if isinstance(backend, RedisBackend):
            backend.client.ping()

        return {"status": "healthy", "message": "API server is running, database and Redis are accessible"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return {"status": "unhealthy", "message": f"Health check failed: {str(e)}"}

# --- Whisper/OpenAI Logic Below ---

# --- Known categories for call categorization ---
KNOWN_CATEGORIES = [
    "Voicemail",
    "Estimate not given",
    "Gave estimate and rejected",
    "Gave estimate and set a follow-up",
    "Gave estimate and sold"
]

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


@app.post("/audio/", status_code=202)
async def upload_audio(
    files: List[UploadFile] = File(...),
    params: str = Form('{}'),
    user: User = Depends(current_user_with_logging),
):
    """
    Updated POST /audio/ endpoint to fire off Celery tasks and return 202 with task IDs.
    """
    start_time = time.perf_counter()
    logger.info(f"User {user.id} uploading {len(files)} files")
    logger.debug(
        f"User details: email={user.email}, is_active={user.is_active}, minutes={user.minutes}")
    if getattr(user, 'minutes', 0) is None or user.minutes <= 0:
        logger.warning(
            f"User {user.id} has insufficient minutes: {user.minutes}")
        raise HTTPException(402, "Insufficient minutes")
    params_dict = json.loads(params)
    custom_red_flags: List[str] = params_dict.get(
        "custom_red_flags", user.custom_red_flags)
    current_reps: list = params_dict.get("current_reps", [])
    os.makedirs("./uploads", exist_ok=True)
    job_ids = []
    tasks = []

    for upload in files:
        async with async_session() as session:
            try:
                job_id = str(uuid.uuid4())
                filepath = f"./uploads/{job_id}_{upload.filename}"
                with open(filepath, "wb") as audio_file:
                    audio_file.write(await upload.read())

                # Check duration and deduct minutes
                duration = get_audio_duration(filepath)
                minutes = max(1, int(round(duration / 60.0)))
                logger.debug(
                    f"File {upload.filename}: duration={duration} seconds, required minutes={minutes}")
                if user.minutes < minutes:
                    os.remove(filepath)
                    raise HTTPException(
                        402, f"Insufficient minutes. This file requires {minutes} minute(s).")
                user.minutes -= minutes

                # Create job in database
                job = Job(
                    id=job_id,
                    user_id=str(user.id),
                    filename=upload.filename,
                    status="pending",
                    result=None,
                    task_id=None
                )
                session.add(job)
                await session.commit()
                job_ids.append(job_id)

                # Update user minutes
                await session.merge(user)
                await session.commit()

                # Fire off Celery task
                from celery_worker import process_audio_job_task
                task = process_audio_job_task.delay(
                    job_id=job_id,
                    filepath=filepath,
                    params=params,
                    user_id=str(user.id),
                    custom_red_flags=custom_red_flags,
                    current_reps=current_reps
                )

                # Update the job with the task_id
                job.task_id = task.id
                await session.merge(job)
                await session.commit()

                tasks.append(task)

            except Exception as e:
                logger.error(
                    f"Failed to process file {upload.filename}: {str(e)}", exc_info=True)
                if os.path.exists(filepath):
                    os.remove(filepath)
                raise HTTPException(
                    status_code=500, detail=f"Failed to process file: {str(e)}")

    elapsed = time.perf_counter() - start_time
    logger.info(
        f"upload_audio processed {len(job_ids)} files in {elapsed:.2f} seconds")
    return {"job_ids": [task.id for task in tasks], "status": "accepted"}


@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str, user: User = Depends(current_user_with_logging)):
    """
    GET /tasks/{task_id} endpoint to check task status and progress.
    Updated to handle both parent and chord tasks by falling back to Celery if no Job is found.
    """
    logger.debug(f"User {user.id} requesting task status for task {task_id}")
    try:
        task_result = AsyncResult(task_id)
        if not task_result:
            logger.warning(f"Task {task_id} not found in Celery backend")
            raise HTTPException(status_code=404, detail="Task not found")

        # First, try to find the Job by task_id in the database (for parent tasks)
        job = None
        async with async_session() as session:
            result = await session.execute(
                select(Job).where(Job.task_id == task_id)
            )
            job = result.scalars().first()
            if job:
                if job.user_id != str(user.id):
                    logger.warning(
                        f"User {user.id} not authorized for job {job.id}")
                    raise HTTPException(
                        status_code=403, detail="Not authorized")
                logger.debug(
                    f"Found Job for task {task_id} in database: {job.id}")

        # Safely retrieve raw task metadata for debugging
        try:
            backend = task_result.backend
            if isinstance(backend, RedisBackend):
                raw_meta = backend.client.get(
                    backend.get_key_for_task(task_id))
                logger.debug(f"Raw task metadata for {task_id}: {raw_meta}")
        except Exception as e:
            logger.warning(
                f"Failed to retrieve raw task metadata for {task_id}: {str(e)}")

        # Safely retrieve task state
        try:
            state = task_result.state
        except Exception as e:
            logger.error(
                f"Failed to retrieve state for task {task_id}: {str(e)}", exc_info=True)
            # Fallback to job status from database if Celery fails and Job exists
            state = job.status.upper() if job and job.status else "UNKNOWN"

        # Initialize defaults
        progress = 0
        error = None
        result = None

        # Handle task info based on state
        if state == "SUCCESS":
            # In SUCCESS state, task_result.info contains the final result
            result = task_result.info if task_result.info else None
            # Progress should be 100% for a successful task
            progress = 100
            # Check if the job has an error in its result (in case of a previous failure)
            if job and job.result and isinstance(job.result, dict) and 'error' in job.result:
                error = job.result.get('error')
        elif state == "FAILURE":
            # In FAILURE state, task_result.info contains the exception info (should be a dict)
            info = task_result.info if task_result.info else {}
            if isinstance(info, dict):
                error = info.get('error', 'Unknown error')
                progress = info.get('progress', 0)
            else:
                error = str(info) if info else "Unknown error"
                progress = 0
        else:
            # In PROGRESS or other states, task_result.info should be a dict with progress/error
            info = task_result.info if task_result.info else {}
            if isinstance(info, dict):
                progress = info.get('progress', 0)
                error = info.get('error') if state == "FAILURE" else None
            else:
                # Fallback to job data if info is not a dict and Job exists
                progress = 0
                error = job.result.get('error') if job and job.result and isinstance(
                    job.result, dict) and 'error' in job.result else f"Unexpected task info format: {str(info)}"

        # If result is still None and the job has a result, use it
        if result is None and job and job.status == "done" and job.result:
            result = job.result if isinstance(job.result, dict) else job.result.get(
                'result') if isinstance(job.result, dict) else None
            if not error and isinstance(job.result, dict) and 'error' in job.result:
                error = job.result.get('error')

        # If no Job was found, return the task status directly (for chord tasks)
        if not job:
            logger.debug(
                f"No Job found for task {task_id}, returning Celery task status directly")
            return {
                "state": state,
                "progress": progress,
                "result": result,
                "error": error
            }

        logger.debug(
            f"Task {task_id} status: state={state}, progress={progress}, error={error}")
        return {
            "state": state,
            "progress": progress,
            "result": result,
            "error": error
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(
            f"Unexpected error in get_task_status for task {task_id}: {str(e)}", exc_info=True)
        return {
            "state": "ERROR",
            "progress": 0,
            "result": None,
            "error": f"Internal server error: {str(e)}"
        }


@app.get("/job-status/{job_id}")
async def job_status(job_id: str, user: User = Depends(current_user_with_logging)):
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
async def update_red_flags(red_flags: List[str], user: User = Depends(current_user_with_logging)):
    logger.info(f"User {user.id} updating red flags")
    user.custom_red_flags = red_flags
    async with async_session() as session:
        await session.merge(user)
        await session.commit()
    logger.info(f"User {user.id} red flags updated")
    return {"custom_red_flags": user.custom_red_flags}


@app.get("/red-flags/")
async def get_red_flags(user: User = Depends(current_user_with_logging)):
    logger.info(f"User {user.id} retrieving red flags")
    logger.info(f"User {user.id} red flags: {user.custom_red_flags}")
    return {"custom_red_flags": user.custom_red_flags}


@app.get("/users/me")
async def get_me(user: User = Depends(current_user_with_logging)):
    logger.info(f"User {user.id} retrieving user info")
    logger.info(
        f"User {user.id} info: {user.email}, {user.minutes}, {user.custom_red_flags}, {user.is_active}, {user.is_superuser}, {user.is_verified}")
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
async def buy_minutes(amount: int = Form(...), user: User = Depends(current_user_with_logging)):
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


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def call_openai(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1024
    )
    return response


@app.post("/api/generate-coaching-plan")
async def generate_coaching_plan(request: Request):
    data = await request.json()
    prompt = data.get("prompt")
    if not prompt:
        return JSONResponse(status_code=400, content={"result": "No prompt provided."})
    try:
        logger.info(f"OpenAI prompt length: {len(prompt)} characters")
        response = call_openai(prompt)
        if 'usage' in response:
            prompt_tokens = response['usage'].get('prompt_tokens', 0)
            completion_tokens = response['usage'].get('completion_tokens', 0)
            input_cost = prompt_tokens * 0.000005
            output_cost = completion_tokens * 0.000015
            total_cost = input_cost + output_cost
            logger.info(
                f"OpenAI usage: prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}, total_cost=${total_cost:.6f}")
        coaching_plan = response.choices[0].message.content.strip()
        return {"result": coaching_plan, "usage": response.get('usage', {}), "cost": total_cost if 'usage' in response else None}
    except Exception as e:

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

# Explicit /auth/register route (moved to after the router inclusions)


@app.post("/auth/register")
async def register_user(user_create: UserCreate, user_manager: UserManager = Depends(get_user_manager)):
    try:
        user = await user_manager.create(user_create)
        return {"message": "User registered successfully", "email": user.email}
    except Exception as e:
        logger.error(f"User registration failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
