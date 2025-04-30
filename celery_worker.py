from celery import Celery

celery_app = Celery("tasks", broker="redis://localhost:6379/0", backend="redis://localhost:6379/0")

# Import your async audio processing function from main.py
from main import process_audio_job, sync_transcribe

@celery_app.task
def process_audio_job_task(job_id, filepath, params, user_id, custom_red_flags, current_reps=None):
    import asyncio
    # Run the async function in the event loop
    return asyncio.run(process_audio_job(job_id, filepath, params, user_id, custom_red_flags, current_reps))

@celery_app.task
def transcribe_audio_chunk_task(chunk_path):
    return sync_transcribe(chunk_path)

@celery_app.task
# This is the callback for the chord. It receives the list of transcripts from all chunks.
def combine_transcripts_task(transcripts, job_id, params, user_id, custom_red_flags):
    full_transcript = '\n'.join(transcripts)
    # Here you can add logic to store the transcript, update DB, trigger further processing, etc.
    # For now, just print/log and return the combined transcript.
    print(f"[Chord Callback] Combined transcript for job {job_id}, length: {len(full_transcript)} characters")
    return full_transcript
