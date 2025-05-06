import os
import asyncio
import time
import logging
import json
import ffmpeg
from faster_whisper import WhisperModel
import openai
import re
from pydub import AudioSegment
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from models import Job
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ---- LOGGING SETUP ----
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("coldcall")

# ---- ASYNC SESSION FOR DB ----
_async_session = None

def set_async_session(session):
    global _async_session
    _async_session = session

# ---- COMPANY NAME ----
COMPANY = os.getenv('COMPANY', 'Greener Living')

# ---- EVAL PROMPT TEMPLATE ----
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
Tip: Suggest specific additional services that could be recommended.
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

FORMATTED TRANSCRIPT
==========

{transcript}
"""

# ---- UTILITY FUNCTIONS ----
def get_audio_duration(filepath):
    try:
        probe = ffmpeg.probe(filepath)
        duration = float(probe['format']['duration'])
        return duration
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg error while getting duration: {str(e)}")
        raise ValueError(f"Failed to get audio duration: {str(e)}")

def split_audio(filepath, chunk_length_sec=30):
    chunks = []
    audio = AudioSegment.from_file(filepath)
    duration_ms = len(audio)
    chunk_length_ms = chunk_length_sec * 1000
    num_chunks = (duration_ms + chunk_length_ms - 1) // chunk_length_ms

    for i in range(num_chunks):
        start_ms = i * chunk_length_ms
        end_ms = min((i + 1) * chunk_length_ms, duration_ms)
        chunk = audio[start_ms:end_ms]
        chunk_path = f"{filepath}_chunk_{i:03d}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    return chunks

def sync_transcribe(audio_path, model):
    try:
        logger.info(f"Processing audio with duration {get_audio_duration(audio_path):.3f}")
        start_time = time.time()
        segments, info = model.transcribe(audio_path, language="en", beam_size=5)
        transcript = " ".join(segment.text for segment in segments).strip()
        elapsed = time.time() - start_time
        logger.info(f"Transcribed {audio_path} in {elapsed:.2f}s")
        return transcript
    except Exception as e:
        logger.error(f"Transcription failed for {audio_path}: {str(e)}", exc_info=True)
        raise ValueError(f"Transcription failed: {str(e)}")

async def transcribe_audio_chunk(chunk_path, model):
    return sync_transcribe(chunk_path, model)

def fuzzy_match_rep_name(transcript, rep_list):
    if not rep_list:
        return "Unknown Rep"
    transcript_lower = transcript.lower()
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(transcript_lower)
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]

    for rep in rep_list:
        rep_lower = rep.lower()
        if rep_lower in transcript_lower:
            return rep
        rep_lemmas = [lemmatizer.lemmatize(word) for word in word_tokenize(rep_lower)]
        if any(lemma in lemmas for lemma in rep_lemmas):
            return rep
    return "Unknown Rep"

async def process_audio_job(job_id, filepath, params, user_id, custom_red_flags, current_reps=None, pre_transcribed=None):
    start_time = time.time()
    logger.info(f"Processing audio job {job_id} for user {user_id}")

    async with _async_session() as session:
        job = await session.get(Job, job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            raise ValueError(f"Job {job_id} not found")

        job.status = "processing"
        await session.commit()

    if pre_transcribed:
        transcript = pre_transcribed
        logger.info(f"Using pre-transcribed text for job {job_id}, length: {len(transcript)}")
    else:
        raise ValueError("Transcription not provided")

    rep_name = fuzzy_match_rep_name(transcript, current_reps or [])
    logger.info(f"Identified rep: {rep_name}")

    custom_red_flag_section = "\n".join(custom_red_flags) if custom_red_flags else "None"
    prompt = EVAL_PROMPT_TEMPLATE.format(
        COMPANY=COMPANY,
        CUSTOM_RED_FLAGS_SECTION=custom_red_flag_section,
        transcript=transcript
    )

    start_analysis = time.time()
    try:
        logger.info(f"Making OpenAI API call for job {job_id}, prompt length: {len(prompt)} characters")
        analysis_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=1024  # Reduced to speed up response
        )
        elapsed_analysis = time.time() - start_analysis
        logger.info(f"OpenAI API call completed in {elapsed_analysis:.2f}s")
        logger.debug(f"OpenAI API response: {analysis_response}")
        if 'usage' in analysis_response:
            prompt_tokens = analysis_response['usage'].get('prompt_tokens', 0)
            completion_tokens = analysis_response['usage'].get('completion_tokens', 0)
            logger.info(f"OpenAI API usage: prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}")
    except Exception as e:
        logger.error(f"OpenAI API call failed for job {job_id}: {str(e)}", exc_info=True)
        raise ValueError(f"OpenAI API call failed: {str(e)}")

    full_analysis_content = analysis_response.choices[0].message.content.strip()
    logger.info(f"Analysis content length: {len(full_analysis_content)} characters")

    # Parse the analysis content into structured fields
    overall_perf_match = re.search(r"Overall Performance: (\d+/10)", full_analysis_content)
    rep_name_match = re.search(r"Rep Name: (.+?)(?:\n|$)", full_analysis_content)
    categorization_match = re.search(r"Call Categorization: (.+?)(?:\n|$)", full_analysis_content)
    red_flags_match = re.search(r"Red Flags: (Yes|No)", full_analysis_content)
    red_flag_reason_match = re.search(r"Reason for Red Flag: (.+?)(?:\n|$)", full_analysis_content)
    red_flag_quotes_match = re.search(r"Red Flag Direct Quotes: \[(.+?)\]", full_analysis_content)
    transcript_match = re.search(r"FORMATTED TRANSCRIPT\n=+\n\n([\s\S]+?)(?:\n\s*Generated automatically|$)", full_analysis_content)
    system_red_flags_match = re.search(r"SystemRedFlagsFound = \[([^\]]*)\]", full_analysis_content)
    custom_red_flags_match = re.search(r"CustomRedFlagsFound = \[([^\]]*)\]", full_analysis_content)
    custom_red_flags_explanations_match = re.search(r"CustomRedFlagsExplanations = \{([^\}]*)\}", full_analysis_content)

    # Log parsing results
    logger.info(f"Parsed overall_perf: {overall_perf_match.group(1) if overall_perf_match else 'N/A'}")
    logger.info(f"Parsed rep_name: {rep_name_match.group(1) if rep_name_match else 'Unknown'}")
    logger.info(f"Parsed categorization: {categorization_match.group(1) if categorization_match else 'Unknown'}")
    logger.info(f"Parsed transcript from OpenAI response: {transcript_match.group(1).strip()[:100] if transcript_match else 'Not found'}")

    # Parse system and custom red flags
    system_red_flags = []
    if system_red_flags_match:
        flags_str = system_red_flags_match.group(1)
        system_red_flags = [flag.strip().strip("'\"") for flag in flags_str.split(",") if flag.strip()]

    custom_red_flags_found = []
    if custom_red_flags_match:
        flags_str = custom_red_flags_match.group(1)
        custom_red_flags_found = [flag.strip().strip("'\"") for flag in flags_str.split(",") if flag.strip()]

    custom_red_flags_explanations = {}
    if custom_red_flags_explanations_match:
        explanations_str = custom_red_flags_explanations_match.group(1)
        pairs = explanations_str.split(",")
        for pair in pairs:
            if ":" in pair:
                key, value = pair.split(":", 1)
                key = key.strip().strip("'\"")
                value = value.strip().strip("'\"")
                custom_red_flags_explanations[key] = value

    # Ensure transcript is always set
    final_transcript = transcript_match.group(1).strip() if transcript_match else transcript
    if not final_transcript:
        logger.warning(f"Transcript is empty for job {job_id}, using placeholder")
        final_transcript = "No transcript available."

    structured_result = {
        "overall_perf": overall_perf_match.group(1) if overall_perf_match else "N/A",
        "rep_name": rep_name_match.group(1) if rep_name_match else "Unknown",
        "categorization": categorization_match.group(1) if categorization_match else "Unknown",
        "red_flags": red_flags_match.group(1) if red_flags_match else "No",
        "red_flag_reason": red_flag_reason_match.group(1) if red_flag_reason_match else "",
        "red_flag_quotes": red_flag_quotes_match.group(1) if red_flag_quotes_match else "",
        "system_red_flags_found": system_red_flags,
        "custom_red_flags_found": custom_red_flags_found,
        "custom_red_flags_explanations": custom_red_flags_explanations,
        "full_analysis_content": full_analysis_content,
        "transcript": final_transcript,
        "title": f"{overall_perf_match.group(1) if overall_perf_match else 'N/A'} {rep_name_match.group(1) if rep_name_match else 'Unknown'} {categorization_match.group(1) if categorization_match else 'Unknown'} {job.filename}.html"
    }

    start_db = time.time()
    async with _async_session() as session:
        job = await session.get(Job, job_id)
        if not job:
            logger.error(f"Job {job_id} not found during final update")
            raise ValueError(f"Job {job_id} not found")
        job.status = "done"
        job.result = structured_result
        await session.commit()
    logger.info(f"DB write took {time.time() - start_db:.2f}s")

    elapsed = time.time() - start_time
    logger.info(f"Job {job_id} completed and result stored.")
    logger.info(f"Total pipeline time for job {job_id}: {elapsed:.2f}s")

    return structured_result