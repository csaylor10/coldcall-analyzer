from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import whisper
import openai
import re
import os
import zipfile
from datetime import datetime

# --- CONFIG ---
openai_client = openai.OpenAI(api_key="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")
whisper_model = whisper.load_model("base")

# FastAPI app setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Cold Call Analyzer API is running!"}

# Evaluation Prompt
EVAL_PROMPT_TEMPLATE = """
You are a cold call analyzer for Greener Living, a lawn care company. Evaluate the following sales call transcript thoroughly, providing accurate feedback using direct quotes from the transcript. Follow the format below exactly.

Overall Performance: X/10
Rep Name: [Name or "Unknown"]
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
\"\"\"
"""

# Extraction function
def extract_details(analysis):
    rating = re.search(r"Overall Performance:\s*(\d+)/10", analysis)
    rating = rating.group(1).strip() if rating else "UnknownRating"

    call_type = re.search(r"Call Categorization:\s*(.+)", analysis)
    call_type = call_type.group(1).strip() if call_type else "UnknownType"
    call_type_sanitized = call_type.replace(" ", "_")

    red_flags = re.search(r"Red Flags:\s*(Yes|No)", analysis, re.I)
    red_flags = red_flags.group(1).capitalize() if red_flags else "No"

    red_flag_reason = re.search(r"Reason for Red Flag:\s*(.+)", analysis)
    red_flag_reason = red_flag_reason.group(1).strip() if red_flag_reason else "None"

    red_flag_quotes = re.search(r"Red Flag Direct Quotes:\s*\[\"(.+)\"\]", analysis)
    red_flag_quotes = red_flag_quotes.group(1).strip() if red_flag_quotes else "None"

    persistence_score_match = re.search(r"Persistence\s*\((\d+)/10\)", analysis)
    persistence_score = persistence_score_match.group(1) if persistence_score_match else "Unknown"

    rep_name = re.search(r"Rep Name:\s*(.+)", analysis)
    rep_name = rep_name.group(1).strip() if rep_name else "Unknown"
    rep_name_sanitized = re.sub(r"\s+", "_", rep_name)

    return {
        "rating": rating,
        "call_type": call_type_sanitized,
        "red_flags": red_flags,
        "red_flag_reason": red_flag_reason,
        "red_flag_quotes": red_flag_quotes,
        "persistence_score": persistence_score,
        "rep_name": rep_name_sanitized
    }

@app.post("/upload/")
async def upload_audio(file: UploadFile = File(...)):
    filepath = f"./{file.filename}"
    with open(filepath, "wb") as audio_file:
        content = await file.read()
        audio_file.write(content)

    transcription_result = whisper_model.transcribe(filepath)
    transcript = transcription_result.get('text', '').strip()
    if not transcript:
        raise HTTPException(status_code=500, detail="Transcription failed.")

    prompt = EVAL_PROMPT_TEMPLATE.format(transcript=transcript)
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    analysis_text = response.choices[0].message.content.strip()
    if not analysis_text:
        raise HTTPException(status_code=500, detail="Analysis generation failed.")

    extracted = extract_details(analysis_text)
    analysis_filename = f"{extracted['rating']}_{extracted['rep_name']}_{extracted['call_type']}_RedFlags{extracted['red_flags']}_{file.filename}_analysis.txt"
    zip_filename = datetime.now().strftime("%Y-%m-%d") + "_analysis_files.zip"

    with zipfile.ZipFile(zip_filename, "a", zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr(analysis_filename, analysis_text + "\n\n--- Full Transcript ---\n\n" + transcript)

    os.remove(filepath)

    return {
        "analysis": analysis_text,
        "overall_perf": extracted["rating"],
        "rep_name": extracted["rep_name"].replace("_", " "),
        "categorization": extracted["call_type"].replace("_", " "),
        "red_flags": extracted["red_flags"],
        "red_flag_reason": extracted["red_flag_reason"],
        "red_flag_quotes": extracted["red_flag_quotes"],
        "persistence_score": extracted["persistence_score"],
        "original_filename": file.filename,
        "transcript": transcript,
        "analysis_file_saved_as": zip_filename
    }
