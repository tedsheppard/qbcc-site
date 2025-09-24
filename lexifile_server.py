import os
import io
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import PyPDF2
import docx
from pathlib import Path

# --- Setup ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI()

# --- FIX: Add CORS Middleware ---
# This resolves the "405 Method Not Allowed" error by allowing
# your frontend to make POST requests to this backend server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, you might want to restrict this to your actual domain
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


# --- Helper Functions ---

def extract_text_from_file(file: io.BytesIO, filename: str) -> str:
    """Extracts text from PDF or DOCX files."""
    text = ""
    try:
        if filename.lower().endswith('.pdf'):
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
        elif filename.lower().endswith('.docx'):
            doc = docx.Document(file)
            for para in doc.paragraphs:
                text += para.text + "\n"
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {filename}")
    except Exception as e:
        print(f"Error extracting text from {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {filename}")
    return text

def get_renaming_system_prompt() -> str:
    """Returns the expert system prompt for the AI to rename documents."""
    return """
You are an expert legal assistant AI named LexiFile. Your task is to analyze text from a legal document and extract key information to suggest a structured filename. The required format is: YYYYMMDD - Strict Name - Looser Description.

Your entire response must be a single, valid JSON object. Do not include any text, notes, or apologies outside of this JSON object.

The JSON object must have the following three keys:
1.  "date": A string representing the primary date found in the document, formatted as YYYYMMDD. Find the execution date, letter date, or filing date. Do not use the file's metadata creation date. If no date can be found, use "00000000".
2.  "docType": A string for the "Strict Name". This should be a concise, formal classification of the document. Examples: "Letter from X to Y", "Contract for Services", "Affidavit of John Smith", "Statement of Claim".
3.  "description": A string for the "Looser Description". This should be a brief, 2-5 word summary of the document's subject matter. Examples: "Security for Costs", "Discovery Request", "Evidence of Service".

Analyze the text and provide the best possible values for these three keys.
"""

# --- API Endpoints ---

@app.post("/rename-document")
async def rename_document(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    try:
        file_content = await file.read()
        file_stream = io.BytesIO(file_content)
        extracted_text = extract_text_from_file(file_stream, file.filename)
        
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract any text from the document.")

        system_prompt = get_renaming_system_prompt()
        user_prompt = f"Please analyze the following document text and return the structured JSON for renaming:\n\n---DOCUMENT TEXT---\n{extracted_text[:12000]}\n---END TEXT---"

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        ai_response_content = response.choices[0].message.content
        return json.loads(ai_response_content)
    except json.JSONDecodeError:
         raise HTTPException(status_code=500, detail="AI returned an invalid JSON response.")
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"ERROR in /rename-document: {e}")
        return JSONResponse(content={"error": f"An unexpected error occurred: {str(e)}"}, status_code=500)

# --- Static File Serving ---
STATIC_DIR = Path(__file__).parent.absolute()

@app.get("/{file_name}")
async def get_html(file_name: str):
    # This now assumes clean URLs and looks for the corresponding .html file
    file_path = os.path.join(STATIC_DIR, f"{file_name}.html")
    if os.path.exists(file_path):
        return FileResponse(file_path)
    
    # Fallback for static assets that might be requested through this route
    static_file_path = os.path.join(STATIC_DIR, file_name)
    if os.path.exists(static_file_path):
        return FileResponse(static_file_path)

    raise HTTPException(status_code=404, detail="File not found")

@app.get("/")
async def get_index():
    return FileResponse(os.path.join(STATIC_DIR, "lexifile_index.html"))

