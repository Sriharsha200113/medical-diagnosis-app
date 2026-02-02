from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from processing import SymptomExtractor, DiagnosisModule, PubMedSearch, Summarizer


app = FastAPI(
    title="Medical Diagnosis API",
    description="AI-powered medical symptom analysis and diagnosis assistance",
    version="1.0.0"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processing modules
symptom_extractor = SymptomExtractor()
diagnosis_module = DiagnosisModule()
pubmed_search = PubMedSearch()
summarizer = Summarizer()


class DiagnoseRequest(BaseModel):
    """Request body for diagnosis endpoint."""
    symptoms: str


class DiagnoseResponse(BaseModel):
    """Response body for diagnosis endpoint."""
    symptoms: list[str]
    duration: str
    severity: str
    diagnosis: Dict[str, Any]
    pubmed_summary: Dict[str, Any]
    disclaimer: str


MEDICAL_DISCLAIMER = (
    "IMPORTANT: This information is for educational purposes only and is NOT medical advice. "
    "Always consult with qualified healthcare professionals for diagnosis and treatment. "
    "If you are experiencing a medical emergency, please call emergency services immediately."
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Medical Diagnosis API",
        "version": "1.0.0",
        "endpoints": {
            "POST /diagnose": "Submit symptoms for analysis",
            "GET /health": "Health check endpoint"
        },
        "disclaimer": MEDICAL_DISCLAIMER
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(request: DiagnoseRequest):
    """
    Analyze symptoms and provide potential diagnoses.

    Takes a symptom description and returns:
    - Extracted symptoms
    - Potential diagnoses with probability
    - Relevant PubMed research summary
    """
    if not request.symptoms or not request.symptoms.strip():
        raise HTTPException(status_code=400, detail="Symptoms description is required")

    try:
        # Step 1: Extract symptoms from user input
        extracted = await symptom_extractor.extract(request.symptoms)

        if not extracted.symptoms:
            raise HTTPException(
                status_code=400,
                detail="Could not extract any symptoms from the provided description"
            )

        # Step 2: Generate diagnosis
        diagnosis_result = await diagnosis_module.diagnose(
            symptoms=extracted.symptoms,
            duration=extracted.duration,
            severity=extracted.severity
        )

        # Step 3: Search PubMed for relevant articles
        condition_names = [c.name for c in diagnosis_result.conditions]
        search_query = pubmed_search.build_search_query(
            symptoms=extracted.symptoms,
            conditions=condition_names
        )
        articles = await pubmed_search.search(search_query, max_results=5)

        # Step 4: Summarize PubMed results
        summary = await summarizer.summarize(
            articles=articles,
            symptoms=extracted.symptoms,
            conditions=condition_names
        )

        return DiagnoseResponse(
            symptoms=extracted.symptoms,
            duration=extracted.duration,
            severity=extracted.severity,
            diagnosis={
                "conditions": [
                    {
                        "name": c.name,
                        "probability": c.probability,
                        "description": c.description
                    }
                    for c in diagnosis_result.conditions
                ],
                "recommendations": diagnosis_result.recommendations,
                "urgency": diagnosis_result.urgency
            },
            pubmed_summary=summary.to_dict(),
            disclaimer=MEDICAL_DISCLAIMER
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    import config

    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
