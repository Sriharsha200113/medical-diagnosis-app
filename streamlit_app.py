import streamlit as st
import httpx
import xml.etree.ElementTree as ET
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Page configuration
st.set_page_config(
    page_title="Medical Diagnosis Assistant",
    page_icon="🏥",
    layout="wide"
)

# Get API key from Streamlit secrets or environment
import os
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
OPENAI_MODEL = st.secrets.get("OPENAI_MODEL", os.getenv("OPENAI_MODEL", "gpt-4"))

# Custom CSS
st.markdown("""
<style>
    .disclaimer-box {
        background-color: #000000;
        color: #ffffff;
        border: 1px solid #333333;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 20px;
    }
    .symptom-tag {
        display: inline-block;
        background-color: #e3f2fd;
        color: #1976d2;
        padding: 5px 12px;
        margin: 3px;
        border-radius: 20px;
        font-size: 14px;
    }
    .condition-card {
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid;
        color: #ffffff;
    }
    .condition-card h4 {
        margin: 0 0 10px 0;
        color: #ffffff;
    }
    .condition-card p {
        margin: 5px 0;
        color: #ffffff;
    }
    .condition-high {
        background-color: #dc3545;
        border-left-color: #a71d2a;
    }
    .condition-medium {
        background-color: #fd7e14;
        border-left-color: #c96000;
    }
    .condition-low {
        background-color: #28a745;
        border-left-color: #1e7e34;
    }
    .urgency-emergency {
        color: #dc3545;
        font-weight: bold;
    }
    .urgency-urgent {
        color: #fd7e14;
        font-weight: bold;
    }
    .urgency-routine {
        color: #ffc107;
    }
    .urgency-self-care {
        color: #28a745;
    }
</style>
""", unsafe_allow_html=True)

# --- Pydantic Models ---
class ExtractedSymptoms(BaseModel):
    symptoms: List[str] = Field(description="List of extracted medical symptoms")
    duration: str = Field(default="", description="Duration of symptoms if mentioned")
    severity: str = Field(default="", description="Severity level if mentioned")

class Condition(BaseModel):
    name: str = Field(description="Name of the condition")
    probability: str = Field(description="Probability level: high, medium, or low")
    description: str = Field(description="Brief description of the condition")

class DiagnosisResult(BaseModel):
    conditions: List[Condition] = Field(description="List of potential conditions")
    recommendations: List[str] = Field(description="General health recommendations")
    urgency: str = Field(description="Urgency level: emergency, urgent, routine, or self-care")

# --- Processing Functions ---
@st.cache_resource
def get_symptom_extractor():
    llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0)
    output_parser = PydanticOutputParser(pydantic_object=ExtractedSymptoms)
    prompt = PromptTemplate(
        template="""You are a medical symptom extraction assistant. Extract all medical symptoms from the user's description.

User Description: {user_input}

{format_instructions}

Important:
- Extract individual symptoms as separate items
- Normalize symptom names (e.g., "headache" not "my head hurts")
- Include duration if mentioned (e.g., "3 days", "1 week")
- Include severity if mentioned (e.g., "mild", "severe", "moderate")
- Only extract actual symptoms, not diagnoses or conditions
""",
        input_variables=["user_input"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()}
    )
    return prompt | llm | output_parser

@st.cache_resource
def get_diagnosis_chain():
    llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0.1)
    output_parser = PydanticOutputParser(pydantic_object=DiagnosisResult)
    prompt = PromptTemplate(
        template="""You are a medical diagnosis assistant. Based on the provided symptoms, generate a list of potential conditions.

IMPORTANT DISCLAIMER: This is for informational purposes only and NOT medical advice. Users should always consult healthcare professionals.

Symptoms: {symptoms}
Duration: {duration}
Severity: {severity}

{format_instructions}

Guidelines:
- List 2-5 most likely conditions based on symptoms
- Rank by probability (high, medium, low)
- Include brief, patient-friendly descriptions
- Provide general health recommendations
- Assess urgency level appropriately
- Be conservative - when in doubt, recommend professional consultation
""",
        input_variables=["symptoms", "duration", "severity"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()}
    )
    return prompt | llm | output_parser

@st.cache_resource
def get_summarizer_chain():
    llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0.2)
    prompt = PromptTemplate(
        template="""You are a medical information summarizer. Create a patient-friendly summary of the following medical research abstracts.

Context:
- Patient symptoms: {symptoms}
- Potential conditions: {conditions}

Research Abstracts:
{abstracts}

Create a summary that:
1. Explains relevant findings in simple, non-technical language
2. Highlights any important insights about the symptoms or conditions
3. Notes any recommended treatments or approaches mentioned in research
4. Maintains accuracy while being accessible to general audience

IMPORTANT: This summary is for informational purposes only. Always recommend consulting healthcare professionals.

Summary:""",
        input_variables=["symptoms", "conditions", "abstracts"]
    )
    return prompt | llm

def search_pubmed(query: str, max_results: int = 5) -> List[Dict]:
    """Search PubMed for relevant articles."""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    try:
        # Search for article IDs
        search_response = httpx.get(
            f"{base_url}esearch.fcgi",
            params={"db": "pubmed", "term": query, "retmax": str(max_results), "retmode": "json", "sort": "relevance"},
            timeout=30.0
        )
        search_data = search_response.json()
        id_list = search_data.get("esearchresult", {}).get("idlist", [])

        if not id_list:
            return []

        # Fetch article details
        fetch_response = httpx.get(
            f"{base_url}efetch.fcgi",
            params={"db": "pubmed", "id": ",".join(id_list), "retmode": "xml", "rettype": "abstract"},
            timeout=30.0
        )

        # Parse XML
        articles = []
        root = ET.fromstring(fetch_response.text)
        for article_elem in root.findall(".//PubmedArticle"):
            pmid = article_elem.findtext(".//PMID", "")
            title = article_elem.findtext(".//ArticleTitle", "")
            abstract = article_elem.findtext(".//Abstract/AbstractText", "")
            year = article_elem.findtext(".//PubDate/Year", "")
            if pmid and title:
                articles.append({"pmid": pmid, "title": title, "abstract": abstract, "year": year})
        return articles
    except Exception:
        return []

def build_search_query(symptoms: List[str], conditions: List[str] = None) -> str:
    terms = [f'"{s}"[Title/Abstract]' for s in symptoms[:3]]
    if conditions:
        terms += [f'"{c}"[Title/Abstract]' for c in conditions[:2]]
    return " OR ".join(terms) + " AND (review[pt] OR clinical trial[pt] OR meta-analysis[pt])"

# --- Main App ---
st.title("Medical Diagnosis Assistant")
st.markdown("---")

# Disclaimer banner
st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ Important Disclaimer:</strong> This tool is for informational purposes only and is NOT medical advice.
    Always consult with qualified healthcare professionals for diagnosis and treatment.
    If you are experiencing a medical emergency, please call emergency services immediately.
</div>
""", unsafe_allow_html=True)

# Check for API key
if not OPENAI_API_KEY:
    st.error("OpenAI API key not configured. Please add OPENAI_API_KEY to your Streamlit secrets.")
    st.stop()

# Main content
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Describe Your Symptoms")
    symptoms_input = st.text_area(
        "Enter your symptoms",
        placeholder="Example: I have been experiencing headaches, fatigue, and mild fever for 3 days...",
        height=200,
        help="Describe your symptoms in detail, including duration and severity if applicable."
    )
    submit_button = st.button("Get Diagnosis", type="primary", use_container_width=True)

with col2:
    if submit_button and symptoms_input:
        with st.spinner("Analyzing your symptoms..."):
            try:
                # Step 1: Extract symptoms
                extractor = get_symptom_extractor()
                extracted = extractor.invoke({"user_input": symptoms_input})

                if not extracted.symptoms:
                    st.error("Could not extract any symptoms from the provided description")
                    st.stop()

                # Step 2: Generate diagnosis
                diagnosis_chain = get_diagnosis_chain()
                diagnosis = diagnosis_chain.invoke({
                    "symptoms": ", ".join(extracted.symptoms),
                    "duration": extracted.duration or "Not specified",
                    "severity": extracted.severity or "Not specified"
                })

                # Step 3: Search PubMed
                condition_names = [c.name for c in diagnosis.conditions]
                query = build_search_query(extracted.symptoms, condition_names)
                articles = search_pubmed(query)

                # Step 4: Summarize
                summary_text = "No relevant medical literature found."
                if articles:
                    abstracts_text = "\n\n".join([
                        f"Title: {a['title']}\nAbstract: {a['abstract'] or 'No abstract'}"
                        for a in articles if a.get('abstract')
                    ])
                    if abstracts_text:
                        summarizer = get_summarizer_chain()
                        result = summarizer.invoke({
                            "symptoms": ", ".join(extracted.symptoms),
                            "conditions": ", ".join(condition_names),
                            "abstracts": abstracts_text
                        })
                        summary_text = result.content

                # Display results
                st.subheader("Analysis Results")

                # Extracted Symptoms
                st.markdown("#### Extracted Symptoms")
                symptoms_html = " ".join([f'<span class="symptom-tag">{s}</span>' for s in extracted.symptoms])
                st.markdown(symptoms_html, unsafe_allow_html=True)

                if extracted.duration:
                    st.markdown(f"**Duration:** {extracted.duration}")
                if extracted.severity:
                    st.markdown(f"**Severity:** {extracted.severity}")

                st.markdown("---")

                # Diagnosis
                st.markdown("#### Potential Conditions")
                urgency = diagnosis.urgency
                urgency_class = f"urgency-{urgency}"
                st.markdown(f'<p class="{urgency_class}">Urgency Level: {urgency.upper()}</p>', unsafe_allow_html=True)

                for condition in diagnosis.conditions:
                    prob = condition.probability
                    st.markdown(f"""
                    <div class="condition-card condition-{prob}">
                        <h4>{condition.name}</h4>
                        <p><strong>Probability:</strong> {prob.capitalize()}</p>
                        <p>{condition.description}</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Recommendations
                st.markdown("#### Recommendations")
                for rec in diagnosis.recommendations:
                    st.markdown(f"- {rec}")

                st.markdown("---")

                # PubMed Research
                st.markdown("#### Medical Research Summary")
                st.info(f"Found {len(articles)} relevant research articles")
                st.markdown(summary_text)

                if articles:
                    with st.expander("View References"):
                        for ref in articles:
                            pmid = ref["pmid"]
                            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                            year = ref.get("year", "N/A")
                            title = ref["title"]
                            st.markdown(f"- [{title}]({url}) ({year})")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    elif submit_button and not symptoms_input:
        st.warning("Please enter your symptoms before clicking 'Get Diagnosis'")
    else:
        st.info("Enter your symptoms in the text area and click 'Get Diagnosis' to receive an analysis.")

# Footer
st.markdown("---")
st.markdown("""
<small>
    <strong>Note:</strong> This application uses AI to analyze symptoms and search medical literature.
    Results should be used for informational purposes only and are not a substitute for professional medical advice.
</small>
""", unsafe_allow_html=True)
