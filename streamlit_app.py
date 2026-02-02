import streamlit as st
import httpx

# Page configuration
st.set_page_config(
    page_title="Medical Diagnosis Assistant",
    page_icon="🏥",
    layout="wide"
)

# API configuration
API_URL = "http://localhost:8000"

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

# Header
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
                # Call the API
                response = httpx.post(
                    f"{API_URL}/diagnose",
                    json={"symptoms": symptoms_input},
                    timeout=60.0
                )

                if response.status_code == 200:
                    data = response.json()

                    # Display results
                    st.subheader("Analysis Results")

                    # Extracted Symptoms Section
                    st.markdown("#### Extracted Symptoms")
                    symptoms_html = " ".join([
                        f'<span class="symptom-tag">{symptom}</span>'
                        for symptom in data["symptoms"]
                    ])
                    st.markdown(symptoms_html, unsafe_allow_html=True)

                    if data.get("duration"):
                        st.markdown(f"**Duration:** {data['duration']}")
                    if data.get("severity"):
                        st.markdown(f"**Severity:** {data['severity']}")

                    st.markdown("---")

                    # Diagnosis Section
                    st.markdown("#### Potential Conditions")

                    diagnosis = data["diagnosis"]

                    # Urgency indicator
                    urgency = diagnosis.get("urgency", "routine")
                    urgency_class = f"urgency-{urgency}"
                    st.markdown(f'<p class="{urgency_class}">Urgency Level: {urgency.upper()}</p>', unsafe_allow_html=True)

                    # Display conditions
                    for condition in diagnosis["conditions"]:
                        prob = condition["probability"]
                        prob_class = f"condition-{prob}"

                        st.markdown(f"""
                        <div class="condition-card {prob_class}">
                            <h4>{condition["name"]}</h4>
                            <p><strong>Probability:</strong> {prob.capitalize()}</p>
                            <p>{condition["description"]}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Recommendations
                    st.markdown("#### Recommendations")
                    for rec in diagnosis["recommendations"]:
                        st.markdown(f"- {rec}")

                    st.markdown("---")

                    # PubMed Research Section
                    st.markdown("#### Medical Research Summary")

                    pubmed = data["pubmed_summary"]
                    st.info(f"Found {pubmed['articles_found']} relevant research articles")

                    st.markdown(pubmed["summary"])

                    # References
                    if pubmed.get("references"):
                        with st.expander("View References"):
                            for ref in pubmed["references"]:
                                pmid = ref["pmid"]
                                url = ref.get("url", f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
                                year = ref.get("year", "N/A")
                                title = ref["title"]
                                st.markdown(f"- [{title}]({url}) ({year})")

                else:
                    error_detail = response.json().get("detail", "Unknown error")
                    st.error(f"Error: {error_detail}")

            except httpx.ConnectError:
                st.error("Could not connect to the API server. Please make sure the server is running on http://localhost:8000")
            except httpx.TimeoutException:
                st.error("Request timed out. Please try again.")
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
