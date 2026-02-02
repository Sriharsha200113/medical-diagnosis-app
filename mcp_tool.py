import json
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from processing import SymptomExtractor, DiagnosisModule, PubMedSearch, Summarizer


# Initialize MCP server
server = Server("medical-diagnosis")

# Initialize processing modules
symptom_extractor = SymptomExtractor()
diagnosis_module = DiagnosisModule()
pubmed_search = PubMedSearch()
summarizer = Summarizer()


MEDICAL_DISCLAIMER = (
    "IMPORTANT: This information is for educational purposes only and is NOT medical advice. "
    "Always consult with qualified healthcare professionals for diagnosis and treatment. "
    "If you are experiencing a medical emergency, please call emergency services immediately."
)


@server.list_tools()
async def list_tools():
    """List available tools."""
    return [
        Tool(
            name="diagnose_symptoms",
            description=(
                "Analyze medical symptoms and provide potential diagnoses. "
                "Takes a description of symptoms and returns extracted symptoms, "
                "potential conditions, and relevant medical research summaries. "
                "NOTE: This is for informational purposes only and is NOT medical advice."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symptoms": {
                        "type": "string",
                        "description": "A description of the symptoms (e.g., 'I have a headache, fever, and fatigue for 3 days')"
                    }
                },
                "required": ["symptoms"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """Handle tool calls."""
    if name != "diagnose_symptoms":
        raise ValueError(f"Unknown tool: {name}")

    symptoms_input = arguments.get("symptoms", "")

    if not symptoms_input or not symptoms_input.strip():
        return [TextContent(
            type="text",
            text="Error: Please provide a description of your symptoms."
        )]

    try:
        # Step 1: Extract symptoms from user input
        extracted = await symptom_extractor.extract(symptoms_input)

        if not extracted.symptoms:
            return [TextContent(
                type="text",
                text="Could not extract any symptoms from the provided description. Please provide more details about your symptoms."
            )]

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

        # Format response
        result = {
            "disclaimer": MEDICAL_DISCLAIMER,
            "extracted_symptoms": {
                "symptoms": extracted.symptoms,
                "duration": extracted.duration or "Not specified",
                "severity": extracted.severity or "Not specified"
            },
            "diagnosis": {
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
            "pubmed_research": summary.to_dict()
        }

        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]

    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Error processing symptoms: {str(e)}"
        )]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
