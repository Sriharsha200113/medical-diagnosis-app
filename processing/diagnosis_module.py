from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

import config


class Condition(BaseModel):
    """A potential medical condition."""
    name: str = Field(description="Name of the condition")
    probability: str = Field(description="Probability level: high, medium, or low")
    description: str = Field(description="Brief description of the condition")


class DiagnosisResult(BaseModel):
    """Schema for diagnosis results."""
    conditions: List[Condition] = Field(description="List of potential conditions")
    recommendations: List[str] = Field(description="General health recommendations")
    urgency: str = Field(description="Urgency level: emergency, urgent, routine, or self-care")


class DiagnosisModule:
    """Generate potential diagnoses based on extracted symptoms."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=0.1
        )
        self.output_parser = PydanticOutputParser(pydantic_object=DiagnosisResult)

        self.prompt = PromptTemplate(
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
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )

        self.chain = self.prompt | self.llm | self.output_parser

    async def diagnose(self, symptoms: List[str], duration: str = "", severity: str = "") -> DiagnosisResult:
        """Generate diagnosis based on symptoms."""
        result = await self.chain.ainvoke({
            "symptoms": ", ".join(symptoms),
            "duration": duration or "Not specified",
            "severity": severity or "Not specified"
        })
        return result

    def diagnose_sync(self, symptoms: List[str], duration: str = "", severity: str = "") -> DiagnosisResult:
        """Synchronous version of diagnose."""
        result = self.chain.invoke({
            "symptoms": ", ".join(symptoms),
            "duration": duration or "Not specified",
            "severity": severity or "Not specified"
        })
        return result
