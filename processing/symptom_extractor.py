from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

import config


class ExtractedSymptoms(BaseModel):
    """Schema for extracted symptoms."""
    symptoms: List[str] = Field(description="List of extracted medical symptoms")
    duration: str = Field(default="", description="Duration of symptoms if mentioned")
    severity: str = Field(default="", description="Severity level if mentioned")


class SymptomExtractor:
    """Extract structured symptoms from user descriptions using GPT-4."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=0
        )
        self.output_parser = PydanticOutputParser(pydantic_object=ExtractedSymptoms)

        self.prompt = PromptTemplate(
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
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )

        self.chain = self.prompt | self.llm | self.output_parser

    async def extract(self, user_input: str) -> ExtractedSymptoms:
        """Extract symptoms from user input."""
        result = await self.chain.ainvoke({"user_input": user_input})
        return result

    def extract_sync(self, user_input: str) -> ExtractedSymptoms:
        """Synchronous version of extract."""
        result = self.chain.invoke({"user_input": user_input})
        return result
