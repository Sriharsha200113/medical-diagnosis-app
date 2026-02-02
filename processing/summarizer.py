from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import config
from .pubmed_search import PubMedArticle


class PubMedSummary:
    """Represents a summarized PubMed search result."""
    def __init__(
        self,
        articles_found: int,
        summary: str,
        references: List[Dict[str, Any]]
    ):
        self.articles_found = articles_found
        self.summary = summary
        self.references = references

    def to_dict(self) -> Dict[str, Any]:
        return {
            "articles_found": self.articles_found,
            "summary": self.summary,
            "references": self.references
        }


class Summarizer:
    """Summarize PubMed articles into patient-friendly information."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=0.2
        )

        self.prompt = PromptTemplate(
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

        self.chain = self.prompt | self.llm

    async def summarize(
        self,
        articles: List[PubMedArticle],
        symptoms: List[str],
        conditions: List[str]
    ) -> PubMedSummary:
        """Summarize PubMed articles."""
        if not articles:
            return PubMedSummary(
                articles_found=0,
                summary="No relevant medical literature found for these symptoms.",
                references=[]
            )

        # Prepare abstracts text
        abstracts_text = "\n\n".join([
            f"Title: {article.title}\nAbstract: {article.abstract or 'No abstract available'}"
            for article in articles
            if article.abstract
        ])

        if not abstracts_text:
            abstracts_text = "No abstracts available for the found articles."

        # Generate summary
        result = await self.chain.ainvoke({
            "symptoms": ", ".join(symptoms),
            "conditions": ", ".join(conditions),
            "abstracts": abstracts_text
        })

        # Prepare references
        references = [
            {
                "title": article.title,
                "pmid": article.pmid,
                "year": article.year,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{article.pmid}/"
            }
            for article in articles
        ]

        return PubMedSummary(
            articles_found=len(articles),
            summary=result.content,
            references=references
        )

    def summarize_sync(
        self,
        articles: List[PubMedArticle],
        symptoms: List[str],
        conditions: List[str]
    ) -> PubMedSummary:
        """Synchronous version of summarize."""
        if not articles:
            return PubMedSummary(
                articles_found=0,
                summary="No relevant medical literature found for these symptoms.",
                references=[]
            )

        # Prepare abstracts text
        abstracts_text = "\n\n".join([
            f"Title: {article.title}\nAbstract: {article.abstract or 'No abstract available'}"
            for article in articles
            if article.abstract
        ])

        if not abstracts_text:
            abstracts_text = "No abstracts available for the found articles."

        # Generate summary
        result = self.chain.invoke({
            "symptoms": ", ".join(symptoms),
            "conditions": ", ".join(conditions),
            "abstracts": abstracts_text
        })

        # Prepare references
        references = [
            {
                "title": article.title,
                "pmid": article.pmid,
                "year": article.year,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{article.pmid}/"
            }
            for article in articles
        ]

        return PubMedSummary(
            articles_found=len(articles),
            summary=result.content,
            references=references
        )
