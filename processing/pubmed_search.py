from typing import List, Dict, Any
import httpx
import xml.etree.ElementTree as ET

import config


class PubMedArticle:
    """Represents a PubMed article."""
    def __init__(self, pmid: str, title: str, abstract: str, year: str, authors: List[str]):
        self.pmid = pmid
        self.title = title
        self.abstract = abstract
        self.year = year
        self.authors = authors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pmid": self.pmid,
            "title": self.title,
            "abstract": self.abstract,
            "year": self.year,
            "authors": self.authors
        }


class PubMedSearch:
    """Search PubMed for relevant medical literature."""

    def __init__(self):
        self.base_url = config.PUBMED_BASE_URL
        self.api_key = config.PUBMED_API_KEY

    def _build_params(self, params: Dict[str, str]) -> Dict[str, str]:
        """Add API key to params if available."""
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    async def search(self, query: str, max_results: int = 5) -> List[PubMedArticle]:
        """Search PubMed for articles matching the query."""
        async with httpx.AsyncClient() as client:
            # Step 1: Search for article IDs
            search_params = self._build_params({
                "db": "pubmed",
                "term": query,
                "retmax": str(max_results),
                "retmode": "json",
                "sort": "relevance"
            })

            search_response = await client.get(
                f"{self.base_url}esearch.fcgi",
                params=search_params,
                timeout=30.0
            )
            search_data = search_response.json()

            id_list = search_data.get("esearchresult", {}).get("idlist", [])

            if not id_list:
                return []

            # Step 2: Fetch article details
            fetch_params = self._build_params({
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "xml",
                "rettype": "abstract"
            })

            fetch_response = await client.get(
                f"{self.base_url}efetch.fcgi",
                params=fetch_params,
                timeout=30.0
            )

            return self._parse_articles(fetch_response.text)

    def search_sync(self, query: str, max_results: int = 5) -> List[PubMedArticle]:
        """Synchronous version of search."""
        with httpx.Client() as client:
            # Step 1: Search for article IDs
            search_params = self._build_params({
                "db": "pubmed",
                "term": query,
                "retmax": str(max_results),
                "retmode": "json",
                "sort": "relevance"
            })

            search_response = client.get(
                f"{self.base_url}esearch.fcgi",
                params=search_params,
                timeout=30.0
            )
            search_data = search_response.json()

            id_list = search_data.get("esearchresult", {}).get("idlist", [])

            if not id_list:
                return []

            # Step 2: Fetch article details
            fetch_params = self._build_params({
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "xml",
                "rettype": "abstract"
            })

            fetch_response = client.get(
                f"{self.base_url}efetch.fcgi",
                params=fetch_params,
                timeout=30.0
            )

            return self._parse_articles(fetch_response.text)

    def _parse_articles(self, xml_text: str) -> List[PubMedArticle]:
        """Parse PubMed XML response into article objects."""
        articles = []

        try:
            root = ET.fromstring(xml_text)

            for article_elem in root.findall(".//PubmedArticle"):
                pmid = ""
                title = ""
                abstract = ""
                year = ""
                authors = []

                # Extract PMID
                pmid_elem = article_elem.find(".//PMID")
                if pmid_elem is not None:
                    pmid = pmid_elem.text or ""

                # Extract title
                title_elem = article_elem.find(".//ArticleTitle")
                if title_elem is not None:
                    title = title_elem.text or ""

                # Extract abstract
                abstract_elem = article_elem.find(".//Abstract/AbstractText")
                if abstract_elem is not None:
                    abstract = abstract_elem.text or ""

                # Extract year
                year_elem = article_elem.find(".//PubDate/Year")
                if year_elem is not None:
                    year = year_elem.text or ""
                else:
                    # Try MedlineDate if Year not available
                    medline_date = article_elem.find(".//PubDate/MedlineDate")
                    if medline_date is not None and medline_date.text:
                        year = medline_date.text[:4]

                # Extract authors
                for author_elem in article_elem.findall(".//Author"):
                    last_name = author_elem.find("LastName")
                    initials = author_elem.find("Initials")
                    if last_name is not None and last_name.text:
                        author_name = last_name.text
                        if initials is not None and initials.text:
                            author_name += f" {initials.text}"
                        authors.append(author_name)

                if pmid and title:
                    articles.append(PubMedArticle(
                        pmid=pmid,
                        title=title,
                        abstract=abstract,
                        year=year,
                        authors=authors[:3]  # Limit to first 3 authors
                    ))

        except ET.ParseError:
            pass

        return articles

    def build_search_query(self, symptoms: List[str], conditions: List[str] = None) -> str:
        """Build an optimized PubMed search query."""
        terms = []

        # Add symptoms
        for symptom in symptoms[:3]:  # Limit to top 3 symptoms
            terms.append(f'"{symptom}"[Title/Abstract]')

        # Add conditions if provided
        if conditions:
            for condition in conditions[:2]:  # Limit to top 2 conditions
                terms.append(f'"{condition}"[Title/Abstract]')

        # Combine with OR for broader results
        query = " OR ".join(terms)

        # Add filters for quality
        query += " AND (review[pt] OR clinical trial[pt] OR meta-analysis[pt])"

        return query
