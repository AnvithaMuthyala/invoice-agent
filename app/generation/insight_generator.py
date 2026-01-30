from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import config
from app.parsers.ocr_parser import ParsedInvoice


class InsightGenerator:
    """Generates insights from parsed invoice data"""

    def __init__(self, provider: str = "groq"):
        """
        Initialize the insight generator.

        Args:
            provider: "groq" or "gemini"
        """
        if provider == "groq":
            self.llm = ChatGroq(
                api_key=config.GROQ_API_KEY,
                model=config.GENERATION_MODEL,
            )
        else:
            self.llm = ChatGoogleGenerativeAI(
                api_key=config.GOOGLE_API_KEY,
                model=config.GENERATION_MODEL,
            )

    def generate(self, parsed_invoice: ParsedInvoice) -> list[str]:
        """
        Generate insights from parsed invoice.
        The LLM decides how many insights to generate.

        Args:
            parsed_invoice: Structured invoice data

        Returns:
            List of insight strings
        """
        system_prompt = """You are an expert financial analyst. Given invoice data,
generate interesting and actionable insights.

IMPORTANT: You must decide how many insights to generate. Choose a random number
that feels appropriate for the invoice complexity - it could be anywhere from 2 to 10.
Don't always pick the same number. Let the data guide you.

Each insight should be:
- Specific and reference actual data from the invoice
- Useful for business decision-making
- Clear and concise (1-2 sentences each)

Return ONLY a numbered list of insights, nothing else."""

        user_prompt = f"""Analyze this invoice data and generate insights.
You decide how many insights are appropriate for this invoice.

Invoice Data:
{parsed_invoice.raw_text}

Generate your insights:"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = self.llm.invoke(messages)
        insights = self._parse_insights(response.content)

        return insights

    def _parse_insights(self, response: str) -> list[str]:
        """Parse LLM response into list of insights"""
        lines = response.strip().split("\n")
        insights = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Remove numbering (1., 2., 1), 2), etc.)
            if line[0].isdigit():
                for i, char in enumerate(line):
                    if char in ".)" and i < 3:
                        line = line[i + 1 :].strip()
                        break
            if line:
                insights.append(line)

        return insights
