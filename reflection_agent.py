# reflection_agent.py

from prompt import reflection_prompt
from vertexai.generative_models import GenerativeModel
import logging

logger = logging.getLogger(__name__)

class ReflectionAgent:
    def __init__(self, model_name: str, temperature: float = 0.0, top_p: float = 0.0, top_k: float = 1):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    def reflect(self, question: str, failed_query: str, error_msg: str) -> str:
        """Generate feedback and suggestions to fix a failed SQL query."""
        try:
            if not question or not failed_query or not error_msg:
                logger.warning("Reflection skipped: insufficient input.")
                return ""

            logger.info("Generating reflection suggestions based on query failure")
            prompt = reflection_prompt(question, failed_query, error_msg)
            logger.debug(f"Reflection Prompt:\n{prompt}")

            model = GenerativeModel(self.model_name)
            config = {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k
            }

            response = model.generate_content(prompt, generation_config=config)
            suggestions = response.text.strip()
            logger.debug(f"Reflection suggestions:\n{suggestions}")
            return suggestions

        except Exception as e:
            logger.error(f"ReflectionAgent failed: {e}", exc_info=True)
            return ""
