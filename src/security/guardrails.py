import logging
from nemoguardrails import LLMRails, RailsConfig

logger = logging.getLogger(__name__)

class SecurityGuardrails:
    def __init__(self, config_path: str = "config/guardrails"):
        self.config = RailsConfig.from_path(config_path)
        self.rails = LLMRails(self.config)

    async def validate_input(self, user_prompt: str) -> dict:
        """
        Validates incoming user query against prompt injection attacks
        and security policy violations using NeMo Guardrails.
        """
        try:
            response = await self.rails.generate_async(
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            # Check if input triggered a guardrail refusal
            content = response.get("content", "")
            if "System Security Error" in content:
                logger.warning(f"Blocked malicious input attempt: {user_prompt[:50]}...")
                return {"is_safe": False, "reason": content}

            return {"is_safe": True, "clean_prompt": user_prompt}

        except Exception as e:
            logger.error(f"Error evaluating input guardrails: {str(e)}")
            return {"is_safe": False, "reason": "Security evaluation failed."}