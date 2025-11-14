"""
Large Language Model Integration for CallAssist

Provides interface to Ollama-hosted LLMs for generating conversational responses
to customer questions. Handles prompt engineering, streaming responses, and
response optimization for real-time call assistance.

Key Features:
- Streaming response generation for real-time conversation
- Intelligent prompt engineering with context and persona
- Response length and token limit management
- Customer name personalization
- Clarification request generation
- Performance metrics and latency tracking
- Model warm-up and connection management

Architecture:
- Uses Ollama REST API for model inference
- Builds structured prompts from Q&A knowledge base
- Implements response streaming for low-latency interaction
- Manages token limits to prevent context overflow
- Tracks performance metrics for optimization

Configuration:
- Model selection via CALLASSIST_MODEL environment variable
- Ollama endpoint via OLLAMA_ENDPOINT environment variable
- Token limits configurable via environment variables
- Generation parameters optimized for conversational AI

Dependencies:
- requests: HTTP client for Ollama API communication
- json: Response parsing and prompt formatting
- os: Environment variable configuration
- time: Performance timing and metrics

Author: Quinn Evans
"""

import json
import os
import time
from typing import Callable, List, Tuple

import requests

# Configuration constants with environment variable fallbacks
MAX_PERSONA_WORDS = int(os.getenv("CALLASSIST_PERSONA_TOKENS", "150"))
MAX_PROMPT_TOKENS = int(os.getenv("CALLASSIST_PROMPT_TOKENS", "700"))
MAX_RESPONSE_CHARS = int(os.getenv("CALLASSIST_RESPONSE_CHARS", "400"))

# Optimized generation parameters for conversational AI
DEFAULT_OPTIONS = {
    "num_predict": 60,      # Maximum tokens to generate
    "temperature": 0.2,     # Low temperature for consistent, factual responses
    "top_k": 20,           # Nucleus sampling parameter
    "top_p": 0.8,          # Top-p sampling for diverse but focused responses
    "repeat_penalty": 1.1,  # Penalize repetition
    "num_ctx": 1024,       # Context window size
}


class LLMManager:
    """
    Manages Large Language Model interactions for conversational AI.

    Interfaces with Ollama-hosted LLMs to generate natural, contextual responses
    for customer service calls. Handles prompt construction, response streaming,
    and performance optimization for real-time conversation assistance.

    The manager builds structured prompts from knowledge base Q&A pairs,
    incorporates conversation context, and ensures responses are concise,
    accurate, and personalized for solar energy consultation.

    Attributes:
        persona (str): Truncated system persona for prompt engineering
        endpoint (str): Ollama API endpoint URL
        model (str): Model name for inference
        session (requests.Session): Persistent HTTP session for efficiency
    """

    def __init__(self, persona: str):
        """
        Initialize the LLM manager with persona and configuration.

        Args:
            persona (str): System persona text describing AI assistant role
        """
        self.persona = self._truncate_persona(persona)
        self.endpoint = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")
        self.model = os.getenv("CALLASSIST_MODEL", "llama3.2:3b")

        # Use persistent session for connection reuse
        self.session = requests.Session()
        self.session.headers.update({"Connection": "keep-alive"})

        # Warm up model connection
        self._warm_up_model()

    # ---------------------------------------------------------------- Response Generation

    def generate_response(
        self,
        qa_bundle: list[dict],
        context: list[str] = None,
        customer_name: str = None,
        stream_callback: Callable[[str], None] = None,
        prompt_override: str | None = None,
    ) -> Tuple[str, bool]:
        """
        Generate a conversational response using the LLM.

        Builds a structured prompt from Q&A knowledge base, conversation context,
        and customer information. Supports streaming responses for real-time
        interaction.

        Args:
            qa_bundle (list[dict]): Q&A pairs with questions, answers, and scores
            context (list[str], optional): Recent conversation context
            customer_name (str, optional): Customer name for personalization
            stream_callback (callable, optional): Function called with each token

        Returns:
            Tuple[str, bool]: (response_text, success_flag)
                - response_text: Generated response or error message
                - success_flag: True if generation succeeded

        Note:
            Returns fallback message and False on generation failure.
            Streaming callback receives individual tokens for real-time display.
        """
        if not qa_bundle and not prompt_override:
            return "No relevant information found.", False

        # Build optimized prompt
        if prompt_override:
            prompt = prompt_override
            token_estimate = len(prompt_override.split())
        else:
            prompt, token_estimate = self._build_prompt(qa_bundle, context, customer_name)

        # Initialize streaming state
        chunks: List[str] = []
        first_token_time = None
        start = time.time()

        try:
            # Make streaming request to Ollama
            response = self.session.post(
                self.endpoint,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": True,
                    "options": DEFAULT_OPTIONS,
                },
                timeout=30,
                stream=True,
            )
            response.raise_for_status()

            # Process streaming response
            for line in response.iter_lines():
                if not line:
                    continue

                data = json.loads(line.decode("utf-8"))
                token = data.get("response", "")

                if token:
                    # Track first token latency
                    if first_token_time is None:
                        first_token_time = time.time()

                    chunks.append(token)

                    # Call streaming callback if provided
                    if callable(stream_callback):
                        stream_callback(token)

                # Check for completion
                if data.get("done"):
                    break

            # Process final response
            response_text = self._truncate_response("".join(chunks).strip())

            # Log performance metrics
            self._log_metrics(token_estimate, start, first_token_time, time.time())

            return response_text, True

        except Exception as exc:
            print(f"LLM error: {exc}")
            return None, False

    def generate_clarification(self, question: str, intent_guess: str = None) -> Tuple[str, bool]:
        """
        Generate a clarification request when question is unclear.

        Creates a polite request for the customer to rephrase their question,
        optionally incorporating intent classification hints.

        Args:
            question (str): Original unclear question
            intent_guess (str, optional): Guessed intent category for context

        Returns:
            Tuple[str, bool]: (clarification_text, success_flag)
        """
        prompt = self._clarification_prompt(question, intent_guess)

        try:
            response = self.session.post(
                self.endpoint,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {**DEFAULT_OPTIONS, "num_predict": 30},
                },
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip(), True

        except Exception as exc:
            print(f"LLM clarify error: {exc}")
            return "I'm not sure I caught that—could you rephrase it?", False

    # ------------------------------------------------------------- Prompt Engineering

    def _build_prompt(
        self,
        qa_bundle: list[dict],
        context: list[str],
        customer_name: str
    ) -> Tuple[str, int]:
        """
        Construct optimized prompt from Q&A data and context.

        Builds a structured prompt that incorporates the most relevant Q&A pairs,
        conversation context, and personalization directives.

        Args:
            qa_bundle: List of Q&A dictionaries with scores
            context: Recent conversation context
            customer_name: Customer name for personalization

        Returns:
            Tuple[str, int]: (constructed_prompt, token_estimate)
        """
        # Select top 2 most relevant Q&A pairs by score
        ranked = sorted(qa_bundle, key=lambda item: item.get("score", 0), reverse=True)[:2]

        # Format questions section
        questions_lines = [f"{idx + 1}. {item['question']}" for idx, item in enumerate(ranked)]
        questions_section = "\n".join(questions_lines)

        # Format facts section (limit to 4 facts total)
        facts_lines = []
        for idx, item in enumerate(ranked, start=1):
            for sub_idx, answer in enumerate((item.get("answers") or [])[:2], start=1):
                facts_lines.append(f"{idx}.{sub_idx} {answer}")
        facts_lines = facts_lines[:4]

        # Format context section
        context_lines = []
        if context:
            context_lines = [f"- {entry}" for entry in context[-2:]]

        # Build name personalization directive
        name_directive = ""
        if customer_name:
            name_directive = (
                f"\nThe caller's name is {customer_name}. Mention their name once naturally "
                "and keep a warm, consultative tone."
            )

        # Compose final prompt
        prompt = self._compose_prompt(questions_section, facts_lines, context_lines, name_directive)

        # Enforce token limits by truncating less critical content
        prompt = self._enforce_prompt_limit(prompt, questions_section, facts_lines, context_lines, name_directive)

        # Estimate token count (rough approximation)
        token_estimate = len(prompt.split())

        return prompt, token_estimate

    def _compose_prompt(
        self,
        questions_section: str,
        facts_lines: list[str],
        context_lines: list[str],
        name_directive: str
    ) -> str:
        """
        Assemble the final prompt with all components.

        Creates a structured prompt that guides the LLM to generate
        appropriate solar consultant responses.

        Args:
            questions_section: Formatted customer questions
            facts_lines: Relevant knowledge base facts
            context_lines: Recent conversation context
            name_directive: Customer name personalization instructions

        Returns:
            str: Complete prompt for LLM generation
        """
        # Build context block if available
        context_block = ""
        if context_lines:
            context_block = f"Recent customer context:\n{chr(10).join(context_lines)}\n\n"

        # Ensure facts section has content
        facts_section = "\n".join(facts_lines) or "No specific facts were available."

        # Construct prompt with clear structure and instructions
        return f"""
{self.persona}

You are speaking live on a call. Answer as a knowledgeable solar consultant.
- 1-2 short sentences.
- Use provided facts.
- Conversational, confident, concise.
- Address every question directly.
{name_directive}

{context_block}Customer asked:
{questions_section}

Facts:
{facts_section}

Response:"""

    def _enforce_prompt_limit(
        self,
        prompt: str,
        questions_section: str,
        facts_lines: list[str],
        context_lines: list[str],
        name_directive: str
    ) -> str:
        """
        Ensure prompt stays within token limits by truncating content.

        Progressively removes less critical content (facts, then context)
        until prompt fits within MAX_PROMPT_TOKENS.

        Args:
            prompt: Current prompt text
            questions_section: Questions (never truncated)
            facts_lines: Facts that can be truncated
            context_lines: Context that can be truncated
            name_directive: Name directive (never truncated)

        Returns:
            str: Truncated prompt within token limits
        """
        while len(prompt.split()) > MAX_PROMPT_TOKENS:
            if facts_lines:
                # Remove least important facts first
                facts_lines.pop()
            elif context_lines:
                # Remove oldest context if no facts left
                context_lines.pop()
            else:
                # Emergency truncation of entire prompt
                words = prompt.split()
                prompt = " ".join(words[:MAX_PROMPT_TOKENS])
                break

            # Rebuild prompt with reduced content
            prompt = self._compose_prompt(questions_section, facts_lines, context_lines, name_directive)

        return prompt

    def _truncate_persona(self, persona: str) -> str:
        """
        Truncate persona text to fit within token limits.

        Args:
            persona: Full persona text

        Returns:
            str: Truncated persona within MAX_PERSONA_WORDS
        """
        words = persona.split()
        if len(words) > MAX_PERSONA_WORDS:
            words = words[:MAX_PERSONA_WORDS]
        return " ".join(words)

    def _truncate_response(self, text: str) -> str:
        """
        Truncate response text to prevent overly long outputs.

        Args:
            text: Generated response text

        Returns:
            str: Truncated response within MAX_RESPONSE_CHARS
        """
        if not text:
            return text

        if len(text) <= MAX_RESPONSE_CHARS:
            return text

        # Truncate and add ellipsis
        return text[:MAX_RESPONSE_CHARS].rstrip() + "..."

    # ------------------------------------------------------------- Performance Monitoring

    def _log_metrics(
        self,
        token_estimate: int,
        start_time: float,
        first_token_time: float,
        end_time: float
    ):
        """
        Log performance metrics for monitoring and optimization.

        Args:
            token_estimate: Estimated input token count
            start_time: Request start timestamp
            first_token_time: First token received timestamp
            end_time: Request completion timestamp
        """
        total_duration = end_time - start_time
        first_token_latency = (first_token_time - start_time) if first_token_time else None

        log = f"LLM latency total={total_duration:.2f}s | prompt_tokens≈{token_estimate}"

        if first_token_latency is not None:
            log += f" | first_token={first_token_latency:.2f}s"

        print(log)

    # --------------------------------------------------------- Model Management

    def _warm_up_model(self):
        """
        Warm up the model connection to reduce first-request latency.

        Makes a minimal request to ensure the model is loaded and ready.
        Failures are ignored as warm-up is not critical.
        """
        try:
            self.session.post(
                self.endpoint,
                json={
                    "model": self.model,
                    "prompt": ".",  # Minimal prompt
                    "stream": False,
                    "options": {"num_predict": 1}  # Generate only 1 token
                },
                timeout=5,
            )
        except Exception:
            pass  # Warm-up failure is not critical

    def _clarification_prompt(self, question: str, intent_guess: str = None) -> str:
        """
        Build prompt for generating clarification requests.

        Args:
            question: Original unclear question
            intent_guess: Optional intent classification hint

        Returns:
            str: Prompt for clarification generation
        """
        intent_line = f"I suspect they might be asking about {intent_guess}." if intent_guess else ""

        return f"""
{self.persona}

The caller just asked: "{question}"
{intent_line}
Politely ask for clarification in 1 short sentence so you can answer accurately.
"""
