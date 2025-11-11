import time
import requests


class LLMManager:
    def __init__(self, persona):
        self.persona = persona
        self.endpoint = "http://localhost:11434/api/generate"
        self.model = "mistral"
        self.session = requests.Session()
        self.session.headers.update({"Connection": "keep-alive"})

    def generate_response(self, qa_bundle, context=None, customer_name=None):
        if not qa_bundle:
            return "No relevant information found.", False

        questions_section = "\n".join(
            f"{idx + 1}. {item['question']}"
            for idx, item in enumerate(qa_bundle)
        )
        facts_lines = []
        for idx, item in enumerate(qa_bundle, start=1):
            for sub_idx, answer in enumerate(item["answers"], start=1):
                facts_lines.append(f"{idx}.{sub_idx} {answer}")

        facts_section = "\n".join(facts_lines)

        context_section = ""
        if context:
            context_lines = "\n".join(f"- {entry}" for entry in context)
            context_section = f"Recent customer context:\n{context_lines}\n\n"

        name_directive = ""
        if customer_name:
            name_directive = (
                f"\nThe caller's name is {customer_name}. Mention their name naturally once in your response "
                "and keep a warm, consultative tone."
            )

        prompt = f"""
{self.persona}

You are speaking live on a call. Answer as a knowledgeable solar consultant.
- 1-2 short sentences.
- Use provided facts.
- Conversational, confident, concise.
- Address all questions.
{name_directive}

{context_section}Customer asked:
{questions_section}

Facts:
{facts_section}

Response:"""
        try:
            start_time = time.time()
            response = self.session.post(
                self.endpoint,
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=25,
            )
            response.raise_for_status()
            data = response.json()
            generated = data.get("response", "").strip()
            elapsed = time.time() - start_time
            print(f"LLM response time: {elapsed:.2f} s")
            return generated, True
        except Exception as exc:
            print(f"LLM error: {exc}")
            return None, False
