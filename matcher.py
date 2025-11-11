from rapidfuzz import fuzz


class Matcher:
    def __init__(self, qa_data):
        self.qa_data = qa_data
        self.threshold = 50  # Lowered threshold for better matching

    def match(self, text):
        matches = []
        lowered = text.lower()
        for item in self.qa_data:
            score = fuzz.ratio(lowered, item["question"].lower())
            if score >= self.threshold:
                matches.append(
                    {"score": score, "question": item["question"], "answer": item["answer"]}
                )
        matches.sort(reverse=True, key=lambda x: x["score"])
        return matches
