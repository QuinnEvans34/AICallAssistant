"""
Question-Answer Matching Engine

Provides fuzzy string matching between user questions and a knowledge base
of Q&A pairs. Uses RapidFuzz library for high-performance approximate string
matching with configurable similarity thresholds.

Key Features:
- Fuzzy string matching with configurable similarity threshold
- Returns ranked matches with confidence scores
- Optimized for solar panel and renewable energy Q&A knowledge base
- Fast matching suitable for real-time applications

Algorithm:
- Uses Levenshtein distance ratio for similarity scoring (0-100)
- Sorts results by descending similarity score
- Returns top N matches regardless of threshold (for fallback handling)
- Case-insensitive matching for robustness

Dependencies:
- rapidfuzz: High-performance fuzzy string matching library

Author: Quinn Evans
"""

from rapidfuzz import fuzz


class Matcher:
    """
    Fuzzy matching engine for question-answer knowledge base lookup.

    Matches user questions against a predefined Q&A dataset using fuzzy string
    similarity. Designed for solar panel and renewable energy domain questions,
    providing ranked matches with confidence scores for intelligent routing.

    The matcher uses case-insensitive fuzzy matching with a configurable
    similarity threshold. It always returns the top matches for fallback
    handling, even if they fall below the threshold.

    Attributes:
        qa_data (list[dict]): Knowledge base of Q&A pairs
            Format: [{"question": str, "answer": str}, ...]
        threshold (int): Minimum similarity score (0-100) for confident matches
    """

    def __init__(self, qa_data: list[dict]):
        """
        Initialize the matcher with Q&A knowledge base.

        Args:
            qa_data (list[dict]): List of Q&A dictionaries with "question"
                and "answer" keys. Questions should be canonical forms.
        """
        self.qa_data = qa_data
        self.threshold = 50  # Lowered threshold for better matching in domain-specific Q&A

    def match(self, text: str) -> list[dict]:
        """
        Find matching Q&A pairs for the input text.

        Performs fuzzy string matching against all questions in the knowledge
        base, returning ranked results by similarity score. Always returns
        the top 5 matches for comprehensive fallback options.

        Args:
            text (str): User question to match against knowledge base

        Returns:
            list[dict]: Top 5 matches sorted by descending similarity score.
                Each match contains:
                - "score" (int): Similarity score (0-100, higher is better)
                - "question" (str): Canonical question from knowledge base
                - "answer" (str): Corresponding answer text

        Note:
            Returns matches even below threshold to ensure fallback options.
            Caller should evaluate scores for confidence-based routing.
        """
        matches = []
        lowered = text.lower()

        # Compare input against all knowledge base questions
        for item in self.qa_data:
            # Calculate fuzzy similarity ratio (0-100)
            score = fuzz.ratio(lowered, item["question"].lower())
            matches.append({
                "score": score,
                "question": item["question"],
                "answer": item["answer"]
            })

        # Sort by descending similarity score
        matches.sort(reverse=True, key=lambda x: x["score"])

        # Return top 5 matches for comprehensive fallback
        return matches[:5]
