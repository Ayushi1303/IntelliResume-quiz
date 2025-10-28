import os
import requests
import json

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "I14m9nTrNhTAiGPaHvLdaqRHrNKDkWDE")
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
DEFAULT_MODEL = "mistral-small"

def _headers():
    return {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

def generate_quiz(skill: str, per_skill: int = 5, model: str = DEFAULT_MODEL, temperature: float = 0.9):
    if not MISTRAL_API_KEY:
        raise RuntimeError("MISTRAL_API_KEY not set in environment")
    prompt = (
        f"Generate {per_skill} multiple-choice questions strictly about this skill: {skill}.\n"
        "Each question must include:\n"
        "- Question text\n"
        "- 4 options (A, B, C, D)\n"
        "- The correct answer letter\n"
        "- A one-line explanation.\n"
        "Format it like this:\n"
        "Question: <text>\nA) ...\nB) ...\nC) ...\nD) ...\nCorrect Answer: <A/B/C/D>\nExplanation: <reason>\n"
        "Vary difficulty (easy/medium/hard) and do not repeat prior phrasings.\n"
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 900
    }
    resp = requests.post(MISTRAL_URL, headers=_headers(), json=payload, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(f"Mistral error {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    questions = []
    blocks = content.split("Question:")[1:]
    for block in blocks:
        lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
        if not lines:
            continue
        question = lines[0]
        options, correct, explanation = [], None, ""
        for line in lines[1:]:
            if line.startswith(("A)", "B)", "C)", "D)")):
                options.append(line[3:].strip())
            elif line.startswith("Correct Answer:"):
                correct = line.split(":")[1].strip()[0]
            elif line.startswith("Explanation:"):
                explanation = line.split(":", 1)[1].strip()
        if question and len(options) == 4 and correct:
            questions.append({
                "question": question,
                "options": options,
                "correct_answer": correct,
                "explanation": explanation,
                "skill": skill
            })
    return questions

def get_explanation(question: str, user_answer: str, correct_answer: str, model: str = DEFAULT_MODEL, temperature: float = 0.4):
    if not MISTRAL_API_KEY:
        raise RuntimeError("MISTRAL_API_KEY not set in environment")
    prompt = (
        f"The user answered '{user_answer}' for the following question:\n{question}\n"
        f"But the correct answer is '{correct_answer}'. Explain briefly why the user's answer is wrong "
        f"and why the correct answer is right in 2-3 sentences."
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 200
    }
    resp = requests.post(MISTRAL_URL, headers=_headers(), json=payload, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(f"Mistral error {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "")
