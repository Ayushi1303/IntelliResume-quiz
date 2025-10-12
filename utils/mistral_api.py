from mistralai import Mistral

client = Mistral(api_key="aKFEMuDwJOvtphHDDOrh2qbfRP7jEA1L")
model = "mistral-large-latest"

def generate_quiz(skill):
    prompt = (
        f"Generate 5 multiple choice questions for the skill '{skill}', with 4 options each. "
        "Return as JSON with fields: question, options, answer."
    )

    response = client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


def get_explanation(question, user_answer, correct_answer):
    prompt = (
        f"The user answered '{user_answer}' for the following question:\n{question}\n"
        f"But the correct answer is '{correct_answer}'. Explain why the user's answer is wrong "
        f"and why the correct answer is right."
    )

    response = client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
