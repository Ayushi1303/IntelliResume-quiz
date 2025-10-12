from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os, requests, fitz, spacy, json, traceback

app = Flask(__name__)
app.secret_key = "your_secret_key"

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

nlp = spacy.load("en_core_web_sm")

MISTRAL_API_KEY = "I14m9nTrNhTAiGPaHvLdaqRHrNKDkWDE"
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
MODEL_NAME = "mistral-small"

skill_keywords = ["Python", "Java", "Machine Learning", "Deep Learning", "Flask", "Django", "React", "SQL", "TensorFlow", "PyTorch", "AI", "NLP", "Data Science", "C++", "JavaScript"]

# -------------------------------
# Extract text from resume
# -------------------------------
def extract_text_from_resume(path):
    print(f"[DEBUG] Extracting text from: {path}")
    if path.endswith('.pdf'):
        try:
            doc = fitz.open(path)
            text = "".join([page.get_text() for page in doc])
            print(f"[DEBUG] Extracted {len(text)} characters from PDF")
            return text
        except Exception as e:
            print(f"[ERROR] PDF read failed: {e}")
            return ""
    return ""

# -------------------------------
# Extract skills
# -------------------------------
def extract_skills_from_text(text):
    print("[DEBUG] Extracting skills...")
    skills = [skill for skill in skill_keywords if skill.lower() in text.lower()]
    print(f"[DEBUG] Found skills: {skills}")
    return skills

# -------------------------------
# Generate quiz (with deep logging)
# -------------------------------
def generate_quiz_questions(skills, num_questions=10):
    print("[DEBUG] Generating quiz with skills:", skills)
    if not skills:
        print("[WARN] No skills found — cannot generate quiz.")
        return []

    skill_list = ', '.join(skills)
    prompt = (
        f"Generate {num_questions} multiple-choice questions based on these skills: {skill_list}.\n"
        "Each question must include:\n"
        "- Question text\n"
        "- 4 options (A, B, C, D)\n"
        "- The correct answer letter\n"
        "- A one-line explanation.\n"
        "Format it like this:\n"
        "Question: <text>\nA) ...\nB) ...\nC) ...\nD) ...\nCorrect Answer: <A/B/C/D>\nExplanation: <reason>\n"
    )

    try:
        print("[DEBUG] Sending request to Mistral API...")
        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 800
        }

        print("[DEBUG] Payload summary:", json.dumps(payload, indent=2)[:300])
        response = requests.post(MISTRAL_URL, headers=headers, json=payload, timeout=40)
        print("[DEBUG] Mistral status code:", response.status_code)

        # Save raw response for debugging
        with open("mistral_debug.json", "w", encoding="utf-8") as f:
            f.write(response.text)

        if response.status_code != 200:
            print("[ERROR] Mistral returned non-200 status.")
            print("[DEBUG] Response body:", response.text)
            return [{"error": f"Mistral API error: {response.status_code}", "raw": response.text}]

        data = response.json()
        if "choices" not in data or not data["choices"]:
            print("[ERROR] No 'choices' key or empty response from Mistral.")
            print("[DEBUG] Full JSON:", data)
            return [{"error": "Empty response from Mistral", "raw": data}]

        content = data["choices"][0]["message"]["content"]
        print("[DEBUG] Received content (first 400 chars):\n", content[:400])

        # -------------------------------
        # Parse the generated quiz text
        # -------------------------------
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
                    "skill": "General"
                })

        print(f"[DEBUG] Successfully parsed {len(questions)} questions.")
        return questions

    except Exception as e:
        print("[ERROR] Exception while generating quiz:")
        traceback.print_exc()
        return [{"error": str(e)}]

# -------------------------------
# Routes
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    print("[DEBUG] Accessed / route")
    if request.method == "POST":
        uploaded_file = request.files.get("resume")
        if not uploaded_file or uploaded_file.filename == "":
            return "⚠ No file uploaded."
        path = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_file.filename)
        uploaded_file.save(path)

        text = extract_text_from_resume(path)
        skills = extract_skills_from_text(text)[:3]
        if not skills:
            return "⚠ No skills found in your resume."

        questions = generate_quiz_questions(skills, 5)

        # If questions contain error info
        if len(questions) == 1 and "error" in questions[0]:
            err_msg = questions[0]["error"]
            raw = questions[0].get("raw", "")
            return f"⚠ Quiz generation failed: {err_msg}<br><pre>{raw}</pre>"

        if not questions:
            return "⚠ Failed to generate quiz questions. Please check your Mistral API key or internet connection."

        session["quiz_questions"] = questions
        return redirect(url_for("quiz"))
    return render_template("index.html")

@app.route("/quiz", methods=["GET", "POST"])
def quiz():
    questions = session.get("quiz_questions", [])
    if not questions:
        return redirect(url_for("index"))
    if request.method == "POST":
        score = 0
        results = []
        for i, q in enumerate(questions):
            user_ans = request.form.get(f"q{i}")
            correct = q["correct_answer"]
            is_correct = user_ans == correct
            if is_correct:
                score += 1
            results.append({
                "question": q["question"],
                "user_answer": user_ans,
                "correct_answer": correct,
                "explanation": q["explanation"],
                "is_correct": is_correct
            })
        return render_template("result.html", score=score, total=len(questions), results=results)
    return render_template("quiz.html", questions=questions)

if __name__ == "__main__":
    print("[INFO] Starting Flask app...")
    app.run(debug=True)