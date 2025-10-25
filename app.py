from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os, requests, fitz, spacy, json, traceback, uuid, re
# Optional OCR deps
try:
    from pdf2image import convert_from_path
    import pytesseract
except Exception:
    convert_from_path = None
    pytesseract = None
from utils.extractor import extract_name

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

# Simple learning resources per skill
RESOURCES = {
    "Python": [
        {"title": "Python Official Docs", "url": "https://docs.python.org/3/"},
        {"title": "Real Python Tutorials", "url": "https://realpython.com/"}
    ],
    "Java": [
        {"title": "Java Tutorials (Oracle)", "url": "https://docs.oracle.com/javase/tutorial/"},
        {"title": "Baeldung Java", "url": "https://www.baeldung.com/"}
    ],
    "JavaScript": [
        {"title": "MDN JavaScript", "url": "https://developer.mozilla.org/en-US/docs/Web/JavaScript"},
        {"title": "JavaScript.info", "url": "https://javascript.info/"}
    ],
    "C++": [
        {"title": "C++ Reference", "url": "https://en.cppreference.com/w/"},
        {"title": "C++ Tutorial (cplusplus.com)", "url": "https://cplusplus.com/doc/tutorial/"}
    ],
    "React": [
        {"title": "React Docs", "url": "https://react.dev/learn"},
        {"title": "Epic React (free articles)", "url": "https://epicreact.dev/articles/"}
    ],
    "Flask": [
        {"title": "Flask Docs", "url": "https://flask.palletsprojects.com/en/latest/"},
        {"title": "Miguel Grinberg Flask Mega-Tutorial", "url": "https://blog.miguelgrinberg.com/category/Flask"}
    ],
    "Django": [
        {"title": "Django Docs", "url": "https://docs.djangoproject.com/en/stable/"},
        {"title": "Django Girls Tutorial", "url": "https://tutorial.djangogirls.org/"}
    ],
    "SQL": [
        {"title": "Mode SQL Tutorial", "url": "https://mode.com/sql-tutorial/"},
        {"title": "SQLBolt", "url": "https://sqlbolt.com/"}
    ],
    "TensorFlow": [
        {"title": "TensorFlow Tutorials", "url": "https://www.tensorflow.org/tutorials"}
    ],
    "PyTorch": [
        {"title": "PyTorch Tutorials", "url": "https://pytorch.org/tutorials/"}
    ],
    "Machine Learning": [
        {"title": "scikit-learn Tutorials", "url": "https://scikit-learn.org/stable/tutorial/index.html"},
        {"title": "Andrew Ng ML Notes", "url": "https://cs229.stanford.edu/notes2021fall/"}
    ],
    "Deep Learning": [
        {"title": "DeepLearning.AI Short Courses", "url": "https://www.deeplearning.ai/short-courses/"}
    ],
    "NLP": [
        {"title": "Hugging Face Course", "url": "https://huggingface.co/learn/nlp-course/"}
    ],
    "Data Science": [
        {"title": "Data Science Handbook (free)", "url": "https://jakevdp.github.io/PythonDataScienceHandbook/"}
    ],
    "AI": [
        {"title": "AI for Everyone (overview)", "url": "https://www.deeplearning.ai/courses/ai-for-everyone/"}
    ]
}

# Offline fallback question bank (5 per common skill)
OFFLINE_QBANK = {
    "Java": [
        {"q": "Which collection does not allow duplicate elements?", "opts": ["List", "Set", "ArrayList", "Queue"], "ans": "B", "exp": "Set enforces uniqueness."},
        {"q": "What is JVM?", "opts": ["Java Virtual Machine", "Java Variable Module", "Just VM", "Java Version Manager"], "ans": "A", "exp": "JVM runs compiled bytecode."},
        {"q": "Which keyword prevents inheritance?", "opts": ["static", "final", "private", "sealed"], "ans": "B", "exp": "final on class stops inheritance."},
        {"q": "Default value of uninitialized int field?", "opts": ["0", "null", "undefined", "-1"], "ans": "A", "exp": "Primitives default to 0."},
        {"q": "Which is not a primitive type?", "opts": ["int", "boolean", "String", "char"], "ans": "C", "exp": "String is a class."}
    ],
    "Machine Learning": [
        {"q": "Which is not a learning paradigm?", "opts": ["Supervised", "Unsupervised", "Reinforcement", "Compilation"], "ans": "D", "exp": "Compilation is not an ML paradigm."},
        {"q": "Overfitting happens when?", "opts": ["Model too simple", "Model too complex", "More data", "Low variance"], "ans": "B", "exp": "High variance fits noise."},
        {"q": "Which metric for classification?", "opts": ["MSE", "Accuracy", "SSE", "RMSE"], "ans": "B", "exp": "Accuracy for classification."},
        {"q": "Which algorithm is linear model?", "opts": ["KNN", "Linear Regression", "DBSCAN", "Random Forest"], "ans": "B", "exp": "LR is linear."},
        {"q": "Train/validation/test split purpose?", "opts": ["Speed", "Regularization", "Evaluation", "Normalization"], "ans": "C", "exp": "Hold-out sets evaluate generalization."}
    ],
    "AI": [
        {"q": "Turing Test evaluates?", "opts": ["Speed", "Memory", "Intelligence", "Energy"], "ans": "C", "exp": "It evaluates intelligent behavior."},
        {"q": "Heuristic search example?", "opts": ["BFS", "DFS", "A*", "Dijkstra"], "ans": "C", "exp": "A* uses heuristics."},
        {"q": "Knowledge representation form?", "opts": ["Graphs", "Arrays", "Stacks", "Heaps"], "ans": "A", "exp": "Graphs represent relations."},
        {"q": "Agent perceives via?", "opts": ["Sensors", "Actuators", "Heuristics", "States"], "ans": "A", "exp": "Sensors perceive environment."},
        {"q": "Rational agent aims to?", "opts": ["Maximize cost", "Random actions", "Maximize performance measure", "Minimize actions"], "ans": "C", "exp": "By definition."}
    ],
    "Python": [
        {"q": "Which creates a list?", "opts": ["{}", "[]", "()", "set()"], "ans": "B", "exp": "[] is list literal."},
        {"q": "Immutable built-in type?", "opts": ["list", "dict", "set", "tuple"], "ans": "D", "exp": "tuples are immutable."},
        {"q": "List comprehension syntax?", "opts": ["[x for x in it]", "(x for x)", "{x:x}", "<x for x>"], "ans": "A", "exp": "Standard list comp."},
        {"q": "PEP 8 refers to?", "opts": ["Package", "Style Guide", "Interpreter", "Profiler"], "ans": "B", "exp": "PEP 8 is style guide."},
        {"q": "dict key must be?", "opts": ["Mutable", "Hashable", "Iterable", "Sorted"], "ans": "B", "exp": "Keys must be hashable."}
    ],
    "SQL": [
        {"q": "Which keyword sorts?", "opts": ["GROUP BY", "ORDER BY", "SORT", "RANK BY"], "ans": "B", "exp": "ORDER BY sorts."},
        {"q": "Which combines rows from two tables?", "opts": ["JOIN", "MERGE", "UNION ALL", "INTERSECT"], "ans": "A", "exp": "JOIN combines horizontally."},
        {"q": "Primary key property?", "opts": ["Nullable", "Duplicate", "Unique", "Text only"], "ans": "C", "exp": "Must be unique and non-null."},
        {"q": "Aggregate function?", "opts": ["COUNT", "WHERE", "LIKE", "BETWEEN"], "ans": "A", "exp": "COUNT aggregates."},
        {"q": "Filter rows after grouping?", "opts": ["WHERE", "HAVING", "LIMIT", "TOP"], "ans": "B", "exp": "HAVING filters groups."}
    ],
    "Flask": [
        {"q": "Flask is a ___ web framework.", "opts": ["full-stack", "micro", "asynchronous", "compiled"], "ans": "B", "exp": "Flask is a micro web framework."},
        {"q": "Default templating engine used with Flask?", "opts": ["Mustache", "Jinja2", "Mako", "EJS"], "ans": "B", "exp": "Flask integrates Jinja2."},
        {"q": "Which creates a route in Flask?", "opts": ["@app.map", "@app.route", "@app.url", "@app.path"], "ans": "B", "exp": "Use @app.route decorator."},
        {"q": "How to access form data in Flask?", "opts": ["request.data", "request.args", "request.form", "request.files only"], "ans": "C", "exp": "Form fields are in request.form."},
        {"q": "How to enable debug reload?", "opts": ["app.run(debug=True)", "app.debug()", "app.reload()", "flask --debug-only"], "ans": "A", "exp": "Pass debug=True to app.run."}
    ],
}

# In-memory store to avoid large client-side cookies
QUIZ_STORE = {}

# Simple company/role recommendations per skill
RECOMMENDATIONS = {
    "Python": [
        {"company": "Google", "role": "Software Engineer (Python)", "url": "https://www.google.com/about/careers/applications/jobs/results/?query=python"},
        {"company": "LinkedIn Jobs", "role": "Python Developer", "url": "https://www.linkedin.com/jobs/search/?keywords=Python%20Developer"}
    ],
    "Java": [
        {"company": "Amazon", "role": "Backend Engineer (Java)", "url": "https://www.amazon.jobs/en/search?base_query=java"},
        {"company": "LinkedIn Jobs", "role": "Java Developer", "url": "https://www.linkedin.com/jobs/search/?keywords=Java%20Developer"}
    ],
    "React": [
        {"company": "Meta", "role": "Frontend Engineer (React)", "url": "https://www.metacareers.com/jobs/?q=frontend"},
        {"company": "LinkedIn Jobs", "role": "React Developer", "url": "https://www.linkedin.com/jobs/search/?keywords=React%20Developer"}
    ],
    "SQL": [
        {"company": "Snowflake", "role": "Data Engineer (SQL)", "url": "https://careers.snowflake.com/us/en"},
        {"company": "LinkedIn Jobs", "role": "SQL Developer", "url": "https://www.linkedin.com/jobs/search/?keywords=SQL%20Developer"}
    ],
    "Machine Learning": [
        {"company": "NVIDIA", "role": "ML Engineer", "url": "https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite"},
        {"company": "LinkedIn Jobs", "role": "Machine Learning Engineer", "url": "https://www.linkedin.com/jobs/search/?keywords=Machine%20Learning%20Engineer"}
    ],
    "Deep Learning": [
        {"company": "OpenAI", "role": "Deep Learning Engineer", "url": "https://openai.com/careers"},
        {"company": "LinkedIn Jobs", "role": "Deep Learning", "url": "https://www.linkedin.com/jobs/search/?keywords=Deep%20Learning"}
    ],
    "Django": [
        {"company": "Canonical", "role": "Backend Engineer (Django)", "url": "https://canonical.com/careers"},
        {"company": "LinkedIn Jobs", "role": "Django Developer", "url": "https://www.linkedin.com/jobs/search/?keywords=Django%20Developer"}
    ],
    "Flask": [
        {"company": "JetBrains", "role": "Backend Engineer (Flask)", "url": "https://www.jetbrains.com/careers/jobs/"},
        {"company": "LinkedIn Jobs", "role": "Flask Developer", "url": "https://www.linkedin.com/jobs/search/?keywords=Flask%20Developer"}
    ],
    "JavaScript": [
        {"company": "Microsoft", "role": "Front-end Engineer (JS)", "url": "https://jobs.careers.microsoft.com/global/en/search?q=javascript"},
        {"company": "LinkedIn Jobs", "role": "JavaScript Developer", "url": "https://www.linkedin.com/jobs/search/?keywords=JavaScript%20Developer"}
    ],
}

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
            # If the PDF is scanned or extraction failed, attempt OCR fallback
            if (text is None) or (len(text.strip()) < 40):
                print("[WARN] Low text yield from PDF, trying OCR fallback...")
                if convert_from_path is None or pytesseract is None:
                    print("[WARN] OCR libraries not available. Install pdf2image and pytesseract to enable OCR.")
                    return text or ""
                try:
                    images = convert_from_path(path, dpi=300)
                    ocr_text_parts = []
                    for i, img in enumerate(images[:8]):  # limit pages for speed
                        try:
                            ocr_text_parts.append(pytesseract.image_to_string(img))
                        except Exception as e:
                            print(f"[ERROR] OCR page {i} failed: {e}")
                    ocr_text = "\n".join(ocr_text_parts)
                    print(f"[DEBUG] OCR extracted {len(ocr_text)} characters")
                    if len(ocr_text.strip()) > len((text or '').strip()):
                        return ocr_text
                except Exception as e:
                    print(f"[ERROR] OCR fallback failed: {e}")
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
def generate_quiz_questions(skills, per_skill=5):
    print("[DEBUG] Generating quiz with skills:", skills)
    if not skills:
        print("[WARN] No skills found — cannot generate quiz.")
        return []

    all_questions = []
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    for skill in skills:
        # Prefer offline bank for speed if available
        bank = OFFLINE_QBANK.get(skill)
        if bank:
            for item in bank[:per_skill]:
                all_questions.append({
                    "question": item["q"],
                    "options": item["opts"],
                    "correct_answer": item["ans"],
                    "explanation": item["exp"],
                    "skill": skill,
                })
            # Move to next skill (skip API for this one)
            continue
        prompt = (
            f"Generate {per_skill} multiple-choice questions strictly about this skill: {skill}.\n"
            "Each question must include:\n"
            "- Question text\n"
            "- 4 options (A, B, C, D)\n"
            "- The correct answer letter\n"
            "- A one-line explanation.\n"
            "Format it like this:\n"
            "Question: <text>\nA) ...\nB) ...\nC) ...\nD) ...\nCorrect Answer: <A/B/C/D>\nExplanation: <reason>\n"
        )
        try:
            print(f"[DEBUG] Sending request to Mistral API for skill: {skill} ...")
            payload = {
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 900
            }
            print("[DEBUG] Payload summary:", json.dumps(payload, indent=2)[:300])
            response = requests.post(MISTRAL_URL, headers=headers, json=payload, timeout=8)
            print("[DEBUG] Mistral status code:", response.status_code)
            with open("mistral_debug.json", "a", encoding="utf-8") as f:
                f.write("\n\n==== RESPONSE FOR SKILL: " + skill + " ====\n")
                f.write(response.text)
            if response.status_code != 200:
                print("[ERROR] Mistral returned non-200 status. Falling back to offline bank for:", skill)
                # Offline fallback
                bank = OFFLINE_QBANK.get(skill, [])
                for item in bank[:per_skill]:
                    all_questions.append({
                        "question": item["q"],
                        "options": item["opts"],
                        "correct_answer": item["ans"],
                        "explanation": item["exp"],
                        "skill": skill,
                    })
                continue
            data = response.json()
            if "choices" not in data or not data["choices"]:
                print("[ERROR] Empty response from Mistral. Falling back to offline bank for:", skill)
                bank = OFFLINE_QBANK.get(skill, [])
                for item in bank[:per_skill]:
                    all_questions.append({
                        "question": item["q"],
                        "options": item["opts"],
                        "correct_answer": item["ans"],
                        "explanation": item["exp"],
                        "skill": skill,
                    })
                continue
            content = data["choices"][0]["message"]["content"]
            print("[DEBUG] Received content (first 400 chars):\n", content[:400])
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
                    all_questions.append({
                        "question": question,
                        "options": options,
                        "correct_answer": correct,
                        "explanation": explanation,
                        "skill": skill
                    })
        except Exception as e:
            print("[ERROR] Exception while generating quiz for skill:", skill)
            traceback.print_exc()
            # Fallback on exception
            bank = OFFLINE_QBANK.get(skill, [])
            for item in bank[:per_skill]:
                all_questions.append({
                    "question": item["q"],
                    "options": item["opts"],
                    "correct_answer": item["ans"],
                    "explanation": item["exp"],
                    "skill": skill,
                })
            continue
    print(f"[DEBUG] Successfully compiled {len(all_questions)} questions across skills.")
    return all_questions

# -------------------------------
# Routes
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    print("[DEBUG] Accessed / route")
    if request.method == "POST":
        uploaded_file = request.files.get("resume")
        if not uploaded_file or uploaded_file.filename == "":
            return "\u26a0 No file uploaded."
        path = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_file.filename)
        uploaded_file.save(path)

        text = extract_text_from_resume(path)
        skills = extract_skills_from_text(text)[:5]
        # Try multiple strategies for name extraction (prefer NER first)
        name = None
        # spaCy PERSON NER on the first 3000 chars; prefer multi-token names
        try:
            doc = nlp(text[:3000])
            persons = [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON" and 3 <= len(ent.text.strip()) <= 60]
            persons = sorted(persons, key=lambda s: len(s.split()), reverse=True)
            if persons:
                name = persons[0]
        except Exception:
            pass
        if not name or name == "Unknown":
            name = extract_name(text)
        if not name or name == "Unknown":
            # Uppercase header heuristic from top of doc with header-word filtering
            head = (text[:800] or "").splitlines()
            header_phrases = {
                "education", "education details", "education qualification", "education qualifications",
                "qualification", "qualifications",
                "work experience", "professional experience", "experience", "projects", "personal projects",
                "skills", "technical skills", "certifications", "achievements", "awards", "contact",
                "interests", "hobbies", "publications", "languages", "responsibilities", "key responsibilities",
                "strengths", "references"
            }
            for line in head[:15]:
                line = line.strip()
                if not line or len(line) > 80:
                    continue
                # skip lines with contacts/links
                low = line.lower().strip(" :;.-")
                if any(k in low for k in ["email", "@", "www", "http", "phone", "+", "linkedin", "github"]):
                    continue
                if low in header_phrases:
                    continue
                tokens = [t for t in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ\-']*", line) if t]
                if 1 <= len(tokens) <= 4:
                    header_token_count = sum(1 for t in tokens if t.lower() in header_phrases)
                    if header_token_count / max(1, len(tokens)) >= 0.5:
                        continue
                    upper_ratio = sum(1 for t in tokens if t.isupper()) / len(tokens)
                    if upper_ratio >= 0.5 or all(t[0].isupper() for t in tokens):
                        name = " ".join(t.title() for t in tokens)
                        break
        if not name or name == "Unknown":
            # Email-based heuristic: extract from local-part like john.doe@, j_doe@, doe-john@
            try:
                m = re.search(r"\b([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+)\.[A-Za-z]{2,}\b", text)
                if m:
                    local = m.group(1)
                    parts = re.split(r"[._\-]+", local)
                    parts = [p for p in parts if p and p.isalpha()]
                    if len(parts) >= 2:
                        cand = parts[:3]
                        name = " ".join(w.title() for w in cand)
            except Exception:
                pass
        if not name or name == "Unknown":
            # Filename-based heuristic
            base = os.path.basename(uploaded_file.filename)
            base = os.path.splitext(base)[0]
            tokens = [t for t in re.split(r"[^A-Za-z]", base) if t]
            if tokens:
                # Take first 2-3 tokens as probable name
                name = " ".join(tokens[:3]).title()
        if not skills:
            return "\u26a0 No skills found in your resume."

        session["resume_name"] = name
        session["resume_skills"] = skills  # small list OK in cookie
        session["resume_path"] = path
        # Issue a small server-side ID for large data
        sid = session.get("sid") or uuid.uuid4().hex
        session["sid"] = sid
        QUIZ_STORE[sid] = {"questions": [], "results": []}
        return render_template("upload.html", name=name, skills=skills)
    return render_template("index.html")

@app.route("/quiz", methods=["GET", "POST"])
def quiz():
    sid = session.get("sid")
    if not sid or sid not in QUIZ_STORE:
        return redirect(url_for("index"))
    store = QUIZ_STORE[sid]
    questions = store.get("questions", [])
    if request.method == "GET" and not questions:
        skills = session.get("resume_skills", [])
        if not skills:
            return redirect(url_for("index"))
        questions = generate_quiz_questions(skills, per_skill=5)
        store["questions"] = questions
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
                "is_correct": is_correct,
                "skill": q.get("skill", "")
            })
        store["results"] = results
        return render_template("result.html", score=score, total=len(questions), results=results)
    return render_template("quiz.html", questions=questions)

@app.route("/resources", methods=["GET"])
def resources():
    sid = session.get("sid")
    if not sid or sid not in QUIZ_STORE:
        return redirect(url_for("quiz"))
    last_results = QUIZ_STORE[sid].get("results")
    if not last_results:
        return redirect(url_for("quiz"))
    # Aggregate by skill
    stats = {}
    for r in last_results:
        skill = r.get("skill", "General") or "General"
        if skill not in stats:
            stats[skill] = {"correct": 0, "total": 0}
        stats[skill]["total"] += 1
        if r.get("is_correct"):
            stats[skill]["correct"] += 1
    weak = []
    for skill, s in stats.items():
        acc = (s["correct"] / s["total"]) if s["total"] else 0
        if acc < 0.7:  # consider weak if <70% accuracy
            weak.append({
                "skill": skill,
                "accuracy": round(acc * 100, 1),
                "resources": RESOURCES.get(skill, [
                    {"title": "FreeCodeCamp", "url": "https://www.freecodecamp.org/learn/"},
                    {"title": "W3Schools", "url": "https://www.w3schools.com/"}
                ])
            })
    # If nothing weak, suggest general improvement links
    if not weak:
        weak = [{
            "skill": "Great job! No weak topics detected.",
            "accuracy": 100,
            "resources": [
                {"title": "Practice on LeetCode (Explore)", "url": "https://leetcode.com/explore/"},
                {"title": "HackerRank Interview Prep", "url": "https://www.hackerrank.com/interview/interview-preparation-kit"}
            ]
        }]
    return render_template("resources.html", weak=weak)

@app.route("/recommendations", methods=["GET"])
def recommendations():
    skills = session.get("resume_skills", [])
    if not skills:
        return redirect(url_for("index"))
    recs = []
    seen = set()
    for skill in skills:
        entries = RECOMMENDATIONS.get(skill, [])
        for e in entries:
            key = (e["company"], e["role"], e["url"])
            if key in seen:
                continue
            seen.add(key)
            recs.append({"skill": skill, **e})
    # Add generic job boards search for coverage
    if not recs:
        for skill in skills:
            recs.append({
                "skill": skill,
                "company": "LinkedIn Jobs",
                "role": f"{skill} roles",
                "url": f"https://www.linkedin.com/jobs/search/?keywords={skill.replace(' ', '%20')}"
            })
    return render_template("recommendations.html", recs=recs)

if __name__ == "__main__":
    print("[INFO] Starting Flask app...")
    app.run(debug=True)