from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os, requests, fitz, spacy, json, traceback, uuid, re, time, random
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

# Load lexicon of names (e.g., Indian names) from names.txt
NAMES_LEXICON = set()
try:
    names_path = os.path.join(os.path.dirname(__file__), "names.txt")
    with open(names_path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                NAMES_LEXICON.add(name.lower())
    print(f"[INFO] Loaded {len(NAMES_LEXICON)} names from names.txt")
except Exception as e:
    print(f"[WARN] Could not load names.txt: {e}")

def find_name_from_lexicon(text: str, names_set: set) -> str:
    if not text or not names_set:
        print("[WARN] No text or names set")
        return ""
    head = (text or "")[:4000]
    # tokenization allowing accents, hyphens, apostrophes
    tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'-]*\.?", head)
    # Prefer earliest hit in header region, then try to expand to a likely full name
    particles = {"de", "da", "del", "della", "van", "von", "bin", "binti", "al", "el", "la", "le", "di", "dos", "das", "do", "du", "mac", "mc", "o'", "d'"}
    locations = {"india", "bengaluru", "bangalore", "mumbai", "delhi", "new delhi", "chennai", "hyderabad", "pune", "kolkata", "ahmedabad", "gurgaon", "noida", "bhopal", "indore", "jaipur", "kochi", "cochin"}
    non_name_stops = {"an", "it", "resume", "cv", "curriculum", "vitae"}
    for i, tok in enumerate(tokens):
        word = tok.rstrip('.')
        wl = word.lower()
        if wl in non_name_stops:
            continue
        if wl in names_set and len(wl) > 1:
            parts = [word]
            j = i + 1
            # include up to 3 following tokens if they look like name parts
            while j < len(tokens) and len(parts) < 4:
                w = tokens[j].rstrip('.')
                wl = w.lower().strip(" :;.-")
                if wl in locations:
                    break
                if wl in particles or (w and (w[0].isupper() or w.isupper())):
                    parts.append(w)
                    j += 1
                    continue
                break
            # require at least two tokens to reduce false positives
            if len([p for p in parts if p]) >= 2:
                return " ".join(p.title() for p in parts if p)
    return ""

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "I14m9nTrNhTAiGPaHvLdaqRHrNKDkWDE")
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

# Simple company/role recommendations per skill (4 per skill)
RECOMMENDATIONS = {
    "Python": [
        {"company": "Google", "role": "Software Engineer (Python)", "url": "https://www.google.com/about/careers/applications/jobs/results/?query=python"},
        {"company": "Microsoft", "role": "Python Developer", "url": "https://jobs.careers.microsoft.com/global/en/search?q=python"},
        {"company": "Netflix", "role": "Backend Engineer (Python)", "url": "https://jobs.netflix.com/search?query=python"},
        {"company": "LinkedIn Jobs", "role": "Python Developer", "url": "https://www.linkedin.com/jobs/search/?keywords=Python%20Developer"}
    ],
    "Java": [
        {"company": "Amazon", "role": "Backend Engineer (Java)", "url": "https://www.amazon.jobs/en/search?base_query=java"},
        {"company": "Oracle", "role": "Java Developer", "url": "https://careers.oracle.com/jobs#/?search=java"},
        {"company": "TCS", "role": "Java Engineer", "url": "https://www.tcs.com/careers"},
        {"company": "LinkedIn Jobs", "role": "Java Developer", "url": "https://www.linkedin.com/jobs/search/?keywords=Java%20Developer"}
    ],
    "React": [
        {"company": "Meta", "role": "Frontend Engineer (React)", "url": "https://www.metacareers.com/jobs/?q=frontend"},
        {"company": "Airbnb", "role": "React Engineer", "url": "https://careers.airbnb.com/positions/"},
        {"company": "Shopify", "role": "Frontend Developer (React)", "url": "https://www.shopify.com/careers/search?keywords=react"},
        {"company": "LinkedIn Jobs", "role": "React Developer", "url": "https://www.linkedin.com/jobs/search/?keywords=React%20Developer"}
    ],
    "SQL": [
        {"company": "Snowflake", "role": "Data Engineer (SQL)", "url": "https://careers.snowflake.com/us/en"},
        {"company": "Databricks", "role": "Data Engineer (SQL)", "url": "https://www.databricks.com/company/careers"},
        {"company": "Oracle", "role": "SQL Developer", "url": "https://careers.oracle.com/jobs#/?search=sql"},
        {"company": "LinkedIn Jobs", "role": "SQL Developer", "url": "https://www.linkedin.com/jobs/search/?keywords=SQL%20Developer"}
    ],
    "Machine Learning": [
        {"company": "NVIDIA", "role": "ML Engineer", "url": "https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite"},
        {"company": "Google", "role": "ML Engineer", "url": "https://www.google.com/about/careers/applications/jobs/results/?query=machine%20learning"},
        {"company": "Uber", "role": "Applied ML", "url": "https://www.uber.com/us/en/careers/"},
        {"company": "LinkedIn Jobs", "role": "Machine Learning Engineer", "url": "https://www.linkedin.com/jobs/search/?keywords=Machine%20Learning%20Engineer"}
    ],
    "Deep Learning": [
        {"company": "OpenAI", "role": "Deep Learning Engineer", "url": "https://openai.com/careers"},
        {"company": "NVIDIA", "role": "Deep Learning Engineer", "url": "https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite"},
        {"company": "Apple", "role": "Deep Learning Researcher", "url": "https://jobs.apple.com/en-us/search?search=deep%20learning"},
        {"company": "LinkedIn Jobs", "role": "Deep Learning", "url": "https://www.linkedin.com/jobs/search/?keywords=Deep%20Learning"}
    ],
    "Django": [
        {"company": "Canonical", "role": "Backend Engineer (Django)", "url": "https://canonical.com/careers"},
        {"company": "Mozilla", "role": "Web Engineer (Django)", "url": "https://www.mozilla.org/en-US/careers/listings/"},
        {"company": "Red Hat", "role": "Software Engineer (Django)", "url": "https://www.redhat.com/en/jobs"},
        {"company": "LinkedIn Jobs", "role": "Django Developer", "url": "https://www.linkedin.com/jobs/search/?keywords=Django%20Developer"}
    ],
    "Flask": [
        {"company": "JetBrains", "role": "Backend Engineer (Flask)", "url": "https://www.jetbrains.com/careers/jobs/"},
        {"company": "Reddit", "role": "Backend Engineer (Flask/Python)", "url": "https://www.redditinc.com/careers"},
        {"company": "Atlassian", "role": "Software Engineer (Flask/Python)", "url": "https://www.atlassian.com/company/careers/all-jobs"},
        {"company": "LinkedIn Jobs", "role": "Flask Developer", "url": "https://www.linkedin.com/jobs/search/?keywords=Flask%20Developer"}
    ],
    "JavaScript": [
        {"company": "Microsoft", "role": "Front-end Engineer (JS)", "url": "https://jobs.careers.microsoft.com/global/en/search?q=javascript"},
        {"company": "Google", "role": "Software Engineer (JS)", "url": "https://www.google.com/about/careers/applications/jobs/results/?query=javascript"},
        {"company": "Stripe", "role": "Frontend Engineer", "url": "https://stripe.com/jobs/search"},
        {"company": "LinkedIn Jobs", "role": "JavaScript Developer", "url": "https://www.linkedin.com/jobs/search/?keywords=JavaScript%20Developer"}
    ],
    "Data Science": [
        {"company": "Airbnb", "role": "Data Scientist", "url": "https://careers.airbnb.com/positions/"},
        {"company": "Meta", "role": "Data Scientist", "url": "https://www.metacareers.com/jobs/?q=data%20scientist"},
        {"company": "Uber", "role": "Data Scientist", "url": "https://www.uber.com/us/en/careers/"},
        {"company": "LinkedIn Jobs", "role": "Data Scientist", "url": "https://www.linkedin.com/jobs/search/?keywords=Data%20Scientist"}
    ],
    "NLP": [
        {"company": "Google", "role": "NLP Engineer", "url": "https://www.google.com/about/careers/applications/jobs/results/?query=nlp"},
        {"company": "Hugging Face", "role": "ML/NLP Engineer", "url": "https://apply.workable.com/huggingface/"},
        {"company": "AWS", "role": "Applied Scientist (NLP)", "url": "https://www.amazon.jobs/en/search?base_query=nlp"},
        {"company": "LinkedIn Jobs", "role": "NLP Engineer", "url": "https://www.linkedin.com/jobs/search/?keywords=NLP%20Engineer"}
    ],
    "TensorFlow": [
        {"company": "Google", "role": "TensorFlow Engineer", "url": "https://www.google.com/about/careers/applications/jobs/results/?query=tensorflow"},
        {"company": "DeepMind", "role": "Research Engineer (TF)", "url": "https://www.deepmind.com/careers"},
        {"company": "Samsung", "role": "ML Engineer (TensorFlow)", "url": "https://www.samsung.com/us/careers/"},
        {"company": "LinkedIn Jobs", "role": "TensorFlow", "url": "https://www.linkedin.com/jobs/search/?keywords=TensorFlow"}
    ],
    "PyTorch": [
        {"company": "Meta", "role": "Research Engineer (PyTorch)", "url": "https://www.metacareers.com/jobs/?q=pytorch"},
        {"company": "NVIDIA", "role": "DL Engineer (PyTorch)", "url": "https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite"},
        {"company": "Microsoft", "role": "AI Engineer (PyTorch)", "url": "https://jobs.careers.microsoft.com/global/en/search?q=pytorch"},
        {"company": "LinkedIn Jobs", "role": "PyTorch", "url": "https://www.linkedin.com/jobs/search/?keywords=PyTorch"}
    ],
    "C++": [
        {"company": "NVIDIA", "role": "Systems/Graphics Engineer (C++)", "url": "https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite"},
        {"company": "Bloomberg", "role": "Software Engineer (C++)", "url": "https://careers.bloomberg.com/job/search/?keyword=c%2B%2B"},
        {"company": "Qualcomm", "role": "Embedded Engineer (C/C++)", "url": "https://careers.qualcomm.com/"},
        {"company": "LinkedIn Jobs", "role": "C++ Developer", "url": "https://www.linkedin.com/jobs/search/?keywords=C%2B%2B%20Developer"}
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

def augment_skills_to_five(skills):
    """Ensure we always have 5 skills by augmenting from a priority list without duplicates."""
    if len(skills) >= 5:
        return skills[:5]
    preferred = [
        "Python", "Java", "Machine Learning", "Deep Learning", "Flask", "SQL", "JavaScript", "React", "Data Science", "AI",
    ]
    pool = [s for s in preferred + skill_keywords if s not in skills]
    for s in pool:
        if len(skills) >= 5:
            break
        skills.append(s)
    return skills[:5]

# -------------------------------
# Generate quiz (with deep logging)
# -------------------------------
def _synthesize_mcq_for_skill(skill: str, n: int):
    """Generate n simple placeholder MCQs for a given skill as a last-resort fallback."""
    items = []
    templates = [
        (f"Basic concept in {skill}?", ["Definition", "Random fact", "Unrelated term", "All of the above"], "A", f"In {skill}, understanding definitions is foundational."),
        (f"Common use of {skill}?", ["Data processing", "Cooking", "Singing", "Driving"], "A", f"{skill} is commonly applied to data/engineering tasks."),
        (f"Choose the correct statement about {skill}.", [f"{skill} has practical industry use", "{skill} is a sport", "{skill} is a fruit", "None"], "A", f"{skill} is a technology/skill area."),
        (f"A typical tool/library for {skill}?", ["Relevant tool", "Hammer", "Paint", "Spoon"], "A", f"Select a tool commonly associated with {skill}."),
        (f"Good practice in {skill} is to?", ["Follow best practices", "Ignore errors", "Avoid docs", "Never test"], "A", f"Best practices improve {skill} outcomes."),
    ]
    for i in range(n):
        t = templates[i % len(templates)]
        items.append({
            "question": t[0],
            "options": t[1],
            "correct_answer": t[2],
            "explanation": t[3],
            "skill": skill,
        })
    return items

def _shuffle_options_from_bank(item):
    """Given an offline bank item with opts and ans letter, return shuffled options and new correct letter."""
    opts = item["opts"][:]
    letters = ['A', 'B', 'C', 'D']
    original_map = dict(zip(letters, opts))
    correct_text = original_map.get(item["ans"], opts[0])
    random.shuffle(opts)
    # find new letter where correct_text ended up
    idx = opts.index(correct_text)
    new_correct = letters[idx]
    return opts, new_correct

def generate_quiz_questions(skills, per_skill=5, nonce: str = ""):
    print("[DEBUG] Generating quiz with skills:", skills)
    if not skills:
        print("[WARN] No skills found — cannot generate quiz.")
        return []

    all_questions = []
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    for skill in skills:
        skill_questions = []
        attempts = 0
        # If no API key, use offline randomized bank directly
        if not MISTRAL_API_KEY:
            bank = OFFLINE_QBANK.get(skill, [])
            sample = bank[:]
            random.shuffle(sample)
            for item in sample[:per_skill]:
                shuffled_opts, new_ans = _shuffle_options_from_bank(item)
                skill_questions.append({
                    "question": item["q"],
                    "options": shuffled_opts,
                    "correct_answer": new_ans,
                    "explanation": item["exp"],
                    "skill": skill,
                })
            if len(skill_questions) < per_skill:
                missing = per_skill - len(skill_questions)
                skill_questions.extend(_synthesize_mcq_for_skill(skill, missing))
            all_questions.extend(skill_questions)
            continue
        while len(skill_questions) < per_skill and attempts < 3:
            needed = per_skill - len(skill_questions)
            prompt = (
                f"Generate {needed} multiple-choice questions strictly about this skill: {skill}.\n"
                "Each question must include:\n"
                "- Question text\n"
                "- 4 options (A, B, C, D)\n"
                "- The correct answer letter\n"
                "- A one-line explanation.\n"
                "Format it like this:\n"
                "Question: <text>\nA) ...\nB) ...\nC) ...\nD) ...\nCorrect Answer: <A/B/C/D>\nExplanation: <reason>\n"
                "Vary difficulty (easy/medium/hard) and do not repeat prior phrasings.\n"
                f"Nonce: {nonce}-{skill}-{attempts}. Ensure the questions differ in wording from any prior outputs.\n"
            )
            try:
                print(f"[DEBUG] Sending request to Mistral API for skill: {skill} (attempt {attempts+1})...")
                payload = {
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "max_tokens": 900
                }
                print("[DEBUG] Payload summary:", json.dumps(payload, indent=2)[:300])
                response = requests.post(MISTRAL_URL, headers=headers, json=payload, timeout=20)
                print("[DEBUG] Mistral status code:", response.status_code)
                with open("mistral_debug.json", "a", encoding="utf-8") as f:
                    f.write("\n\n==== RESPONSE FOR SKILL: " + skill + f" (attempt {attempts+1}) ====\n")
                    f.write(response.text)
                if response.status_code != 200:
                    raise RuntimeError(f"Mistral non-200 status: {response.status_code}")
                data = response.json()
                if "choices" not in data or not data["choices"]:
                    raise RuntimeError("Mistral returned empty choices")
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
                        skill_questions.append({
                            "question": question,
                            "options": options,
                            "correct_answer": correct,
                            "explanation": explanation,
                            "skill": skill
                        })
            except Exception as e:
                print("[ERROR] Mistral generation failed for skill:", skill, "error:", e)
            finally:
                attempts += 1
                if len(skill_questions) < per_skill:
                    # simple exponential backoff
                    time.sleep(min(1.0 * attempts, 3.0))
        if len(skill_questions) < per_skill:
            raise RuntimeError(f"Insufficient questions from Mistral for {skill}: {len(skill_questions)}/{per_skill}")
        all_questions.extend(skill_questions[:per_skill])
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
        skills = extract_skills_from_text(text)
        skills = augment_skills_to_five(skills)
        # Try multiple strategies for name extraction: prefer lexicon, then NER, then heuristics
        name = None
        # 0) Fast lexicon match over the header region
        try:
            lex_name = find_name_from_lexicon(text, NAMES_LEXICON)
            print(f'lex_name is {lex_name}')
            if lex_name:
                name = lex_name
        except Exception:
            pass
        # 1) spaCy PERSON NER; prioritize earliest PERSON near the top, break ties by longer names
        try:
            if len(text) > nlp.max_length:
                nlp.max_length = len(text) + 1000
            window = text[:6000]
            doc = nlp(window)
            person_ents = [
                (ent.start_char, ent.text.strip())
                for ent in doc.ents
                if ent.label_ == "PERSON" and 2 <= len(ent.text.strip()) <= 80
            ]
            # Prefer earliest occurrence (header area), then multi-token names
            person_ents.sort(key=lambda t: (t[0], -len(t[1].split())))
            if (not name) and person_ents:
                name = person_ents[0][1]
        except Exception:
            pass
        # 2) Heuristic extractor
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
        session["resume_filename"] = uploaded_file.filename
        # Issue a small server-side ID for large data
        sid = session.get("sid") or uuid.uuid4().hex
        session["sid"] = sid
        QUIZ_STORE[sid] = {"questions": [], "results": []}
        return render_template("upload.html", name=name, skills=skills, filename=uploaded_file.filename)
    return render_template("index.html")

@app.route("/quiz", methods=["GET", "POST"])
def quiz():
    sid = session.get("sid")
    if not sid or sid not in QUIZ_STORE:
        return redirect(url_for("index"))
    store = QUIZ_STORE[sid]
    questions = store.get("questions", [])
    if request.method == "GET":
        skills = session.get("resume_skills", [])
        if not skills:
            return redirect(url_for("index"))
        # If no API key, generate via offline randomized bank (handled inside generate_quiz_questions)
        # Always regenerate fresh questions on Start Quiz
        fresh_nonce = f"{uuid.uuid4().hex}-{time.time_ns()}"
        try:
            questions = generate_quiz_questions(skills, per_skill=5, nonce=fresh_nonce)
        except Exception as e:
            return render_template(
                "error.html",
                heading="Quiz Generation Failed",
                message="We couldn't generate questions right now.",
                details=str(e),
                primary_url=url_for("quiz"),
                primary_label="Try Again",
                secondary_url=url_for("index"),
                secondary_label="Back to Upload"
            )
        store["questions"] = questions
        store["results"] = []
        store["quiz_id"] = fresh_nonce
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
    return render_template("quiz.html", questions=questions, quiz_id=store.get("quiz_id"))

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
    # Group openings by company with roles list
    groups = {}
    seen = set()
    for skill in skills:
        for e in RECOMMENDATIONS.get(skill, [])[:6]:
            key = (e["company"], e["role"], e["url"])  # dedupe exact role at company
            if key in seen:
                continue
            seen.add(key)
            comp = e["company"]
            if comp not in groups:
                groups[comp] = {"company": comp, "roles": []}
            groups[comp]["roles"].append({
                "role": e["role"],
                "skill": skill,
                "url": e["url"],
            })
    # Fallback: if nothing matched, add generic job board entries grouped under 'LinkedIn Jobs'
    if not groups:
        groups["LinkedIn Jobs"] = {"company": "LinkedIn Jobs", "roles": []}
        for skill in skills:
            groups["LinkedIn Jobs"]["roles"].append({
                "role": f"{skill} roles",
                "skill": skill,
                "url": f"https://www.linkedin.com/jobs/search/?keywords={skill.replace(' ', '%20')}"
            })
    companies = sorted(groups.values(), key=lambda x: x["company"].lower())
    return render_template("recommendations.html", companies=companies)

if __name__ == "__main__":
    print("[INFO] Starting Flask app...")
    app.run(debug=True)