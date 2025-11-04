import re
import docx2txt
from PyPDF2 import PdfReader

SKILLS_DB = ["Python", "Java", "Machine Learning", "SQL", "C++", "HTML", "CSS", "JavaScript",
"Bootstrap", "Tailwind", "React", "Angular", "Vue.js", "Node.js", "Express.js", "Flask", "Django",
"MySQL", "PostgreSQL", "MongoDB", "SQLite", "Firebase", "Pandas", "NumPy", "Matplotlib", "Seaborn",
"Scikit-learn", "TensorFlow", "Keras", "PyTorch", "Data Analysis", "Data Visualization", "Deep Learning",
"Artificial Intelligence", "Natural Language Processing", "Git", "GitHub", "Docker", "AWS", "Google Cloud",
"Azure", "Jira", "VS Code", "Jupyter", "Linux", "Power BI", "Tableau", "OOP", "DSA", "REST API", "Microservices",
"Agile", "CI/CD", "Unit Testing", "Cloud Computing", "Communication", "Leadership", "Teamwork",
"Problem Solving", "Time Management", "Creativity", "Critical Thinking"]

def extract_text_from_resume(file_path):
    if file_path.endswith('.pdf'):
        reader = PdfReader(file_path)
        return " ".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif file_path.endswith('.docx'):
        return docx2txt.process(file_path)
    else:
        raise ValueError("Unsupported file type")

def extract_skills(text):
    text = text.lower()
    return [skill for skill in SKILLS_DB if skill.lower() in text]

def extract_name(text):
    # Normalize
    text = re.sub(r"[\t\r]+", " ", text)
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # Quick regex for explicit labels like: Name: FIRST M. LAST (match line-only)
    label_pat = re.compile(r"(?im)^\s*name\s*[:\-]?\s*([A-Za-z][A-Za-z\-']+(?:\s+[A-Za-z]\.)?(?:\s+[A-Za-z][A-Za-z\-']+){1,2})\s*$")
    m = label_pat.search("\n".join(lines[:40]))
    if m:
        return m.group(1).strip().title()

    # Heuristic scan of the top of document
    blacklist = {"resume", "curriculum vitae", "cv", "profile", "summary", "objective", "portfolio"}
    # Common resume section headers to skip entirely
    header_phrases = {
        "education", "education details", "education qualification", "education qualifications",
        "work experience", "professional experience", "experience", "projects", "personal projects",
        "skills", "technical skills", "certifications", "achievements", "awards", "contact", "portfolio",
        "interests", "hobbies", "publications", "languages", "responsibilities", "key responsibilities",
        "strengths", "references"
    }
    job_words = {"engineer", "developer", "software", "data", "scientist", "analyst", "student", "intern", "consultant", "manager", "architect"}
    contact_markers = ["email", "@", "www", "http", "phone", "+", "linkedin", "github"]

    def tokenize(line: str):
        # keep words with letters (including basic accented ranges), allow hyphen/apostrophe and initials like "M."
        return [w for w in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ\-']*\.?", line) if any(c.isalpha() for c in w)]

    particles = {"de", "da", "del", "della", "van", "von", "bin", "binti", "al", "el", "la", "le", "di", "dos", "das", "do", "du", "mac", "mc", "o'", "d'"}

    candidates = []
    # 1) Prefer lines near contact info (common resume pattern: name above email/phone)
    email_re = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    phone_re = re.compile(r"(?:\+?\d[\d\s().-]{6,}\d)")
    first_contact_idx = None
    for i, l in enumerate(lines[:80]):
        if email_re.search(l) or phone_re.search(l):
            first_contact_idx = i
            break
    def score_line(idx, line, bonus=0):
        low = line.lower().strip(" :;.-")
        if any(b in low for b in blacklist):
            return None
        if low in header_phrases:
            return None
        # skip contact lines entirely (e.g., 'Email: x@y')
        if any(k in low for k in contact_markers):
            return None
        # remove contacts and split segments
        working = re.sub(r"\S+@\S+", " ", line)
        working = re.sub(r"https?://\S+", " ", working)
        working = re.sub(r"www\.\S+", " ", working)
        working = re.sub(r"(?:\+?\d[\d\s().-]{6,}\d)", " ", working)
        parts = re.split(r"[|•·•·,;\/\\\-–—]", working)
        first_clean = next((p.strip() for p in parts if p.strip()), working.strip())
        words = tokenize(first_clean)
        if words:
            header_token_count = sum(1 for w in words if w.lower() in header_phrases)
            if header_token_count / max(1, len(words)) >= 0.5:
                return None
        words_clean = [w.rstrip('.') for w in words]
        # reject obvious job-title lines (short lines dominated by job words)
        job_count = sum(1 for w in words_clean if w.lower() in job_words)
        if job_count >= 1 and len(words_clean) <= 3:
            return None
        if len(words_clean) == 1:
            w = words_clean[0]
            wl = w.lower().strip(" :;.-")
            if (3 <= len(w) <= 30) and w[0].isalpha() and (w[0].isupper() or w.isupper()) and (wl not in header_phrases) and (wl not in job_words):
                score = 3 + max(0, 5 - idx//3) + bonus
                return (score, [w])
        # Accept 2-5 tokens to allow middle names/suffixes
        if 2 <= len(words_clean) <= 5:
            caps_words = 0
            for w in words_clean:
                wl = w.lower()
                if wl in particles:
                    caps_words += 1
                elif w and (w[0].isupper() or w.isupper()):
                    caps_words += 1
            all_caps = sum(1 for w in words_clean if w.isupper()) == len(words_clean)
            # Slightly favor very top lines
            score = caps_words + max(0, 6 - idx//4) + bonus
            # Accept if enough words look like names or whole line is caps (common in headers)
            if all_caps or caps_words >= 2:
                return (score, words_clean)
        return None

    if first_contact_idx is not None:
        start = max(0, first_contact_idx - 6)
        for i in range(start, first_contact_idx):
            if len(lines[i]) > 120:
                continue
            res = score_line(i, lines[i], bonus=3)
            if res:
                candidates.append(res)
    for idx, line in enumerate(lines[:35]):
        if len(line) > 120:
            continue
        res = score_line(idx, line)
        if res:
            # penalize lines that look like job titles
            words_clean = res[1]
            if sum(1 for w in words_clean if w.lower() in job_words) >= 1:
                res = (res[0] - 2, words_clean)
            candidates.append(res)

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        chosen = candidates[0][1][:4]
        name = " ".join(w.title() for w in chosen)
        try:
            print(f"[NAME] selected: {name}")
        except Exception:
            pass
        return name

    # Fallback generic two-capitalized-words pattern anywhere near the top
    generic_pat = re.compile(r"\b([A-Z][a-zA-Z\-']+\s+[A-Z][a-zA-Z\-']+(?:\s+[A-Z][a-zA-Z\-']+)?)\b")
    for line in lines[:80]:
        low = line.lower().strip(" :;.-")
        if any(k in low for k in contact_markers):
            continue
        if low in blacklist or low in header_phrases:
            continue
        if any(w in low.split() for w in job_words):
            continue
        g = generic_pat.search(line)
        if g:
            name = g.group(1).strip().title()
            try:
                print(f"[NAME] fallback selected: {name}")
            except Exception:
                pass
            return name

    # Final fallback: accept a strong single-token name near the top
    for idx, line in enumerate(lines[:35]):
        # Skip obvious contact lines
        if any(k in line.lower() for k in contact_markers):
            continue
        words = [w.rstrip('.').strip(" :;.-") for w in tokenize(line)]
        words = [w for w in words if w and w[0].isalpha()]
        if len(words) == 1:
            w = words[0]
            wl = w.lower()
            if (3 <= len(w) <= 30) and (w[0].isupper() or w.isupper()) and (wl not in header_phrases) and (wl not in job_words) and (wl not in blacklist):
                try:
                    print(f"[NAME] single-token selected: {w.title()}")
                except Exception:
                    pass
                return w.title()

    return "Unknown"