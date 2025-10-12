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
    match = re.search(r'(?i)(name[:\-]?\s*)([A-Z][a-z]+\s[A-Z][a-z]+)', text)
    return match.group(2) if match else "Unknown"
