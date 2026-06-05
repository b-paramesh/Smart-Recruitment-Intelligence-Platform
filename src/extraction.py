import re

class InformationExtractor:
    def __init__(self):
        # Basic list of skills for keyword matching
        self.known_skills = [
            "python", "java", "c++", "machine learning", "deep learning", 
            "nlp", "data analysis", "sql", "react", "node.js", "pytorch", 
            "tensorflow", "aws", "docker", "kubernetes", "agile", "scrum",
            "streamlit", "pandas", "numpy", "html", "css", "javascript",
            "git", "github", "linux", "rest api", "graphql"
        ]
        
    def extract_skills(self, text):
        text_lower = text.lower()
        extracted_skills = []
        for skill in self.known_skills:
            # Word boundary matching
            if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
                extracted_skills.append(skill)
        return list(set(extracted_skills))

    def extract_experience_years(self, text):
        # Simple regex to find "X years" or "X+ years"
        match = re.search(r'(\d+)\+?\s*years?\s*of\s*experience', text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        # Fallback to general mentions of years
        matches = re.findall(r'(\d+)\+?\s*years?', text, re.IGNORECASE)
        if matches:
            try:
                # return the highest number found that is reasonable (<40)
                nums = [int(m) for m in matches if int(m) < 40]
                return max(nums) if nums else 0
            except:
                return 0
        return 0

    def extract_education(self, text):
        education = []
        degrees = ['bachelor', 'master', 'phd', 'bsc', 'msc', 'b.tech', 'm.tech', 'b.e.', 'mba']
        text_lower = text.lower()
        for degree in degrees:
            if degree in text_lower:
                education.append(degree)
        return list(set(education))

    def extract_certifications(self, text):
        certifications = []
        if re.search(r'(certified|certification|aws|azure|gcp|coursera|udemy)', text, re.IGNORECASE):
            # very naive extraction
            matches = re.findall(r'([A-Z][a-zA-Z\s]+Certification)', text)
            certifications.extend(matches)
        return certifications

    def extract_projects(self, text):
        # Mock project extraction
        if "project" in text.lower():
            return ["Extracted Project 1", "Extracted Project 2"]
        return []

    def extract_all(self, text):
        return {
            "skills": self.extract_skills(text),
            "experience_years": self.extract_experience_years(text),
            "education": self.extract_education(text),
            "certifications": self.extract_certifications(text),
            "projects": self.extract_projects(text)
        }
