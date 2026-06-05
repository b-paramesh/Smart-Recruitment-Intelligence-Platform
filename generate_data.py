import pandas as pd
import random

def generate_synthetic_resumes(num_resumes=50):
    categories = ['Data Scientist', 'Software Engineer', 'Product Manager', 'HR Specialist', 'Web Developer']
    skills_pool = [
        "python", "java", "c++", "machine learning", "deep learning", 
        "nlp", "data analysis", "sql", "react", "node.js", "pytorch", 
        "tensorflow", "aws", "docker", "kubernetes", "agile", "scrum",
        "html", "css", "javascript", "git", "linux", "rest api"
    ]
    
    data = []
    for i in range(1, num_resumes + 1):
        cat = random.choice(categories)
        exp = random.randint(0, 15)
        
        # Select random skills
        num_skills = random.randint(3, 8)
        skills = random.sample(skills_pool, num_skills)
        
        text = f"Resume of Candidate {i}. I am a highly motivated {cat} with {exp} years of experience. "
        text += f"My core expertise includes {', '.join(skills)}. "
        
        if random.random() > 0.5:
            text += "I hold a Bachelor degree in Computer Science. "
        else:
            text += "I hold a Master degree in Engineering. "
            
        if random.random() > 0.7:
            text += "I have an AWS Certification. "
            
        data.append({
            'id': i,
            'candidate_name': f"Candidate {i}",
            'category': cat,
            'experience_years': exp,
            'extracted_skills': skills,
            'text': text
        })
        
    df = pd.DataFrame(data)
    import os
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/synthetic_resumes.csv', index=False)
    print(f"Generated {num_resumes} synthetic resumes at data/synthetic_resumes.csv")

if __name__ == "__main__":
    generate_synthetic_resumes()
