class SimilarityEngine:
    def __init__(self):
        pass

    def calculate_skill_match(self, jd_skills, resume_skills):
        if not jd_skills:
            return 0.0
        jd_skills_set = set([s.lower() for s in jd_skills])
        resume_skills_set = set([s.lower() for s in resume_skills])
        
        match_count = len(jd_skills_set.intersection(resume_skills_set))
        return (match_count / len(jd_skills_set)) * 100

    def calculate_experience_match(self, jd_exp, resume_exp):
        if jd_exp == 0:
            return 100.0 if resume_exp > 0 else 0.0
        
        if resume_exp >= jd_exp:
            return 100.0
        return (resume_exp / jd_exp) * 100

    def calculate_project_match(self, jd_projects, resume_projects):
        # A simple mock match - if both have projects, we give a score, otherwise 0
        if not jd_projects:
            return 100.0 if resume_projects else 0.0
        if resume_projects:
            return 100.0
        return 0.0

    def rank_candidates(self, jd_extracted, resumes_extracted):
        """
        jd_extracted: dict with 'skills', 'experience_years', 'projects'
        resumes_extracted: list of dicts with 'id', 'skills', 'experience_years', 'projects'
        """
        ranked_list = []
        for resume in resumes_extracted:
            skill_score = self.calculate_skill_match(jd_extracted.get('skills', []), resume.get('skills', []))
            exp_score = self.calculate_experience_match(jd_extracted.get('experience_years', 0), resume.get('experience_years', 0))
            proj_score = self.calculate_project_match(jd_extracted.get('projects', []), resume.get('projects', []))
            
            # Weighted average
            total_score = (skill_score * 0.6) + (exp_score * 0.3) + (proj_score * 0.1)
            
            ranked_list.append({
                'id': resume.get('id'),
                'candidate_name': resume.get('candidate_name', f"Candidate {resume.get('id')}"),
                'skill_match': round(skill_score, 2),
                'experience_match': round(exp_score, 2),
                'project_match': round(proj_score, 2),
                'total_score': round(total_score, 2)
            })
            
        # Sort by total_score descending
        ranked_list = sorted(ranked_list, key=lambda x: x['total_score'], reverse=True)
        return ranked_list[:10] # Top 10 candidates
