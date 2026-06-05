import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

class ResumeAnalytics:
    def __init__(self, dataset_df):
        """
        Expects a DataFrame with columns: 
        'id', 'text', 'category', 'extracted_skills', 'experience_years'
        """
        self.df = dataset_df

    def plot_category_distribution(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(y='category', data=self.df, order=self.df['category'].value_counts().index, ax=ax, palette='viridis')
        ax.set_title("Resume Category Distribution")
        ax.set_xlabel("Count")
        ax.set_ylabel("Category")
        return fig

    def plot_experience_distribution(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(self.df['experience_years'], bins=10, kde=True, ax=ax, color='teal')
        ax.set_title("Experience Distribution (Years)")
        ax.set_xlabel("Years of Experience")
        ax.set_ylabel("Frequency")
        return fig

    def plot_skill_wordcloud(self):
        # Flatten all skills into a single list
        all_skills = []
        for skills_list in self.df['extracted_skills']:
            if isinstance(skills_list, list):
                all_skills.extend(skills_list)
        
        skill_text = " ".join(all_skills)
        
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='tab20').generate(skill_text)
        
        fig, ax = plt.subplots(figsize=(15, 7.5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title("Skill Word Cloud")
        return fig

    def plot_resume_length_analysis(self):
        self.df['text_length'] = self.df['text'].apply(lambda x: len(str(x).split()))
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='category', y='text_length', data=self.df, ax=ax, palette='Set2')
        ax.set_title("Resume Length (Word Count) by Category")
        ax.set_ylabel("Word Count")
        ax.set_xlabel("Category")
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
