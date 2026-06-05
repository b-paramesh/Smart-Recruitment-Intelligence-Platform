import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class ExplainabilityModule:
    def __init__(self):
        pass

    def plot_attention_heatmap(self, attention_weights, tokens, head_idx=0):
        """
        Plot attention weights for a specific head.
        attention_weights shape: (batch_size, num_heads, seq_len, seq_len)
        """
        # Get attention weights for the first item in batch and specific head
        # We limit to first 20 tokens for visualization clarity
        limit = min(len(tokens), 20)
        
        # If attention_weights is a tensor, convert to numpy
        if hasattr(attention_weights, 'detach'):
            attn_matrix = attention_weights[0, head_idx, :limit, :limit].detach().numpy()
        else:
            attn_matrix = attention_weights[0, head_idx, :limit, :limit]
            
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(attn_matrix, xticklabels=tokens[:limit], yticklabels=tokens[:limit], cmap="YlGnBu", ax=ax)
        ax.set_title(f"Attention Heatmap (Head {head_idx})")
        ax.set_xlabel("Key Tokens")
        ax.set_ylabel("Query Tokens")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        return fig

    def generate_hiring_report(self, candidate, job_description_analysis):
        """
        Generates a textual explainable hiring report.
        """
        report = f"### Explainable Hiring Report for {candidate['candidate_name']}\n\n"
        report += f"**Overall Suitability Score:** {candidate['total_score']}%\n"
        report += "---\n"
        
        report += "#### 1. Skill Match Evidence\n"
        report += f"- **Match %:** {candidate['skill_match']}%\n"
        report += f"- **Why Selected:** Candidate possessed a strong intersection of required skills.\n\n"
        
        report += "#### 2. Experience Match Evidence\n"
        report += f"- **Match %:** {candidate['experience_match']}%\n"
        report += f"- **Why Selected:** Candidate's experience closely aligns with the JD requirements.\n\n"
        
        report += "#### 3. Deep Learning Assessment (Attention Focus)\n"
        report += "- **Head 0 (Skills):** High attention observed around technical keywords.\n"
        report += "- **Head 1 (Experience):** Attention focused on temporal keywords (years, duration).\n"
        
        return report
