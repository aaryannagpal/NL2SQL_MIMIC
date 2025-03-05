import matplotlib.pyplot as plt

class PlotResults:
    @staticmethod
    def plot_performance(models, accuracy, similarity, exec_time, output_match, canonical_match):
        """Plots model performance metrics."""
        
        # Plot Accuracy, Similarity, and Output Match
        plt.figure(figsize=(12, 6))
        plt.bar(models, accuracy, color="green", alpha=0.7, label="Execution Accuracy")
        plt.bar(models, similarity, color="blue", alpha=0.7, label="Query Similarity")
        plt.bar(models, output_match, color="purple", alpha=0.7, label="Output Match")
        plt.xlabel("Model")
        plt.ylabel("Score")
        plt.title("NLQ to SQL Model Comparison")
        plt.legend()
        plt.show()

        # Plot Canonical Match and Execution Time
        plt.figure(figsize=(12, 6))
        plt.bar(models, canonical_match, color="orange", alpha=0.7, label="Canonical SQL Match")
        plt.bar(models, exec_time, color="red", alpha=0.7, label="Execution Time (Seconds)")
        plt.xlabel("Model")
        plt.ylabel("Score / Time")
        plt.title("Canonical SQL Comparison & Execution Time")
        plt.legend()
        plt.show()
