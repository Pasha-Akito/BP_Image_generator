import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def count_results(dataset):
    counts = {}
    for status in ["CLEAR", "PARTIAL", "FAILED"]:
        counts[status] = (dataset["left_image"].value_counts().get(status) + dataset["right_image"].value_counts().get(status))
    return counts

if __name__ == "__main__":
    symbolic_results_df = pd.read_csv("symbolic_results.csv")
    english_results_df = pd.read_csv("english_results.csv")
    minimal_results_df = pd.read_csv("minimal_results.csv")
    
    symbolic_results_count = count_results(symbolic_results_df)
    english_results_count = count_results(english_results_df)
    minimal_results_count = count_results(minimal_results_df)
    
    total_clear_counts = [symbolic_results_count["CLEAR"], english_results_count["CLEAR"], minimal_results_count["CLEAR"]]
    total_failed_counts = [symbolic_results_count["FAILED"], english_results_count["FAILED"], minimal_results_count["FAILED"]]
    total_partial_counts = [symbolic_results_count["PARTIAL"], english_results_count["PARTIAL"], minimal_results_count["PARTIAL"]]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    y_positions_list = np.arange(3)
    
    ax.barh(y_positions_list, total_clear_counts,0.6, label="Clear", color="teal")
    ax.barh(y_positions_list, total_failed_counts, 0.6, left=total_clear_counts, label="Failed", color="darkmagenta")
    ax.barh(y_positions_list, total_partial_counts, 0.6, left=[total_clear_counts[i] + total_failed_counts[i] for i in range(3)], label="Partial", color="coral")
    
    for i in range(3):
        ax.text(total_clear_counts[i] / 2, i, str(total_clear_counts[i]), ha="center", va="center", fontweight="bold", color="white")
        ax.text(total_clear_counts[i] + total_failed_counts[i] / 2, i, str(total_failed_counts[i]), ha="center", va="center",fontweight="bold", color="white")
        ax.text(total_clear_counts[i] + total_failed_counts[i] + total_partial_counts[i] / 2, i, str(total_partial_counts[i]), ha="center", va="center", fontweight="bold", color="white")
    
    ax.set_yticks(y_positions_list)
    ax.set_yticklabels(["Symbolic", "English", "Minimal"])
    ax.set_xlabel("Number of Image Evaluations")
    ax.set_title("Results by Dataset")
    ax.legend()
    
    plt.tight_layout()
    plt.show()
