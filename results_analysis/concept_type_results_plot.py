import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    symbolic_results_df = pd.read_csv('symbolic_results.csv')
    english_results_df = pd.read_csv('english_results.csv')
    minimal_results_df = pd.read_csv('minimal_results.csv')

    combined_data = []
    for df in [symbolic_results_df, english_results_df, minimal_results_df]:
        for _, row in df.iterrows():
            combined_data.append({'concept_type': row['concept_type'], 'result': row['left_image']})
            combined_data.append({'concept_type': row['concept_type'], 'result': row['right_image']})

    combined_df = pd.DataFrame(combined_data)
    total_results_count = combined_df.groupby(['concept_type', 'result']).size().unstack()

    concept_types_list = ['size', 'visual_properties', 'numerosity', 'shape_geometry', 'spatial_relationship']
    clear_counts = [total_results_count.loc[concept_type, 'CLEAR'] for concept_type in concept_types_list]
    failed_counts = [total_results_count.loc[concept_type, 'FAILED'] for concept_type in concept_types_list]
    partial_counts = [total_results_count.loc[concept_type, 'PARTIAL'] for concept_type in concept_types_list]

    _, ax = plt.subplots(figsize=(12, 8))
    y_positions = np.arange(5)

    ax.barh(y_positions, clear_counts, 0.6, label='Clear', color='teal')
    ax.barh(y_positions, failed_counts, 0.6, left=clear_counts, label='Failed', color='darkmagenta')
    ax.barh(y_positions, partial_counts, 0.6, left=[clear_counts[i] + failed_counts[i] for i in range(5)], label='Partial', color='coral')

    for i in range(5):
        ax.text(clear_counts[i]/2, i, str(clear_counts[i]), ha='center', va='center', fontweight='bold', color='white')
        ax.text(clear_counts[i] + failed_counts[i]/2, i, str(failed_counts[i]), ha='center', va='center', fontweight='bold', color='white')
        ax.text(clear_counts[i] + failed_counts[i] + partial_counts[i]/2, i, str(partial_counts[i]), ha='center', va='center', fontweight='bold', color='white')

    ax.set_xlabel('Number of Evaluations')
    ax.set_title('Results by Concept Type (All Datasets Combined)')
    ax.set_yticks(y_positions)
    ax.set_yticklabels(["Size", "Visual Properties", "Numerosity", "Shape and Geometry", "Spatial Relationships"])
    ax.legend()

    plt.tight_layout()
    plt.show()