import pandas as pd

# DATASET_RESULTS = "english"
# DATASET_RESULTS = "symbolic"
DATASET_RESULTS = "minimal"

if __name__ == "__main__":
    results_df = pd.read_csv(f'{DATASET_RESULTS}_results.csv')

    results_df = results_df[['bp_number', 'text_input', 'concept_type', 'left_image', 'right_image']].copy()
    results_df.columns = ['BP#', 'Text Input', 'Concept Type', 'Left Image Result', 'Right Image Result']
    
    results_df['Concept Type'] = results_df['Concept Type'].str.replace('_', '\\_')

    latex_code = results_df.to_latex(
        index=False,
        escape=False,
        column_format='|p{0.6cm}|p{4.5cm}|p{3.3cm}|p{1.8cm}|p{1.8cm}|',
        caption=f'{DATASET_RESULTS} Analysis Results',
        label=f'tab:{DATASET_RESULTS}_analysis_results',
        longtable=True
    )

    with open(f'{DATASET_RESULTS}_table_results.txt', 'w') as f:
        f.write(latex_code)

    print(f"Latex saved to {DATASET_RESULTS}_table_results.txt")