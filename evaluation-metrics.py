import pandas as pd
import sys


def sample_csv(input_file, output_file, n=1000):
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"❌ Error: File '{input_file}' not found.")
        sys.exit(1)

    sample_size = min(n, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)
    sample_df.to_csv(output_file, index=False)
    print(f"✅ Sampled {sample_size} rows and saved to '{output_file}'")

def validate_and_score(csv_path):
    global df  # make it accessible to the new function
    # Load CSV
    df = pd.read_csv(csv_path)

    # Required columns
    required_cols = ['label', 'gpt_answer', 'gemini_answer', 'llama_answer']

    # Strip whitespace and normalize answers
    for col in required_cols:
        if df[col].isnull().any():
            bad_rows = df[df[col].isnull()].index.tolist()
            print(f"❌ Error: Column '{col}' has missing values at rows: {bad_rows}")
            sys.exit(1)

        df[col] = df[col].astype(str).str.strip().str.upper()

        # Check for invalid entries
        invalid_mask = ~df[col].isin(['A', 'B'])
        if invalid_mask.any():
            bad_rows = df[invalid_mask][[col]].reset_index()
            for _, row in bad_rows.iterrows():
                print(f"❌ Error: Invalid value '{row[col]}' in column '{col}' at row {row['index']}")
            sys.exit(1)

    print("✅ All answers are valid.\n")

    # Calculate accuracy and track wrong answers
    global wrong_rows_by_model
    wrong_rows_by_model = {}

    print("\n📊 Accuracy:")
    for model in ['gpt_answer', 'gemini_answer', 'llama_answer']:
        correct = (df[model] == df['label'])
        total = len(df)
        accuracy = correct.sum() / total * 100
        print(f"{model}: {accuracy:.2f}% ({correct.sum()}/{total})")

        wrong = df[~correct].copy()
        wrong["wrong_model"] = model
        wrong_rows_by_model[model] = wrong

    # Combine all wrong rows and save to CSV
    if wrong_rows_by_model:
        all_wrong = pd.concat(wrong_rows_by_model.values(), ignore_index=True)
        all_wrong.to_csv("wrong_answers.csv", index=False)
        print(f"\n📁 Saved all incorrect answers to 'wrong_answers.csv' ({len(all_wrong)} rows).")
def validate_and_score_COT(csv_path):
    global df  # make it accessible to the new function
    # Load CSV
    df = pd.read_csv(csv_path)

    # Required columns
    required_cols = ['label', 'gpt_answer_COT', 'gemini_answer_COT', 'llama_answer_COT']

    # Strip whitespace and normalize answers
    for col in required_cols:
        if df[col].isnull().any():
            bad_rows = df[df[col].isnull()].index.tolist()
            print(f"❌ Error: Column '{col}' has missing values at rows: {bad_rows}")
            sys.exit(1)

        df[col] = df[col].astype(str).str.strip().str.upper()

        # Check for invalid entries
        invalid_mask = ~df[col].isin(['A', 'B'])
        if invalid_mask.any():
            bad_rows = df[invalid_mask][[col]].reset_index()
            for _, row in bad_rows.iterrows():
                print(f"❌ Error: Invalid value '{row[col]}' in column '{col}' at row {row['index']}")
            sys.exit(1)

    print("✅ All answers are valid.\n")

    # Calculate accuracy and track wrong answers
    global wrong_rows_by_model
    wrong_rows_by_model = {}

    print("\n📊 Accuracy:")
    for model in ['gpt_answer_COT', 'gemini_answer_COT', 'llama_answer_COT']:
        correct = (df[model] == df['label'])
        total = len(df)
        accuracy = correct.sum() / total * 100
        print(f"{model}: {accuracy:.2f}% ({correct.sum()}/{total})")

        wrong = df[~correct].copy()
        wrong["wrong_model"] = model
        wrong_rows_by_model[model] = wrong

    # Combine all wrong rows and save to CSV
    if wrong_rows_by_model:
        all_wrong = pd.concat(wrong_rows_by_model.values(), ignore_index=True)
        all_wrong.to_csv("wrong_answers.csv", index=False)
        print(f"\n📁 Saved all incorrect answers to 'wrong_answers.csv' ({len(all_wrong)} rows).")

def print_distractor_length_distribution_by_model():
    print("\n📊 Distractor Length Distribution (for wrong answers only):")
    for model, df_wrong in wrong_rows_by_model.items():
        print(f"\n🔍 {model} — wrong predictions by distractor_length:")
        counts = df_wrong['distractor_distance'].value_counts().sort_index()
        for length, count in counts.items():
            print(f"  Length {length}: {count}")

if __name__ == "__main__":
    print('ENGLISH TEST')
    print(' - WITHOUT COT')
    validate_and_score('NSP_QUESTIONS_WITH_ANSWERS_EN.csv')
    # print(' - WITH COT')
    # validate_and_score_COT('NSP_QUESTIONS_WITH_ANSWERS_1000_EN - FINAL.csv')

    # print('SWAHILI TEST')
    # print(' - WITHOUT COT')
    # validate_and_score('NSP_QUESTIONS_WITH_ANSWERS_1000_SW - FINAL.csv')
    # print(' - WITH COT')
    # validate_and_score_COT('NSP_QUESTIONS_WITH_ANSWERS_1000_SW - FINAL.csv')
    #
    # print('HAUSA TEST')
    # print(' - WITHOUT COT')
    # validate_and_score('NSP_QUESTIONS_WITH_ANSWERS_1000_HA - FINAL.csv')
    # print(' - WITH COT')
    # validate_and_score_COT('NSP_QUESTIONS_WITH_ANSWERS_1000_HA - FINAL.csv')

