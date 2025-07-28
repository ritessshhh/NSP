import os
import time
import pandas as pd
import sys
import re
from openai import OpenAI
from google import genai
from together import Together
from dotenv import load_dotenv
def extract_answer_and_reasoning(response_text):
    """
    Extracts the reasoning and the final single-letter answer from a model's response.
    It assumes the last line containing just 'A' or 'B' is the answer.
    """
    if not response_text:
        return "NO_RESPONSE", ""

    lines = response_text.strip().split('\n')
    answer = "PARSE_ERROR"
    reasoning = response_text  # Default reasoning is the full response

    # Iterate from the end to find the first valid answer line
    for i in range(len(lines) - 1, -1, -1):
        cleaned_line = lines[i].strip().upper().strip('"').strip("'")
        if cleaned_line == 'A' or cleaned_line == 'B':
            answer = cleaned_line
            reasoning = "\n".join(lines[:i]).strip()
            break

    # Fallback: check for pattern like "The answer is A"
    if answer == "PARSE_ERROR":
        match = re.search(r'\b(A|B)\b["\']?$', response_text.strip(), re.IGNORECASE)
        if match:
            answer = match.group(1).upper()
            reasoning = response_text.strip()

    return answer, reasoning



def main(input_file):
    # ——— CONFIG ———
    # IMPORTANT: Replace with your actual API keys
    openai_api_key = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
    gemini_api_key = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
    llama3_api_key = os.environ.get("TOGETHER_API_KEY", "YOUR_TOGETHER_API_KEY")

    try:
        openai_client = OpenAI(api_key=openai_api_key)
        genai_client = genai.Client(api_key=gemini_api_key)
        together_client = Together(api_key=llama3_api_key)
    except Exception as e:
        print(f"❌ Error initializing API clients. Make sure your API keys are set correctly. Error: {e}")
        sys.exit(1)


    GPT_MODEL = "gpt-4-turbo"
    GEMINI_MODEL = "gemini-1.5-flash"
    LLAMA_MODEL = "meta-llama/Meta-Llama-3-70B-Instruct-Turbo"

    INPUT_CSV = input_file
    OUTPUT_CSV = input_file # Save back to the same file
    CHECKPOINT = 5  # Save every 5 rows
    REASONING_LANG_CODE = 'EN'

    # ——— LOAD DATA ———
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"❌ Error: The file '{INPUT_CSV}' was not found.")
        sys.exit(1)

    # Add new columns for COT answers and reasoning if they don't exist
    new_cols = [
        "gpt_answer_COT", "gpt_reasoning_COT",
        "gemini_answer_COT", "gemini_reasoning_COT",
        "llama_answer_COT", "llama_reasoning_COT"
    ]
    for col in new_cols:
        if col not in df.columns:
            df[col] = ""
    # Also ensure original columns exist
    for col in ("gpt_answer", "gemini_answer", "llama_answer"):
         if col not in df.columns:
            df[col] = ""


    # ——— PROMPT BUILDER (CHAIN OF THOUGHT) ———
    def build_cot_prompt(context, opt_a, opt_b):
        lang_instructions = {
            "EN": "First, provide a step-by-step reasoning in English explaining which sentence is a more logical continuation of the story. After your reasoning, on a new line, state your final answer as only a single letter: A or B.",
            "HA": "Da farko, bayar da dalili mataki-mataki a cikin harshen Hausa da ke bayanin wace jumla ce ta fi dacewa da ci gaban labarin. Bayan ka bayar da dalilinka, a kan sabon layi, bayyana amsarka ta karshe da harafi guda daya kawai: A ko B.",
            "SW": "Kwanza, toa sababu hatua kwa hatua kwa lugha ya Kiswahili ukieleza ni sentensi ipi inayoendeleza hadithi kimantiki zaidi. Baada ya maelezo yako, kwenye mstari mpya, taja jibu lako la mwisho kama herufi moja tu: A au B."
        }
        instruction = lang_instructions[REASONING_LANG_CODE]

        return (
            f"Given the following story context:\n\n{context}\n\n"
            "Which sentence comes next?\n\n"
            f"A: {opt_a}\n\n"
            f"B: {opt_b}\n\n"
            f"{instruction}"
        )

    # ——— MODEL QUERY FUNCTIONS ———
    def ask_gpt(prompt):
        try:
            response = openai_client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ GPT Error: {e}")
            return f"ERROR: {e}"

    def ask_gemini(prompt, retries=5, delay=60):
        for attempt in range(retries):
            try:
                response = genai_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt
                )
                ans = response.text.strip().upper()
                return ans
            except Exception as e:
                if '503' in str(e) or 'UNAVAILABLE' in str(e).upper():
                    print(f"❌ Gemini 503 error (attempt {attempt + 1}/{retries}): {e}")
                    print(f"⏳ Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"❌ Non-retryable error: {e}")
                    break
        return "ERROR: Retry failed"

    def ask_llama(prompt):
        try:
            response = together_client.chat.completions.create(
                model=LLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ Llama Error: {e}")
            return f"ERROR: {e}"

    # ——— MAIN LOOP ———
    try:
        # You can set a start_index if you need to resume a partial run
        start_index = 244
        print(f"🚀 Starting processing for {INPUT_CSV}. Reasoning language: {REASONING_LANG_CODE}")

        for idx, row in df.iloc[start_index:].iterrows():
            print(f"➡️ Row {idx}: Processing...")

            prompt = build_cot_prompt(row["context"], row["option_A"], row["option_B"])

            # --- GPT ---
            if row.get("gpt_answer_COT", "") not in ("A", "B"):
                print("  - Querying GPT-4...")
                gpt_response = ask_gpt(prompt)
                if gpt_response.startswith("ERROR:"):
                    print("❌ GPT-4 failed. Exiting.")
                    sys.exit(1)
                answer, reasoning = extract_answer_and_reasoning(gpt_response)
                df.at[idx, "gpt_answer_COT"] = answer
                df.at[idx, "gpt_reasoning_COT"] = reasoning
                print(f"  - GPT-4 Answer: {answer}")

            # --- Gemini ---
            if row.get("gemini_answer_COT", "") not in ("A", "B"):
                print("  - Querying Gemini 1.5 Flash...")
                gemini_response = ask_gemini(prompt)
                if gemini_response.startswith("ERROR:"):
                    print("❌ Gemini failed. Exiting.")
                    sys.exit(1)
                answer, reasoning = extract_answer_and_reasoning(gemini_response)
                df.at[idx, "gemini_answer_COT"] = answer
                df.at[idx, "gemini_reasoning_COT"] = reasoning
                print(f"  - Gemini Answer: {answer}")

            # --- Llama ---
            if row.get("llama_answer_COT", "") not in ("A", "B"):
                print("  - Querying Llama 3...")
                llama_response = ask_llama(prompt)
                if llama_response.startswith("ERROR:"):
                    print("❌ Llama failed. Exiting.")
                    sys.exit(1)
                answer, reasoning = extract_answer_and_reasoning(llama_response)
                df.at[idx, "llama_answer_COT"] = answer
                df.at[idx, "llama_reasoning_COT"] = reasoning
                print(f"  - Llama 3 Answer: {answer}")

            if (idx + 1) % CHECKPOINT == 0:
                df.to_csv(OUTPUT_CSV, index=False)
                print(f"💾 Checkpoint saved at row {idx}")


    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
    finally:
        # ——— FINAL SAVE ———
        print("\n💾 Saving final results...")
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"✅ Done! Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    load_dotenv()
    main('FILENAME')