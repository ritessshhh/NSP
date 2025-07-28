import os
import time
import pandas as pd
import sys
from openai import OpenAI
from google import genai
from together import Together
from dotenv import load_dotenv

def main(input_file):
    # ——— CONFIG ———
    openai_api_key = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
    gemini_api_key = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
    llama3_api_key = os.environ.get("TOGETHER_API_KEY", "YOUR_TOGETHER_API_KEY")

    openai_client = OpenAI(api_key=openai_api_key)
    genai_client = genai.Client(api_key=gemini_api_key)
    together_client = Together(api_key=llama3_api_key)  # Optional: load from env

    GPT_MODEL = "gpt-4-turbo"
    GEMINI_MODEL = "gemini-1.5-flash"
    LLAMA_MODEL = "meta-llama/Meta-Llama-3-70B-Instruct-Turbo"

    INPUT_CSV = input_file
    OUTPUT_CSV = input_file
    CHECKPOINT = 5  # Save every 5 rows

    # ——— LOAD DATA ———
    df = pd.read_csv(INPUT_CSV)
    for col in ("gpt_answer", "gemini_answer", "llama_answer"):
        if col not in df.columns:
            df[col] = ""

    # ——— PROMPT BUILDER ———
    def build_prompt(context, opt_a, opt_b):
        return (
            f"Given the following story context:\n\n{context}\n\n"
            "Which sentence comes next?\n\n"
            f"A: {opt_a}\n\n"
            f"B: {opt_b}\n\n"
            "Only reply with a single letter: A or B. Do not say Neither, you have to reply with a single letter."
        )

    # ——— GPT QUERY ———
    def ask_gpt(prompt):
        response = openai_client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        ans = response.choices[0].message.content.strip().upper()
        # print(ans)
        return ans

    # ——— GEMINI QUERY ———
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

    # ——— LLAMA QUERY (Together API) ———
    def ask_llama(prompt):
        response = together_client.chat.completions.create(
            model=LLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        ans = response.choices[0].message.content.strip().upper()
        # print(ans)
        return ans

    # ——— MAIN LOOP ———
    try:
        start_index = 9170  # for example
        for idx, row in df.iloc[start_index:].iterrows():
            if (
                row["gpt_answer"] in ("A", "B") and
                row["gemini_answer"] in ("A", "B") and
                row["llama_answer"] in ("A", "B")
            ):
                continue

            print(f"➡️ Row {idx}: querying models...")

            prompt = build_prompt(row["context"], row["option_A"], row["option_B"])
            df.at[idx, "gpt_answer"] = ask_gpt(prompt)
            df.at[idx, "gemini_answer"] = ask_gemini(prompt)
            df.at[idx, "llama_answer"] = ask_llama(prompt)

            if idx % CHECKPOINT == 0:
                df.to_csv(OUTPUT_CSV, index=False)
                print(f"💾 Checkpoint saved at row {idx}")

    except Exception as e:
        print("❌ Error occurred, saving and exiting:", e)
        df.to_csv(OUTPUT_CSV, index=False)
        sys.exit(1)

    # ——— FINAL SAVE ———
    df.to_csv(OUTPUT_CSV, index=False)
    print("✅ Done! Results saved to", OUTPUT_CSV)

if __name__ == '__main__':
    load_dotenv()
    main('FILENAME')