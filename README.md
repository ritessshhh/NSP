# Testing Cross-Lingual Text Comprehension in LLMs Using Next Sentence Prediction

This repository contains the codebase for the research paper, "Testing Cross-Lingual Text Comprehension in LLMs Using Next Sentence Prediction". It includes everything from dataset generation to testing all three Large Language Models (LLMs).

## About The Project

Large language models show stunning fluency in English, but is this true understanding or just a reflection of massive training data? To find out, we tested their comprehension in a setting where they couldn't rely on data abundance: low-resource languages. Building on prior work (Agarwal et al., AAAI-25) that used Next Sentence Prediction (NSP) as a test, we created a large-scale benchmark with 10,000 questions each for English (a high-resource language), Swahili (medium-resource), and Hausa (low-resource). We then tested several top models, including GPT-4 Turbo, Gemini 1.5 Flash, and LLaMA 3 70B, to see how their comprehension holds up. The results painted a clear picture of how language resources impact performance. While all models excelled in English, their accuracy dropped in Swahili and fell sharply in Hausa, with LLaMA 3 struggling the most. The story became even more interesting when we introduced Chain-of-Thought (CoT) prompting. For the struggling LLaMA 3, CoT acted as a helpful guide, significantly boosting its accuracy. However, for the more capable GPT-4 and Gemini, the same technique often backfired, leading to a kind of "overthinking" that hurt their performance in the cross-lingual context.This reveals that Chain-of-Thought is not a universal solution; its effectiveness depends heavily on the model's baseline capability and the specific context of the task. Our framework pinpoints LLM weaknesses, highlights when CoT helps or hinders cross‑lingual comprehension, and factors influencing their decisions. The goal is to build models that can understand all human languages, not just the most common ones.
## File Structure

Here is an overview of the files and directories in this project.

### `datasets/`
This directory contains the datasets used for generating and evaluation NSP questions.

* **`txt-en/all_books.txt`**: Contains all the English stories downloaded from African Stories Website. 
* **`txt-ha/all_books.txt`**: Contains all the Hausa stories downloaded from African Stories Website. 
* **`txt-sw/all_books.txt`**: Contains all the Swahili stories downloaded from African Stories Website. 

### `*.csv` Files
These files contain questions and answers for the Next Sentence Prediction (NSP) task.

* **`NSP_QUESTIONS_WITH_ANSWERS_EN.csv`**: Contains the entire 10,000 NSP questions dataset for English. 
* **`NSP_QUESTIONS_WITH_ANSWERS_EN_1000_COT.csv`**: Contains randomly sampled 1000 questions for English which were used for COT based evaluation.
* **`NSP_QUESTIONS_WITH_ANSWERS_HA.csv`**: Contains the entire 10,000 NSP questions dataset for Hausa. 
* **`NSP_QUESTIONS_WITH_ANSWERS_HA_1000_COT.csv`**: Contains randomly sampled 1000 questions for Hausa which were used for COT based evaluation. 
* **`NSP_QUESTIONS_WITH_ANSWERS_SW.csv`**: Contains the entire 10,000 NSP questions dataset for Swahili. 
* **`NSP_QUESTIONS_WITH_ANSWERS_SW_1000_COT.csv`**: Contains randomly sampled 1000 questions for Swahili which were used for COT based evaluation. 

### Python Scripts

* **`generate-nsp.py`**
    * **Description**: This script processes raw text files to generate Next Sentence Prediction (NSP) questions. It splits stories into sentences, creates question contexts, and pairs a correct next sentence with a plausible "distractor" sentence from later in the story.
    * **Functions**:
        * `clean_text(s: str) -> str`: Takes a string and replaces all whitespace (like newlines, tabs, and multiple spaces) with a single space.
        * `load_stories(filepath)`: Reads a text file and splits it into a list of individual stories. It assumes stories are separated by lines of three or more dashes.
        * `split_sentences(text)`: Takes a block of text (a story) and splits it into a list of sentences based on punctuation marks (., !, ?).
        * `generate_nsp_items(...)`: The core function that generates NSP questions from a list of sentences. For each possible context, it creates a correct answer (the next sentence) and a distractor (a sentence from further in the text), then randomizes their order (A/B). It also records metadata like context length and distractor distance.

* **`gpt-gemini-llama.py`**
    * **Description**: This script runs the standard NSP evaluation. It reads a CSV file of questions, sends them to the GPT, Gemini, and Llama APIs, and records their single-letter answers (A or B) back into the same CSV.
    * **Functions**:
        * `main(input_file)`: The main function that orchestrates the entire process. It handles API client initialization, data loading, iterating through questions, calling the models, and saving the results.
        * `build_prompt(context, opt_a, opt_b)`: Creates the simple, direct prompt that asks the model to choose the next sentence, instructing it to reply with only a single letter.
        * `ask_gpt(prompt)`: Sends the prompt to the OpenAI (GPT) API and returns the model's response.
        * `ask_gemini(prompt, ...)`: Sends the prompt to the Google (Gemini) API. It includes a retry mechanism with a delay to handle potential server errors (like 503).
        * `ask_llama(prompt)`: Sends the prompt to the Together API (for Llama) and returns the model's response.

* **`gpt-gemini-llama-COT.py`**
    * **Description**: This script runs the Chain-of-Thought (CoT) NSP evaluation. It is similar to the standard script but uses a more complex prompt that asks the models to provide step-by-step reasoning before giving their final answer. It then parses this detailed response to extract both the reasoning and the final single-letter answer.
    * **Functions**:
        * `extract_answer_and_reasoning(response_text)`: Parses the model's full text response. It finds the final single-letter answer (A or B) and separates it from the preceding text, which is considered the reasoning.
        * `main(input_file)`: The main function that orchestrates the CoT evaluation process, similar to the standard script but using the CoT prompt and the answer extraction logic.
        * `build_cot_prompt(context, opt_a, opt_b)`: Creates the detailed CoT prompt. It includes instructions in the target language (English, Hausa, or Swahili) for the model to provide reasoning first, followed by the single-letter answer on a new line.
        * `ask_gpt(prompt)`, `ask_gemini(prompt, ...)` and `ask_llama(prompt)`: These functions work identically to the ones in the standard script, sending the CoT prompt to their respective APIs.

* **`evaluation-metrics.py`**
    * **Description**: This script contains functions to analyze the results of the model evaluations. It can sample data, calculate accuracy scores, and provide deeper analysis of where models went wrong.
    * **Functions**:
        * `sample_csv(input_file, output_file, n=1000)`: Creates a smaller, random sample from a larger CSV file. This is useful for creating consistent test sets.
        * `validate_and_score(csv_path)`: Calculates and prints the accuracy for each model based on the standard evaluation results. It checks for valid 'A' or 'B' answers and compares them to the correct 'label'. It also saves all incorrect answers to a `wrong_answers.csv` file.
        * `validate_and_score_COT(csv_path)`: Does the same as the function above, but for the Chain-of-Thought results (using the `_COT` columns in the CSV).
        * `print_distractor_length_distribution_by_model()`: Analyzes the `wrong_answers.csv` file to show how often models failed based on the `distractor_distance` feature. This helps identify if models struggle more when the wrong answer is closer to the context.

* **`Testing_Cross_Lingual_Text_Comprehension_in_LLMs_Using_Next_Sentence_Prediction.ipynb`**
    * **Description**: This Jupyter Notebook is used for in-depth feature engineering and data analysis. It computes semantic similarity and perplexity scores for the NSP question pairs to understand the underlying characteristics of the dataset and how they might influence model performance. It also contains functions for visualizing these features and analyzing model errors.
    * **Functions**:
        * `addNewFeatures(filePath, ...)`: Reads an NSP question CSV, samples it, and computes two new features: semantic similarity (using SentenceTransformers) and perplexity (using various causal LMs like UlizaLlama3, Mistral-7B, and HausaLlama). It saves the augmented data to a new CSV.
        * `get_embedding(text)`: A helper function within `addNewFeatures` to encode text into a vector embedding.
        * `compute_ppl(context, option)`: A helper function within `addNewFeatures` to calculate the perplexity of an option sentence given a context.
        * `plot_data(data_path, title)`: Reads a feature-rich CSV and generates a series of plots to validate the dataset quality. These plots visualize relationships like distractor perplexity vs. distance and perplexity vs. context length.
        * `save_wrong_answers_by_model(csv_path)`: Filters and saves the rows where each model answered incorrectly into separate CSV files for detailed error analysis.
        * `main(csv_path)`: Orchestrates the error analysis by generating and plotting wrong predictions based on features like context length and distractor distance.



## Getting Started

Instructions on how to set up the project locally.

### Prerequisites

List any software or libraries that need to be installed.
* python
* pandas
* openai
* google-generativeai
* together
* python-dotenv

```bash
pip install pandas openai google-generativeai together python-dotenv
```

### Installation

1.  Clone the repo
    ```sh
    git clone https://github.com/ritessshhh/NSP.git
    ```
2.  Create a `.env` file in the root directory and add your API keys:
    ```
    OPENAI_API_KEY="your_openai_key"
    GEMINI_API_KEY="your_gemini_key"
    TOGETHER_API_KEY="your_together_key"
    ```

## Usage

1.  **Generate Questions**: Run `generate-nsp.py` (configure the `INPUT_TXT` and `OUTPUT_CSV` variables inside the script for each language).
2.  **Run Evaluations**: Run `gpt-gemini-llama.py` and `gpt-gemini-llama-COT.py`, passing the path to your question CSV as an argument to the `main` function.
3.  **Analyze Results**: Run `evaluation-metrics.py` to see the accuracy scores and other analyses.
