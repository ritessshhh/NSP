# Testing Cross-Lingual Text Comprehension in LLMs Using Next Sentence Prediction

The contains the codebase for everything we did in the research paper. From generating Dataset to Testing all 3 LLMs.

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
    * **Description**: [Overall purpose of this script. e.g., This script generates Next Sentence Prediction (NSP) questions from the text corpora.]
    * **Functions**:
        * `function_name_1()`: [Explain what this function does.]
        * `function_name_2()`: [Explain what this function does.]
        * `main()`: [Explain the main execution flow.]

* **`gpt-gemini-llama-COT.py`**
    * **Description**: [Overall purpose of this script. e.g., This script uses GPT, Gemini, and Llama models to answer questions using a Chain-of-Thought prompting strategy.]
    * **Functions**:
        * `load_model()`: [Explain what this function does.]
        * `generate_answer()`: [Explain what this function does.]
        * `evaluate_results()`: [Explain what this function does.]

* **`gpt-gemini-llama.py`**
    * **Description**: [Overall purpose of this script. e.g., This script uses GPT, Gemini, and Llama models for standard question answering.]
    * **Functions**:
        * `function_name_1()`: [Explain what this function does.]
        * `function_name_2()`: [Explain what this function does.]
        * `function_name_3()`: [Explain what this function does.]

## Getting Started

Instructions on how to set up the project locally.

### Prerequisites

List any software or libraries that need to be installed.
* python
* pandas
* transformers
```bash
pip install pandas transformers
```

### Installation

1.  Clone the repo
    ```sh
    git clone [https://github.com/your_username/your_repository.git](https://github.com/your_username/your_repository.git)
    ```
2.  Install packages
    ```sh
    pip install -r requirements.txt
    ```

## Usage

Provide examples of how to run your scripts and what the expected output is.

_For more examples, please refer to the [Documentation](link_to_docs)_

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## License

Distributed under the [MIT License](https://opensource.org/licenses/MIT). See `LICENSE` for more information.

## Contact

Your Name - [@your_twitter](https://twitter.com/your_twitter) - email@example.com

Project Link: [https://github.com/your_username/your_repository](https://github.com/your_username/your_repository)
