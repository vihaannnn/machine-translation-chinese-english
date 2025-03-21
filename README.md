# Chinese English Machine Translation

## A Guardrails based approach to Chinese to English translation

Using guardrails of translations from less accurate but non-hallucinogenic models, we use GPT4 to generate new translations with increased context from these translation models to decrease likelihood of hallucinations.

Live Website can be found [here](https://machine-translation-chinese-english.streamlit.app/).

## Dataset
The data used to train this model can be found [here](https://www.kaggle.com/datasets/qianhuan/translation/code). A more detailed discussion of its limitations and potential pitfalls can be found in the [Ethics statement](Ethics_statement.md).

### Pipeline
1.⁠ ⁠Dataset Acquisition

    Source: Kaggle dataset (qianhuan/translation)
    Download Process:
        Uses the Kaggle API to authenticate and fetch the dataset.
        The dataset is extracted from a .zip file.

2.⁠ ⁠Data Loading & Preprocessing

    Loading the JSON dataset:
        Reads the translation2019zh_train.json and translation2019zh_valid.json files into Pandas DataFrames.
        Converts JSON lines into structured tabular format.

3.⁠ ⁠Tokenization

    BERT Tokenization
        Tokenizes Chinese and English sentences using pre-trained BERT tokenizers.
        Adds special tokens ([CLS], [SEP]).

4.⁠ ⁠Data Preparation for Transformer

    Splitting into Inputs and Labels for Sequence Prediction
        Removes the last token ([:-1]) to create input sequences.
        Removes the first token ([1:]) to create target sequences.
    Padding Input Sequences（Ensures all sequences are the same length.）
5.⁠ ⁠TensorFlow Dataset Preparation

    Convert Sequences into Tensors
        Convert tokenized sequences into TensorFlow tensors.

## Scripts

### Naïve model
 The [Naïve model](scripts-final/Naive.py) approach follows a Transformer-based architecture, incorporating key components like positional embedding, attention mechanisms, feed-forward networks, and an encoder-decoder structure. The model consists of 4 encoder and 4 decoder layers, each leveraging multi-head attention with 8 heads to capture linguistic dependencies effectively.

### Classical model

In the [Classical model](scripts-final/Trad_2.py) We use a dictionary based translation approach anda  heuristics based rearrangement approach. After this we then use a logistic regression model to choose between the reordered translation and the raw translation. With increased resources a Hidden Markov Model might improve this approach. 

### Deep Learning model

The [Deep Learning Model](scripts-final/Deep-Learning.py) Uses the Naive and ML translations as guardrails to prevent hallucinations
Aim - the light weight models creates guardrails in the form of candidate translations, which help the model stay on track.
Also scores each candidate model to understand what went wrong, and know where to focus to fix it.

## Setup and running of the main model

### Setting Up a Virtual Environment

To manage dependencies, it's recommended to use a virtual environment.

#### 1. Create a Virtual Environment

```bash
python -m venv venv
```

#### 2. Activate the Virtual Environment

- **On macOS/Linux:**

  ```bash
  source venv/bin/activate
  ```

- **On Windows (Command Prompt):**

  ```bash
  venv\Scripts\activate
  ```

- **On Windows (PowerShell):**

  ```bash
  venv\Scripts\Activate.ps1
  ```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Run [main](main.py)

## Models

The models are stored in the 'models' folder, they can be loaded and used directly.

## Notebooks

Rough notebook for exploring the Naïve approach model is included in the notebooks folder.

## Web App

The web app is included [here](https://city-segmentation-inpainting-gcpa28fv6enxw9ategareh.streamlit.app/), this demonstrates the entire project as an interactive experience.
