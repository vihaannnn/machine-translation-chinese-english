import ssl
import json
import pandas as pd
import numpy as np
import nltk
import re
from collections import defaultdict
import spacy
import joblib
from sklearn.linear_model import LogisticRegression


def load_tools():
    """
    Downloads necessary NLTK data and loads spaCy's English model.
    Returns the spaCy nlp object.
    """
    try:  # Chat GPT4o utilised to create workaround for ssl issue 03/19/10:40
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context # end

    nltk.download('punkt')
    nlp = spacy.load("en_core_web_sm")
    return nlp

def load_data():
    """
    Loads the Chinese→English dataset and the CC‑CEDICT dictionary.
    Assumes the dataset JSON has columns "chinese" (source) and "english" (ground truth translation).
    Returns a tuple (df, zh_eng_dict) where:
      - df is a DataFrame sampled for speed.
      - zh_eng_dict maps simplified Chinese tokens to lists of English definitions.
    """
    # Not included in repo, must be downloaded (see README).
    file_path = "translation2019zh_train.json"
    data_list = []
    with open(file_path, 'r') as f:
        for line in f:
            data_list.append(json.loads(line))
    df = pd.DataFrame(data_list)
    df.columns = df.columns.str.strip().str.lower()
    if "chinese" not in df.columns or "english" not in df.columns:
        raise KeyError("Dataset must contain 'chinese' and 'english' columns!")
    # For speed, sample 5000 sentences (adjust as needed)
    df = df.sample(n=5000, random_state=0)

    cedict_path = "cedict_ts.u8"
    zh_to_eng_dict = defaultdict(list)
    with open(cedict_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = re.split(r"\s+", line.strip(), maxsplit=3)
            if len(parts) < 4:
                continue
            simplified, traditional, pinyin, definition = parts
            definitions = definition.strip("/").split("/")
            zh_to_eng_dict[simplified] = definitions
    zh_eng_dict = dict(zh_to_eng_dict)
    return df, zh_eng_dict


def translate_and_filter(sentence, zh_eng_dict):
    """
    For each Chinese character in the sentence, if a translation exists in CC‑CEDICT,
    include it using only the first option (i.e. text before the first semicolon);
    otherwise, skip it.
    Returns a tuple: (filtered Chinese tokens, translated English tokens).
    """
    tokens = list(sentence)
    filtered_zh = []
    translated = []
    for token in tokens:
        if token in zh_eng_dict:
            filtered_zh.append(token)
            first_def = zh_eng_dict[token][0]
            first_option = first_def.split(';')[0].strip()
            translated.append(first_option)
    return filtered_zh, translated

def filter_pairs(df, zh_eng_dict):
    """
    Applies translate_and_filter to every Chinese sentence in the DataFrame.
    Adds two new columns:
      - 'Filtered_Chinese'
      - 'Filtered_Translated_Words'
    Returns the updated DataFrame.
    """
    filtered_pairs = df['chinese'].apply(lambda s: translate_and_filter(s, zh_eng_dict))
    df['Filtered_Chinese'] = filtered_pairs.apply(lambda x: " ".join(x[0]))
    df['Filtered_Translated_Words'] = filtered_pairs.apply(lambda x: " ".join(x[1]))
    return df


def reorder_with_dependencies(sentence, nlp):
    """
    Uses spaCy's dependency parser to reorder words.
    It extracts subjects, verbs, objects, and others, then recombines them as:
      Subject(s) → Verb(s) → Object(s) → Others.
    """
    if not sentence:
        return ""
    doc = nlp(sentence)
    subjects, verbs, objects, others = [], [], [], []
    for token in doc:
        if token.dep_ in {"nsubj", "nsubjpass"}:
            subjects.append(token.text)
        elif token.pos_ == "VERB":
            verbs.append(token.text)
        elif token.dep_ in {"dobj", "attr", "pobj"}:
            objects.append(token.text)
        else:
            others.append(token.text)
    reordered = subjects + verbs + objects + others
    return " ".join(reordered)

def translate_chinese_text(chinese_text, zh_eng_dict, nlp):
    """
    Given a Chinese text, this function:
      1. Translates the text character-by-character using CC‑CEDICT.
      2. Joins the translations into a raw English sentence.
      3. Reorders the raw translation using dependency parsing.
    Returns a tuple: (raw_translation, dependency_reordered_translation)
    """
    filtered_zh, translated_tokens = translate_and_filter(chinese_text, zh_eng_dict)
    if not translated_tokens:
        return "", ""
    raw_translation = " ".join(translated_tokens)
    reordered_translation = reorder_with_dependencies(raw_translation, nlp)
    return raw_translation, reordered_translation

def extract_candidate_features(candidate, ground_truth):
    """
    Extracts simple features comparing candidate translation to the ground truth.
    Features:
      1. Absolute difference in number of words.
      2. Absolute difference in average word length.
      3. Absolute difference in number of unique words.
      4. Absolute difference in ratio (unique words / total words).
    Returns a list of features.
    """
    cand_words = candidate.split()
    gt_words = ground_truth.split()
    n_cand = len(cand_words)
    n_gt = len(gt_words)
    # Avoid division by zero.
    if n_cand == 0 or n_gt == 0:
        return [0, 0, 0, 0]
    avg_cand = np.mean([len(w) for w in cand_words])
    avg_gt = np.mean([len(w) for w in gt_words])
    unique_cand = len(set(cand_words))
    unique_gt = len(set(gt_words))
    ratio_cand = unique_cand / n_cand
    ratio_gt = unique_gt / n_gt
    return [abs(n_cand - n_gt),
            abs(avg_cand - avg_gt),
            abs(unique_cand - unique_gt),
            abs(ratio_cand - ratio_gt)]

def prepare_logistic_data(df):
    """
    For each row in the DataFrame, extracts features for both candidate translations 
    (the raw filtered translation and the dependency-reordered version) relative to the ground truth.
    Computes a difference vector: features(candidate_A, GT) - features(candidate_B, GT).
    Sets the label to 1 if candidate A (raw) is closer (i.e. lower sum of differences) than candidate B (reordered),
    and 0 otherwise.
    Returns X (feature differences) and y (labels).
    """
    X, y = [], []
    for idx, row in df.iterrows():
        ground_truth = row["english"].lower().strip()
        cand_A = row["Filtered_Translated_Words"].lower().strip()  # raw translation
        cand_B = row["Dependency_Reordered"].lower().strip()       # reordered translation
        # Skip rows where ground truth or candidates are missing.
        if not ground_truth or not cand_A or not cand_B:
            continue
        feat_A = extract_candidate_features(cand_A, ground_truth)
        feat_B = extract_candidate_features(cand_B, ground_truth)
        diff = np.array(feat_A) - np.array(feat_B)
        X.append(diff.tolist())
        # Label: 1 if candidate A is closer (i.e., sum(feat_A) < sum(feat_B)), else 0.
        label = 1 if sum(feat_A) < sum(feat_B) else 0
        y.append(label)
    return X, y

def train_logistic_model(X, y):
    """
    Trains a logistic regression model
    Returns the trained model.
    """
    model = LogisticRegression()
    model.fit(X, y)
    return model

def choose_candidate(original, reordered, ground_truth, model):
    """
    extracts features for each relative to the ground truth, computes the difference vector,
    and uses the logistic regression model to choose the better candidate.
    Returns "Original" if the model predicts label 1, else "Reordered".
    """
    feat_orig = extract_candidate_features(original.lower().strip(), ground_truth.lower().strip())
    feat_reord = extract_candidate_features(reordered.lower().strip(), ground_truth.lower().strip())
    diff = np.array(feat_orig) - np.array(feat_reord)
    diff = diff.reshape(1, -1)
    pred = model.predict(diff)
    return "Original" if pred[0] == 1 else "Reordered"


def example():
    nlp = load_tools()
    df, zh_eng_dict = load_data()
    # Ensure the ground truth exists.
    if "english" not in df.columns:
        raise KeyError("Ground truth translation (column 'english') is required in the dataset!")
    df = filter_pairs(df, zh_eng_dict)
    
    # Apply dependency-based reordering.
    df['Dependency_Reordered'] = df['Filtered_Translated_Words'].apply(lambda s: reorder_with_dependencies(s, nlp))
    
    # For demonstration, print sample translations.
    print("=== Sample Translations ===")
    print(df[['chinese', 'english', 'Filtered_Translated_Words', 'Dependency_Reordered']].head(10))
    
    # Prepare training data for logistic regression.
    X, y = prepare_logistic_data(df)
    if len(set(y)) < 2:
        raise ValueError("Training data does not have two classes. Check your feature extraction and labeling.")
    
    logreg_model = train_logistic_model(X, y)

    joblib.dump(logreg_model, "logistic_model.pkl") # Save model 




## Examples generated using below code 

# print(translate_chinese_text("我喜欢学习新语言"))
# print((""))
# print(translate_chinese_text("今天天气很好"))
# print((""))
# print(translate_chinese_text("她正在看书"))
# print((""))
# print(translate_chinese_text("这杯咖啡味道很好"))
# print((""))
# print(translate_chinese_text("他每天早上跑步"))
# print((""))
# print(translate_chinese_text("请关灯"))
# print((""))
# print(translate_chinese_text("我稍后给你打电话"))
# print((""))
# print(translate_chinese_text("时间过得真快"))
# print((""))
# print(translate_chinese_text("最近的地铁站在哪里"))
# print((""))
# print(translate_chinese_text("我们出去吃晚饭吧"))