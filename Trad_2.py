

import json
import pandas as pd
import numpy as np
import nltk
import re
from collections import defaultdict
import spacy

def load_tools():

    # Download necessary NLTK data (we only need the tokenizer)
    nltk.download('punkt')

    # Load spaCy English model
    nlp = spacy.load("en_core_web_sm")

    return nlp

    ###############################################
    # 1. Load Data and CC-CEDICT Dictionary
    ###############################################

def load_data():
    # Path to your Chinese→English dataset (each line is a JSON object)
    file_path = "translation2019zh_train.json"

    # Load data into a DataFrame
    data_list = []
    with open(file_path, 'r') as file:
        for line in file:
            data_list.append(json.loads(line))
    df = pd.DataFrame(data_list)

    # Standardize column names and ensure 'chinese' exists.
    df.columns = df.columns.str.strip().str.lower()
    if "chinese" not in df.columns:
        raise KeyError("Column 'chinese' not found in DataFrame!")

    # For speed, sample 5k sentences (adjust as needed)
    df = df.sample(n=5000, random_state=42)

    # Path to your downloaded CC-CEDICT file (e.g., "cedict_ts.u8")
    cedict_path = "cedict_ts.u8"

    # Load the CC-CEDICT dictionary.
    # Keys are simplified Chinese tokens; values are lists of English definitions.
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

    # Convert to a regular dict for easier lookup.
    zh_eng_dict = dict(zh_to_eng_dict)

    return df, zh_eng_dict

###############################################
# 2. Translate & Filter Chinese Sentences
###############################################

def translate_and_filter(sentence):
    """
    For each Chinese character in the sentence, if a translation exists in CC‑CEDICT,
    include it using only the first option (i.e. text before the first semicolon);
    otherwise, skip the symbol.
    Returns a tuple: (filtered Chinese tokens, translated English tokens).
    """
    df, zh_eng_dict = load_data()
    tokens = list(sentence)
    filtered_zh = []
    translated = []
    for token in tokens:
        if token in zh_eng_dict:
            filtered_zh.append(token)
            # Get the first definition and split at semicolon.
            first_def = zh_eng_dict[token][0]
            first_option = first_def.split(';')[0].strip()
            translated.append(first_option)
    return filtered_zh, translated

def filter_pairs(df):
    # Apply the translation & filtering function.
    filtered_pairs = df['chinese'].apply(translate_and_filter)

    # Save filtered results for inspection.
    df['Filtered_Chinese'] = filtered_pairs.apply(lambda x: " ".join(x[0]))
    df['Filtered_Translated_Words'] = filtered_pairs.apply(lambda x: " ".join(x[1]))

###############################################
# 3. Reorder Translated English with Dependency Parsing
###############################################

def reorder_with_dependencies(sentence):
    """
    Uses spaCy's dependency parser to reorder words.
    Subject(s) → Verb(s) → Object(s) → Remaining tokens.
    """
    if not sentence:
        return ""
    nlp = load_tools()
    doc = nlp(sentence)
    subjects = []
    verbs = []
    objects = []
    others = []
    
    for token in doc:
        # Nominal subjects and passive subjects.
        if token.dep_ in {"nsubj", "nsubjpass"}:
            subjects.append(token.text)
        # Verbs.
        elif token.pos_ == "VERB":
            verbs.append(token.text)
        # Direct objects, attributes, or objects of prepositions.
        elif token.dep_ in {"dobj", "attr", "pobj"}:
            objects.append(token.text)
        else:
            others.append(token.text)
    
    # Combine tokens in a rough natural order.
    reordered = subjects + verbs + objects + others
    return " ".join(reordered)


def translate_chinese_text(chinese_text):
    """
    Given a Chinese text, this function:
      1. Translates the text character-by-character using CC‑CEDICT (skipping characters with no translation).
      2. Joins the translations into a raw English sentence.
      3. Reorders the raw English sentence using spaCy's dependency parser.
    
    Returns a tuple: (raw_translation, dependency_reordered_translation)
    """
    # Use the existing translate_and_filter function from your pipeline.
    filtered_zh, translated_tokens = translate_and_filter(chinese_text)
    
    # If no token was translated, return empty strings.
    if not translated_tokens:
        return "", ""
    
    # Form the raw translation.
    raw_translation = " ".join(translated_tokens)
    
    # Use the dependency-based reordering function to reorder the translation.
    reordered_translation = reorder_with_dependencies(raw_translation)
    
    return raw_translation, reordered_translation


print(translate_chinese_text("我喜欢学习新语言"))
print((""))
print(translate_chinese_text("今天天气很好"))
print((""))
print(translate_chinese_text("她正在看书"))
print((""))
print(translate_chinese_text("这杯咖啡味道很好"))
print((""))
print(translate_chinese_text("他每天早上跑步"))
print((""))
print(translate_chinese_text("请关灯"))
print((""))
print(translate_chinese_text("我稍后给你打电话"))
print((""))
print(translate_chinese_text("时间过得真快"))
print((""))
print(translate_chinese_text("最近的地铁站在哪里"))
print((""))
print(translate_chinese_text("我们出去吃晚饭吧"))