#!/usr/bin/env python
"""
This script builds a transformer-based sequence-to-sequence model for machine translation between Chinese and English using TensorFlow and Hugging Face's tokenizers.
It includes data preparation, custom transformer architecture, training, and saving the model.
"""

import os
import subprocess
import json
import re
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from transformers import BertTokenizer

def setup_kaggle_and_download():
    """Sets up Kaggle API credentials and downloads the translation dataset."""
    os.system("pip install -q kaggle")
    os.system("mkdir -p ~/.kaggle")
    os.system("""bash -c 'echo "{\"username\":\"\",\"key\":\"\"}" > ~/.kaggle/kaggle.json'""")
    os.system("chmod 600 ~/.kaggle/kaggle.json")
    os.system("kaggle datasets download -d qianhuan/translation")
    os.system("unzip -o translation.zip")

def jsontodf(json_filepath):
    """Converts a JSON file to a pandas DataFrame.

    Args:
        json_filepath (str): Path to the JSON file.

    Returns:
        pd.DataFrame: DataFrame containing parsed JSON objects.
    """
    json_list = []
    with open(json_filepath, 'r') as file:
        for line in file:
            json_list.append(line.strip())
    json_objects = [json.loads(json_str) for json_str in json_list if json_str]
    return pd.DataFrame(json_objects)

def contains_english_or_number(text):
    """Checks if the given text contains English letters or numbers.

    Args:
        text (str): Input text.

    Returns:
        bool: True if text contains English or digits, False otherwise.
    """
    pattern = r"^(?=.*[a-zA-Z])|(?=.*\d).+$"
    return bool(re.match(pattern, text))

def add_padding(token_list, max_length):
    """Pads or truncates a list of tokens to a fixed length.

    Args:
        token_list (List[int]): List of token IDs.
        max_length (int): Desired length.

    Returns:
        List[int]: Padded or truncated token list.
    """
    if len(token_list) < max_length:
        padding_length = max_length - len(token_list)
        token_list += [0] * padding_length
    else:
        token_list = token_list[:max_length]
    return token_list

class PositionalEmbedding(tf.keras.layers.Layer):
    """Embedding layer that includes learned word embeddings and fixed positional encodings."""
    def __init__(self, vocab_size, d_model, max_length):
        super().__init__()
        self.embed_layer = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.pos_encoding = self._build_pos_encoding(max_length, d_model)

    def _build_pos_encoding(self, max_len, d_model):
        pos = np.arange(max_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = pos * angle_rates
        pos_enc = np.zeros((max_len, d_model))
        pos_enc[:, 0::2] = np.sin(angle_rads[:, 0::2])
        pos_enc[:, 1::2] = np.cos(angle_rads[:, 1::2])
        return tf.cast(pos_enc, tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        x = self.embed_layer(x)
        x *= tf.math.sqrt(tf.cast(tf.shape(x)[-1], tf.float32))
        return x + self.pos_encoding[tf.newaxis, :seq_len, :]

class AttentionBaseLayer(tf.keras.layers.Layer):
    """Base layer for attention with normalization and skip connection."""
    def __init__(self, num_heads, key_dim, dropout_rate=0.1):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout_rate)
        self.norm = tf.keras.layers.LayerNormalization()
        self.skip = tf.keras.layers.Add()

class CrossAttentionLayer(AttentionBaseLayer):
    """Applies multi-head cross-attention between query and context."""
    def call(self, query, context):
        attn_output, attn_scores = self.mha(query=query, key=context, value=context, return_attention_scores=True)
        self.last_scores = attn_scores
        x = self.skip([query, attn_output])
        return self.norm(x)

class GlobalSelfAttentionLayer(AttentionBaseLayer):
    """Applies global self-attention on the input."""
    def call(self, x):
        attn_output = self.mha(query=x, key=x, value=x)
        return self.norm(self.skip([x, attn_output]))

class CausalSelfAttentionLayer(AttentionBaseLayer):
    """Applies causal self-attention (for decoder input)."""
    def call(self, x):
        attn_output = self.mha(query=x, key=x, value=x, use_causal_mask=True)
        return self.norm(self.skip([x, attn_output]))

class FeedForwardNetwork(tf.keras.layers.Layer):
    """Feedforward neural network with residual connection and normalization."""
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.ffn_seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.norm = tf.keras.layers.LayerNormalization()
        self.skip = tf.keras.layers.Add()

    def call(self, x):
        return self.norm(self.skip([x, self.ffn_seq(x)]))

class TransformerEncoderLayer(tf.keras.layers.Layer):
    """Single transformer encoder layer with self-attention and FFN."""
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.self_attn = GlobalSelfAttentionLayer(num_heads=num_heads, key_dim=d_model, dropout_rate=dropout_rate)
        self.ffn = FeedForwardNetwork(d_model, dff, dropout_rate)

    def call(self, x):
        x = self.self_attn(x)
        return self.ffn(x)

class TransformerEncoder(tf.keras.layers.Layer):
    """Full transformer encoder stack."""
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1):
        super().__init__()
        max_length = 128
        self.pos_embed = PositionalEmbedding(vocab_size, d_model, max_length)
        self.enc_layers = [TransformerEncoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        x = self.pos_embed(x)
        x = self.dropout(x)
        for layer in self.enc_layers:
            x = layer(x)
        return x

class TransformerDecoderLayer(tf.keras.layers.Layer):
    """Single transformer decoder layer with causal and cross-attention."""
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.causal_attn = CausalSelfAttentionLayer(num_heads, d_model, dropout_rate)
        self.cross_attn = CrossAttentionLayer(num_heads, d_model, dropout_rate)
        self.ffn = FeedForwardNetwork(d_model, dff, dropout_rate)

    def call(self, x, encoder_out):
        x = self.causal_attn(x)
        x = self.cross_attn(x, encoder_out)
        self.last_scores = self.cross_attn.last_scores
        return self.ffn(x)

class TransformerDecoder(tf.keras.layers.Layer):
    """Full transformer decoder stack."""
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1):
        super().__init__()
        max_length = 128
        self.pos_embed = PositionalEmbedding(vocab_size, d_model, max_length)
        self.dec_layers = [TransformerDecoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.last_scores = None

    def call(self, x, encoder_out):
        x = self.pos_embed(x)
        x = self.dropout(x)
        for layer in self.dec_layers:
            x = layer(x, encoder_out)
        self.last_scores = self.dec_layers[-1].last_scores
        return x

@tf.keras.utils.register_keras_serializable()
class Transformer(tf.keras.Model):
    """Complete transformer model combining encoder and decoder."""
    def __init__(self, *, num_layers, d_model, num_heads, dff,
                 input_vocab_size, target_vocab_size, dropout_rate=0.1):
        super().__init__()
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, dff, input_vocab_size, dropout_rate)
        self.decoder = TransformerDecoder(num_layers, d_model, num_heads, dff, target_vocab_size, dropout_rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, training=False):
        context, x = inputs
        context = self.encoder(context)
        x = self.decoder(x, context)
        logits = self.final_layer(x)
        try:
            del logits._keras_mask
        except AttributeError:
            pass
        return logits

@tf.keras.utils.register_keras_serializable()
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Implements the learning rate schedule used in the original Transformer paper."""
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        d_model_val = int(self.d_model.numpy()) if hasattr(self.d_model, 'numpy') else self.d_model
        return {'d_model': d_model_val, 'warmup_steps': self.warmup_steps}

@tf.keras.utils.register_keras_serializable()
class CustomAdam(tf.keras.optimizers.Adam):
    """Custom Adam optimizer that includes additional parameters for serialization."""
    def __init__(self, custom_param, **kwargs):
        super().__init__(**kwargs)
        self.custom_param = custom_param

    def get_config(self):
        config = super().get_config()
        config.update({'custom_param': self.custom_param})
        return config

def main():
    setup_kaggle_and_download()
    train_df = jsontodf("translation2019zh/translation2019zh_train.json")
    valid_df = jsontodf("translation2019zh/translation2019zh_valid.json")
    sampled_df = train_df.sample(frac=0.03, random_state=38)
    sampled_df["contains_english_or_number"] = sampled_df["chinese"].apply(contains_english_or_number)
    filtered_sampled_df = sampled_df[~sampled_df["contains_english_or_number"]].drop(columns=["contains_english_or_number"])

    valid_df["contains_english_or_number"] = valid_df["chinese"].apply(contains_english_or_number)
    filtered_valid_df = valid_df[~valid_df["contains_english_or_number"]].drop(columns=["contains_english_or_number"])
    filtered_df = pd.concat([filtered_valid_df, filtered_sampled_df], ignore_index=True)

    tokenizer_en = BertTokenizer.from_pretrained("bert-base-cased")
    tokenizer_cn = BertTokenizer.from_pretrained("bert-base-chinese")

    filtered_df["english_tokenized"] = filtered_df["english"].apply(
        lambda x: tokenizer_en.encode(x, add_special_tokens=True)
    )
    filtered_df["chinese_tokenized"] = filtered_df["chinese"].apply(
        lambda x: tokenizer_cn.encode(x, add_special_tokens=True)
    )
    english_seqs = filtered_df["english_tokenized"]
    chinese_seqs = filtered_df["chinese_tokenized"]

    MAX_TOKENIZE_LENGTH = max(english_seqs.apply(len).max(), chinese_seqs.apply(len).max())
    MAX_TOKENIZE_LENGTH = int(pow(2, math.ceil(math.log(MAX_TOKENIZE_LENGTH, 2))))
    EMBEDDING_DEPTH = 256

    print("MAX_TOKENIZE_LENGTH:", MAX_TOKENIZE_LENGTH)
    print("EMBEDDING_DEPTH:", EMBEDDING_DEPTH)

    cn_set_start = chinese_seqs.apply(lambda x: x[:-1])
    cn_set_end = chinese_seqs.apply(lambda x: x[1:])
    en_set_start = english_seqs.apply(lambda x: x[:-1])
    en_set_end = english_seqs.apply(lambda x: x[1:])

    chinese_seqs = chinese_seqs.apply(lambda x: add_padding(x, MAX_TOKENIZE_LENGTH))
    english_seqs = english_seqs.apply(lambda x: add_padding(x, MAX_TOKENIZE_LENGTH))
    cn_set_start = cn_set_start.apply(lambda x: add_padding(x, MAX_TOKENIZE_LENGTH - 1))
    cn_set_end = cn_set_end.apply(lambda x: add_padding(x, MAX_TOKENIZE_LENGTH - 1))
    en_set_start = en_set_start.apply(lambda x: add_padding(x, MAX_TOKENIZE_LENGTH - 1))
    en_set_end = en_set_end.apply(lambda x: add_padding(x, MAX_TOKENIZE_LENGTH - 1))

    print("===== Chinese tokenized example =====")
    print(chinese_seqs.iloc[0])
    print(cn_set_start.iloc[0])
    print(cn_set_end.iloc[0])
    print("===== English tokenized example =====")
    print(english_seqs.iloc[0])
    print(en_set_start.iloc[0])
    print(en_set_end.iloc[0])

    data_size = len(filtered_df)
    train_size = int(0.95 * data_size)
    print("Train size:", train_size)
    print("Test size:", data_size - train_size)

    batch_size = 32
    en_tensor = tf.convert_to_tensor(list(english_seqs))
    cn_tensor = tf.convert_to_tensor(list(chinese_seqs))
    cn_start_tensor = tf.convert_to_tensor(list(cn_set_start))
    cn_end_tensor = tf.convert_to_tensor(list(cn_set_end))
    en_start_tensor = tf.convert_to_tensor(list(en_set_start))
    en_end_tensor = tf.convert_to_tensor(list(en_set_end))

    en_train, en_test = en_tensor[:train_size], en_tensor[train_size:]
    cn_train, cn_test = cn_tensor[:train_size], cn_tensor[train_size:]
    cn_start_train, cn_start_test = cn_start_tensor[:train_size], cn_start_tensor[train_size:]
    cn_end_train, cn_end_test = cn_end_tensor[:train_size], cn_end_tensor[train_size:]
    en_start_train, en_start_test = en_start_tensor[:train_size], en_start_tensor[train_size:]
    en_end_train, en_end_test = en_end_tensor[:train_size], en_end_tensor[train_size:]

    en_to_cn_train_set = tf.data.Dataset.from_tensor_slices(((en_train, cn_start_train), cn_end_train)).batch(batch_size)
    cn_to_en_train_set = tf.data.Dataset.from_tensor_slices(((cn_train, en_start_train), en_end_train)).batch(batch_size)

    en_to_cn_test_set = tf.data.Dataset.from_tensor_slices(((en_test, cn_start_test), cn_end_test)).shuffle(train_size, reshuffle_each_iteration=True).batch(batch_size)
    cn_to_en_test_set = tf.data.Dataset.from_tensor_slices(((cn_test, en_start_test), en_end_test)).batch(batch_size)

    print("EN to CN Train Set:")
    for (en_batch, cn_batch), cn_label in en_to_cn_train_set.take(1):
        print(en_batch.shape, cn_batch.shape, cn_label.shape)
    print("CN to EN Train Set:")
    for (cn_batch, en_batch), en_label in cn_to_en_train_set.take(1):
        print(cn_batch.shape, en_batch.shape, en_label.shape)

    sample_en = en_tensor[:batch_size]
    sample_cn = cn_tensor[:batch_size]

    if tf.config.list_physical_devices("GPU"):
        print("GPU is available")
        tf.config.set_visible_devices(tf.config.list_physical_devices("GPU"), "GPU")
        logical_devices = tf.config.list_logical_devices("GPU")
        print(len(logical_devices), "GPU(s) are available.")
    else:
        print("GPU is NOT available")
        tf.config.set_visible_devices(tf.config.list_physical_devices("CPU"), "CPU")

    encoder = TransformerEncoder(
        num_layers=4,
        d_model=EMBEDDING_DEPTH,
        num_heads=8,
        dff=MAX_TOKENIZE_LENGTH,
        vocab_size=tokenizer_cn.vocab_size,
        dropout_rate=0.1)
    encoder_output = encoder(sample_cn)
    print("Encoder Input shape:", sample_cn.shape)
    print("Encoder Output shape:", encoder_output.shape)

    decoder = TransformerDecoder(
        num_layers=4,
        d_model=EMBEDDING_DEPTH,
        num_heads=8,
        dff=MAX_TOKENIZE_LENGTH,
        vocab_size=tokenizer_en.vocab_size,
        dropout_rate=0.1)
    decoder_output = decoder(sample_en, encoder_output)
    print("Decoder Input shape:", sample_en.shape)
    print("Decoder Output shape:", decoder_output.shape)
    print("Last attention scores shape:", decoder.last_scores.shape)

    tf.keras.utils.get_custom_objects().clear()
    num_layers = 1
    d_model = EMBEDDING_DEPTH
    dff = MAX_TOKENIZE_LENGTH
    num_heads = 3
    dropout_rate = 0.1

    cn_to_en_transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=tokenizer_cn.vocab_size,
        target_vocab_size=tokenizer_en.vocab_size,
        dropout_rate=dropout_rate)

    output = cn_to_en_transformer((sample_cn, sample_en))
    print("Test model output shape:", output.shape)

    learning_rate = CustomSchedule(EMBEDDING_DEPTH)
    optimizer = CustomAdam(
        custom_param=learning_rate,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )
    cn_to_en_transformer.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=optimizer,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    history = cn_to_en_transformer.fit(
        cn_to_en_train_set,
        epochs=10,
        validation_data=cn_to_en_test_set
    )

    model_path = 'cn_to_en_transformer-test.h5'
    cn_to_en_transformer.save(model_path, save_format='h5')
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    main()
