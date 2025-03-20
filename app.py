import streamlit as st
import os
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import openai

# Load environment variables
load_dotenv()

# Set up OpenAI API
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OPENAI_API_KEY not found in environment variables. Please add it to your .env file.")
else:
    openai.api_key = openai_api_key

# Define the model ID
MODEL_ID = "vihaannnn/Chinese-English-Transformer"

@st.cache_resource
def load_model():
    """Load model and tokenizer from HuggingFace."""
    try:
        st.info(f"Loading model from {MODEL_ID}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def translate_with_huggingface(text, model, tokenizer):
    """Generate translation using HuggingFace model."""
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=100)
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translation
    except Exception as e:
        st.error(f"Translation error: {e}")
        return None

def correct_with_openai(source_text, candidate_translation):
    """Refine translation using OpenAI API."""
    try:
        prompt = f"""translate - {source_text} 
Use this as a candidate translation - {candidate_translation} 
If you feel the candidate translation is wrong then correct it, and also print out the new translation phrase. 
Explain why the candidate translation was wrong by giving it a score from 0-10. 
Do not use emojis"""

        response = openai.chat.completions.create(
            model="gpt-4-turbo",  # or your preferred model
            messages=[
                {"role": "system", "content": "You are a helpful assistant specializing in Chinese to English translation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return None

# App interface
st.title("Chinese to English Translation with AI Correction")
st.write(f"Using HuggingFace model: {MODEL_ID}")
st.write("This app generates an initial translation and then refines it using OpenAI.")

# Main interface
input_text = st.text_area("Enter Chinese text:", height=150, 
                         placeholder="请在这里输入中文文本...")

col1, col2 = st.columns(2)
with col1:
    if st.button("Translate", type="primary"):
        if not input_text:
            st.warning("Please enter some text to translate.")
        else:
            # Load model
            model, tokenizer = load_model()
            
            if model and tokenizer:
                # Get candidate translation
                with st.spinner("Getting candidate translation..."):
                    candidate = translate_with_huggingface(input_text, model, tokenizer)
                
                if candidate:
                    st.subheader("Candidate Translation")
                    st.info(candidate)
                    
                    # Get corrected translation from OpenAI
                    with st.spinner("AI is refining the translation..."):
                        correction = correct_with_openai(input_text, candidate)
                    
                    if correction:
                        st.subheader("Refined Translation and Analysis")
                        st.markdown(correction)
                        
                        # Display download button for the final translation
                        st.download_button(
                            label="Download Translation",
                            data=correction,
                            file_name="translation_result.txt",
                            mime="text/plain"
                        )

with col2:
    st.subheader("Example")
    st.code("美国缓慢地开始倾听，但并非没有艰难曲折。", language="markdown")
    st.caption("Try copying this example to test the translation system.")

# Instructions for .env file
st.sidebar.header("Configuration")
st.sidebar.info(
    "Make sure to create a `.env` file in the same directory with:\n"
    "```\n"
    "OPENAI_API_KEY=your-openai-api-key-here\n"
    "```"
)

# Requirements
st.sidebar.header("Requirements")
st.sidebar.code("pip install streamlit torch transformers openai python-dotenv")

# How it works
st.sidebar.header("How it works")
st.sidebar.markdown(
    "1. Your Chinese text is sent to the Chinese-English Transformer\n"
    "2. The initial translation is generated\n"
    "3. OpenAI reviews and refines the translation\n"
    "4. You get both translations and an analysis"
)