import streamlit as st
import openai
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_openai_refinement(chinese_text, candidate_translation):
    """Get translation refinement from OpenAI API using JSON response format"""
    try:
        response = openai.chat.completions.create(
            model="gpt-4-turbo",  # You can change this to the model you want to use
            messages=[
                {"role": "system", "content": "You are a helpful translation assistant. Return your response as JSON."},
                {"role": "user", "content": f"""
                translate - {chinese_text} 
                Use this as a candidate translation - {candidate_translation} 
                If you feel the candidate translation is wrong then correct it, and also print out the new translation phrase. 
                Explain why the candidate translation was wrong by giving it a score from 0-10. 
                Do not use emojis.
                
                Respond with a JSON object using this exact structure:
                {{
                    "refined_translation": "your corrected translation or the original if correct",
                    "score": number between 0-10,
                    "comments": "your explanation of why the translation was wrong or right"
                }}
                """}
            ],
            temperature=0.3,
            max_tokens=500,
            response_format={"type": "json_object"}  # Use the JSON response format
        )
        
        # The response will be in JSON format already
        result = response.choices[0].message.content
        
        # Parse the JSON
        return json.loads(result)
        
    except Exception as e:
        return {
            "refined_translation": candidate_translation,
            "score": "N/A",
            "comments": f"Error with OpenAI API: {str(e)}"
        }

# Sample dictionary of Chinese sentences and their candidate translations
translations_dict = {
    "我喜欢学习新语言。": "I like study new language.",
    "今天天气很好。": "⁠Today weather very good.",
    "她正在看书。": "She is read book.",
    "这杯咖啡味道很好。": "This cup coffee taste very good.",
    "他每天早上跑步。": "He every morning run.",
    "请关灯。": "Please close light.",
    "我稍后给你打电话。": "⁠I later give you call.",
    "时间过得真快。": "⁠Time pass really fast.",
    "最近的地铁站在哪里？": "Nearest subway station where?",
    "我们出去吃晚饭吧。": "We go out eat dinner."
}

# Streamlit app
st.title("Chinese Translation Refinement App")

# Sidebar for selecting Chinese text
st.sidebar.header("Select Chinese Sentence")
selected_chinese = st.sidebar.selectbox(
    "Choose a sentence:",
    options=list(translations_dict.keys())
)

# Display selected Chinese text
st.header("Chinese Text")
st.text(selected_chinese)

# Display candidate translation
st.header("Candidate Translation")
candidate_translation = translations_dict[selected_chinese]
st.text(candidate_translation)

# Add system message instructions to OpenAI
st.sidebar.markdown("---")
st.sidebar.subheader("OpenAI Response Format")
with st.sidebar.expander("JSON Structure"):
    st.code('''{
  "refined_translation": "corrected or original translation",
  "score": 0-10,
  "comments": "explanation of the translation quality"
}''', language="json")

# Button to get OpenAI refinement
if st.button("Get Translation Refinement"):
    with st.spinner("Getting refinement from OpenAI..."):
        refinement_json = get_openai_refinement(selected_chinese, candidate_translation)
        
    st.header("OpenAI Refinement")
    
    # Display the results in a structured way
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Refined Translation")
        st.info(refinement_json.get("refined_translation", "No translation provided"))
        
        st.subheader("Score")
        st.metric("Translation Score", refinement_json.get("score", "N/A"))
    
    with col2:
        st.subheader("Comments")
        st.text_area("Feedback", refinement_json.get("comments", "No comments provided"), height=200)
    
    # Display the raw JSON
    with st.expander("Show Raw JSON"):
        st.json(refinement_json)

# Add information about API key setup
st.sidebar.markdown("---")
st.sidebar.subheader("Setup")
st.sidebar.info(
    "This app requires an OpenAI API key. "
    "Create a .env file with OPENAI_API_KEY=your_key_here "
    "or set it as an environment variable."
)

# Add instructions
st.sidebar.markdown("---")
st.sidebar.subheader("Instructions")
st.sidebar.info(
    "1. Select a Chinese sentence from the dropdown\n"
    "2. View the candidate translation\n"
    "3. Click 'Get Translation Refinement' to get OpenAI's analysis\n"
    "4. The result will show the refined translation, score, and comments"
)