import streamlit as st
import openai
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_openai_refinement(chinese_text, ML_candidate_translation, Naive_candidate_translation):
    """Get translation refinement from OpenAI API using JSON response format"""
    try:
        response = openai.chat.completions.create(
            model="gpt-4-turbo",  # You can change this to the model you want to use
            messages=[
                {"role": "system", "content": "You are a helpful translation assistant. Return your response as JSON."},
                {"role": "user", "content": f"""
                I have used 2 models to translate a chinese sentence to english. Use their translations as guardrails, to keep you on track while translating.
                translate - {chinese_text} 
                Use this as a candidate translation from the machine learning model - {ML_candidate_translation} 
                Use this as a candidate translation from the naive model - {Naive_candidate_translation}
                If you feel the candidate translation is wrong then correct it, and also print out the new translation phrase. 
                Explain why the candidate translation was wrong by giving it a score from 0-10. 
                Do not use emojis.
                
                Respond with a JSON object using this exact structure:
                {{
                    "refined_translation": "your corrected translation or the original if correct",
                    "ML_score": "number between 0-10 judging the machine learning translation",
                    "ML_comments": "your explanation of why the machine learning translation was wrong or right"
                    "Naive_score": "number between 0-10 judging the naive translation"
                    "Naive_comments": "your explanation of why the naive translation was wrong or right"
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
            "refined_translation": ML_candidate_translation,
            "score": "N/A",
            "comments": f"Error with OpenAI API: {str(e)}"
        }

# Sample dictionary of Chinese sentences and their candidate translations
ML_translations_dict = {
    "我喜欢学习新语言。": "I words to be fond of new",
    "今天天气很好。": "now day day very to be fond of",
    "她正在看书。": "she exist see straight to to",
    "这杯咖啡味道很好。": "coffee used taste loanwords value road cup very to be fond of",
    "他每天早上跑步。": "he each day early up to run a step",
    "请关灯。": "'",
    "我稍后给你打电话。": "i empress you to beat",
    "时间过得真快。": "have to to really rapid",
    "最近的地铁站在哪里？": "most near bull's-eye earth station to exist which?",
    "我们出去吃晚饭吧。": "I go go eat evening to out to to"
}

Naive_translations_dict = {
    "我喜欢学习新语言。": "I think that the new learning and learning, and learning, and I have a new spirit of the new spirit of the spirit of the spirit of learning.",
    "今天天气很好。": "The weather, the weather, the weather, the weather, the weather, the good weather, good weather, good weather, good weather, good weather, good, good, good, good, good, good, good, good, good weather, good, good, good, good, good, good weather, good weather, good weather, good weather, good, good, good, good, good, good, good, good, good, good, good, good, good, good, good, good, good, good, good, good, good, good, good, good, good weather, good weather, good weather, good, good, good, good, good, good, good, good, good, good, good, good, good, good, good, good, good, good, good, good, good, good, good, good, good, good, good, good, good, good, good, good, good, good, good, good, good weather, good weather, good weather, good weather, good weather, good weather, good weather, good weather, good weather, good, good, good, good, good, good weather, good weather, good weather, good weather, good weather, good, good",
    "她正在看书。": "She saw her, she saw her, she saw her, she saw her.",
    "这杯咖啡味道很好。": "It's a good taste, and you'll just a good taste, and you are good, and you are good, and you are good, you are good, you are good, good.",
    "他每天早上跑步。": "When he saw the day, he saw the day, he saw the day, and the day he went to the day, and the day, he went to the day, and the day, and the day.",
    "请关灯。": "Please Please Please, the shoes, and the shoes, and the shoes, please, please, and the shoes, please, and the shoes, please.",
    "我稍后给你打电话。": " I'll just you with you, you can't even you like you like to buy a TV, you can't even more like you like to TV, you can't even you like to you like.",
    "时间过得真快。": "The old one.",
    "最近的地铁站在哪里？": "Is the iron iron iron iron iron iron iron iron iron iron - copper iron ore, the iron the iron iron iron iron iron iron iron ore?",
    "我们出去吃晚饭吧。": "We'll eat, you'll eat, you eat, you eat, you eat, you'll eat, you eat the eat, you eat, you eat, you eat the eat, you eat, you'll eat, and you '.'and you'and you'and you'and you'and you'and you'and you'and you'and you eat, you'and you'll eat, you'll eat the night, you'and you eat, you'and you'll eat the eat the eat the eat, you'and you'll eat the eat the eat, you eat, you'll eat, you'and you'and you'and you'and you eat the eat, you'll eat, you'and you'and you'and you'll eat, you'and you'll eat, you'and you'and you'll eat, you'll eat, you'and you'll eat, you'll eat, you'll eat, you'and you'll eat, you'll eat, you'll eat, you'll eat, you'll eat, you'll eat, you'and you'll eat, you"
}

# Streamlit app
st.title("Chinese Translation Refinement App")

# Sidebar for selecting Chinese text
st.sidebar.header("Select Chinese Sentence")
selected_chinese = st.sidebar.selectbox(
    "Choose a sentence:",
    options=list(ML_translations_dict.keys())
)

# Display selected Chinese text
st.header("Chinese Text")
st.text(selected_chinese)

# Display candidate translation
st.header("Candidate Translation")
ML_candidate_translation = ML_translations_dict[selected_chinese]
Naive_candidate_translation = Naive_translations_dict[selected_chinese]
st.text("Naive Transaltion")
st.text(Naive_candidate_translation)
st.text("Machine Learning Translation")
st.text(ML_candidate_translation)

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
        refinement_json = get_openai_refinement(selected_chinese, ML_candidate_translation, Naive_candidate_translation)
        
    st.header("OpenAI Refinement")

    st.subheader("Refined Translation")
    st.info(refinement_json.get("refined_translation", "No translation provided"))
    
    # Display the results in a structured way
    col1, col2 = st.columns(2)
    
    
    st.subheader("Naive Model Score")
    st.metric("Naive Model Translation Score", refinement_json.get("Naive_score", "N/A"))
    
    
    st.subheader("Naive Model Comments")
    st.text_area("Naive Model Feedback", refinement_json.get("Naive_comments", "No comments provided"), height=200)

    
    st.subheader("Machine Learning Model Score")
    st.metric("Machine Learning Model Translation Score", refinement_json.get("ML_score", "N/A"))
    
    
    st.subheader("Machine Learning Model Comments")
    st.text_area("Machine Learning Model Feedback", refinement_json.get("ML_comments", "No comments provided"), height=200)
    
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