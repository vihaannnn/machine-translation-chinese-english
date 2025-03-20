import streamlit as st
import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_openai_refinement(chinese_text, candidate_translation):
    """Get translation refinement from OpenAI API"""
    prompt = f"""
    translate - {chinese_text} 
    Use this as a candidate translation - {candidate_translation} 
    Explain why the candidiate translation was wrong by giving it a score form 0-10. 
    Do not use emojis.
    Provide a refined version of the response as well.
    """
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4",  # You can change this to the model you want to use
            messages=[
                {"role": "system", "content": "You are a helpful translation assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error with OpenAI API: {str(e)}"

# Sample dictionary of Chinese sentences and their candidate translations
translations_dict = {
    "美国缓慢地开始倾听，但并非没有艰难曲折。": "hi hello",
    "他们终于开始认真对待我们的担忧。": "They finally started to take our concerns seriously.",
    "这个项目的完成需要更多的时间和资源。": "This project requires more time and resource to finish.",
    "我们必须找到一个平衡点来解决这个问题。": "We must locate a balance point to solve this issue.",
    "新政策将从下个月开始实施。": "The new policy will begin next month."
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

# Button to get OpenAI refinement
if st.button("Get Translation Refinement"):
    with st.spinner("Getting refinement from OpenAI..."):
        refinement = get_openai_refinement(selected_chinese, candidate_translation)
        
    st.header("OpenAI Refinement")
    st.text_area("Refinement Result", refinement, height=200)

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
    "4. The result will show a corrected translation if needed, with a score and explanation"
)