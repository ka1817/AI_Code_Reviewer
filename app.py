import os
import streamlit as st
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
# Set up API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(api_key=GROQ_API_KEY, model_name="gemma2-9b-it")


# Define example responses for few-shot prompting
examples = [
    {
        "input": "def add(a, b):\nreturn a + b",
        "output": "Your function 'add' is missing proper indentation. Here's a corrected version:\n\ndef add(a, b):\n    return a + b\n"
    },
    {
        "input": "def divide(a, b):\n    return a / b",
        "output": "Potential bug detected: Division by zero error. You should handle this case:\n\ndef divide(a, b):\n    if b == 0:\n        return 'Error: Division by zero'\n    return a / b\n"
    }
]

# Define example template
example_template = PromptTemplate(
    input_variables=["input", "output"],
    template="Code: \n{input}\n\nFeedback:\n{output}\n"
)
prefix="""You are a highly skilled Python code reviewer. 
Your task is to analyze the given Python code, identify potential bugs, suggest improvements, and provide a corrected version of the code if necessary. Ensure that your feedback is clear, precise, and actionable.
First you have to specify where and what the error is.
Next give the correct code
If the code is out of context replay "Out of Context"

"""

# Create a few-shot prompt template
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    prefix=prefix,
    suffix="Code:\n{input}\n\nFeedback:",
    input_variables=["input"]
)

# Create the LLMChain
llm_chain = LLMChain(llm=llm, prompt=few_shot_prompt)

# Streamlit App
st.title("🤖 AI Code Reviewer 📝")
st.markdown("### Get instant feedback on your Python code! 🚀")

code_snippet = st.text_area("✍️ Enter Python code below:", height=200)

if st.button("🔍 Review Code"):
    if code_snippet.strip():
        response = llm_chain.run(input=code_snippet)
        st.subheader("🧐 Review Feedback:")
        st.code(response, language="python")
    else:
        st.warning("⚠️ Please enter some Python code to review!")
