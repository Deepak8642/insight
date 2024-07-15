import streamlit as st
import pandas as pd
import openai
import os
import random
import time

LOG = "questions.log"

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = ""#add your key

@st.cache_data()
def load_data(file):
    """
    Load the data.
    """
    # Load data from the uploaded CSV file
    df = pd.read_csv(file)
    return pre_process(df)

def add_to_log(message):
    """
    Log the message.
    """
    with open(LOG, "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S") + " ")
        f.write(message + "\n")
        f.flush()

def pre_process(df):
    """
    Pre-process the data.
    """
    # Drop columns that start with "Unnamed"
    for col in df.columns:
        if col.startswith("Unnamed"):
            df = df.drop(col, axis=1)
    return df

def ask_question(question, system="You are a data scientist."):
    """
    Ask a question and return the answer.
    """ 
    openai.api_key = os.getenv("OPENAI_API_KEY")
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0
    )
    answer = response.choices[0].message['content']
    return answer

def ask_question_with_retry(question, system="You are a data scientist.", retries=5, base_delay=3):
    """
    Wrapper around ask_question that retries if it fails.
    Proactively wait for the rate limit to reset. Eg for a rate limit of 20 calls per minutes, wait for at least 2 seconds
    Compute delay using an exponential backoff, so we don't exceed the rate limit.
    """
    delay = base_delay * (1 + random.random())
    for i in range(retries):
        try:
            time.sleep(delay)
            return ask_question(question, system=system)
        except Exception as e:
            add_to_log(f"Error: {e}")
            delay *= 2
    return "Request timed out. Please wait and resubmit your question."

def describe_dataframe(df):
    """
    Describe the dataframe.
    """
    description = []
    # List the columns of the dataframe
    description.append(f"The dataframe df has the following columns: {', '.join(df.columns)}.")
    try:
        # For each column with a categorical variable, list the unique values
        if cols := check_categorical_variables(df):
            return f"ERROR: All values in a categorical variable must be strings: {', '.join(cols)}." 
        for column in df.columns:
            if df[column].dtype == "object" and len(df[column].unique()) < 10:
                description.append(f"Column {column} has the following levels: {', '.join(df[column].dropna().unique())}.")
            elif df[column].dtype == "int64" or df[column].dtype == "float64":
                description.append(f"Column {column} is a numerical variable.")
    except Exception as e:
        add_to_log(f"Error: {e}")
        return "Unexpected error with the dataset."
    return "\n".join(description)

def check_categorical_variables(df):
    """
    Check that all values of categorical variables are strings.
    """
    # Return [] if all values of categorical variables are strings
    # Return columns if not all values of categorical variables are strings
    return [column for column in df.columns if df[column].dtype == "object" 
        and not all(isinstance(x, str) for x in df[column].dropna().unique())]

def list_non_categorical_values(df, column):
    """
    List the non-categorical values in a column.
    """
    return [x for x in df[column].unique() if not isinstance(x, str)]

def generate_placeholder_question(df):
    return "Show the relationship between Valuation ($B) and Founded Year."

def test_ask_question():
    system = "You are a data scientist. Answer the question based on the dataset."
    question = "What is the average valuation of the companies?"
    answer = ask_question_with_retry(question, system=system)
    print(answer)

def test_describe_dataframe():
    import pandas as pd
    df = pd.DataFrame({
        "a": ["male", "female", "male"], 
        "b": [4, 5, 6],
        "c": ["yes", "no", "yes"]})
    description = describe_dataframe(df)
    print(description)

def test_answer_with_chat():
    import pandas as pd
    df = pd.DataFrame({
        "Company": ["Company A", "Company B", "Company C", "Company D"],
        "Valuation ($B)": [10.5, 15.3, 7.8, 22.1],
        "Date Joined": ["2020-01-01", "2019-05-15", "2021-06-20", "2018-11-30"],
        "Country": ["USA", "Canada", "Germany", "UK"],
        "City": ["San Francisco", "Toronto", "Berlin", "London"],
        "Industry": ["Tech", "Finance", "Health", "Retail"],
        "Select Investors": ["Investor 1", "Investor 2", "Investor 3", "Investor 4"],
        "Founded Year": [2015, 2012, 2018, 2010],
        "Total Raised": [500, 700, 300, 900],
        "Financial Stage": ["Late", "Late", "Early", "Late"],
        "Investors Count": [5, 6, 4, 7]
    })
    question = "What is the average valuation of the companies?"
    description = describe_dataframe(df)
    answer = ask_question_with_retry(f"Context: {description} Question: {question}")
    print(answer)

st.title("Chat with your data")

uploaded_file = st.sidebar.file_uploader("Upload a dataset", type="csv")

if uploaded_file:
    df = load_data(uploaded_file)

    with st.chat_message("assistant"):
        st.markdown("Here is a table with the data:")
        st.dataframe(df, height=200)

    question = st.chat_input(placeholder=generate_placeholder_question(df))

    if question:
        with st.chat_message("user"):
            st.markdown(question)

        add_to_log(f"Question: {question}")
            
        description = describe_dataframe(df)
        if "ERROR" in description:
            with st.chat_message("assistant"):
                st.markdown(description)
        else:
            with st.spinner("Thinking..."):
                answer = ask_question_with_retry(f"Context: {description} Question: {question}")
            with st.chat_message("assistant"):
                if answer:
                    st.markdown(answer)
                else:
                    add_to_log("Error: Request timed out.")
                    st.markdown("Request timed out. Please wait and resubmit your question.")
else:
    with st.chat_message("assistant"):
        st.markdown("Upload a dataset to get started.")

# if __name__ == "__main__":    
#     # test_ask_question()
#     # test_describe_dataframe()
#     # test_answer_with_chat()
