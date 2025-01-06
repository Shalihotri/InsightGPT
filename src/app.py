import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEndpoint
import re

# Configure Streamlit page
st.set_page_config(
    page_title="Data Analysis Assistant",
    page_icon="📊",
    layout="wide"
)

# Initialize session state variables
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'api_key_set' not in st.session_state:
    st.session_state.api_key_set = False

# Sidebar content
with st.sidebar:
    st.title("📊 Data Analysis Assistant")
    st.subheader("Powered by LLM")
    st.write("""
    This project allows you to analyze data using natural language queries.
    Simply upload your data, ask questions, and get instant visualizations
    and insights powered by advanced language models.
    
    Features:
    - Natural language data analysis
    - Automatic visualization generation
    - Interactive data exploration
    - Secure API key management
    """)

def initialize_llm(api_key):
    """Initialize the LLM model"""
    repo_id = 'mistralai/Mistral-7B-Instruct-v0.3'
    # repo_id = 'deepset/roberta-base-squad2'
    return HuggingFaceEndpoint(
        repo_id=repo_id,
        max_length=128,
        temperature=0.7,
        token=api_key
    )

def create_visualization(df, plot_type, columns, title=""):
    """Create visualization based on plot type and columns"""
    try:
        if plot_type in ["histogram", "box"]:
            if len(columns) >= 1:
                if plot_type == "histogram":
                    fig = px.histogram(df, x=columns[0], title=title)
                else:
                    fig = px.box(df, y=columns[0], title=title)
                return fig
        
        elif plot_type == "bar" and len(columns) == 1:
            value_counts = df[columns[0]].value_counts().reset_index()
            fig = px.bar(value_counts, x='index', y=columns[0], title=title)
            return fig
            
        elif len(columns) >= 2:
            if plot_type == "scatter":
                fig = px.scatter(df, x=columns[0], y=columns[1], title=title)
            elif plot_type == "line":
                fig = px.line(df, x=columns[0], y=columns[1], title=title)
            elif plot_type == "bar":
                fig = px.bar(df, x=columns[0], y=columns[1], title=title)
            elif plot_type == "pie":
                values = df[columns[1]].value_counts() if len(columns) == 1 else df[columns[1]]
                names = df[columns[0]]
                fig = px.pie(df, values=values, names=names, title=title)
            else:
                return None
            return fig
            
        return None
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None

def clean_column_name(column):
    """Clean and validate column name"""
    return column.strip().strip('"').strip("'").strip()

def parse_visualization_request(response_text, available_columns):
    """Parse LLM response to extract visualization requirements"""
    viz_pattern = r"VISUALIZATION:(.*?)(?:ANALYSIS:|$)"
    analysis_pattern = r"ANALYSIS:(.*?)$"
    
    viz_match = re.search(viz_pattern, response_text, re.DOTALL)
    analysis_match = re.search(analysis_pattern, response_text, re.DOTALL)
    
    if viz_match:
        viz_text = viz_match.group(1).strip()
        plot_types = ["scatter", "line", "bar", "histogram", "box", "pie"]
        plot_type = None
        for pt in plot_types:
            if pt in viz_text.lower():
                plot_type = pt
                break
        
        columns = []
        quoted_columns = re.findall(r'["\']([^"\']+)["\']', viz_text)
        if quoted_columns:
            columns = [clean_column_name(col) for col in quoted_columns]
        else:
            for col in available_columns:
                if col in viz_text:
                    columns.append(col)
        
        columns = [col for col in columns if col in available_columns]
        
        return {
            'plot_type': plot_type,
            'columns': columns,
            'analysis': analysis_match.group(1).strip() if analysis_match else ""
        }
    return None

def analyze_data_with_prompt(df, llm, user_question):
    """Analyze data and suggest visualization based on user's prompt"""
    data_context = f"""
    Here is information about the dataset:
    - Columns: {', '.join(df.columns)}
    - Number of rows: {len(df)}
    - Data types: {df.dtypes.to_dict()}
    - Sample of data (first 5 rows): 
    {df.head().to_string()}
    
    Based on this data, please answer: {user_question}
    
    Important: When suggesting visualizations, use the exact column names from the dataset and specify them in quotes.
    
    Provide your response in the following format:
    VISUALIZATION: Specify the type of plot (scatter/line/bar/histogram/box/pie) and the exact column names in quotes
    ANALYSIS: Provide detailed analysis of the visualization and the data
    """
    
    template = """
    You are a data visualization and analysis expert. Using the provided dataset information, please suggest an appropriate visualization and provide analysis.
    
    Dataset Information:
    {data_context}
    
    Provide a visualization suggestion and analysis that best answers the question.
    Remember to format your response with VISUALIZATION: and ANALYSIS: sections.
    Use exact column names in quotes for the visualization.
    """
    
    prompt = PromptTemplate(
        template = template,
        input_variables = ["data_context"]
    )
    
    llm_chain = prompt | llm
    return llm_chain.invoke({"data_context": data_context})

# Login section
def login_section():
    st.header("Login")
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if email == "harshv.shalihotri@blend360.com" and password == "1234":
                st.session_state.logged_in = True
                st.success("Successfully logged in!")
                st.rerun()
            else:
                st.error("Invalid credentials!")

# API Key section
def api_key_section():
    st.header("Set API Key")
    with st.form("api_key_form"):
        api_key = st.text_input("Enter your HuggingFace API Key", type="password")
        submit = st.form_submit_button("Set API Key")
        
        if submit and api_key:
            st.session_state.api_key = api_key
            st.session_state.api_key_set = True
            st.success("API Key set successfully!")
            st.rerun()

# Main analysis section
def analysis_section():
    st.header("Data Analysis")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        
        # Show data preview
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Initialize LLM
        llm = initialize_llm(st.session_state.api_key)
        
        # Question input
        user_question = st.text_input("What would you like to analyze?")
        
        if user_question:
            with st.spinner("Analyzing data..."):
                # Get analysis and visualization suggestion
                result = analyze_data_with_prompt(df, llm, user_question)
                
                # Parse the response
                viz_details = parse_visualization_request(str(result), df.columns)
                
                if viz_details and viz_details['plot_type'] and viz_details['columns']:
                    # Create visualization
                    fig = create_visualization(
                        df,
                        viz_details['plot_type'],
                        viz_details['columns'],
                        title=user_question
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show analysis
                    if viz_details['analysis']:
                        st.subheader("Analysis")
                        st.write(viz_details['analysis'])
                else:
                    st.write("Couldn't create visualization. Here's the analysis:")
                    st.write(result)

# Main app flow
def main():
    if not st.session_state.logged_in:
        login_section()
    elif not st.session_state.api_key_set:
        api_key_section()
    else:
        analysis_section()

if __name__ == "__main__":
    main()