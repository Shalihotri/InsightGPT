from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEndpoint
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re

def initialize_llm():
    """Initialize the LLM model"""
    load_dotenv()
    sec_key = os.getenv('HF_TOKEN')
    repo_id = 'mistralai/Mistral-7B-Instruct-v0.3'
    return HuggingFaceEndpoint(
        repo_id=repo_id,
        max_length=128,
        temperature=0.7,
        token=sec_key
    )

def create_visualization(df, plot_type, columns, title=""):
    """Create visualization based on plot type and columns"""
    try:
        # For single column visualizations
        if plot_type in ["histogram", "box"]:
            if len(columns) >= 1:
                if plot_type == "histogram":
                    fig = px.histogram(df, x=columns[0], title=title)
                else:  # box plot
                    fig = px.box(df, y=columns[0], title=title)
                return fig
        
        # For bar charts with counts
        elif plot_type == "bar" and len(columns) == 1:
            value_counts = df[columns[0]].value_counts().reset_index()
            fig = px.bar(value_counts, x='index', y=columns[0], title=title)
            return fig
            
        # For two-column visualizations
        elif len(columns) >= 2:
            if plot_type == "scatter":
                fig = px.scatter(df, x=columns[0], y=columns[1], title=title)
            elif plot_type == "line":
                fig = px.line(df, x=columns[0], y=columns[1], title=title)
            elif plot_type == "bar":
                fig = px.bar(df, x=columns[0], y=columns[1], title=title)
            elif plot_type == "pie":
                # For pie charts, we'll use value counts if not specified otherwise
                values = df[columns[1]].value_counts() if len(columns) == 1 else df[columns[1]]
                names = df[columns[0]]
                fig = px.pie(df, values=values, names=names, title=title)
            else:
                return None
            return fig
            
        return None
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        return None

def clean_column_name(column):
    """Clean and validate column name"""
    # Remove any quotes or extra spaces
    clean_name = column.strip().strip('"').strip("'").strip()
    return clean_name

def parse_visualization_request(response_text, available_columns):
    """Parse LLM response to extract visualization requirements"""
    viz_pattern = r"VISUALIZATION:(.*?)(?:ANALYSIS:|$)"
    analysis_pattern = r"ANALYSIS:(.*?)$"
    
    viz_match = re.search(viz_pattern, response_text, re.DOTALL)
    analysis_match = re.search(analysis_pattern, response_text, re.DOTALL)
    
    if viz_match:
        viz_text = viz_match.group(1).strip()
        
        # Extract plot type
        plot_types = ["scatter", "line", "bar", "histogram", "box", "pie"]
        plot_type = None
        for pt in plot_types:
            if pt in viz_text.lower():
                plot_type = pt
                break
        
        # Extract column names more robustly
        columns = []
        # Try to find quoted column names first
        quoted_columns = re.findall(r'["\']([^"\']+)["\']', viz_text)
        if quoted_columns:
            columns = [clean_column_name(col) for col in quoted_columns]
        else:
            # If no quoted columns, try to match with available columns
            for col in available_columns:
                if col in viz_text:
                    columns.append(col)
        
        # Validate columns exist in the dataset
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
        template=template,
        input_variables=["data_context"]
    )
    
    llm_chain = prompt | llm
    return llm_chain.invoke({"data_context": data_context})

def main():
    # Initialize LLM
    llm = initialize_llm()
    
    # Load the CSV file
    df = pd.read_csv("Data.csv")
    
    while True:
        print("\nWhat would you like to analyze? (type 'exit' to quit)")
        user_question = input("> ")
        
        if user_question.lower() == 'exit':
            break
            
        try:
            # Get analysis and visualization suggestion from LLM
            result = analyze_data_with_prompt(df, llm, user_question)
            
            # Parse the response
            viz_details = parse_visualization_request(str(result), df.columns)
            
            if viz_details and viz_details['plot_type'] and viz_details['columns']:
                print(f"\nCreating {viz_details['plot_type']} plot using columns: {viz_details['columns']}")
                
                # Create and show visualization
                fig = create_visualization(
                    df, 
                    viz_details['plot_type'],
                    viz_details['columns'],
                    title=user_question
                )
                if fig:
                    fig.show()
                else:
                    print("Couldn't create visualization with the specified parameters")
                
                # Show analysis
                if viz_details['analysis']:
                    print("\nAnalysis:")
                    print(viz_details['analysis'])
            else:
                print("\nCouldn't determine visualization type or columns. Here's the raw analysis:")
                print(result)
                
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()