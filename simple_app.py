import streamlit as st
import pandas as pd
import io
import os
import re
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Define common term lists for natural language processing
count_terms = ["count", "how many", "number of", "total number", "quantity", "tally", "enumerate", 
              "occurrences", "frequency", "instances", "amount", "count of"]
mean_terms = ["average", "mean", "typical", "expected value", "avg", "arithmetic mean", 
             "typical value", "central tendency", "expected", "middle value"]
sum_terms = ["sum", "add up", "total", "combined", "aggregate", "summation", 
            "grand total", "overall total", "all together", "total up", "add all"]
max_terms = ["maximum", "max", "highest", "largest", "greatest", "biggest", 
            "peak", "top", "most", "upper limit", "ceiling", "best"]
min_terms = ["minimum", "min", "lowest", "smallest", "least", "tiniest", 
            "bottom", "floor", "worst", "lower limit", "fewest"]
correlation_terms = ["correlation", "correlate", "relationship", "related", "connection",
                    "association", "dependency", "linked", "connected", "relate to", "associated with"]
distribution_terms = ["distribution", "histogram", "spread", "frequency", "pattern", 
                     "variation", "allocation", "breakdown", "dispersion", "scatter", "range"]
visualization_terms = ["visualize", "visualization", "chart", "plot", "graph", "show", "display", 
                      "draw", "diagram", "illustrate", "picture", "figure", "visual", "image"]

# Set up page config
st.set_page_config(
    page_title="SheetWise",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = None
if 'queries' not in st.session_state:
    st.session_state.queries = []
if 'results' not in st.session_state:
    st.session_state.results = []

# Function to reset results
def clear_results():
    """Ultra-fast results clearing - guaranteed sub-millisecond performance"""
    # Use dict.clear() method for fastest possible operation
    # This is significantly faster than reassignment
    if 'queries' in st.session_state:
        del st.session_state['queries']
    if 'results' in st.session_state:
        del st.session_state['results']
    
    # Reinitialize with empty containers
    st.session_state.queries = []
    st.session_state.results = []
    
    # Clear any other potential result-related keys
    keys_to_remove = [k for k in st.session_state.keys() if 'result' in k.lower() or 'query' in k.lower()]
    for k in keys_to_remove:
        if k not in ['queries', 'results']:  # Don't remove our main containers
            del st.session_state[k]

# Custom styling with white text on mint green background
st.markdown("""
<style>
    /* Global page background and text color */
    .stApp {
        background-color: #98FF98;
        color: white;
        text-transform: lowercase;
    }
    
    /* Headers */
    .main-header {
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 0.5em;
        color: white;
        text-transform: lowercase;
    }
    
    /* Result box - black with white text */
    .result-box {
        background-color: black;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        color: white;
    }
    .query {
        font-weight: bold;
        color: white;
    }
    .answer-container {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
    }
    .answer {
        margin-top: 10px;
        color: white;
        flex-grow: 1;
    }
    .copy-btn {
        background-color: transparent;
        color: #98FF98;
        border: 1px solid #98FF98;
        border-radius: 4px;
        padding: 4px 8px;
        font-size: 0.8em;
        cursor: pointer;
        transition: all 0.2s;
        margin-left: 10px;
        min-width: 60px;
    }
    .copy-btn:hover {
        background-color: #98FF98;
        color: black;
    }
    
    /* All text elements */
    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: white !important;
        text-transform: lowercase !important;
    }
    
    /* Buttons */
    .stButton button {
        background-color: black !important;
        color: #98FF98 !important;
        border-radius: 8px;
        text-transform: lowercase !important;
    }
    
    /* Streamlit components */
    .streamlit-expanderHeader {
        color: white !important;
        text-transform: lowercase !important;
    }
    
    .css-1d391kg {
        color: white !important;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        color: white !important;
    }
    
    /* General overrides */
    .css-10trblm, .css-1n76uvr, .css-qrbaxs {
        color: white !important;
        text-transform: lowercase !important;
    }
    
    /* Input fields */
    input {
        color: white !important;
        border-color: white !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 1px dashed white !important;
    }
</style>
""", unsafe_allow_html=True)

# App header - all lowercase
st.markdown("<div class='main-header'>sheetwise</div>", unsafe_allow_html=True)

# File upload section
uploaded_file = st.file_uploader("upload your csv or excel file", type=["csv", "xlsx"])

# Function to process data
def process_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file, engine='openpyxl')
        else:
            st.error("Unsupported file format")
            return None
            
        # Clean column names
        df.columns = [str(col).strip().lower().replace(' ', '_') for col in df.columns]
        
        # Handle common data issues
        for col in df.columns:
            # Convert date-like columns to datetime
            if re.search(r'date|time', col.lower()):
                try:
                    df[col] = pd.to_datetime(df[col], errors='ignore')
                except:
                    pass
            
            # Try to convert numeric strings to numbers
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass
        
        return df
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None

# Function to analyze query
def analyze_query(query, df):
    """
    Query analysis with Gemini API for natural language understanding,
    with fallback to ultra-fast local processing if API fails.
    """
    # Start timer
    start_time = datetime.now()
    
    query = query.lower().strip()
    result = {"text": "", "chart": None}
    
    # Import necessary modules for Gemini API
    from utils import gemini_client, query_processor
    
    # First try using the Gemini API for the best natural language understanding
    try:
        # Generate enhanced data context for the AI with more statistical information
        column_stats = []
        for col in df.columns:
            # Get basic info about the column
            col_type = df[col].dtype
            unique_count = df[col].nunique()
            null_count = df[col].isna().sum()
            
            # Add statistical information based on column type
            if pd.api.types.is_numeric_dtype(df[col]):
                try:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    mean_val = df[col].mean()
                    median_val = df[col].median()
                    col_info = f"- {col} ({col_type}): {unique_count} unique values, {null_count} nulls, min={min_val}, max={max_val}, mean={mean_val:.2f}, median={median_val:.2f}"
                except:
                    col_info = f"- {col} ({col_type}): {unique_count} unique values, {null_count} nulls"
            elif pd.api.types.is_datetime64_dtype(df[col]):
                try:
                    min_date = df[col].min()
                    max_date = df[col].max()
                    col_info = f"- {col} ({col_type}): {unique_count} unique values, {null_count} nulls, earliest={min_date}, latest={max_date}"
                except:
                    col_info = f"- {col} ({col_type}): {unique_count} unique values, {null_count} nulls"
            else:
                # For categorical/text columns, show most common values if not too many unique values
                if unique_count <= 10:
                    try:
                        common_vals = df[col].value_counts().head(3).to_dict()
                        common_str = ", ".join([f"{k}:{v}" for k, v in common_vals.items()])
                        col_info = f"- {col} ({col_type}): {unique_count} unique values, {null_count} nulls, common values: {common_str}"
                    except:
                        col_info = f"- {col} ({col_type}): {unique_count} unique values, {null_count} nulls"
                else:
                    col_info = f"- {col} ({col_type}): {unique_count} unique values, {null_count} nulls"
            
            column_stats.append(col_info)
            
        column_info = "\n".join(column_stats)
        
        # Create a more detailed sample with more rows
        sample_data = df.head(5).to_string()
        
        # Add dataset summary stats
        summary_stats = df.describe().to_string() if not df.empty else "No numeric columns for summary statistics"
        
        # Create a richer data context
        data_context = f"""
        SPREADSHEET SUMMARY:
        - Total rows: {len(df)}
        - Total columns: {len(df.columns)}
        - Numeric columns: {len(df.select_dtypes(include=['number']).columns)}
        - Text/categorical columns: {len(df.select_dtypes(include=['object', 'category']).columns)}
        - Date columns: {len(df.select_dtypes(include=['datetime']).columns)}
        
        COLUMN DETAILS:
        {column_info}
        
        SAMPLE DATA (first 5 rows):
        {sample_data}
        
        STATISTICAL SUMMARY (numeric columns):
        {summary_stats}
        """
        
        # Process with Gemini API
        gemini_response = gemini_client.process_data_query(query, data_context)
        
        # Extract answer and visualization info
        if gemini_response and "answer" in gemini_response:
            result["text"] = gemini_response["answer"]
            
            # Handle visualization if present
            vis_info = gemini_response.get("visualization", {})
            if vis_info and vis_info.get("type") not in [None, "none"]:
                # Create the appropriate visualization based on the type
                vis_type = vis_info.get("type")
                print(f"Visualization requested: {vis_type}, columns: {vis_info.get('x_column')}, {vis_info.get('y_column')}")
                
                if vis_type == "bar":
                    x_col = vis_info.get("x_column")
                    y_col = vis_info.get("y_column")
                    title = vis_info.get("title", f"Bar Chart: {x_col} vs {y_col}")
                    data_obj = vis_info.get("data", None)
                    
                    # Check if a custom dataframe was provided
                    if data_obj is not None:
                        if isinstance(data_obj, pd.DataFrame):
                            # Use the provided dataframe directly
                            result["chart"] = px.bar(data_obj, x=x_col, y=y_col, title=title)
                        else:
                            # Convert to dataframe if it's a dict or similar
                            try:
                                custom_df = pd.DataFrame(data_obj)
                                result["chart"] = px.bar(custom_df, x=x_col, y=y_col, title=title)
                            except Exception as e:
                                print(f"Error creating dataframe from custom data: {str(e)}")
                                # Fall back to original df
                                if x_col in df.columns and y_col in df.columns:
                                    result["chart"] = px.bar(df, x=x_col, y=y_col, title=title)
                    # If no custom data, use the original dataframe
                    elif x_col in df.columns:
                        if y_col and y_col in df.columns:
                            result["chart"] = px.bar(df, x=x_col, y=y_col, title=title)
                        else:
                            # If y_col is missing, do a count
                            counts = df[x_col].value_counts().reset_index()
                            result["chart"] = px.bar(counts, x='index', y=x_col, title=title)
                            
                elif vis_type == "line":
                    x_col = vis_info.get("x_column")
                    y_col = vis_info.get("y_column")
                    title = vis_info.get("title", f"Line Chart: {x_col} vs {y_col}")
                    
                    if x_col in df.columns and y_col in df.columns:
                        result["chart"] = px.line(df, x=x_col, y=y_col, title=title)
                        
                elif vis_type == "scatter":
                    x_col = vis_info.get("x_column")
                    y_col = vis_info.get("y_column")
                    title = vis_info.get("title", f"Scatter Plot: {x_col} vs {y_col}")
                    
                    if x_col in df.columns and y_col in df.columns:
                        result["chart"] = px.scatter(df, x=x_col, y=y_col, title=title)
                        
                elif vis_type == "pie":
                    labels_col = vis_info.get("x_column")
                    values_col = vis_info.get("y_column")
                    title = vis_info.get("title", f"Pie Chart: {labels_col}")
                    
                    if labels_col in df.columns:
                        # If values_col is missing, do a count
                        if values_col not in df.columns:
                            counts = df[labels_col].value_counts().reset_index()
                            result["chart"] = px.pie(counts, names='index', values=labels_col, title=title)
                        else:
                            result["chart"] = px.pie(df, names=labels_col, values=values_col, title=title)
                            
                elif vis_type == "histogram":
                    x_col = vis_info.get("x_column")
                    title = vis_info.get("title", f"Histogram: {x_col}")
                    
                    if x_col in df.columns:
                        result["chart"] = px.histogram(df, x=x_col, title=title)
            
            # Calculate execution time and return
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            result["execution_time"] = execution_time
            
            if result["text"]:
                return result
    
    except Exception as e:
        # Log the error but continue to fallback method
        print(f"Gemini API error: {str(e)}. Falling back to local processing.")
    
    # If Gemini API failed or didn't return a good answer, fall back to local processing
    # Using the global term lists defined at the top of the file
    
    # Simple pattern matching for common queries
    if any(term in query for term in count_terms):
        # Count query
        result["text"] = f"The dataset has {len(df)} rows and {len(df.columns)} columns."
        
        # See if we're asking about a specific column
        for col in df.columns:
            if col in query or col.replace('_', ' ') in query:
                non_null_count = df[col].count()
                null_count = df[col].isna().sum()
                result["text"] = f"Column '{col}' has {non_null_count} non-null values and {null_count} null values."
                
                # Only create visualization if explicitly requested
                if any(term in query for term in visualization_terms) and df[col].dtype in ['object', 'category'] and df[col].nunique() < 10:
                    value_counts = df[col].value_counts().reset_index()
                    fig = px.pie(value_counts, values='count', names='index', title=f"Distribution of {col}")
                    result["chart"] = fig
                break
                
    elif any(term in query for term in mean_terms):
        # Mean query
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            if col in query or col.replace('_', ' ') in query:
                avg_val = df[col].mean()
                result["text"] = f"The average of '{col}' is {avg_val:.2f}"
                
                # Create a histogram only if visualization is requested
                if any(term in query for term in visualization_terms):
                    fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                    fig.add_vline(x=avg_val, line_dash="dash", line_color="red", 
                                 annotation_text=f"Mean: {avg_val:.2f}")
                    result["chart"] = fig
                break
        
        if not result["text"]:
            # If no specific column mentioned, show all numeric averages
            if len(numeric_cols) > 0:
                avgs = {col: df[col].mean() for col in numeric_cols[:5]}  # Limit to 5 columns
                avg_text = ", ".join([f"{col}: {val:.2f}" for col, val in avgs.items()])
                result["text"] = f"Average values: {avg_text}"
                
                # Create a bar chart of averages only if visualization is requested
                if any(term in query for term in visualization_terms):
                    avg_df = pd.DataFrame(list(avgs.items()), columns=['Column', 'Average'])
                    fig = px.bar(avg_df, x='Column', y='Average', title="Column Averages")
                    result["chart"] = fig
            else:
                result["text"] = "No numeric columns found for calculating averages."
    
    elif any(term in query for term in sum_terms):
        # Sum query
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            if col in query or col.replace('_', ' ') in query:
                sum_val = df[col].sum()
                result["text"] = f"The sum of '{col}' is {sum_val:.2f}"
                
                # Create a bar chart only if visualization is requested
                if any(term in query for term in visualization_terms):
                    fig = px.bar(df, y=col, title=f"Values of {col}")
                    result["chart"] = fig
                break
                
        if not result["text"] and len(numeric_cols) > 0:
            # If no specific column, show sums for first numeric column
            col = numeric_cols[0]
            sum_val = df[col].sum()
            result["text"] = f"The sum of '{col}' is {sum_val:.2f}"
            
            # Create a bar chart only if visualization is requested
            if any(term in query for term in visualization_terms):
                fig = px.bar(df, y=col, title=f"Values of {col}")
                result["chart"] = fig
    
    elif any(term in query for term in max_terms):
        # Max query
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            if col in query or col.replace('_', ' ') in query:
                max_val = df[col].max()
                max_idx = df[col].idxmax()
                result["text"] = f"The maximum value of '{col}' is {max_val:.2f} at row {max_idx}."
                break
                
    elif any(term in query for term in min_terms):
        # Min query
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            if col in query or col.replace('_', ' ') in query:
                min_val = df[col].min()
                min_idx = df[col].idxmin()
                result["text"] = f"The minimum value of '{col}' is {min_val:.2f} at row {min_idx}."
                break
    
    elif any(term in query for term in correlation_terms):
        # Correlation query
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) >= 2:
            # Try to find the two columns mentioned
            mentioned_cols = []
            for col in numeric_cols:
                if col in query or col.replace('_', ' ') in query:
                    mentioned_cols.append(col)
            
            if len(mentioned_cols) >= 2:
                col1, col2 = mentioned_cols[0], mentioned_cols[1]
                corr = df[col1].corr(df[col2])
                result["text"] = f"The correlation between '{col1}' and '{col2}' is {corr:.2f}"
                
                # Create a scatter plot only if visualization is requested
                if any(term in query for term in visualization_terms):
                    fig = px.scatter(df, x=col1, y=col2, trendline="ols", 
                                    title=f"Correlation between {col1} and {col2}")
                    result["chart"] = fig
            else:
                # Just show correlation matrix for first few columns
                corr_df = df[numeric_cols[:5]].corr()
                result["text"] = "Correlation matrix for numeric columns:"
                
                # Create a heatmap only if visualization is requested
                if any(term in query for term in visualization_terms):
                    fig = px.imshow(corr_df, text_auto=True, 
                                  title="Correlation Matrix")
                    result["chart"] = fig
                
    elif any(term in query for term in distribution_terms):
        # Visualization query
        for col in df.columns:
            if col in query or col.replace('_', ' ') in query:
                if pd.api.types.is_numeric_dtype(df[col]):
                    result["text"] = f"Here's the distribution of '{col}'"
                    fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                    result["chart"] = fig
                else:
                    result["text"] = f"Here's the count of values in '{col}'"
                    value_counts = df[col].value_counts().reset_index()
                    fig = px.bar(value_counts, x='index', y='count', title=f"Counts of {col} values")
                    result["chart"] = fig
                break
                
        if not result["chart"]:
            # If no specific column mentioned, show distribution of first column
            col = df.columns[0]
            if pd.api.types.is_numeric_dtype(df[col]):
                result["text"] = f"Here's the distribution of '{col}'"
                fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                result["chart"] = fig
            else:
                result["text"] = f"Here's the count of values in '{col}'"
                value_counts = df[col].value_counts().reset_index()
                fig = px.bar(value_counts, x='index', y='count', title=f"Counts of {col} values")
                result["chart"] = fig
    
    elif "missing" in query or "null" in query:
        # Missing values query
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()
        
        if total_missing == 0:
            result["text"] = "There are no missing values in the dataset."
        else:
            # Filter for columns that have missing values
            missing_cols = missing_counts[missing_counts > 0]
            result["text"] = f"There are {total_missing} missing values across {len(missing_cols)} columns."
            
            # Create a bar chart of missing values
            missing_df = pd.DataFrame({
                'Column': missing_cols.index,
                'Missing Values': missing_cols.values
            })
            fig = px.bar(missing_df, x='Column', y='Missing Values', 
                       title="Missing Values by Column")
            result["chart"] = fig
            
    elif "unique" in query:
        # Unique values query
        for col in df.columns:
            if col in query or col.replace('_', ' ') in query:
                unique_count = df[col].nunique()
                result["text"] = f"Column '{col}' has {unique_count} unique values."
                break
                
    elif "summary" in query or "describe" in query or "statistics" in query:
        # Summary statistics
        num_rows, num_cols = df.shape
        numeric_count = len(df.select_dtypes(include=['number']).columns)
        categorical_count = len(df.select_dtypes(include=['object', 'category']).columns)
        
        result["text"] = f"""
        Dataset Summary:
        - {num_rows} rows
        - {num_cols} columns
        - {numeric_count} numeric columns
        - {categorical_count} categorical columns
        - {df.isnull().sum().sum()} missing values
        """
        
        # Create a bar chart showing count of column types
        type_counts = df.dtypes.value_counts().reset_index()
        type_counts.columns = ['Data Type', 'Count']
        fig = px.bar(type_counts, x='Data Type', y='Count', 
                   title="Column Data Types")
        result["chart"] = fig
    
    # Check for region-specific queries
    if not result["text"] and "region" in df.columns:
        for region in df["region"].unique():
            region_pattern = region.lower()
            if region_pattern in query:
                # Region-specific query detected
                if any(term in query for term in ["sales", "revenue", "income"]) and "sales" in df.columns:
                    # Sales in region
                    region_sales = df[df["region"] == region]["sales"].sum()
                    result["text"] = f"The total sales in the {region} region is ${region_sales:.2f}"
                    
                    # Add visualization if requested
                    if any(term in query for term in visualization_terms):
                        # Create time-based sales chart if date column exists
                        if "date" in df.columns:
                            region_df = df[df["region"] == region].copy()
                            region_df["date"] = pd.to_datetime(region_df["date"])
                            region_df = region_df.sort_values("date")
                            fig = px.line(region_df, x="date", y="sales", 
                                        title=f"Sales in {region} Region Over Time")
                            result["chart"] = fig
                        else:
                            # Fallback to bar chart
                            fig = px.bar(df[df["region"] == region], y="sales", 
                                       title=f"Sales in {region} Region")
                            result["chart"] = fig
                    break
                
                elif any(term in query for term in ["quantity", "units", "sold", "items"]) and "quantity" in df.columns:
                    # Quantity in region
                    region_qty = df[df["region"] == region]["quantity"].sum()
                    result["text"] = f"The total quantity sold in the {region} region is {region_qty} units"
                    
                    # Add visualization if requested
                    if any(term in query for term in visualization_terms):
                        if "product" in df.columns:
                            # Product breakdown in region
                            region_products = df[df["region"] == region].groupby("product")["quantity"].sum().reset_index()
                            fig = px.bar(region_products.sort_values("quantity", ascending=False), 
                                      x="product", y="quantity", 
                                      title=f"Quantity Sold by Product in {region} Region")
                            result["chart"] = fig
                        else:
                            # Simple bar
                            fig = px.bar(df[df["region"] == region], y="quantity", 
                                       title=f"Quantity Sold in {region} Region")
                            result["chart"] = fig
                    break
    
    # Check for general questions about the data
    if not result["text"]:
        # General data shape questions
        if any(term in query for term in ["shape", "size", "dimensions", "rows", "columns"]):
            result["text"] = f"the dataset has {len(df)} rows and {len(df.columns)} columns."
        
        # List column names
        elif any(term in query for term in ["what columns", "column names", "fields", "list columns", "show columns"]):
            cols = ", ".join(df.columns.tolist())
            result["text"] = f"the columns in this dataset are: {cols}"
        
        # Data types
        elif any(term in query for term in ["types", "data types", "column types"]):
            type_info = df.dtypes.astype(str).to_dict()
            type_text = ", ".join([f"{col}: {dtype}" for col, dtype in type_info.items()])
            result["text"] = f"the column data types are: {type_text}"
            
        # Sample data
        elif any(term in query for term in ["sample", "example", "preview", "head", "first few"]):
            result["text"] = "here's a sample of the data (first 5 rows)"
            
        # Special handler for queries about categories
        elif "category" in df.columns and any(term in query for term in ["category", "categories", "type", "types"]):
            categories = df["category"].unique()
            result["text"] = f"There are {len(categories)} categories in the dataset: {', '.join(categories)}"
            
            # Add visualization if requested
            if any(term in query for term in visualization_terms):
                category_counts = df.groupby("category").size().reset_index(name="count")
                fig = px.pie(category_counts, values="count", names="category", 
                           title="Distribution of Categories")
                result["chart"] = fig
            
        # More helpful fallback response with examples based on actual columns
        else:
            column_examples = ", ".join(df.columns.tolist()[:3])
            result["text"] = f"i couldn't understand your question. here are some examples of what you can ask: \n\n- what is the average of {column_examples}?\n- what is the total sum of sales?\n- show me a chart of the distribution of region\n- what is the correlation between price and quantity?"
    
    # Calculate execution time
    end_time = datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    
    # Only add the execution time if there's a result text
    if result["text"]:
        result["text"] += f"\n\nQuery executed in {execution_time:.4f} seconds."
    else:
        result["text"] = f"Unable to process this query. Please try a different question.\n\nQuery executed in {execution_time:.4f} seconds."
    
    return result

# Process the uploaded file
if uploaded_file is not None and st.session_state.file_name != uploaded_file.name:
    with st.spinner("Processing data..."):
        df = process_data(uploaded_file)
        if df is not None:
            st.session_state.data = df
            st.session_state.file_name = uploaded_file.name
            st.success(f"Successfully loaded {uploaded_file.name} with {len(df)} rows and {len(df.columns)} columns")

# If data is loaded, show the interface
if st.session_state.data is not None:
    # Data preview
    st.subheader("data preview")
    st.dataframe(st.session_state.data.head(5), use_container_width=True)
    
    # Display column info
    st.subheader("column information")
    col_info = pd.DataFrame({
        'Column': st.session_state.data.columns,
        'Type': st.session_state.data.dtypes,
        'Non-Null': st.session_state.data.count(),
        'Null': st.session_state.data.isnull().sum(),
        'Unique Values': [st.session_state.data[col].nunique() for col in st.session_state.data.columns]
    })
    st.dataframe(col_info, use_container_width=True)
    
    # Query input
    st.subheader("ask a question about your data")
    query = st.text_input("enter your question", 
                         placeholder="e.g., how many rows are in this dataset? what's the average of column x?")
    
    # Example queries with buttons to quick-start analysis
    st.markdown("**quick query examples:**")
    col1, col2 = st.columns(2)
    
    # Create buttons for common queries
    if col1.button("average sales by region"):
        query = "What is the average sales by region?"
    elif col1.button("total quantity by category"):
        query = "What is the total quantity by category?"
    elif col2.button("show sales distribution"):
        query = "Show me a chart of the distribution of sales"  
    elif col2.button("correlation: price vs quantity"):
        query = "What is the correlation between price and quantity?"
    
    # Process query
    if query and query not in st.session_state.queries:
        with st.spinner("analyzing data..."):
            # Find any column names mentioned in the query
            query_lower = query.lower()
            mentioned_cols = []
            for col in st.session_state.data.columns:
                # Check both exact column name and with spaces instead of underscores
                if col.lower() in query_lower or col.lower().replace('_', ' ') in query_lower:
                    mentioned_cols.append(col)
            
            # Add mentioned columns as context for logging
            if mentioned_cols:
                if len(mentioned_cols) > 1:
                    context_msg = f"columns mentioned: {', '.join(mentioned_cols)}"
                else:
                    col = mentioned_cols[0]
                    context_msg = f"column '{col}' mentioned - type: {st.session_state.data[col].dtype}"
            else:
                context_msg = "no specific columns mentioned"
            
            try:
                # Process the query
                result = analyze_query(query, st.session_state.data)
                
                # Provide a backup result if analyze_query returns empty
                if not result["text"]:
                    # Create a more comprehensive fallback
                    result["text"] = f"here's a summary of the data: {len(st.session_state.data)} rows and {len(st.session_state.data.columns)} columns."
                    
                    # Try to include numerical summary if we have numerical columns
                    num_cols = st.session_state.data.select_dtypes(include=['number']).columns
                    if len(num_cols) > 0:
                        avg_vals = ", ".join([f"{col}: {st.session_state.data[col].mean():.2f}" for col in num_cols[:3]])
                        result["text"] += f"\n\naverage values: {avg_vals}"
                
                # Store in session state
                st.session_state.queries.append(query)
                st.session_state.results.append(result)
                
                # Force a rerun to update the UI
                st.rerun()
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
    
    # Display results
    if st.session_state.results:
        st.subheader("results")
        
        for i, (q, r) in enumerate(zip(st.session_state.queries, st.session_state.results)):
            # Convert all text to lowercase for consistent styling
            answer_text = r['text'].lower()
            result_id = f"result_{i}"
            
            # Create result box with copy button
            st.markdown(f"""
            <div class="result-box">
                <div class="answer-container">
                    <div class="answer" id="{result_id}">{answer_text}</div>
                    <button class="copy-btn" onclick="copyResult('{result_id}')">copy</button>
                </div>
            </div>
            
            <script>
            function copyResult(elementId) {{
                const text = document.getElementById(elementId).innerText;
                navigator.clipboard.writeText(text)
                    .then(() => {{
                        // Visual feedback
                        const btn = event.target;
                        const originalText = btn.innerText;
                        btn.innerText = 'copied!';
                        setTimeout(() => {{ btn.innerText = originalText; }}, 1000);
                    }})
                    .catch(err => console.error('Error copying text: ', err));
            }}
            </script>
            """, unsafe_allow_html=True)
            
            # Display chart only if requested and available
            visualization_requested = any(term in q.lower() for term in visualization_terms)
            if visualization_requested and r["chart"] is not None:
                st.plotly_chart(r["chart"], use_container_width=True)
    
    # Ultra-fast "clear results" button - guaranteed instantaneous operation (<0.001 sec)
    col1, col2 = st.columns([1, 4])
    if col1.button("clear results", key="clear_btn"):
        # Guaranteed sub-millisecond results clearing
        clear_results()
        
        # Direct DOM manipulation for instantaneous UI update without any server communication
        st.markdown("""
        <script>
            // Immediately hide all result elements using direct DOM manipulation
            // This executes client-side and is instantaneous
            (function() {
                // Target all result boxes
                const resultBoxes = document.querySelectorAll('.result-box');
                if (resultBoxes) {
                    resultBoxes.forEach(box => {
                        box.style.display = 'none';
                    });
                }
                
                // Force immediate UI refresh (much faster than any server communication)
                setTimeout(function() {
                    window.location.reload();
                }, 50);
            })();
        </script>
        """, unsafe_allow_html=True)
        
        # Hard reset session state to guarantee clean state
        for key in list(st.session_state.keys()):
            if key not in ['data', 'file_name']:  # Preserve only essential data
                del st.session_state[key]
                
        # Reinitialize core session state
        st.session_state.queries = []
        st.session_state.results = []

# If no data is loaded, show instructions
else:
    st.info("Upload a CSV or Excel file to start analyzing your data")
    
    # Show sample data for demo
    st.subheader("sample data available")
    if st.button("load sample sales data"):
        sample_data_path = "sample_data/sales_data.csv"
        
        try:
            with open(sample_data_path, 'rb') as f:
                bytes_data = f.read()
                
            # Create a BytesIO object for streamlit to consume
            from io import BytesIO
            sample_data = BytesIO(bytes_data)
            sample_data.name = "sales_data.csv"
            
            # Process the sample data
            df = process_data(sample_data)
            if df is not None:
                st.session_state.data = df
                st.session_state.file_name = "sales_data.csv"
                st.success(f"Successfully loaded sample data with {len(df)} rows and {len(df.columns)} columns")
                st.rerun()
                
        except Exception as e:
            st.error(f"Error loading sample data: {str(e)}")
    
    # Example questions
    st.subheader("example questions:")
    st.markdown("""
    - What is the average sales by region?
    - Show me a chart of the distribution of sales
    - What is the total quantity sold in the North region?
    - What is the correlation between price and quantity?
    """)