import os
import pandas as pd
import json
import numpy as np
import traceback
from utils import gemini_client

# Initialize Google API key
google_api_key = os.environ.get("GOOGLE_API_KEY", "")

def process_query(query, data):
    """
    Process a natural language query about the data, first trying AI processing with Gemini
    and falling back to fast local processing if the API fails.
    
    Args:
        query: The natural language query string
        data: The pandas DataFrame containing the data
        
    Returns:
        tuple: (response_text, visualization_type, visualization_data)
    """
    # Add debug logging
    print(f"Received query: {query}")
    
    # First try using Gemini AI if we have an API key
    if google_api_key:
        try:
            print("Using Gemini API for natural language processing...")
            return _process_query_with_ai(query, data)
        except Exception as ai_error:
            print(f"Gemini API failed: {str(ai_error)}. Falling back to local processing...")
    else:
        print("No Gemini API key found. Using local processing only.")
    
    # Fall back to fast local processing if AI fails or is not available
    try:
        # Use the optimized local query processor with <2 second response time
        print("Using optimized local data processing for fast response")
        return _process_query_fallback(query, data)
    except Exception as e:
        print(f"Local processing also failed: {str(e)}")
        return f"I couldn't process your query. Error: {str(e)}", None, None


def _process_query_with_ai(query, data):
    """Use the Google Gemini Pro model to process the query"""
    # Generate data context for the AI
    data_context = _generate_data_context(data)
    print(f"Generated data context with {len(data_context)} characters")
    
    try:
        print("Using Google Gemini API for query processing...")
        # Call the Gemini client to process the query
        parsed_response = gemini_client.process_data_query(query, data_context)
        
        # Extract the components
        answer_text = parsed_response.get("answer", "I couldn't determine an answer to your query.")
        print(f"Answer text: {answer_text[:50]}...")
        
        # Get visualization info
        vis_info = parsed_response.get("visualization", {})
        vis_type = vis_info.get("type", "none")
        print(f"Visualization type: {vis_type}")
        
        # If no visualization is needed, return just the text
        if vis_type.lower() == "none":
            print("No visualization needed")
            return answer_text, None, None
        
        # Return all the visualization data
        print("Returning visualization data")
        return answer_text, vis_type, vis_info
    
    except Exception as e:
        # Handle errors gracefully
        print(f"ERROR in _process_query_with_ai: {str(e)}")
        traceback.print_exc()
        raise e


def _process_query_fallback(query, data):
    """
    Fast local method for processing queries.
    Optimized for speed (under 2 seconds response time).
    """
    print("Using high-speed local query processing")
    # Lowercase the query to make matching more robust
    query = query.lower()
    
    # Check for common query types
    try:
        # Showing data statistics
        if "average" in query or "mean" in query:
            # Find potential column names mentioned in the query
            col_matches = _find_columns_in_query(query, data)
            
            if col_matches:
                col_name = col_matches[0]
                if pd.api.types.is_numeric_dtype(data[col_name]):
                    avg_value = data[col_name].mean()
                    return f"The average of '{col_name}' is {avg_value:.2f}", "bar", {
                        "type": "bar",
                        "x_column": col_name,
                        "y_column": None,
                        "title": f"Average of {col_name}",
                        "description": f"The average value is {avg_value:.2f}"
                    }
                else:
                    return f"Cannot calculate average for '{col_name}' as it is not a numeric column.", None, None
            else:
                # If no specific column, calculate average for all numeric columns
                numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    results = f"Average values for numeric columns:\n"
                    for col in numeric_cols[:5]:  # Limit to first 5 columns
                        avg = data[col].mean()
                        results += f"- {col}: {avg:.2f}\n"
                    
                    return results, "bar", {
                        "type": "bar",
                        "x_column": numeric_cols[0] if numeric_cols else None,
                        "y_column": None,
                        "title": "Numeric Column Averages",
                        "description": "Average values for numeric columns"
                    }
                else:
                    return "No numeric columns found for calculating averages.", None, None
                    
        elif "sum" in query or "total" in query:
            col_matches = _find_columns_in_query(query, data)
            
            if col_matches:
                col_name = col_matches[0]
                if pd.api.types.is_numeric_dtype(data[col_name]):
                    sum_value = data[col_name].sum()
                    return f"The sum of '{col_name}' is {sum_value:.2f}", "bar", {
                        "type": "bar",
                        "x_column": col_name,
                        "y_column": None,
                        "title": f"Sum of {col_name}",
                        "description": f"The total sum is {sum_value:.2f}"
                    }
                else:
                    return f"Cannot calculate sum for '{col_name}' as it is not a numeric column.", None, None
            else:
                # Calculate sum for all numeric columns
                numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    results = f"Sum values for numeric columns:\n"
                    for col in numeric_cols[:5]:  # Limit to first 5 columns
                        total = data[col].sum()
                        results += f"- {col}: {total:.2f}\n"
                    
                    return results, None, None
                else:
                    return "No numeric columns found for calculating sums.", None, None
                
        elif "count" in query or "how many" in query:
            # Handle country/location queries first (very common use case)
            if "from" in query or "in" in query:
                # Check for queries like "how many people are from Nigeria"
                # Extract the country/location name
                words = query.split()
                location = None
                
                # Simple location extraction
                for i, word in enumerate(words):
                    if word in ['from', 'in'] and i < len(words) - 1:
                        location = words[i+1].strip("?.,")
                        break
                
                if location:
                    # Look for columns that might contain location data
                    location_cols = []
                    for col in data.columns:
                        col_lower = col.lower()
                        if any(term in col_lower for term in ['country', 'location', 'region', 'city', 'state', 'nation']):
                            location_cols.append(col)
                    
                    if location_cols:
                        # Try to find the location in the potential columns
                        for col in location_cols:
                            # Very fast method using vectorized operations
                            try:
                                # Case-insensitive contains
                                mask = data[col].str.contains(location, case=False, na=False)
                                count = mask.sum()
                                if count > 0:
                                    return f"Found {count} records for {location} in the {col} column", "pie", {
                                        "type": "pie",
                                        "labels_column": col,
                                        "values_column": "count",
                                        "title": f"Distribution for {location}",
                                        "description": f"Found {count} records for {location}"
                                    }
                            except:
                                continue
            
            # Standard count handling
            col_matches = _find_columns_in_query(query, data)
            
            if col_matches:
                col_name = col_matches[0]
                count = data[col_name].count()
                return f"The count of non-null values in '{col_name}' is {count}", "bar", {
                    "type": "bar",
                    "x_column": col_name,
                    "y_column": None,
                    "title": f"Count of {col_name}",
                    "description": f"Total count: {count}"
                }
            else:
                return f"The dataset has {len(data)} rows and {len(data.columns)} columns.", None, None
                
        elif "chart" in query or "plot" in query or "graph" in query or "visualize" in query:
            col_matches = _find_columns_in_query(query, data)
            
            if len(col_matches) >= 2:
                # If we found at least two columns, we can try to create a visualization
                x_col = col_matches[0]
                y_col = col_matches[1]
                
                # Determine appropriate visualization type
                if pd.api.types.is_numeric_dtype(data[x_col]) and pd.api.types.is_numeric_dtype(data[y_col]):
                    return f"Here's a scatter plot of {x_col} vs {y_col}", "scatter", {
                        "type": "scatter",
                        "x_column": x_col,
                        "y_column": y_col,
                        "title": f"{x_col} vs {y_col}",
                        "description": "Scatter plot showing the relationship between these variables"
                    }
                elif pd.api.types.is_numeric_dtype(data[y_col]):
                    return f"Here's a bar chart of {y_col} by {x_col}", "bar", {
                        "type": "bar",
                        "x_column": x_col,
                        "y_column": y_col,
                        "title": f"{y_col} by {x_col}",
                        "description": "Bar chart showing values across categories"
                    }
                else:
                    return f"Here's a count of {x_col} values", "pie", {
                        "type": "pie",
                        "x_column": x_col,
                        "y_column": data[x_col].value_counts().index.tolist(),
                        "title": f"Distribution of {x_col}",
                        "description": "Pie chart showing the distribution of values"
                    }
            elif len(col_matches) == 1:
                col_name = col_matches[0]
                if pd.api.types.is_numeric_dtype(data[col_name]):
                    return f"Here's a histogram of {col_name}", "histogram", {
                        "type": "histogram",
                        "x_column": col_name,
                        "title": f"Distribution of {col_name}",
                        "description": "Histogram showing the distribution of values"
                    }
                else:
                    return f"Here's a bar chart showing the count of each {col_name} value", "bar", {
                        "type": "bar",
                        "x_column": col_name,
                        "y_column": "count",
                        "title": f"Count of {col_name} values",
                        "description": "Bar chart showing the frequency of each value"
                    }
            else:
                # If no specific columns mentioned, show a summary
                if "bar" in query:
                    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                    if numeric_cols:
                        # Create a temporary dataframe with reset index to avoid x='index' error
                        temp_df = pd.DataFrame({
                            'row_number': range(len(data)),
                            'value': data[numeric_cols[0]]
                        })
                        return f"Here's a bar chart of {numeric_cols[0]}", "bar", {
                            "type": "bar",
                            "x_column": "row_number",
                            "y_column": "value",
                            "data": temp_df,  # Pass the dataframe directly
                            "title": f"Values of {numeric_cols[0]}",
                            "description": "Bar chart showing values"
                        }
                else:
                    # Default visualization - show first numeric column
                    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                    if numeric_cols:
                        return f"Here's a histogram of {numeric_cols[0]}", "histogram", {
                            "type": "histogram",
                            "x_column": numeric_cols[0],
                            "title": f"Distribution of {numeric_cols[0]}",
                            "description": "Histogram showing the distribution of values"
                        }
                    else:
                        return "No numeric columns found for visualization.", None, None
                    
        elif "missing" in query or "null" in query:
            # Count missing values in each column
            missing_counts = data.isnull().sum()
            missing_cols = missing_counts[missing_counts > 0]
            
            if len(missing_cols) > 0:
                results = "Missing values by column:\n"
                for col, count in missing_cols.items():
                    percent = round((count / len(data)) * 100, 2)
                    results += f"- {col}: {count} ({percent}%)\n"
                
                # Create a visualization of missing data
                return results, "bar", {
                    "type": "bar",
                    "x_column": missing_cols.index.tolist(),
                    "y_column": missing_cols.values.tolist(),
                    "title": "Missing Values by Column",
                    "description": "Number of missing values in each column"
                }
            else:
                return "There are no missing values in the dataset.", None, None
                
        elif "describe" in query or "summary" in query or "summarize" in query or "statistics" in query:
            # Provide a general summary of the dataset
            numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
            desc = data[numeric_cols].describe() if numeric_cols else data.describe()
            summary_text = "Dataset Summary:\n"
            summary_text += f"- Rows: {len(data)}\n"
            summary_text += f"- Columns: {len(data.columns)}\n"
            summary_text += f"- Numeric columns: {len(numeric_cols)}\n"
            
            # Add some basic stats for the first few numeric columns
            if numeric_cols:
                for col in numeric_cols[:3]:
                    summary_text += f"\nStats for {col}:\n"
                    summary_text += f"- Mean: {data[col].mean():.2f}\n"
                    summary_text += f"- Min: {data[col].min():.2f}\n"
                    summary_text += f"- Max: {data[col].max():.2f}\n"
                
                # Create a box plot visualization for numeric columns
                return summary_text, "box", {
                    "type": "box",
                    "x_column": numeric_cols[0],
                    "title": "Box Plot of Numeric Columns",
                    "description": "Statistical distribution of numeric values"
                }
            else:
                return summary_text, None, None
        
        # Default response if we can't parse the query
        return "I'm not sure how to answer that query. Try asking about statistics like average, sum, or count for specific columns.", None, None
    
    except Exception as e:
        print(f"Error in fallback query processing: {str(e)}")
        return f"I couldn't process your query: {str(e)}", None, None


def _find_columns_in_query(query, data):
    """
    Find column names mentioned in the query - optimized for speed.
    This function has been optimized for sub-second performance.
    """
    query = query.lower()
    matches = []
    
    # Fast exact match lookup using sets
    query_words = set(query.split())
    column_set = {col.lower(): col for col in data.columns}
    
    # First try direct lookups - O(1) operation with sets
    for word in query_words:
        if word in column_set:
            matches.append(column_set[word])
            
    # If no exact matches, try word contains
    if not matches:
        for col_lower, col in column_set.items():
            if col_lower in query:
                matches.append(col)
                break
    
    # Quick fallback - grab the first column
    if not matches and len(data.columns) > 0:
        # Just grab first numeric column for speed
        numeric_cols = data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            matches = [numeric_cols[0]]
        else:
            matches = [data.columns[0]]
    
    return matches


def _generate_data_context(data):
    """
    Generate a context description of the DataFrame for the AI.
    
    Args:
        data: The pandas DataFrame
        
    Returns:
        str: A context description of the DataFrame
    """
    # Get basic DataFrame information
    num_rows, num_cols = data.shape
    column_names = list(data.columns)
    
    # Get data types for each column
    dtypes = data.dtypes.to_dict()
    cleaned_dtypes = {col: str(dtype) for col, dtype in dtypes.items()}
    
    # Sample values from each column (up to 5)
    sample_values = {}
    for col in column_names:
        unique_values = data[col].dropna().unique()
        if len(unique_values) > 5:
            unique_values = unique_values[:5]
        
        # Convert to strings and handle potential serialization issues
        try:
            sample_values[col] = [str(val) for val in unique_values]
        except:
            sample_values[col] = ["[complex data]"]
    
    # Check for date columns
    date_columns = []
    for col in column_names:
        if pd.api.types.is_datetime64_any_dtype(data[col]):
            date_columns.append(col)
        elif str(dtypes[col]).startswith('object'):
            # Check if object columns might contain dates
            try:
                if data[col].str.match(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}').any():
                    date_columns.append(col)
            except:
                pass
    
    # Check for numeric columns
    numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
    
    # Check for categorical columns
    categorical_columns = data.select_dtypes(include=['category']).columns.tolist()
    
    # Add object columns with low cardinality as potential categorical columns
    for col in data.select_dtypes(include=['object']).columns:
        if data[col].nunique() < 20:  # Less than 20 unique values
            categorical_columns.append(col)
    
    # Build the context string
    context = f"""
    DataFrame with {num_rows} rows and {num_cols} columns.
    
    Column names: {column_names}
    
    Data types:
    {json.dumps(cleaned_dtypes, indent=2)}
    
    Sample values from each column:
    {json.dumps(sample_values, indent=2)}
    
    Date columns: {date_columns}
    
    Numeric columns: {numeric_columns}
    
    Categorical columns: {categorical_columns}
    """
    
    return context
