"""
DeepSeek API client integration for natural language processing of data queries.
"""
import os
import json
from openai import OpenAI

# Get the API key from environment variables
api_key = os.environ.get("DEEPSEEK_API_KEY", "")

# Create a client instance with the DeepSeek API base URL
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com/v1"  # DeepSeek API endpoint
)

def process_data_query(query, data_context):
    """
    Use DeepSeek to process a natural language query about data.
    
    Args:
        query: The natural language query string
        data_context: Context information about the data
        
    Returns:
        dict: The parsed JSON response from DeepSeek
    """
    if not api_key:
        raise ValueError("DeepSeek API key not found. Please set the DEEPSEEK_API_KEY environment variable.")
    
    system_prompt = f"""
    You are an AI assistant specialized in data analysis. You help users analyze their spreadsheet data.
    The user will give you questions about their data. You should respond with:
    1. A natural language answer to their question
    2. The type of visualization that would best represent the answer (if applicable)
    3. The data for the visualization

    The data is a pandas DataFrame with the following characteristics:
    {data_context}
    
    Respond in the following JSON format:
    {{
        "answer": "Your natural language answer to the query",
        "visualization": {{
            "type": "One of: bar, line, pie, scatter, histogram, heatmap, none",
            "x_column": "Column name for x-axis if applicable",
            "y_column": "Column name for y-axis if applicable",
            "category": "Category column if applicable",
            "title": "Chart title",
            "description": "Brief chart description"
        }}
    }}
    """
    
    # Call DeepSeek API using their recommended model
    try:
        print("Sending request to DeepSeek API...")
        response = client.chat.completions.create(
            model="deepseek-chat",  # Use the appropriate DeepSeek model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.2,
            max_tokens=2000,
            response_format={"type": "json_object"}  # Request JSON response
        )
        
        # Extract the content from the response
        content = response.choices[0].message.content
        print(f"Raw response: {content[:100]}...")
        
        # Parse the JSON response
        try:
            parsed_response = json.loads(content)
            print("JSON parsed successfully")
            return parsed_response
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {str(e)}")
            # Create a basic response for error
            return {
                "answer": f"I couldn't process this query properly. JSON parsing error: {str(e)}",
                "visualization": {"type": "none"}
            }
    
    except Exception as e:
        print(f"DeepSeek API error: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return error response
        return {
            "answer": f"I encountered an error processing your query: {str(e)}",
            "visualization": {"type": "none"}
        }