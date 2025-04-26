"""
Google Gemini API client integration for natural language processing of data queries.
"""
import os
import json
import google.generativeai as genai
import traceback

def process_data_query(query, data_context):
    """
    Use Google Gemini Pro to process a natural language query about data.
    
    Args:
        query: The natural language query string
        data_context: Context information about the data
        
    Returns:
        dict: The parsed JSON response from Gemini
    """
    # Check for API key
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError("Google API Key is required but not found in environment variables")
    
    # Configure the Gemini API client
    genai.configure(api_key=api_key)
    
    # Use the specific Gemini model requested by the user
    try:
        # Use the specified model: gemini-2.5-flash-preview-04-17
        model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17') 
        print(f"SUCCESS! Using requested model: gemini-2.5-flash-preview-04-17")
    except Exception as e:
        print(f"Failed to load model gemini-2.5-flash-preview-04-17: {str(e)}. Trying fallback models.")
        try:
            # Fallback to newer model name
            model = genai.GenerativeModel('gemini-1.5-pro')
            print(f"SUCCESS! Using fallback model: gemini-1.5-pro")
        except Exception as e2:
            print(f"Failed to load model gemini-1.5-pro: {str(e2)}. Trying older model version.")
            try:
                # Fallback to older model name
                model = genai.GenerativeModel('gemini-pro')
                print(f"SUCCESS! Using fallback model: gemini-pro")
            except Exception as e3:
                # If all specific models fail, try to list available models and use the first one
                print(f"Failed to load model gemini-pro: {str(e3)}. Listing available models...")
                try:
                    models = genai.list_models()
                    available_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
                    if available_models:
                        selected_model = available_models[0]
                        print(f"Available models: {available_models}")
                        print(f"SUCCESS! Using auto-detected model: {selected_model}")
                        model = genai.GenerativeModel(selected_model)
                    else:
                        raise ValueError("No suitable models found for text generation")
                except Exception as e4:
                    raise ValueError(f"Could not initialize any Gemini model: {str(e4)}")
    
    # Build the prompt with the query and data context
    print("Sending request to Gemini API...")
    prompt = f"""
    You are an expert data analyst working with spreadsheet data. Your task is to translate ANY natural language query into specific data operations and return precise answers with calculated values directly from the data.
    
    USER QUERY: {query}
    
    SPREADSHEET DATA INFORMATION:
    {data_context}
    
    INSTRUCTIONS:
    1. INTERPRET the user's query to identify what information they want from the spreadsheet
    2. EXTRACT the relevant data from the spreadsheet information provided
    3. CALCULATE the exact values requested (even if not explicitly asked for statistical calculations)
    4. RETURN a specific, concrete answer with numbers and facts extracted from the data
    5. NEVER respond with "I would need to" or "I could calculate" - actually perform the calculation using the data provided
    
    QUERY HANDLING INSTRUCTIONS:
    - For questions about averages/means: Calculate and return the actual numerical average
    - For questions about sums/totals: Calculate and return the actual sum
    - For questions about counts: Count and return the actual number
    - For questions about min/max: Find and return the actual minimum/maximum values
    - For comparisons: Perform the comparison and return the specific result
    - For relationships: Calculate correlations or provide specific examples from the data
    - For trends: Analyze the data and describe the specific trend with supporting numbers
    
    You must respond in valid JSON format only, using this exact structure:
    
    {{
        "answer": "specific answer with actual calculated values from the data",
        "visualization": {{
            "type": "bar", // options: bar, line, scatter, pie, histogram, box, none
            "x_column": "column_name", // x-axis column or labels column for pie
            "y_column": "column_name", // y-axis column or values column for pie, can be null
            "title": "chart title"
        }}
    }}
    
    CRITICAL REQUIREMENTS:
    - ALWAYS return concrete numbers and facts extracted directly from the data
    - ALWAYS calculate the exact values rather than describing what could be calculated
    - NEVER return generic responses that don't answer the specific question
    - ALWAYS choose the most appropriate visualization for the data if relevant
    - DO NOT include comments, explanations, or markdown formatting in your JSON
    - USE only column names that are actually in the dataset 
    - RETURN only raw JSON - no explanations, preambles, or additional text
    
    EXAMPLE QUERY: "what's the average of column sales?"
    WRONG ANSWER: "I can calculate the average of the sales column from the dataset."
    CORRECT ANSWER: "The average value of the sales column is 1,250.43."
    """
    
    # Add additional logging
    try:
        # Send the request to Gemini
        response = model.generate_content(prompt)
        
        # Extract just the content
        response_text = response.text
        
        # Make sure we have a JSON string by removing any markdown code block formatting
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
            
        # Parse the JSON response
        parsed_response = json.loads(response_text)
        
        # Validate that our response has the required fields
        if not isinstance(parsed_response, dict):
            raise ValueError("Response is not a dictionary")
        
        if "answer" not in parsed_response:
            raise ValueError("Response is missing 'answer' field")
            
        # Return the parsed response
        return parsed_response
        
    except Exception as e:
        # Log detailed error information for debugging
        print(f"Error in Gemini API request: {str(e)}")
        traceback.print_exc()
        raise e