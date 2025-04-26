import pandas as pd
import numpy as np
import re
from datetime import datetime


def clean_data(df):
    """
    Clean and preprocess the dataframe, handling common data issues.
    
    Args:
        df: The input pandas DataFrame
        
    Returns:
        tuple: (cleaned_df, issues_list) - the cleaned DataFrame and a list of issues found
    """
    # Make a copy to avoid modifying the original
    cleaned_df = df.copy()
    issues_list = []
    
    # Handle column names
    cleaned_df = clean_column_names(cleaned_df, issues_list)
    
    # Check for and handle nulls
    cleaned_df, null_issues = handle_nulls(cleaned_df)
    issues_list.extend(null_issues)
    
    # Look for and fix date columns
    cleaned_df, date_issues = fix_date_formats(cleaned_df)
    issues_list.extend(date_issues)
    
    # Fix numeric columns
    cleaned_df, numeric_issues = fix_numeric_columns(cleaned_df)
    issues_list.extend(numeric_issues)
    
    # Handle duplicate rows
    cleaned_df, duplicate_issue = handle_duplicates(cleaned_df)
    if duplicate_issue:
        issues_list.append(duplicate_issue)
    
    # Infer data types
    cleaned_df = infer_data_types(cleaned_df, issues_list)
    
    return cleaned_df, issues_list


def clean_column_names(df, issues_list):
    """Clean column names to be more usable"""
    # Get original column names
    original_columns = df.columns.tolist()
    
    # Clean column names: lowercase, spaces to underscores, remove special chars
    new_columns = []
    for col in original_columns:
        # Convert to string if not already
        col_str = str(col)
        # Lowercase and strip
        cleaned = col_str.lower().strip()
        # Replace spaces and special chars with underscore
        cleaned = re.sub(r'[^\w\s]', '', cleaned)
        cleaned = re.sub(r'\s+', '_', cleaned)
        # Ensure no empty column names or duplicates
        if cleaned == '':
            cleaned = f'column_{len(new_columns)}'
        
        # Check for duplicates
        if cleaned in new_columns:
            cleaned = f'{cleaned}_{new_columns.count(cleaned) + 1}'
        
        new_columns.append(cleaned)
    
    # Record column name changes as issues
    for i, (old, new) in enumerate(zip(original_columns, new_columns)):
        if old != new:
            issues_list.append(f"Column name changed: '{old}' → '{new}'")
    
    # Rename columns in dataframe
    df.columns = new_columns
    return df


def handle_nulls(df):
    """Handle null values in the dataframe"""
    issues = []
    
    # Check for nulls in each column
    null_counts = df.isnull().sum()
    null_columns = null_counts[null_counts > 0]
    
    if len(null_columns) > 0:
        for col, count in null_columns.items():
            percent = round(count / len(df) * 100, 2)
            issues.append(f"Column '{col}' has {count} missing values ({percent}%)")
    
    return df, issues


def fix_date_formats(df):
    """Identify and convert date columns to datetime format"""
    issues = []
    
    # Common date patterns for detection
    date_patterns = [
        # ISO format
        r'^\d{4}-\d{2}-\d{2}$',
        # US format (MM/DD/YYYY)
        r'^\d{1,2}/\d{1,2}/\d{4}$',
        r'^\d{1,2}-\d{1,2}-\d{4}$',
        # European format (DD/MM/YYYY)
        r'^\d{1,2}\.\d{1,2}\.\d{4}$',
        # Short year
        r'^\d{1,2}/\d{1,2}/\d{2}$',
        r'^\d{1,2}-\d{1,2}-\d{2}$'
    ]
    
    for col in df.columns:
        # Skip if column is already datetime
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        
        # Skip non-string columns
        if not pd.api.types.is_string_dtype(df[col]):
            continue
        
        # Skip columns with too many null values
        if df[col].isnull().sum() > len(df) * 0.5:
            continue
        
        # Sample non-null values to check patterns
        sample = df[col].dropna().astype(str).sample(min(10, df[col].dropna().shape[0])).values
        
        # Check if column values match date patterns
        date_matches = 0
        for value in sample:
            if any(re.match(pattern, value) for pattern in date_patterns):
                date_matches += 1
        
        # If more than 70% of the sample match date patterns, consider it a date column
        if date_matches / len(sample) >= 0.7:
            try:
                # Attempt to convert to datetime
                df[col] = pd.to_datetime(df[col], errors='coerce')
                issues.append(f"Converted column '{col}' to datetime format")
            except Exception as e:
                issues.append(f"Failed to convert column '{col}' to datetime: {str(e)}")
    
    return df, issues


def fix_numeric_columns(df):
    """Identify and fix numeric columns with formatting issues"""
    issues = []
    
    for col in df.columns:
        # Skip datetime columns
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        
        # Skip already numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        
        # Check if column might be numeric but stored as text
        if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            # Sample non-null values
            sample = df[col].dropna().astype(str).sample(min(10, df[col].dropna().shape[0]))
            
            # Look for common numeric patterns with commas, dollar signs, etc.
            numeric_pattern = r'^[$€£¥]?\s*-?\s*[0-9,]+(\.[0-9]+)?\s*%?$'
            numeric_matches = sum(bool(re.match(numeric_pattern, str(val))) for val in sample)
            
            # If more than 70% match, try to convert
            if numeric_matches / len(sample) >= 0.7:
                try:
                    # Remove currency symbols, commas, and percent signs
                    temp_col = df[col].astype(str).replace(r'[$€£¥,]', '', regex=True)
                    temp_col = temp_col.replace(r'%', '', regex=True)
                    
                    # Convert to numeric
                    df[col] = pd.to_numeric(temp_col, errors='coerce')
                    issues.append(f"Converted column '{col}' to numeric format")
                except Exception as e:
                    issues.append(f"Failed to convert column '{col}' to numeric: {str(e)}")
    
    return df, issues


def handle_duplicates(df):
    """Check for and remove duplicate rows"""
    duplicate_count = df.duplicated().sum()
    
    if duplicate_count > 0:
        df_deduped = df.drop_duplicates()
        removed = len(df) - len(df_deduped)
        issue = f"Removed {removed} duplicate rows"
        return df_deduped, issue
    
    return df, None


def infer_data_types(df, issues_list):
    """Try to infer appropriate data types for columns"""
    # Check for boolean columns
    for col in df.select_dtypes(include=['object']).columns:
        # Skip columns with too many unique values
        if df[col].nunique() > 5:
            continue
        
        # Convert to lowercase for comparison
        lower_values = df[col].dropna().astype(str).str.lower()
        
        # Check if values look like booleans
        if set(lower_values.unique()).issubset({'true', 'false', 'yes', 'no', 't', 'f', 'y', 'n', '1', '0', 'true', 'false'}):
            try:
                # Map values to booleans
                bool_map = {
                    'true': True, 'yes': True, 't': True, 'y': True, '1': True, 1: True,
                    'false': False, 'no': False, 'f': False, 'n': False, '0': False, 0: False
                }
                
                # Convert to lowercase and map
                df[col] = df[col].astype(str).str.lower().map(bool_map)
                issues_list.append(f"Converted column '{col}' to boolean type")
            except Exception:
                # If conversion fails, leave as is
                pass
    
    # Convert integers stored as floats (1.0, 2.0) to integers
    for col in df.select_dtypes(include=['float64']).columns:
        # Check if all non-null values are integers
        if df[col].dropna().apply(lambda x: x.is_integer()).all():
            df[col] = df[col].astype('Int64')  # Use Int64 to preserve NaN values
            issues_list.append(f"Converted column '{col}' from float to integer")
    
    return df
