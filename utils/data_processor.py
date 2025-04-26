import pandas as pd
import io
import requests
import re
import os


def load_data(uploaded_file):
    """
    Load data from an uploaded file (CSV or Excel)
    
    Args:
        uploaded_file: The uploaded file object
        
    Returns:
        tuple: (data, sheet_names) where data is either a DataFrame or 
               a dict of DataFrames for Excel files with multiple sheets
    """
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension == 'csv':
        # Try different encodings and delimiters for CSV
        try:
            # First try UTF-8
            data = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            # If UTF-8 fails, try Latin-1
            uploaded_file.seek(0)
            data = pd.read_csv(uploaded_file, encoding='latin1')
        except pd.errors.ParserError:
            # If comma delimiter fails, try with tab or semicolon
            uploaded_file.seek(0)
            data = pd.read_csv(uploaded_file, sep=None, engine='python')
        
        return data, None
    
    elif file_extension == 'xlsx':
        # Handle Excel files
        xlsx_data = {}
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_names = excel_file.sheet_names
        
        for sheet_name in sheet_names:
            xlsx_data[sheet_name] = pd.read_excel(excel_file, sheet_name=sheet_name)
        
        return xlsx_data, sheet_names
    
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")


def get_sheet_from_url(url):
    """
    Extract data from a Google Sheets URL
    
    Args:
        url: The Google Sheets URL
        
    Returns:
        tuple: (data, sheet_names) where data is a dict of DataFrames
    """
    # Extract the sheet ID from the URL
    pattern = r'https://docs.google.com/spreadsheets/d/([a-zA-Z0-9-_]+)'
    match = re.search(pattern, url)
    
    if not match:
        raise ValueError("Invalid Google Sheets URL format")
    
    sheet_id = match.group(1)
    
    # Construct the export URL (CSV format)
    export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet="
    
    # Try to fetch sheet names
    try:
        # Request just the main page to get sheet names
        response = requests.get(url)
        
        # Extract sheet names (this is a simplified approach and might not be 100% reliable)
        sheet_names_pattern = r'aria-label="([^"]+) \(tab\)"'
        found_sheets = re.findall(sheet_names_pattern, response.text)
        
        if not found_sheets:
            # If we can't extract sheet names, try to get just the first sheet
            response = requests.get(export_url)
            if response.status_code == 200:
                sheet_data = {"Sheet1": pd.read_csv(io.StringIO(response.text))}
                return sheet_data, list(sheet_data.keys())
            else:
                raise ValueError(f"Failed to fetch Google Sheet data: {response.status_code}")
        
        # Fetch each sheet
        sheet_data = {}
        for sheet_name in found_sheets:
            sheet_url = export_url + sheet_name.replace(' ', '%20')
            response = requests.get(sheet_url)
            
            if response.status_code == 200:
                sheet_data[sheet_name] = pd.read_csv(io.StringIO(response.text))
            else:
                # Skip sheets that can't be loaded
                continue
        
        if not sheet_data:
            raise ValueError("No valid sheets found in the Google Sheet")
        
        return sheet_data, list(sheet_data.keys())
    
    except Exception as e:
        raise ValueError(f"Error fetching Google Sheet: {str(e)}")
