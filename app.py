import streamlit as st
import pandas as pd
import os
import io
import base64
from utils.data_processor import load_data, get_sheet_from_url
from utils.query_processor import process_query
from utils.visualizer import create_visualization
from utils.cleaning import clean_data

# Set up page config
st.set_page_config(
    page_title="sheetwise",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling - minimal adjustments while keeping Streamlit defaults
st.markdown("""
<style>
    .main-header {
        font-weight: 300;
        text-transform: lowercase;
    }
    .stButton button {
        background-color: #98FF98;
        border-radius: 8px;
        text-transform: lowercase;
    }
    div[data-testid="stFileUploader"] {
        border: 1px dashed #98FF98;
        border-radius: 8px;
        padding: 10px;
    }
    .custom-card {
        border-radius: 8px;
        padding: 20px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'data_issues' not in st.session_state:
    st.session_state.data_issues = None
if 'sheet_names' not in st.session_state:
    st.session_state.sheet_names = None
if 'current_sheet' not in st.session_state:
    st.session_state.current_sheet = None

def reset_app():
    """Reset the app state"""
    st.session_state.data = None
    st.session_state.file_name = None
    st.session_state.chat_history = []
    st.session_state.cleaned_data = None
    st.session_state.data_issues = None
    st.session_state.sheet_names = None
    st.session_state.current_sheet = None


# Main app layout
st.markdown('<h1 class="main-header">sheetwise</h1>', unsafe_allow_html=True)
st.markdown('*ask questions about your spreadsheet data using natural language*')

# File upload section
st.markdown("### upload your data")
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "drag and drop your spreadsheet here (csv, xlsx)", 
        type=["csv", "xlsx"],
        label_visibility="collapsed"
    )

with col2:
    sheet_url = st.text_input("or paste a google sheets url", placeholder="https://docs.google.com/spreadsheets/d/...")
    validate_url = st.button("validate & load")

# Process uploaded file or Google Sheet URL
if uploaded_file is not None and st.session_state.file_name != uploaded_file.name:
    try:
        # Load and process the data
        data, sheet_names = load_data(uploaded_file)
        
        # Store in session state
        st.session_state.data = data
        st.session_state.file_name = uploaded_file.name
        st.session_state.sheet_names = sheet_names
        
        if sheet_names and len(sheet_names) > 0:
            st.session_state.current_sheet = sheet_names[0]
        
        # Clean data and detect issues
        current_data = data[st.session_state.current_sheet] if sheet_names else data
        st.session_state.cleaned_data, st.session_state.data_issues = clean_data(current_data)
        
        st.success(f"'{uploaded_file.name}' successfully loaded!")
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")

elif validate_url and sheet_url:
    try:
        # Load and process the Google Sheet
        data, sheet_names = get_sheet_from_url(sheet_url)
        
        # Store in session state
        st.session_state.data = data
        st.session_state.file_name = sheet_url
        st.session_state.sheet_names = sheet_names
        
        if sheet_names and len(sheet_names) > 0:
            st.session_state.current_sheet = sheet_names[0]
        
        # Clean data and detect issues
        current_data = data[st.session_state.current_sheet] if sheet_names else data
        st.session_state.cleaned_data, st.session_state.data_issues = clean_data(current_data)
        
        st.success(f"Google Sheet successfully loaded!")
    except Exception as e:
        st.error(f"Error loading Google Sheet: {str(e)}")

# If data is loaded, show the interface
if st.session_state.data is not None:
    st.markdown("---")
    
    # Sheet selector (if multiple sheets)
    if st.session_state.sheet_names and len(st.session_state.sheet_names) > 1:
        selected_sheet = st.selectbox(
            "select sheet:", 
            st.session_state.sheet_names,
            index=st.session_state.sheet_names.index(st.session_state.current_sheet)
        )
        
        if selected_sheet != st.session_state.current_sheet:
            st.session_state.current_sheet = selected_sheet
            current_data = st.session_state.data[selected_sheet]
            st.session_state.cleaned_data, st.session_state.data_issues = clean_data(current_data)
    
    # Get the current dataset
    current_data = st.session_state.data[st.session_state.current_sheet] if st.session_state.sheet_names else st.session_state.data
    cleaned_data = st.session_state.cleaned_data
    
    # Data preview
    st.markdown("### data preview")
    st.dataframe(cleaned_data.head(10), use_container_width=True)
    
    # Show data information
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("rows", f"{len(cleaned_data)}")
    with col2:
        st.metric("columns", f"{len(cleaned_data.columns)}")
    with col3:
        if st.session_state.data_issues:
            st.metric("data issues", f"{len(st.session_state.data_issues)}")
    
    # Show data issues if any
    if st.session_state.data_issues:
        with st.expander("view data issues"):
            for issue in st.session_state.data_issues:
                st.markdown(f"- {issue}")
    
    # Query input
    st.markdown("### ask about your data")
    query = st.text_input("enter your question in natural language", placeholder="what is the average of column A?")
    
    # Process query
    if query:
        with st.spinner("analyzing your data..."):
            try:
                # Process the query
                response, vis_type, vis_data = process_query(query, cleaned_data)
                
                # Add to chat history
                st.session_state.chat_history.append({"query": query, "response": response, "vis_type": vis_type, "vis_data": vis_data})
                
                # Scroll to bottom to show new response
                st.rerun()
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### results")
        
        for i, chat in enumerate(st.session_state.chat_history):
            st.markdown(f"""
            <div class="custom-card">
                <p><strong>you:</strong> {chat["query"]}</p>
                <p><strong>sheetwise:</strong> {chat["response"]}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display visualization if available
            if chat["vis_type"] and chat["vis_data"]:
                st.markdown("#### visualization")
                fig = create_visualization(
                    chat["vis_type"], 
                    chat["vis_data"],
                    cleaned_data
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Export options
    st.markdown("### export options")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("export cleaned data (csv)"):
            csv = cleaned_data.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="cleaned_data.csv">download csv file</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    with col2:
        if st.button("export analysis summary"):
            if st.session_state.chat_history:
                summary = "# SheetWise Analysis Summary\n\n"
                for chat in st.session_state.chat_history:
                    summary += f"## Query: {chat['query']}\n"
                    summary += f"Answer: {chat['response']}\n\n"
                
                b64 = base64.b64encode(summary.encode()).decode()
                href = f'<a href="data:text/markdown;base64,{b64}" download="analysis_summary.md">download summary</a>'
                st.markdown(href, unsafe_allow_html=True)
            else:
                st.warning("No analysis to export. Ask questions first!")
    
    # Reset app button
    if st.button("start over"):
        reset_app()
        st.rerun()

# If no data is loaded, show welcome animation
else:
    st.markdown("""
    <div style="text-align: center; padding: 50px;">
        <h2 style="font-weight: 300; text-transform: lowercase;">welcome to sheetwise</h2>
        <p>upload your spreadsheet and start asking questions</p>
        <div style="font-size: 60px; margin: 30px;">📊</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sample queries
    st.markdown("### sample questions you can ask:")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("- what's the total revenue this year?")
        st.markdown("- show me a chart of expenses by department")
        st.markdown("- how many missing values are in column b?")
    
    with col2:
        st.markdown("- summarize this data for me")
        st.markdown("- give me top 5 products by sales")
        st.markdown("- what's the correlation between price and quantity?")
