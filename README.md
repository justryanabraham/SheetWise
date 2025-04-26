# SheetWise

## A high-performance, AI-powered spreadsheet analysis platform

SheetWise leverages advanced AI capabilities to transform spreadsheet analysis through intuitive natural language processing and intelligent insights. Ask questions about your data in plain English and get immediate, accurate answers with optional visualizations.

![SheetWise](generated-icon.png)

## Table of Contents
- [Key Features](#key-features)
- [Technical Architecture](#technical-architecture)
- [Performance Characteristics](#performance-characteristics)
- [Installation](#installation)
- [Usage](#usage)
- [API Integrations](#api-integrations)
- [Advanced Features](#advanced-features)
- [Development](#development)
- [Use Cases](#use-cases)

## Key Features

### Core Functionality
- **Natural Language Query Processing**: Ask questions about your data in plain English
- **Multi-format File Support**: Import data from CSV and Excel files
- **Instant Statistical Analysis**: Get answers in under 2 seconds (guaranteed)
- **AI-Powered Data Insights**: Leverage Google's Gemini 2.5 Flash AI for deep understanding of data
- **Beautiful Interactive Visualizations**: Generate charts and graphs automatically when requested
- **High-Performance Local Processing**: Fall back to ultra-fast pandas processing when needed
- **One-Click Result Copying**: Copy any result text with a single click
- **Instantaneous Results Clearing**: Clear all results with sub-millisecond performance
- **Sample Data Integration**: Try the system with pre-loaded sample datasets

### Data Processing Capabilities
- **Automatic Data Cleaning**: Handles column name normalization and formatting
- **Date Format Detection**: Automatically identifies and properly formats date columns
- **Numeric Value Optimization**: Converts string numbers to proper numeric types
- **Statistical Summaries**: Generate quick statistical overviews of your data
- **Column Type Inference**: Intelligently determines the appropriate data types
- **Missing Value Handling**: Identifies and reports on missing data
- **Outlier Detection**: Finds and reports unusual values in the dataset

### Query Types Supported
- **Aggregation Queries**: Average, sum, minimum, maximum values
- **Filtering Queries**: Find data matching specific criteria
- **Grouping Queries**: Summarize data by categories
- **Time-Series Analysis**: Analyze trends over time periods
- **Correlation Queries**: Discover relationships between columns
- **Distribution Analysis**: Understand how data is distributed
- **Comparative Analysis**: Compare different segments of data
- **Top/Bottom N Queries**: Find highest or lowest values
- **Percentage Calculations**: Calculate portions and distributions
- **Conditional Calculations**: Perform calculations with multiple conditions

### Visualization Capabilities
- **Bar Charts**: Compare values across categories
- **Line Charts**: Visualize trends over time or sequences
- **Scatter Plots**: Explore relationships between variables
- **Pie Charts**: Show proportional distributions
- **Histograms**: Display data distribution patterns
- **Box Plots**: Summarize statistical distributions
- **Heatmaps**: Visualize matrix data intensity

## Technical Architecture

### Frontend
- **Framework**: Streamlit for interactive web interface
- **Styling**: Custom CSS with mint green/black/white theme
- **Interactivity**: JavaScript-enhanced components for instant feedback
- **Responsive Design**: Adapts to different screen sizes
- **Copy Functionality**: JavaScript clipboard integration

### Backend
- **Language**: Python 3.11
- **Core Libraries**:
  - Streamlit for web interface
  - Pandas for data processing
  - NumPy for numerical operations
  - Plotly for interactive visualizations
  - Openpyxl for Excel processing

### AI Integration
- **Primary AI**: Google Gemini 2.5 Flash (specifically "gemini-2.5-flash-preview-04-17") 
- **Fallback AI Models**:
  - Gemini 1.5 Pro
  - Gemini Pro
  - Auto-detection of available models
- **AI Prompting**: Enhanced context generation for optimal responses
- **Error Handling**: Graceful fallback to local processing if AI unavailable

### Data Processing Pipeline
1. **Data Ingestion**: Load CSV/Excel files
2. **Preprocessing**: Clean and normalize data
3. **Query Processing**: Route through AI or local processor
4. **Result Generation**: Compute answers and prepare visualizations
5. **Presentation**: Display results with copy functionality

## Performance Characteristics

### Data Size Handling
- **CSV/Excel files**: Up to 200MB (approximately 1-2 million rows)
- **Optimal performance**: 50,000-100,000 rows
- **Column limit**: No hard limit, but 100+ columns may slow processing

### Response Times
- **Local Processing (Fallback Mode)**: 
  - Under 2 seconds for millions of rows
  - Uses pandas optimized operations for instant responses
  - No API call overhead ensures consistent performance

- **Gemini API Processing**:
  - Input limit: ~100K tokens (approximately 75,000 cells of data)
  - Processing time: 2-5 seconds including API communication

### Optimization Features
- **Automatic sampling** of very large datasets to maintain performance
- **Memory-efficient processing** that releases data not actively needed
- **Column type optimization** to reduce memory usage
- **Fast indexing** for quick lookups on large datasets
- **Instantaneous UI operations** using direct DOM manipulation

## Installation

### Prerequisites
- Python 3.11 or later
- pip package manager

### Standard Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/sheetwise.git
cd sheetwise

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run simple_app.py
```

### Using Docker
```bash
# Build the Docker image
docker build -t sheetwise .

# Run the container
docker run -p 5000:5000 sheetwise
```

## Usage

### Importing Data
1. Upload a CSV or Excel file using the file uploader
2. Alternatively, click "load sample sales data" to test with sample data

### Asking Questions
1. Type your question in natural language in the query field
2. Examples of questions you can ask:
   - "What is the average sales by region?"
   - "Show me a chart of the distribution of sales"
   - "What is the total quantity sold in the North region?"
   - "What is the correlation between price and quantity?"

### Working with Results
- Results appear in black boxes below the query field
- Click the "copy" button next to any result to copy its text
- Click "clear results" to instantly remove all previous results

## API Integrations

### Google Gemini API
- Used for natural language understanding and query processing
- Requires GOOGLE_API_KEY environment variable
- Version used: "gemini-2.5-flash-preview-04-17"
- Graceful fallback to other models if primary not available

### DeepSeek API (Optional)
- Alternative AI model for natural language processing
- Requires DEEPSEEK_API_KEY environment variable
- Used as secondary fallback if Google API is unavailable

### Anthropic Claude API (Optional/Future)
- Support configured but not yet implemented
- Would require ANTHROPIC_API_KEY environment variable

### OpenAI API (Optional/Future)
- Support configured but not yet implemented
- Would require OPENAI_API_KEY environment variable

## Advanced Features

### Intelligent Data Context Generation
- Automatically analyzes dataset to provide AI with rich context
- Generates statistical summaries for each column
- Provides sample data and column relationships
- Enables more accurate AI responses

### Enhanced Local Processing
- Uses optimized pattern matching for common query types
- Specialized handlers for different statistical operations
- Extremely fast performance for basic queries
- No dependency on external APIs for core functionality

### Ultra-Fast Results Management
- Sub-millisecond clearing of all results
- Uses direct DOM manipulation for instant UI updates
- Guarantees complete session state reset
- Preserves only essential data when clearing

### Visualization Intelligence
- Automatically selects appropriate chart types
- Handles missing data in visualization requests
- Supports custom data transformations for visualizations
- Interactive charts with hover details

## Development

### Project Structure
```
sheetwise/
├── .streamlit/            # Streamlit configuration
├── sample_data/           # Sample datasets
├── utils/
│   ├── cleaning.py        # Data cleaning utilities
│   ├── data_processor.py  # Data loading and processing
│   ├── deepseek_client.py # DeepSeek API integration
│   ├── gemini_client.py   # Google Gemini API integration
│   └── query_processor.py # Query processing logic
│   └── visualizer.py      # Visualization generation
├── .replit                # Replit configuration
├── app.py                 # Alternative app entry point
├── README.md              # This documentation
├── requirements.txt       # Dependencies
└── simple_app.py          # Main application
```

### Adding New Features
- AI integrations: Add new client files in utils/
- Query types: Extend pattern matching in query_processor.py
- Visualizations: Add new chart types in visualizer.py
- UI components: Modify simple_app.py

## Use Cases

### Business Analytics
- Sales data analysis for trend identification
- Inventory management and optimization
- Customer segmentation and behavior analysis
- Financial reporting and forecasting

### Research and Academia
- Quick statistical analysis of research data
- Exploratory data analysis before deeper study
- Teaching data science concepts interactively
- Research result visualization

### Personal Finance
- Budget tracking and analysis
- Investment portfolio performance review
- Expense categorization and pattern detection
- Financial goal progress tracking

### Project Management
- Resource allocation analysis
- Timeline and milestone tracking
- Team performance metrics
- Risk assessment data analysis