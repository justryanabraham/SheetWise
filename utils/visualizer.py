import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_visualization(vis_type, vis_data, dataframe):
    """
    Create a visualization based on the specified type and data.
    
    Args:
        vis_type: The type of visualization to create
        vis_data: The data needed for the visualization
        dataframe: The pandas DataFrame containing the data
        
    Returns:
        A plotly figure object
    """
    # Extract visualization parameters
    x_column = vis_data.get('x_column')
    y_column = vis_data.get('y_column')
    category = vis_data.get('category')
    title = vis_data.get('title', 'Data Visualization')
    description = vis_data.get('description', '')
    
    # Create visualization based on type
    if vis_type.lower() == 'bar':
        return create_bar_chart(dataframe, x_column, y_column, category, title, description)
    
    elif vis_type.lower() == 'line':
        return create_line_chart(dataframe, x_column, y_column, category, title, description)
    
    elif vis_type.lower() == 'pie':
        return create_pie_chart(dataframe, x_column, y_column, title, description)
    
    elif vis_type.lower() == 'scatter':
        return create_scatter_plot(dataframe, x_column, y_column, category, title, description)
    
    elif vis_type.lower() == 'histogram':
        return create_histogram(dataframe, x_column, title, description)
    
    elif vis_type.lower() == 'heatmap':
        return create_heatmap(dataframe, x_column, y_column, title, description)
    
    else:
        # Default to a simple bar chart if type is unknown
        return create_bar_chart(dataframe, x_column, y_column, category, 
                               "Default Visualization", "Unknown chart type requested")


def create_bar_chart(df, x_column, y_column, category=None, title='Bar Chart', description=''):
    """Create a bar chart visualization"""
    # Theme settings
    theme_color = '#98FF98'
    
    if category:
        fig = px.bar(
            df, x=x_column, y=y_column, color=category,
            title=title,
            color_discrete_sequence=px.colors.qualitative.Pastel,
            template='plotly_white'
        )
    else:
        fig = px.bar(
            df, x=x_column, y=y_column,
            title=title,
            color_discrete_sequence=[theme_color],
            template='plotly_white'
        )
    
    # Add description as annotation if provided
    if description:
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0, y=-0.15,
            text=description,
            showarrow=False,
            font=dict(size=12),
            align="left"
        )
    
    # Update layout
    fig.update_layout(
        plot_bgcolor='white',
        margin=dict(t=100, l=40, r=40, b=40),
        title_font=dict(size=18, color='#333333'),
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14)
    )
    
    return fig


def create_line_chart(df, x_column, y_column, category=None, title='Line Chart', description=''):
    """Create a line chart visualization"""
    # Theme settings
    theme_color = '#98FF98'
    
    if category:
        fig = px.line(
            df, x=x_column, y=y_column, color=category,
            title=title,
            color_discrete_sequence=px.colors.qualitative.Pastel,
            template='plotly_white'
        )
    else:
        fig = px.line(
            df, x=x_column, y=y_column,
            title=title,
            color_discrete_sequence=[theme_color],
            template='plotly_white'
        )
    
    # Add description as annotation if provided
    if description:
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0, y=-0.15,
            text=description,
            showarrow=False,
            font=dict(size=12),
            align="left"
        )
    
    # Update layout
    fig.update_layout(
        plot_bgcolor='white',
        margin=dict(t=100, l=40, r=40, b=40),
        title_font=dict(size=18, color='#333333'),
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14)
    )
    
    return fig


def create_pie_chart(df, labels_column, values_column, title='Pie Chart', description=''):
    """Create a pie chart visualization"""
    # Aggregate data by labels column if needed
    if labels_column and values_column:
        # Get unique values and their sums
        agg_data = df.groupby(labels_column)[values_column].sum().reset_index()
        
        fig = px.pie(
            agg_data, names=labels_column, values=values_column,
            title=title,
            color_discrete_sequence=px.colors.sequential.Mint,
            template='plotly_white'
        )
    else:
        # If columns aren't specified properly, create an empty chart with a warning
        fig = go.Figure(go.Pie(
            labels=["No Data"], 
            values=[1],
            textinfo="label"
        ))
        fig.update_layout(title="Invalid data for pie chart")
    
    # Add description as annotation if provided
    if description:
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0, y=-0.15,
            text=description,
            showarrow=False,
            font=dict(size=12),
            align="left"
        )
    
    # Update layout
    fig.update_layout(
        margin=dict(t=100, l=40, r=40, b=40),
        title_font=dict(size=18, color='#333333')
    )
    
    return fig


def create_scatter_plot(df, x_column, y_column, category=None, title='Scatter Plot', description=''):
    """Create a scatter plot visualization"""
    # Theme settings
    theme_color = '#98FF98'
    
    if category:
        fig = px.scatter(
            df, x=x_column, y=y_column, color=category,
            title=title,
            color_discrete_sequence=px.colors.qualitative.Pastel,
            template='plotly_white'
        )
    else:
        fig = px.scatter(
            df, x=x_column, y=y_column,
            title=title,
            color_discrete_sequence=[theme_color],
            template='plotly_white'
        )
    
    # Add description as annotation if provided
    if description:
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0, y=-0.15,
            text=description,
            showarrow=False,
            font=dict(size=12),
            align="left"
        )
    
    # Update layout
    fig.update_layout(
        plot_bgcolor='white',
        margin=dict(t=100, l=40, r=40, b=40),
        title_font=dict(size=18, color='#333333'),
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14)
    )
    
    return fig


def create_histogram(df, column, title='Histogram', description=''):
    """Create a histogram visualization"""
    # Theme settings
    theme_color = '#98FF98'
    
    fig = px.histogram(
        df, x=column,
        title=title,
        color_discrete_sequence=[theme_color],
        template='plotly_white'
    )
    
    # Add description as annotation if provided
    if description:
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0, y=-0.15,
            text=description,
            showarrow=False,
            font=dict(size=12),
            align="left"
        )
    
    # Update layout
    fig.update_layout(
        plot_bgcolor='white',
        margin=dict(t=100, l=40, r=40, b=40),
        title_font=dict(size=18, color='#333333'),
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14)
    )
    
    return fig


def create_heatmap(df, x_column, y_column, title='Heatmap', description=''):
    """Create a heatmap visualization"""
    # Check if both columns are provided
    if not x_column or not y_column:
        fig = go.Figure()
        fig.update_layout(title="Invalid data for heatmap - columns not specified")
        return fig
    
    try:
        # Create a pivot table or correlation based on column types
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if x_column in numeric_cols and y_column in numeric_cols:
            # For numeric columns, show correlation
            corr_df = df[[x_column, y_column]].corr()
            
            fig = px.imshow(
                corr_df,
                title=title,
                color_continuous_scale='Mint',
                template='plotly_white',
                text_auto=True
            )
        else:
            # For categorical columns, create a cross-tabulation
            pivot = pd.crosstab(df[y_column], df[x_column])
            
            fig = px.imshow(
                pivot,
                title=title,
                color_continuous_scale='Mint',
                template='plotly_white',
                text_auto=True
            )
        
        # Add description as annotation if provided
        if description:
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0, y=-0.15,
                text=description,
                showarrow=False,
                font=dict(size=12),
                align="left"
            )
        
        # Update layout
        fig.update_layout(
            margin=dict(t=100, l=40, r=40, b=40),
            title_font=dict(size=18, color='#333333')
        )
        
        return fig
    
    except Exception as e:
        # If error occurs, return an empty figure with error message
        fig = go.Figure()
        fig.update_layout(title=f"Error creating heatmap: {str(e)}")
        return fig
