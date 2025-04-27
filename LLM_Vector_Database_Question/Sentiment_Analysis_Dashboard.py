import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
import numpy as np

def create_sentiment_dashboard(df):
    """
    Create a Dash app for visualizing sentiment trends.
    
    Args:
        df: DataFrame containing review data
    
    Returns:
        dash.Dash: Dash app
    """
    # Prepare data
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M').astype(str)
    
    # Calculate sentiment counts by month and category
    sentiment_by_month = df.groupby(['month', 'sentiment']).size().unstack(fill_value=0)
    sentiment_by_category = df.groupby(['category', 'sentiment']).size().unstack(fill_value=0)
    
    # Calculate average rating by month and category
    rating_by_month = df.groupby('month')['rating'].mean().reset_index()
    rating_by_category = df.groupby('category')['rating'].mean().reset_index()
    
    # Create Dash app
    app = dash.Dash(__name__)
    
    app.layout = html.Div([
        html.H1("Product Review Sentiment Analysis Dashboard"),
        
        html.Div([
            html.Div([
                html.H3("Filter by Category"),
                dcc.Dropdown(
                    id='category-filter',
                    options=[{'label': category, 'value': category} for category in df['category'].unique()],
                    value='All',
                    clearable=False
                ),
            ], style={'width': '30%', 'display': 'inline-block'}),
            
            html.Div([
                html.H3("Filter by Date Range"),
                dcc.DatePickerRange(
                    id='date-range',
                    min_date_allowed=df['date'].min().date(),
                    max_date_allowed=df['date'].max().date(),
                    start_date=df['date'].min().date(),
                    end_date=df['date'].max().date()
                ),
            ], style={'width': '70%', 'display': 'inline-block'})
        ]),
        
        html.Div([
            html.Div([
                html.H3("Sentiment Trends Over Time"),
                dcc.Graph(id='sentiment-time-trend')
            ], style={'width': '50%', 'display': 'inline-block'}),
            
            html.Div([
                html.H3("Sentiment Distribution by Category"),
                dcc.Graph(id='sentiment-category-dist')
            ], style={'width': '50%', 'display': 'inline-block'})
        ]),
        
        html.Div([
            html.Div([
                html.H3("Average Rating Over Time"),
                dcc.Graph(id='rating-time-trend')
            ], style={'width': '50%', 'display': 'inline-block'}),
            
            html.Div([
                html.H3("Top Features by Sentiment"),
                dcc.Graph(id='feature-sentiment')
            ], style={'width': '50%', 'display': 'inline-block'})
        ]),
        
        html.Div([
            html.H3("Product Sentiment Comparison"),
            dcc.Graph(id='product-comparison')
        ])
    ])
    
    @app.callback(
        [Output('sentiment-time-trend', 'figure'),
         Output('sentiment-category-dist', 'figure'),
         Output('rating-time-trend', 'figure'),
         Output('feature-sentiment', 'figure'),
         Output('product-comparison', 'figure')],
        [Input('category-filter', 'value'),
         Input('date-range', 'start_date'),
         Input('date-range', 'end_date')]
    )
    def update_graphs(selected_category, start_date, end_date):
        # Filter data
        filtered_df = df.copy()
        
        if selected_category != 'All':
            filtered_df = filtered_df[filtered_df['category'] == selected_category]
        
        filtered_