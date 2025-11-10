# ============================================================================
# STREAMLIT DASHBOARD - CAR PRICE ANALYSIS
# Enhanced with Descriptive, Correlation & Geographic Analysis
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import os

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Car Price Analysis Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .main-title { font-size: 2.5em; font-weight: bold; color: #1f77b4; }
        .section-title { font-size: 1.8em; font-weight: bold; color: #2c3e50; margin-top: 20px; }
        .metric-box { background-color: #f0f2f6; padding: 15px; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD DATA
# =============================================================================

@st.cache_data
def load_data():
    # Get the absolute path to your CSV file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, '..', 'Data', 'Cleaned_Car_Price.csv')

    # Read the file
    df = pd.read_csv(csv_path)
    return df

df = load_data()

# =============================================================================
# SIDEBAR FILTERS
# =============================================================================

st.sidebar.markdown("### 🔧 FILTERS")
st.sidebar.markdown("---")

# Company filter
brands = sorted(df['Brand'].unique())
selected_brands = st.sidebar.multiselect(
    "Select Car Brands",
    brands,
    default=brands #[:5]
)

# Region filter
regions = sorted(df['region'].unique())
selected_regions = st.sidebar.multiselect(
    "Select Regions",
    regions,
    default=regions
)

# Body type filter
body_types = sorted(df['carbody'].unique())
selected_body_types = st.sidebar.multiselect(
    "Select Body Types",
    body_types,
    default=body_types
)

# Drive wheel filter
drive_wheels = sorted(df['drivewheel'].unique())
selected_drive_wheels = st.sidebar.multiselect(
    "Select Drive Wheels",
    drive_wheels,
    default=drive_wheels
)

# Price range filter - FIXED: Unpack the tuple
price_range = st.sidebar.slider(
    "Price Range ($)",
    float(df['price'].min()),
    float(df['price'].max()),
    (float(df['price'].min()), float(df['price'].max()))
)

# Apply filters - FIXED: Properly unpack price_range tuple
min_price, max_price = price_range  # Unpack the tuple

df_filtered = df[
    (df['Brand'].isin(selected_brands)) &
    (df['region'].isin(selected_regions)) &
    (df['carbody'].isin(selected_body_types)) &
    (df['drivewheel'].isin(selected_drive_wheels)) &
    (df['price'] >= min_price) &  # Use min_price
    (df['price'] <= max_price)    # Use max_price
]

st.sidebar.markdown("---")
st.sidebar.metric("Filtered Records", len(df_filtered))
st.sidebar.metric("Avg Price", f"${df_filtered['price'].mean():,.0f}")

# =============================================================================
# MAIN DASHBOARD - TABS
# =============================================================================

tab1, tab2, tab3 = st.tabs([
    "📊 Descriptive Statistics",
    "🔗 Correlation Analysis", 
    "🌍 Geographic Analysis"
])

# =============================================================================
# TAB 1: DESCRIPTIVE STATISTICS
# =============================================================================

with tab1:
    st.markdown("# 📊 Descriptive Statistics Analysis")
    st.markdown("Analyze average car prices by Brand and Type")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Cars", len(df_filtered))
    with col2:
        st.metric("Avg Price", f"${df_filtered['price'].mean():,.0f}")
    with col3:
        st.metric("Median Price", f"${df_filtered['price'].median():,.0f}")
    with col4:
        st.metric("Price Std Dev", f"${df_filtered['price'].std():,.0f}")
    
    st.markdown("---")
    
    # Average price by Brand
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Average Price by Brand")
        price_by_company = df_filtered.groupby('Brand')['price'].agg(['mean', 'count']).sort_values('mean', ascending=False).head(10)
        
        fig = px.bar(
            price_by_company,
            x=price_by_company.index,
            y='mean',
            title="Top 10 Most Expensive Brands",
            labels={'mean': 'Average Price ($)', 'Brand': 'Car Brand'},
            hover_data={'mean': ':.0f', 'count': True}
        )
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 10 LEAST Expensive Makes
        st.subheader(" ")
        least_expensive = df_filtered.groupby('Brand')['price'].agg(['mean', 'count']).sort_values('mean').head(10)
        fig = px.bar(
            least_expensive,
            x=least_expensive.index,
            y='mean',
            title="10 Least Expensive Brands",
            labels={'mean': 'Average Price ($)', 'Brand': 'Car Brand'},
            hover_data={'mean': ':.0f', 'count': True}
        )
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)


   # Average price by Body Type 
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Average Price by Type")
        price_by_company = df_filtered.groupby('carbody')['price'].agg(['mean', 'count']).sort_values('mean', ascending=False).head(10)
        
        fig = px.bar(
            price_by_company,
            x=price_by_company.index,
            y='mean',
            title="Price by Types",
            labels={'mean': 'Average Price ($)', 'carbody': 'Car Body Type'},
            hover_data={'mean': ':.0f', 'count': True}
        )
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Car Body Type vs Price Analysis")
        least_expensive = df_filtered.groupby('carbody')['price'].agg(['mean', 'count']).sort_values('mean').head(10)

        fig = px.scatter(
            least_expensive,
            x=least_expensive.index,
            y='mean',
            title="Average price per Car Body Types",
            labels={'mean': 'Average Price ($)', 'index': 'Car Body Type'},
            size='count',  # Use count for bubble size
            hover_data={'mean': ':.0f', 'count': True}
        )
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)



    # Detailed statistics table
    st.subheader("Descriptive Statistics by Brand")
    stats_columns = {
        'price': ['count', 'mean', 'median', 'min', 'max', 'std'],
        'horsepower': 'mean',
        'enginesize': 'mean'
    }

    
    # Add fuel_efficiency only if it exists
    if 'fuel_efficiency' in df_filtered.columns:
        stats_columns['fuel_efficiency'] = 'mean'
    
    stats_table = df_filtered.groupby('Brand').agg(stats_columns).round(2)
    
    # Flatten column names
    stats_table.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in stats_table.columns]
    
    # Rename columns for better readability
    column_rename = {
        'price_count': 'Count',
        'price_mean': 'Avg Price', 
        'price_median': 'Median Price',
        'price_min': 'Min Price',
        'price_max': 'Max Price',
        'price_std': 'Std Dev',
        'horsepower_mean': 'Avg HP',
        'enginesize_mean': 'Avg Engine'
    }
    
    if 'fuel_efficiency_mean' in stats_table.columns:
        column_rename['fuel_efficiency_mean'] = 'Avg MPG'
    
    stats_table = stats_table.rename(columns=column_rename)
    st.dataframe(stats_table, use_container_width=True)

# =============================================================================
# TAB 2: CORRELATION ANALYSIS  
# =============================================================================

with tab2:
    st.markdown("# 🔗 Correlation Analysis")
    st.markdown("Explore relationships between features and car prices")

    # Create a meaningful mapping dictionary for correlation analysis
    column_mapping = {
        'price': 'Price',
        'carlength': 'Vehicle length',
        'carwidth': 'Vehicle width', 
        'carheight': 'Vehicle height',
        'car_size': 'Vehicle Size Index',
        'cylindernumber': 'Cylinder count',
        'enginesize': 'Engine Size (cc)',
        'volume': 'Engine Volume (L)',
        'engine_performance': 'Engine Performance',
        'citympg': 'City Fuel MPG',
        'highwaympg': 'Highway Fuel MPG',
        'fuel_efficiency': 'Overall Fuel Efficiency'
    }

    # Apply the renaming
    df_filtered_corr = df_filtered.rename(columns=column_mapping)

    # Scatter plots for top correlations
    st.subheader("Feature-Price Relationships")

    # Calculate correlation matrix
    # numerical_cols = df_filtered.select_dtypes(include=[np.number]).columns
    # numerical_cols = numerical_cols['']  # Selected columns for correlation
    #corr_matrix = df_filtered[['price','carlength', 'carwidth', 'carheight', 'car_size', 'cylindernumber', 'enginesize', 'volume', 'engine_performance', 'citympg', 'highwaympg', 'fuel_efficiency']].corr()
    corr_matrix = df_filtered_corr[['Price', 'Vehicle length', 'Vehicle width', 'Vehicle height', 'Vehicle Size Index', 'Cylinder count', 'Engine Size (cc)', 'Engine Volume (L)', 'Engine Performance', 'City Fuel MPG', 'Highway Fuel MPG', 'Overall Fuel Efficiency']].corr()


    # Correlation Heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.round(2).values,
        texttemplate='%{text:.2f}',
        textfont={"size": 8}
    ))
    fig.update_layout(height=500, width=500)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        #st.subheader("Feature Correlation with Price")
        
        if 'Price' in corr_matrix.columns:
            price_corr = corr_matrix['Price'].sort_values(ascending=False)
            # Remove price itself and get top 10
            price_corr = price_corr[price_corr.index != 'Price'].head(20)
            
            fig = px.bar(
                x=price_corr.values,
                y=price_corr.index,
                orientation='h',
                title="Top Features Correlated with Price",
                labels={'x': 'Correlation Coefficient', 'y': 'Feature'},
                color=price_corr.values,
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Price column not available for correlation analysis")
    
    with col2:
        if 'Engine Performance' in df_filtered_corr.columns:
            fig = px.scatter(
                df_filtered_corr,
                x='Engine Performance',
                y='Price',
                color='wheelbase',
                title="Engine Performance vs Price",
                labels={'Price': 'Price ($)', 'Engine Performance': 'Engine Performance'}
            )
            st.plotly_chart(fig, use_container_width=True)


    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Vehicle Size Index' in df_filtered_corr.columns:
            fig = px.scatter(
                df_filtered_corr,
                x='Vehicle Size Index',
                y='Price',
                color='wheelbase',
                title="Vehicle Size vs Price",
                labels={'Price': 'Price ($)', 'Vehicle Size Index': 'Vehicle Size'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'Overall Fuel Efficiency' in df_filtered_corr.columns:
            fig = px.scatter(
                df_filtered_corr,
                x='Overall Fuel Efficiency',
                y='Price',
                color='wheelbase', 
                title="Fuel Efficiency vs Price",
                labels={'Price': 'Price ($)', 'Overall Fuel Efficiency': 'Fuel Efficiency (MPG)'}
            )
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 3: GEOGRAPHIC ANALYSIS
# =============================================================================

with tab3:
    st.markdown("# 🌍 Geographic Analysis")
    st.markdown("Compare prices across manufacturing regions and markets")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Average Price by Region")
        
        region_stats = df_filtered.groupby('region').agg({
            'price': ['mean', 'count'],
            'Brand': 'nunique'
        }).round(2)
        region_stats.columns = ['Avg Price', 'Cars', 'Brands']
        
        fig = px.bar(
            x=region_stats.index,
            y=region_stats['Avg Price'],
            title="Average Price by Region",
            labels={'x': 'Region', 'y': 'Average Price ($)'},
            color=region_stats['Avg Price'],
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Market Composition by Make")
        
        # Use available category column
        region_segment = pd.crosstab(df_filtered['region'], df_filtered['Brand'], normalize='index') * 100
        
        fig = px.bar(
            region_segment,
            x=region_segment.index,
            y=region_segment.columns,
            title=f"Market Composition by Region Make %",
            labels={'value': 'Percentage (%)', 'index': 'Region'},
            barmode='stack'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Regional statistics table
    st.subheader("Regional Market Statistics")
    
    geo_stats_agg = {
        'price': ['count', 'mean', 'min', 'max', 'std'],
        'Brand': 'nunique',
        'horsepower': 'mean',
        'enginesize': 'mean'
    }
    
    # Add fuel efficiency if available
    if 'fuel_efficiency' in df_filtered.columns:
        geo_stats_agg['fuel_efficiency'] = 'mean'
    
    geo_stats = df_filtered.groupby('region').agg(geo_stats_agg).round(2)
    
    # Flatten and rename columns
    geo_stats.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in geo_stats.columns]
    geo_stats = geo_stats.rename(columns={
        'price_count': 'Count',
        'price_mean': 'Avg Price',
        'price_min': 'Min Price',
        'price_max': 'Max Price', 
        'price_std': 'Std Dev',
        'car_company_nunique': 'Brands',
        'horsepower_mean': 'Avg HP',
        'enginesize_mean': 'Avg Engine'
    })
    
    if 'fuel_efficiency_mean' in geo_stats.columns:
        geo_stats = geo_stats.rename(columns={'fuel_efficiency_mean': 'Avg MPG'})
    
    st.dataframe(geo_stats, use_container_width=True)

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #888;'>
        <p>🚗 Car Price Analysis Dashboard</p>
        <p>Developed for hackathon data analysis project</p>
    </div>
""", unsafe_allow_html=True)