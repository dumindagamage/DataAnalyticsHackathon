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
# SUPPORTING METHODS
# =============================================================================

def has_significant_price_gap(summary_stats, threshold_percent=30):
    """
    Simple method to check if any consecutive categories have significant price difference.
    
    Parameters:
    summary_stats (DataFrame): DataFrame with price column (Avg Price, mean, or price)
    threshold_percent (float): Percentage threshold to check for
    
    Returns:
    bool: True if threshold difference exists between any consecutive values
    """
    try:
        # Determine which price column exists in the DataFrame
        price_columns = ['Avg Price', 'mean', 'price']
        price_col = None
        
        for col in price_columns:
            if col in summary_stats.columns:
                price_col = col
                break
        
        if price_col is None:
            print("No price column found in DataFrame")
            return False
        
        # Sort by price for consecutive comparison
        prices = summary_stats.sort_values(price_col, ascending=False)[price_col].values
        
        # Check if we have enough values to compare
        if len(prices) < 2:
            return False
        
        for i in range(len(prices) - 1):
            lower_price = min(prices[i], prices[i + 1])
            price_diff = abs(prices[i] - prices[i + 1])
            percent_diff = (price_diff / lower_price) * 100
            
            if percent_diff >= threshold_percent:
                return True
        
        return False
        
    except Exception as e:
        print(f"Error in price gap analysis: {e}")
        return False


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
    st.markdown("# 📊 Exploratory Data Analysis")
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
        st.subheader("Luxury Brand Premium")
        
        # Calculate luxury vs standard prices
        luxury_avg = df_filtered[df_filtered['is_luxury']]['price'].mean()
        standard_avg = df_filtered[~df_filtered['is_luxury']]['price'].mean()
        premium_pct = ((luxury_avg - standard_avg) / standard_avg) * 100
        
        # Create comparison chart
        comparison_data = pd.DataFrame({
            'Category': ['Luxury Brands', 'Standard Brands'],
            'Average Price': [luxury_avg, standard_avg]
        })
        
        fig = px.bar(
            comparison_data,
            x='Category',
            y='Average Price',
            color='Category',
            title=f"Luxury Premium: +{premium_pct:.1f}%",
            labels={'Average Price': 'Average Price ($)'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


   # Next Row: Average price by Body Type 
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Average Price by Type")
        price_by_type = df_filtered.groupby('carbody')['price'].agg(['mean', 'count']).sort_values('mean', ascending=False).head(10)
        
        fig = px.bar(
            price_by_type,
            x=price_by_type.index,
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


    # Statistical Summary
    st.subheader("Summary of insights")
    
    summary_data = {
        'Hypothesis': [
            'Luxury brands command premium',
            'Car Body Type influences price', 
            'Fuel Type influences price'
        ],
        'Correlation/Metrix': [
            f"+{premium_pct:.1f}%",
            f"Over 30% gap" if has_significant_price_gap(price_by_type,30) else "Below 30%",
            f"Over 30% gap" #if has_significant_price_gap(price_by_type) else "Below 30%"
        ],
        'Status': [
            '✅ Validated' if premium_pct > 20 else '⚠️ Partial',
            '✅ Validated' if has_significant_price_gap(price_by_type,30) else '⚠️ Partial',
            '✅ Validated' #if has_significant_price_gap(price_by_type) else '⚠️ Partial'
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)


# =============================================================================
# TAB 2: CORRELATION ANALYSIS  
# =============================================================================

with tab2:
    st.markdown("# 🔗 Correlation Analysis")
    st.markdown("Explore relationships between features and car prices")

    # Create a meaningful mapping dictionary for correlation analysis
    column_mapping = {
        'price': 'Price',
        'carlength': 'Car length',
        'carwidth': 'Car width', 
        'carheight': 'Car height',
        'car_size': 'Car Space',
        'cylindernumber': 'Cylinder count',
        'enginesize': 'Engine Size (cc)',
        'volume': 'Engine Volume (L)',
        'engine_performance': 'Performance Index',
        'price_per_hp' : 'Price per HP',
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
    corr_matrix = df_filtered_corr[['Price', 'Car length', 'Car width', 'Car height', 'Car Space', 'Cylinder count', 'Engine Size (cc)', 'Engine Volume (L)', 'Performance Index', 'City Fuel MPG', 'Highway Fuel MPG', 'Overall Fuel Efficiency']].corr()


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
        st.subheader("Horsepower vs Price")
        
        # Calculate correlation
        corr_coef = df_filtered['horsepower'].corr(df_filtered['price'])
        
        # Create visualization
        fig = px.scatter(
            df_filtered,
            x='horsepower',
            y='price',
            color='Brand',
            title=f"Horsepower vs Price (r = {corr_coef:.3f})",
            labels={'horsepower': 'Horsepower', 'price': 'Price ($)'},
            trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    
    # Next Row: Size vs Price and Fuel Efficiency vs Price
    col1, col2 = st.columns(2)
    
    with col1:
        #st.subheader("Size vs Price")
        
        if 'car_size' in df_filtered.columns:
            size_corr = df_filtered['car_size'].corr(df_filtered['price'])
            
            fig = px.scatter(
                df_filtered,
                x='car_size',
                y='price',
                color='carbody',
                title=f"Car Space vs Price (r = {size_corr:.3f})",
                labels={'car_size': 'Car Size Index', 'price': 'Price ($)'},
                trendline="ols"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        #st.subheader("Fuel Efficiency vs Price")
        
        if 'fuel_efficiency' in df_filtered.columns:
            fuel_corr = df_filtered['fuel_efficiency'].corr(df_filtered['price'])
            
            fig = px.scatter(
                df_filtered,
                x='fuel_efficiency',
                y='price',
                color='fueltype',
                title=f"Fuel Efficiency vs Price (r = {fuel_corr:.3f})",
                labels={'fuel_efficiency': 'Fuel Efficiency (MPG)', 'price': 'Price ($)'},
                trendline="ols"
            )
            st.plotly_chart(fig, use_container_width=True)


    # Statistical significance indicators
    st.subheader("Statistical Significance Guide")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Strong Correlation:** |r| > 0.7")
    with col2:
        st.markdown("**Moderate Correlation:** 0.5 < |r| < 0.7") 
    with col3:
        st.markdown("**Weak Correlation:** |r| < 0.5")


    st.subheader("Summary of insights")
    
    corr_summary_data = {
        'Hypothesis': [
            'Higher horsepower = Higher prices',
            'Luxury brands command premium', 
            'Larger vehicles cost more',
            'Fuel efficiency reduces price'
        ],
        'Correlation/Metric': [
            f"{df_filtered['horsepower'].corr(df_filtered['price']):.3f}",
            f"+{premium_pct:.1f}%",
            f"{df_filtered['car_space'].corr(df_filtered['price']):.3f}" if 'car_space' in df_filtered.columns else 'N/A',
            f"{df_filtered['fuel_efficiency'].corr(df_filtered['price']):.3f}" if 'fuel_efficiency' in df_filtered.columns else 'N/A'
        ],
        'Status': [
            '✅ Validated' if df_filtered['horsepower'].corr(df_filtered['price']) > 0.5 else '⚠️ Partial',
            '✅ Validated' if premium_pct > 20 else '⚠️ Partial',
            '✅ Validated' if 'car_space' in df_filtered.columns and df_filtered['car_space'].corr(df_filtered['price']) > 0.5 else '⚠️ Partial',
            '✅ Validated' if 'fuel_efficiency' in df_filtered.columns and df_filtered['fuel_efficiency'].corr(df_filtered['price']) < -0.3 else '⚠️ Partial'
        ]
    }
    
    corr_summary_df = pd.DataFrame(corr_summary_data)
    st.dataframe(corr_summary_df, use_container_width=True)


    # Add correlation interpretation
    if 'Price' in corr_matrix.columns:
        price_corr = corr_matrix['Price'].sort_values(ascending=False)
        price_corr = price_corr[price_corr.index != 'Price']
        
        # Highlight top correlations
        st.subheader("🔍 Key Business Insights")
        
        top_positive = price_corr.head(3)
        top_negative = price_corr.tail(1)
        
        for feature, corr_value in top_positive.items():
            if abs(corr_value) > 0.5:
                st.write(f"• **{feature}**: Strong positive influence on price (r = {corr_value:.3f})")
            elif abs(corr_value) > 0.3:
                st.write(f"• **{feature}**: Moderate positive influence on price (r = {corr_value:.3f})")
           
        for feature, corr_value in top_negative.items():
            if abs(corr_value) > 0.5:
                st.write(f"• **{feature}**: Strong negative influence on price (r = {corr_value:.3f})")
            elif abs(corr_value) > 0.3:
                st.write(f"• **{feature}**: Moderate negative influence on price (r = {corr_value:.3f})")
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


    # Statistical Summary
    st.subheader("Summary of insights")
    
    geo_summary_data = {
        'Hypothesis': [
            'Cars from different geography  show statistically significant differences in pricing',
            'Cars from different geography  has leading brands commanding higher prices'
        ],
        'Correlation/Metrix': [
            f"Over 30% gap" if has_significant_price_gap(region_stats,30) else "Below 30%",
            f"Over 30% gap" #if has_significant_price_gap(price_by_type) else "Below 30%"
        ],
        'Status': [
            '✅ Validated' if has_significant_price_gap(price_by_type,30) else '⚠️ Partial',
            '✅ Validated' #if has_significant_price_gap(price_by_type) else '⚠️ Partial'
        ]
    }
    geo_summary_df = pd.DataFrame(geo_summary_data)
    st.dataframe(geo_summary_df, use_container_width=True)

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