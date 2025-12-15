"""
Campaign Performance Dashboard
Bigabid Analytics - Interactive Dashboard for Campaign Monitoring
Author: Shahar Margulis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Campaign Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    .alert-box {
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        background-color: #FFF3CD;
        border-left: 4px solid #F4A261;
    }
    h1 {
        color: #264653;
        padding-bottom: 10px;
        border-bottom: 3px solid #E63946;
    }
    h2 {
        color: #457B9D;
        margin-top: 30px;
    }
    h3 {
        color: #2A9D8F;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# STEP 1: DATA LOADING AND HELPER FUNCTIONS
# ============================================================================

@st.cache_data
def load_data():
    """
    Load unified campaign data from CSV
    Returns: DataFrame with parsed dates
    """
    try:
        df = pd.read_csv('unified_campaign_data.csv')
        df['impression_date'] = pd.to_datetime(df['impression_date'])
        return df
    except FileNotFoundError:
        st.error("❌ Error: unified_campaign_data.csv not found. Please run unify_data.py first.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()


def calculate_metrics(df):
    """
    Calculate all campaign metrics from aggregated data

    Args:
        df: DataFrame with columns: cost, impressions, installs, purchasers, revenue_d7

    Returns:
        Dictionary with calculated metrics
    """
    # Aggregate totals
    total_cost = df['cost'].sum()
    total_impressions = df['impressions'].sum()
    total_installs = df['installs'].sum()
    total_purchasers = df['purchasers'].sum()
    total_revenue = df['revenue_d7'].sum()

    # Calculate metrics (handle division by zero)
    metrics = {
        'cost': total_cost,
        'impressions': total_impressions,
        'installs': total_installs,
        'purchasers': total_purchasers,
        'revenue': total_revenue,
        'CPI': total_cost / total_installs if total_installs > 0 else 0,
        'CTR': (total_installs / total_impressions * 100) if total_impressions > 0 else 0,
        'CVR': (total_purchasers / total_installs * 100) if total_installs > 0 else 0,
        'ROAS': total_revenue / total_cost if total_cost > 0 else 0,
        'RPI': total_revenue / total_installs if total_installs > 0 else 0,
        'eCPM': (total_revenue / total_impressions * 1000) if total_impressions > 0 else 0,
        'Payback_Weeks': total_cost / total_revenue if total_revenue > 0 else 0
    }

    return metrics


def get_comparison_period(df, start_date, end_date):
    """
    Calculate the previous period of equal length for comparison

    Args:
        df: Full DataFrame
        start_date: Start of selected period
        end_date: End of selected period

    Returns:
        tuple: (comparison_start, comparison_end) or (None, None) if not possible
    """
    # Calculate period length
    period_length = (end_date - start_date).days + 1  # +1 to include end date

    # Calculate comparison period end (day before selected start)
    comparison_end = start_date - timedelta(days=1)
    comparison_start = comparison_end - timedelta(days=period_length - 1)

    # Check if comparison period exists in data
    min_date = df['impression_date'].min()

    if comparison_start < min_date:
        return None, None

    return comparison_start, comparison_end


def calculate_percentage_change(current, previous):
    """
    Calculate percentage change between two values

    Args:
        current: Current period value
        previous: Previous period value

    Returns:
        float: Percentage change (can be positive or negative)
    """
    if previous == 0:
        return 0 if current == 0 else 100
    return ((current - previous) / previous) * 100


def format_metric_value(value, metric_name):
    """
    Format metric value for display

    Args:
        value: Numeric value
        metric_name: Name of metric (for appropriate formatting)

    Returns:
        str: Formatted string
    """
    if metric_name in ['CPI', 'RPI', 'cost']:
        return f"${value:,.2f}"
    elif metric_name in ['CVR', 'CTR']:
        return f"{value:.2f}%"
    elif metric_name in ['ROAS', 'Payback_Weeks']:
        return f"{value:.2f}"
    elif metric_name in ['installs', 'impressions', 'purchasers']:
        return f"{int(value):,}"
    else:
        return f"{value:,.2f}"


# ============================================================================
# LOAD DATA
# ============================================================================

# Load the unified dataset
df_raw = load_data()

# Get date range from data
min_date = df_raw['impression_date'].min().date()
max_date = df_raw['impression_date'].max().date()

# Display header
st.title("📊 Campaign Performance Dashboard")
st.markdown(f"**Bigabid Analytics** | Data Range: {min_date} to {max_date}")
st.markdown("---")


# ============================================================================
# STEP 2: SIDEBAR CONTROLS
# ============================================================================

with st.sidebar:
    st.header("⚙️ Dashboard Controls")

    # Date Range Selector
    st.subheader("📅 Select Time Period")

    # Default to last 7 days
    default_end = max_date
    default_start = max_date - timedelta(days=6)

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=default_start,
            min_value=min_date,
            max_value=max_date
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=default_end,
            min_value=min_date,
            max_value=max_date
        )

    # Validate date range
    if start_date > end_date:
        st.error("❌ Start date must be before end date")
        st.stop()

    # Calculate period info
    period_days = (end_date - start_date).days + 1
    st.info(f"📊 Selected period: **{period_days} days**")

    st.markdown("---")

    # Filters Section
    st.subheader("🎯 Filters")

    # Get unique values for filters
    all_audiences = sorted(df_raw['audience'].unique())
    all_formats = sorted(df_raw['ad_format'].unique())

    # Audience filter
    selected_audiences = st.multiselect(
        "Audience Tiers",
        options=all_audiences,
        default=all_audiences,
        help="Select one or more audience tiers to analyze"
    )

    # Ad Format filter
    selected_formats = st.multiselect(
        "Ad Formats",
        options=all_formats,
        default=all_formats,
        help="Select one or more ad formats to analyze"
    )

    if not selected_audiences or not selected_formats:
        st.warning("⚠️ Please select at least one audience and format")
        st.stop()

    st.markdown("---")

    # Show breakdown option
    st.subheader("📋 Display Options")
    show_breakdown = st.checkbox(
        "Show detailed breakdown",
        value=False,
        help="Show metrics broken down by audience and ad format"
    )

    st.markdown("---")

    # Info box
    st.info("""
    💡 **How to use:**
    1. Select date range
    2. Choose filters (audiences & formats)
    3. View aggregated metrics
    4. Enable breakdown for details
    """)


# ============================================================================
# STEP 3: DATA PROCESSING
# ============================================================================

# Convert dates to datetime for filtering
start_datetime = pd.Timestamp(start_date)
end_datetime = pd.Timestamp(end_date)

# Filter data by selected period and filters
df_selected = df_raw[
    (df_raw['impression_date'] >= start_datetime) &
    (df_raw['impression_date'] <= end_datetime) &
    (df_raw['audience'].isin(selected_audiences)) &
    (df_raw['ad_format'].isin(selected_formats))
].copy()

# Check if we have data
if df_selected.empty:
    st.error("❌ No data available for selected filters and date range")
    st.stop()

# Calculate metrics for selected period
selected_metrics = calculate_metrics(df_selected)

# Get comparison period
comp_start, comp_end = get_comparison_period(df_raw, start_datetime, end_datetime)

# Calculate comparison metrics if available
comparison_metrics = None
has_comparison = False

if comp_start is not None and comp_end is not None:
    df_comparison = df_raw[
        (df_raw['impression_date'] >= comp_start) &
        (df_raw['impression_date'] <= comp_end) &
        (df_raw['audience'].isin(selected_audiences)) &
        (df_raw['ad_format'].isin(selected_formats))
    ].copy()

    if not df_comparison.empty:
        comparison_metrics = calculate_metrics(df_comparison)
        has_comparison = True

# Calculate tier distribution for tier mix chart
tier_distribution = df_selected.groupby('audience').agg({
    'installs': 'sum',
    'cost': 'sum'
}).reset_index()

# Prepare breakdown data if requested (for detailed table at bottom)
breakdown_by_audience = []
for audience in selected_audiences:
    df_group = df_selected[df_selected['audience'] == audience]
    if not df_group.empty:
        metrics = calculate_metrics(df_group)
        metrics['Dimension'] = 'Audience'
        metrics['Group'] = audience
        breakdown_by_audience.append(metrics)

breakdown_by_format = []
for fmt in selected_formats:
    df_group = df_selected[df_selected['ad_format'] == fmt]
    if not df_group.empty:
        metrics = calculate_metrics(df_group)
        metrics['Dimension'] = 'Ad Format'
        metrics['Group'] = fmt
        breakdown_by_format.append(metrics)


# ============================================================================
# STEP 4: EXECUTIVE KPI CARDS
# ============================================================================

st.subheader("📈 Executive Summary")

# Display comparison period info if available
if has_comparison:
    st.caption(f"🔴 **Selected Period:** {start_date} to {end_date} | 🔵 **Comparison Period:** {comp_start.date()} to {comp_end.date()}")
else:
    st.caption(f"🔴 **Selected Period:** {start_date} to {end_date} (No comparison period available)")

# Create 4 KPI cards
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

# KPI 1: CPI
with kpi_col1:
    cpi_value = selected_metrics['CPI']
    if has_comparison:
        cpi_change = calculate_percentage_change(cpi_value, comparison_metrics['CPI'])
        st.metric(
            label="Cost Per Install (CPI)",
            value=f"${cpi_value:.2f}",
            delta=f"{cpi_change:+.1f}%",
            delta_color="inverse"  # Lower is better for CPI
        )
    else:
        st.metric(
            label="Cost Per Install (CPI)",
            value=f"${cpi_value:.2f}"
        )

# KPI 2: CVR
with kpi_col2:
    cvr_value = selected_metrics['CVR']
    if has_comparison:
        cvr_change = calculate_percentage_change(cvr_value, comparison_metrics['CVR'])
        st.metric(
            label="Conversion Rate (CVR)",
            value=f"{cvr_value:.2f}%",
            delta=f"{cvr_change:+.1f}%",
            delta_color="normal"  # Higher is better for CVR
        )
    else:
        st.metric(
            label="Conversion Rate (CVR)",
            value=f"{cvr_value:.2f}%"
        )

# KPI 3: ROAS
with kpi_col3:
    roas_value = selected_metrics['ROAS']
    if has_comparison:
        roas_change = calculate_percentage_change(roas_value, comparison_metrics['ROAS'])
        st.metric(
            label="Return on Ad Spend (ROAS)",
            value=f"{roas_value:.3f}",
            delta=f"{roas_change:+.1f}%",
            delta_color="normal"  # Higher is better for ROAS
        )
    else:
        st.metric(
            label="Return on Ad Spend (ROAS)",
            value=f"{roas_value:.3f}"
        )

# KPI 4: Installs
with kpi_col4:
    installs_value = selected_metrics['installs']
    if has_comparison:
        installs_change = calculate_percentage_change(installs_value, comparison_metrics['installs'])
        st.metric(
            label="Total Installs",
            value=f"{int(installs_value):,}",
            delta=f"{installs_change:+.1f}%",
            delta_color="normal"  # Higher is better for installs
        )
    else:
        st.metric(
            label="Total Installs",
            value=f"{int(installs_value):,}"
        )

st.markdown("---")


# ============================================================================
# STEP 5: ALERT PANEL
# ============================================================================

# Check for threshold violations
alerts = []

# Overall CPI threshold
if selected_metrics['CPI'] > 18:
    alerts.append(f"⚠️ Overall CPI (${selected_metrics['CPI']:.2f}) is above $18 threshold")

# Check tier-specific metrics for alerts
for audience in selected_audiences:
    df_tier = df_selected[df_selected['audience'] == audience]
    if not df_tier.empty:
        tier_metrics = calculate_metrics(df_tier)

        # Tier 2 specific alerts
        if audience == "Tier 2":
            if tier_metrics['CPI'] > 65:
                alerts.append(f"⚠️ Tier 2 CPI (${tier_metrics['CPI']:.2f}) is above $65 threshold")

            # Daily spend check for Tier 2
            tier2_daily_avg = tier_metrics['cost'] / period_days
            if tier2_daily_avg > 15000:
                alerts.append(f"⚠️ Tier 2 daily spend (${tier2_daily_avg:,.0f}) is above $15K threshold")

        # Tier 3 specific alerts
        if audience == "Tier 3":
            if tier_metrics['CVR'] < 1.0:
                alerts.append(f"⚠️ Tier 3 CVR ({tier_metrics['CVR']:.2f}%) is below 1% threshold")

            # Tier 3 share check
            tier3_installs = tier_metrics['installs']
            total_installs = selected_metrics['installs']
            tier3_share = (tier3_installs / total_installs * 100) if total_installs > 0 else 0

            if tier3_share > 80:
                alerts.append(f"⚠️ Tier 3 share ({tier3_share:.1f}%) is above 80% threshold")

# Daily spend check
daily_avg_spend = selected_metrics['cost'] / period_days
if daily_avg_spend > 25000:
    alerts.append(f"⚠️ Daily average spend (${daily_avg_spend:,.0f}) is above $25K threshold")

# Display alerts if any
if alerts:
    st.subheader("🔔 Performance Alerts")
    for alert in alerts:
        st.markdown(f'<div class="alert-box">{alert}</div>', unsafe_allow_html=True)
    st.markdown("---")


# ============================================================================
# STEP 6: TIER MIX VISUALIZATION
# ============================================================================

st.subheader("🎯 Audience Tier Distribution")

col_pie1, col_pie2 = st.columns(2)

# Tier mix by installs
with col_pie1:
    fig_tier_installs = px.pie(
        tier_distribution,
        values='installs',
        names='audience',
        title='Distribution by Installs',
        color='audience',
        color_discrete_map={
            'Tier 1': '#264653',
            'Tier 2': '#E76F51',
            'Tier 3': '#2A9D8F'
        }
    )
    fig_tier_installs.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_tier_installs, use_container_width=True)

# Tier mix by spend
with col_pie2:
    fig_tier_spend = px.pie(
        tier_distribution,
        values='cost',
        names='audience',
        title='Distribution by Spend',
        color='audience',
        color_discrete_map={
            'Tier 1': '#264653',
            'Tier 2': '#E76F51',
            'Tier 3': '#2A9D8F'
        }
    )
    fig_tier_spend.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_tier_spend, use_container_width=True)

st.markdown("---")


# ============================================================================
# STEP 7: CPI TREND OVER TIME
# ============================================================================

st.subheader("📉 CPI Trend Over Time")

# Calculate daily CPI for all data (for context)
df_daily_all = df_raw[
    (df_raw['audience'].isin(selected_audiences)) &
    (df_raw['ad_format'].isin(selected_formats))
].groupby('impression_date').agg({
    'cost': 'sum',
    'installs': 'sum'
}).reset_index()

df_daily_all['CPI'] = df_daily_all['cost'] / df_daily_all['installs']

# Create figure
fig_cpi_trend = go.Figure()

# Add baseline data (grey)
if has_comparison:
    df_baseline = df_daily_all[
        (df_daily_all['impression_date'] >= comp_start) &
        (df_daily_all['impression_date'] <= comp_end)
    ]
    fig_cpi_trend.add_trace(go.Scatter(
        x=df_baseline['impression_date'],
        y=df_baseline['CPI'],
        mode='lines+markers',
        name='Comparison Period',
        line=dict(color='#457B9D', width=2),
        marker=dict(size=6)
    ))

# Add selected period data (red)
df_selected_daily = df_daily_all[
    (df_daily_all['impression_date'] >= start_datetime) &
    (df_daily_all['impression_date'] <= end_datetime)
]
fig_cpi_trend.add_trace(go.Scatter(
    x=df_selected_daily['impression_date'],
    y=df_selected_daily['CPI'],
    mode='lines+markers',
    name='Selected Period',
    line=dict(color='#E63946', width=3),
    marker=dict(size=8)
))

# Add shaded regions
if has_comparison:
    fig_cpi_trend.add_vrect(
        x0=comp_start, x1=comp_end,
        fillcolor="#457B9D", opacity=0.1,
        layer="below", line_width=0,
        annotation_text="Comparison", annotation_position="top left"
    )

fig_cpi_trend.add_vrect(
    x0=start_datetime, x1=end_datetime,
    fillcolor="#E63946", opacity=0.1,
    layer="below", line_width=0,
    annotation_text="Selected", annotation_position="top left"
)

fig_cpi_trend.update_layout(
    title='Daily CPI Evolution',
    xaxis_title='Date',
    yaxis_title='CPI ($)',
    hovermode='x unified',
    height=400
)

st.plotly_chart(fig_cpi_trend, use_container_width=True)

st.markdown("---")


# ============================================================================
# STEP 8: MULTI-METRIC GRID
# ============================================================================

st.subheader("📊 Multi-Metric Comparison")

# Prepare data for comparison chart
metrics_to_compare = ['CPI', 'CVR', 'ROAS', 'RPI', 'CTR', 'Payback_Weeks']
metric_labels = {
    'CPI': 'Cost Per Install ($)',
    'CVR': 'Conversion Rate (%)',
    'ROAS': 'Return on Ad Spend',
    'RPI': 'Revenue Per Install ($)',
    'CTR': 'Click-Through Rate (%)',
    'Payback_Weeks': 'Payback Period (Weeks)'
}

# Create 2x3 subplot grid
from plotly.subplots import make_subplots

fig_metrics = make_subplots(
    rows=2, cols=3,
    subplot_titles=[metric_labels[m] for m in metrics_to_compare],
    vertical_spacing=0.15,
    horizontal_spacing=0.1
)

# Add each metric
for idx, metric in enumerate(metrics_to_compare):
    row = (idx // 3) + 1
    col = (idx % 3) + 1

    # Prepare data
    if has_comparison:
        categories = ['Comparison', 'Selected']
        values = [comparison_metrics[metric], selected_metrics[metric]]
        colors = ['#457B9D', '#E63946']
    else:
        categories = ['Selected']
        values = [selected_metrics[metric]]
        colors = ['#E63946']

    fig_metrics.add_trace(
        go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            showlegend=False,
            text=[f'{v:.2f}' for v in values],
            textposition='outside'
        ),
        row=row, col=col
    )

fig_metrics.update_layout(
    height=600,
    showlegend=False,
    title_text="Key Metrics: Selected vs Comparison Period"
)

st.plotly_chart(fig_metrics, use_container_width=True)

st.markdown("---")


# ============================================================================
# STEP 9: DETAILED METRICS TABLE
# ============================================================================

if show_breakdown:
    st.subheader("📋 Detailed Breakdown")

    # Combine audience and format breakdowns
    all_breakdown = breakdown_by_audience + breakdown_by_format

    if all_breakdown:
        breakdown_df = pd.DataFrame(all_breakdown)

        # Prepare display DataFrame
        display_columns = ['Dimension', 'Group', 'CPI', 'CVR', 'ROAS', 'RPI', 'CTR', 'installs', 'cost', 'revenue']
        display_df = breakdown_df[display_columns].copy()

        # Format columns for better display
        display_df['CPI'] = display_df['CPI'].apply(lambda x: f"${x:.2f}")
        display_df['CVR'] = display_df['CVR'].apply(lambda x: f"{x:.2f}%")
        display_df['ROAS'] = display_df['ROAS'].apply(lambda x: f"{x:.3f}")
        display_df['RPI'] = display_df['RPI'].apply(lambda x: f"${x:.2f}")
        display_df['CTR'] = display_df['CTR'].apply(lambda x: f"{x:.2f}%")
        display_df['installs'] = display_df['installs'].apply(lambda x: f"{int(x):,}")
        display_df['cost'] = display_df['cost'].apply(lambda x: f"${x:,.2f}")
        display_df['revenue'] = display_df['revenue'].apply(lambda x: f"${x:,.2f}")

        # Rename columns for display
        display_df.columns = ['Dimension', 'Group', 'CPI', 'CVR (%)', 'ROAS', 'RPI', 'CTR (%)', 'Installs', 'Total Cost', 'Total Revenue']

        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No breakdown data available for current selection")

    st.markdown("---")


# ============================================================================
# STEP 10: EXPORT FUNCTIONALITY
# ============================================================================

# Add export button to sidebar
with st.sidebar:
    st.markdown("---")
    st.subheader("💾 Export Data")

    # Prepare export data (aggregated metrics)
    export_data = {
        'period_start': [start_date],
        'period_end': [end_date],
        'period_days': [period_days],
        'filters_audiences': [', '.join(selected_audiences)],
        'filters_formats': [', '.join(selected_formats)],
        **selected_metrics
    }
    export_df = pd.DataFrame([export_data])

    # Convert to CSV
    csv_data = export_df.to_csv(index=False)

    st.download_button(
        label="📥 Download CSV",
        data=csv_data,
        file_name=f"campaign_metrics_{start_date}_to_{end_date}.csv",
        mime="text/csv",
        help="Download aggregated metrics as CSV file"
    )

# Footer
st.markdown("---")
st.caption("📊 Campaign Performance Dashboard | Built with Streamlit | Data source: Bigabid DA Test")

