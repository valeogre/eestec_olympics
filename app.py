import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import sqlite3
from streamlit_autorefresh import st_autorefresh
import time

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        padding: 0.5rem 0;
        border-bottom: 2px solid #3498db;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

DB_NAME = "transactions.db"

# Helper function to load and process data
# NOTE: we DO NOT cache this function because the DB is written concurrently
# by another process (detect.py). Caching caused "stale" totals for non-fraud
# rows in the original app.
def load_transaction_data(retries: int = 3, wait_sec: float = 0.2) -> pd.DataFrame:
    """
    Citeste toate tranzactiile din baza SQLite si returneaza un DataFrame curat.
    - foloseste WAL si busy_timeout pentru a permite scrierea concurenta
    - nu este memorat in cache pentru a reflecta imediat modificarile
    """
    attempt = 0
    while attempt < retries:
        try:
            conn = sqlite3.connect(DB_NAME, timeout=5)
            # allow concurrent writer (useful when detect.py writes)
            try:
                conn.execute('PRAGMA journal_mode=WAL;')
            except Exception:
                pass
            conn.execute('PRAGMA busy_timeout = 3000;')

            df = pd.read_sql_query("SELECT * FROM transactions", conn)
            conn.close()

            if df.empty:
                return pd.DataFrame()

            # Normalize / convert columns safely
            # Some rows might have missing unix_time -> coerce to NaT
            if 'unix_time' in df.columns:
                df['unix_time'] = pd.to_numeric(df['unix_time'], errors='coerce')
                df['timestamp'] = pd.to_datetime(df['unix_time'], unit='s', errors='coerce')
            else:
                df['timestamp'] = pd.NaT

            df['amt'] = pd.to_numeric(df.get('amt', 0), errors='coerce').fillna(0)
            df['is_fraud'] = pd.to_numeric(df.get('is_fraud', 0), errors='coerce').fillna(0).astype(int)
            df['lat'] = pd.to_numeric(df.get('lat', np.nan), errors='coerce')
            df['long'] = pd.to_numeric(df.get('long', np.nan), errors='coerce')

            # If datetime is NaT for many rows, attempt to parse trans_date+trans_time
            if df['timestamp'].isna().all() and 'trans_date' in df.columns:
                try:
                    df['timestamp'] = pd.to_datetime(df['trans_date'].astype(str) + ' ' + df.get('trans_time', '').astype(str), errors='coerce')
                except Exception:
                    pass

            # Keep rows even if datetime is NaT (we don't drop them) -- they count towards totals
            # but many dashboard features rely on datetime. For time-limited views we filter.
            print("first 5 rows", df.head())
            return df

        except sqlite3.OperationalError as e:
            attempt += 1
            time.sleep(wait_sec)
            if attempt >= retries:
                st.error(f"Could not read DB after {retries} attempts: {e}")
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Failed to load data from database: {e}")
            return pd.DataFrame()


def filter_last_n_minutes(df, minutes, time_column='timestamp'):
    """Filter dataframe to include only last N minutes"""
    if df.empty or time_column not in df.columns:
        return pd.DataFrame(columns=df.columns)

    current_time = datetime.now()
    cutoff_time = current_time - timedelta(minutes=minutes)

    # If datetime column has NaT values, they will be excluded (they can't be recent)
    return df[df[time_column] >= cutoff_time]


def calculate_fraud_rate(df):
    if len(df) == 0:
        return 0.0
    return (df['is_fraud'].sum() / len(df)) * 100


def get_fraud_patterns(df, hours=1):
    # fraud_df = filter_last_n_minutes(df, hours * 60)
    # fraud_df = fraud_df[fraud_df['is_fraud'] == 1]

    # if fraud_df.empty:
    #     return []

    # patterns = []
    # if len(fraud_df) > 0:
    #     high_amount = fraud_df[fraud_df['amt'] > fraud_df['amt'].quantile(0.75)]
    #     if len(high_amount) > 0:
    #         patterns.append(f"High-value transactions (>${high_amount['amt'].mean():.2f} avg)")

    # if 'city' in fraud_df.columns:
    #     most_common_city = fraud_df['city'].mode()
    #     if len(most_common_city) > 0 and pd.notna(most_common_city[0]):
    #         patterns.append(f"Geographic: {most_common_city[0]}")

    # if 'trans_time' in fraud_df.columns:
    #     fraud_df = fraud_df.copy()
    #     fraud_df['hour'] = fraud_df['datetime'].dt.hour
    #     common_hour = fraud_df['hour'].mode()
    #     if len(common_hour) > 0:
    #         patterns.append(f"Time pattern: {common_hour[0]}:00-{common_hour[0]+1}:00")

    # if 'category' in fraud_df.columns:
    #     top_category = fraud_df['category'].mode()
    #     if len(top_category) > 0 and pd.notna(top_category[0]):
    #         patterns.append(f"Category: {top_category[0]}")

    # return patterns[:5]
    fraud_df = filter_last_n_minutes(df, hours * 60)
    fraud_df = fraud_df[fraud_df['is_fraud'] == 1]
    # fraud_df = df[df['is_fraud'] == 1]

    if fraud_df.empty:
        return []

    patterns = []
    if len(fraud_df) > 0:
        high_amount = fraud_df[fraud_df['amt'] > fraud_df['amt'].quantile(0.75)]
        if len(high_amount) > 0:
            patterns.append(f"High-value transactions (>${high_amount['amt'].mean():.2f} avg)")

    if 'city' in fraud_df.columns:
        most_common_city = fraud_df['city'].mode()
        if len(most_common_city) > 0 and pd.notna(most_common_city[0]):
            patterns.append(f"Geographic: {most_common_city[0]}")

    if 'trans_time' in fraud_df.columns:
        fraud_df = fraud_df.copy()
        fraud_df['hour'] = fraud_df['timestamp'].dt.hour
        common_hour = fraud_df['hour'].mode()
        if len(common_hour) > 0:
            patterns.append(f"Time pattern: {common_hour[0]}:00-{common_hour[0]+1}:00")

    if 'category' in fraud_df.columns:
        top_category = fraud_df['category'].mode()
        if len(top_category) > 0 and pd.notna(top_category[0]):
            patterns.append(f"Category: {top_category[0]}")

    return patterns[:5]


# Main dashboard
def main():
    st.markdown('<div class="main-header">🔍 Real-Time Fraud Detection Dashboard</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        auto_refresh = st.checkbox("Auto-refresh (every 30s)", value=True)
        if auto_refresh:
            st_autorefresh(interval=30000, key="datarefresh")

        if st.button("Force reload now"):
            # Use simple state toggle to force rerun/read
            st.session_state['force_reload'] = not st.session_state.get('force_reload', False)

        st.markdown("---")
        st.info("💡 This dashboard monitors real-time transaction data and detects fraudulent activities.")

    # Load data (no caching)
    df = load_transaction_data()

    if df.empty:
        st.warning("⚠️ No transaction data available. Waiting for data from 'detect.py'...")
        cols = ['trans_time', 'merchant', 'category', 'amt', 'is_fraud', 'city', 'state', 'lat', 'long', 'dob', 'unix_time', 'datetime']
        df = pd.DataFrame(columns=cols)

    # SECTION 1: Recent Activity & Metrics
    st.markdown('<div class="section-header">📊 Recent Activity Overview</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        last_5_min = filter_last_n_minutes(df, 5)
        fraud_last_5_min = last_5_min[last_5_min['is_fraud'] == 1]
        st.metric(
            label="📝 Transactions (Last 5 min)",
            value=len(last_5_min),
            delta=f"{len(fraud_last_5_min)} fraudulent" if len(fraud_last_5_min) > 0 else None,
            # delta = len(last_5_min)
        )
        # data_count = df.shape[0]
        # st.metric(
        #     label="📝 Recent Transactions",
        #     value=data_count,
        #     # delta=f"{len(fraud_last_5_min)} fraudulent" if len(fraud_last_5_min) > 0 else None
        #     # delta = len(data_count)
        # )

    with col2:
        fraud_count = df[df['is_fraud'] == 1].shape[0]
        st.metric(
            label="🚨 Total Frauds Detected",
            value=fraud_count,
            delta=f"{calculate_fraud_rate(df):.2f}% fraud rate",
            delta_color="inverse"
        )

    with col3:
        fraud_rate = calculate_fraud_rate(df)
        st.metric(
            label="📈 Overall Fraud Rate",
            value=f"{fraud_rate:.2f}%",
            delta="Real-time monitoring",
            delta_color="off"
        )

    # Tables
    st.markdown("### 📋 Transaction Tables")
    table_col1, table_col2 = st.columns(2)

    with table_col1:
        st.markdown("**Recent Transactions (Last 5 Minutes)**")
        recent_trans = last_5_min[['trans_date', 'trans_time', 'merchant', 'category', 'amt', 'state', 'is_fraud']].head(10)
        if not recent_trans.empty:
            def highlight_fraud(row):
                return ['background-color: #ffcccc' if row['is_fraud'] == 1 else '' for _ in row]
            st.dataframe(recent_trans.style.apply(highlight_fraud, axis=1), height=300, use_container_width=True)
        else:
            st.info("No transactions in the last 5 minutes")
        # st.markdown("**Recent Transactions (Last 5 Minutes)**")
        # recent_trans = df.sort_values('datetime', ascending=False)
        # recent_trans_display = recent_trans[['trans_time', 'merchant', 'amt', 'city', 'state']].head(10)
        # if not recent_trans_display.empty:
        #     st.dataframe(recent_trans_display, height=300, use_container_width=True)
        # else:
        #     st.success("✅ No transactions detected!")

    with table_col2:
        st.markdown("**Detected Fraudulent Transactions**")
        fraud_trans = df[df['is_fraud'] == 1].sort_values('timestamp', ascending=False)
        fraud_trans_display = fraud_trans[['trans_date', 'trans_time', 'merchant', 'amt', 'city', 'state']].head(10)
        if not fraud_trans_display.empty:
            st.dataframe(fraud_trans_display, height=300, use_container_width=True)
        else:
            st.success("✅ No fraud detected!")

    # Map
    st.markdown('<div class="section-header">🗺️ Transaction Location Map</div>', unsafe_allow_html=True)
    map_df = filter_last_n_minutes(df, 180)

    if not map_df.empty and 'lat' in map_df.columns and 'long' in map_df.columns:
        map_df = map_df.dropna(subset=['lat', 'long'])
        if not map_df.empty:
            center_lat = map_df['lat'].mean()
            center_lon = map_df['long'].mean()
            m = folium.Map(location=[center_lat, center_lon], zoom_start=6)
            for idx, row in map_df.iterrows():
                color = 'red' if row['is_fraud'] == 1 else 'green'
                folium.CircleMarker(
                    location=[row['lat'], row['long']],
                    radius=5,
                    popup=f"""
                        <b>Merchant:</b> {row.get('merchant', 'N/A')}<br>
                        <b>Amount:</b> ${row.get('amt', 0):.2f}<br>
                        <b>Category:</b> {row.get('category', 'N/A')}<br>
                        <b>City:</b> {row.get('city', 'N/A')}<br>
                        <b>Fraud:</b> {'YES' if row.get('is_fraud', 0) == 1 else 'NO'}
                    """,
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.6
                ).add_to(m)
            st_folium(m, width=1400, height=500)
        else:
            st.warning("No recent geographic data available")
    else:
        st.warning("No geographic data available")

    # Fraud Alerts & Stats
    st.markdown('<div class="section-header">⚠️ Fraud Alerts & Statistics</div>', unsafe_allow_html=True)
    alert_col1, alert_col2, alert_col3 = st.columns(3)

    with alert_col1:
        last_2_hours = filter_last_n_minutes(df, 120)
        fraud_2h = last_2_hours[last_2_hours['is_fraud'] == 1]
        st.metric(
            label="🚨 Fraud Alerts (Last 2 Hours)",
            value=len(fraud_2h),
            delta=f"${fraud_2h['amt'].sum():.2f} total value" if len(fraud_2h) > 0 else "No frauds"
        )

    with alert_col2:
        fraud_df = df[df['is_fraud'] == 1]
        if len(fraud_df) > 0 and 'dob' in fraud_df.columns:
            fraud_df_copy = fraud_df.copy()
            fraud_df_copy['dob'] = pd.to_datetime(fraud_df_copy['dob'], errors='coerce')
            fraud_df_copy = fraud_df_copy.dropna(subset=['dob'])
            if not fraud_df_copy.empty:
                fraud_df_copy['age'] = (datetime.now() - fraud_df_copy['dob']).dt.days / 365.25
                avg_age = fraud_df_copy['age'].mean()
                st.metric(
                    label="👤 Avg Age of Fraudsters",
                    value=f"{avg_age:.1f} years" if not np.isnan(avg_age) else "N/A",
                    delta=f"{len(fraud_df_copy)} individuals"
                )
            else:
                st.metric(label="👤 Avg Age of Fraudsters", value="N/A", delta="No valid DOB data")
        else:
            st.metric(label="👤 Avg Age of Fraudsters", value="N/A", delta="No data")

    with alert_col3:
        total_fraud_value = df[df['is_fraud'] == 1]['amt'].sum()
        st.metric(
            label="💰 Total Fraud Value",
            value=f"${total_fraud_value:,.2f}",
            delta="All time",
            delta_color="inverse"
        )

    # Fraud Patterns
    st.markdown('<div class="section-header">🔍 Common Fraud Patterns (Last Hour)</div>', unsafe_allow_html=True)
    patterns = get_fraud_patterns(df, hours=1)
    if patterns:
        for i, pattern in enumerate(patterns, 1):
            st.markdown(f"**{i}.** {pattern}")
    else:
        st.info("No significant fraud patterns detected in the last hour")

    # Category Analysis
    st.markdown('<div class="section-header">📊 Fraud by Category</div>', unsafe_allow_html=True)
    fraud_by_category = df[df['is_fraud'] == 1]['category'].value_counts().head(10)
    if len(fraud_by_category) > 0:
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            fig_bar = px.bar(
                x=fraud_by_category.index,
                y=fraud_by_category.values,
                labels={'x': 'Category', 'y': 'Fraud Count'},
                title='Top Categories with Most Frauds',
                color=fraud_by_category.values,
                color_continuous_scale='Reds'
            )
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        with chart_col2:
            fig_pie = px.pie(
                values=fraud_by_category.values,
                names=fraud_by_category.index,
                title='Fraud Distribution by Category'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No fraud data available for category analysis")

    # Time Series
    st.markdown('<div class="section-header">📈 Fraud Rate Over Time</div>', unsafe_allow_html=True)
    if 'timestamp' in df.columns and not df.empty:
        df_time = df.copy()
        # Keep rows with datetime only for the time-series
        df_time = df_time.dropna(subset=['timestamp'])
        if not df_time.empty:
            df_time['hour'] = df_time['timestamp'].dt.floor('H')
            hourly_stats = df_time.groupby('hour').agg({'is_fraud': ['sum', 'count']}).reset_index()
            hourly_stats.columns = ['hour', 'fraud_count', 'total_count']
            hourly_stats['fraud_rate'] = (hourly_stats['fraud_count'] / hourly_stats['total_count']) * 100

            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(
                x=hourly_stats['hour'],
                y=hourly_stats['fraud_rate'],
                mode='lines+markers',
                name='Fraud Rate (%)',
                line=dict(width=2),
                marker=dict(size=8)
            ))
            fig_time.update_layout(title='Fraud Rate Trend (Hourly)', xaxis_title='Time', yaxis_title='Fraud Rate (%)', hovermode='x unified')
            st.plotly_chart(fig_time, use_container_width=True)

    st.markdown("---")
    st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")


if __name__ == "__main__":
    main()
