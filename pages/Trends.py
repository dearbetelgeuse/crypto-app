import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from binance.client import Client
from PIL import Image
import matplotlib.dates as mdates
from fastapi import FastAPI

app = FastAPI()

# Binance API credentials
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')

# Initialize Binance client
client = Client(API_KEY, API_SECRET)

# List of crypto pairs
crypto_pairs = [ 'BTCUSDT', 'USDTTRY', 'USDTARS', 'USDTCOP', 'ETHUSDT', 'USDCUSDT',
                'FDUSDUSDT', 'XRPUSDT', 'SOLUSDT', 'DOGEUSDT', 'PENGUUSDT', 'BNBUSDT',
                'MOVEUSDT', 'PEPEUSDT', 'USUALUSDT', 'FDUSDTRY', 'ZENUSDT', 'SUIUSDT',
                'USDTBRL', 'HBARUSDT', 'TRXUSDT', 'VANAUSDT', 'PHAUSDT', 'ENAUSDT',
                'ADAUSDT', 'MEUSDT', 'LINKUSDT', 'VIBUSDT', 'AAVEUSDT', 'AVAXUSDT']

# Function to fetch historical data
def get_binance_data(symbol, interval, lookback):
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=lookback)
    klines = client.get_historical_klines(symbol, interval, start_time.strftime("%d %b %Y %H:%M:%S"),
                                          end_time.strftime("%d %b %Y %H:%M:%S"))
    df = pd.DataFrame(klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
                                       'Close time', 'Quote asset volume', 'Number of trades',
                                       'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
    df['Close'] = df['Close'].astype(float)
    return df[['Open time', 'Close']]

# Function to run regression and categorize trends
def categorize_trends(crypto_pairs, interval, lookback):
    results = []
    for pair in crypto_pairs:
        try:
            df = get_binance_data(pair, interval, lookback)
            df['time_index'] = range(len(df))
            X = sm.add_constant(df['time_index'])
            y = df['Close']
            model = sm.OLS(y, X).fit()
            trend = model.params['time_index']

            if trend > 0.001:
                category = 'Positive Trend'
            elif trend < -0.001:
                category = 'Negative Trend'
            else:
                category = 'Flat'

            results.append({
                'Symbol': pair,
                'Trend Coefficient': trend,
                'Category': category,
                'R-squared': model.rsquared
            })
        except Exception:
            results.append({
                'Symbol': pair,
                'Trend Coefficient': None,
                'Category': 'Error',
                'R-squared': None
            })

    return pd.DataFrame(results)

# Streamlit App
def main():
    st.set_page_config(layout="wide", page_title= 'Trends', page_icon="📈")
    
    st.sidebar.success("Select a page above to navigate.")
    st.sidebar.title("Enter details")
    # Sidebar setup
    #st.sidebar.image("Pic1.png", use_column_width=True)
    symbol_to_visualize = st.sidebar.selectbox("Crypto Pair", crypto_pairs)
    interval = st.sidebar.selectbox("Interval", ['1m', '5m', '15m', '1h', '1d'])
    lookback = st.sidebar.slider("Lookback (days)", min_value=1, max_value=30, step=1)
    
    
    st.markdown("""
    <style>
        /* Remove extra top padding from the main page */
        .block-container {
            padding-top: 3rem !important;
            padding-bottom: 0rem !important;

        /* Force sidebar to move to the very top */
        section[data-testid="stSidebar"] {
            padding-top: 0px !important;
            margin-top: 0px !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Create placeholders for chart & data
    st.image("banner.png", use_container_width=True)
    st.markdown(f"<h5 style='text-align: Left; margin-bottom: -300px;'>Crypto Regression Analysis (OLS) for <span style='color: blue;'><b>{symbol_to_visualize}</b></span></h5>", unsafe_allow_html=True)
    st.markdown("<style>.block-container {padding-top: 0;}</style>", unsafe_allow_html=True)

    colx, coly = st.columns(2)
    with colx:
        st.markdown("<div style='display:none;>Hidden Row</div>", unsafe_allow_html=True)
        st.empty()

    with coly:
        st.markdown("<div style='display:none;'>Hidden Row</div>", unsafe_allow_html=True)
        st.empty()

    col1, col2 = st.columns(2)
    
    st.markdown(
    """
    <style>
        .dataframe-container {
            width: 100% !important;  /* Forces full width */
            overflow-x: auto;  /* Enables horizontal scrolling only if necessary */
            font-size: 12px !important;  /* Makes text more compact */
            margin-bottom: 0px !important;  /* Reduces extra spacing */
        }
        .dataframe-container table {
            width: 100% !important;
        }
        .block-container {
            padding-left: 32px !important;  /* Reduces left padding */
            padding-right: 32px !important; /* Reduces right padding */
        }
    </style>
    """,
    unsafe_allow_html=True
    )
    with col1:
        st.markdown("<h6 style='text-align: left;padding-top: 16px;'>Regression Results</h6>", unsafe_allow_html=True)
        df_placeholder = st.empty()  # Placeholder for regression table

    with col2:
        st.markdown(f"<h6 style='text-align: left;padding-top: 16px;'>Scatterplot of <span style='color: blue;'><b>{interval}</b></span> Trend</h6>", unsafe_allow_html=True)
        chart_placeholder = st.empty()  # Placeholder for scatterplot

    # Keep "Run Regressions" button in its original place
    if st.sidebar.button("Run Regressions"):
        df_results = categorize_trends(crypto_pairs, interval, lookback)

        selected_trend_data = df_results[df_results['Symbol'] == symbol_to_visualize]

        if not selected_trend_data.empty:
            trend_coefficient = selected_trend_data.iloc[0]['Trend Coefficient']
            category = selected_trend_data.iloc[0]['Category']
            rsquared = selected_trend_data.iloc[0]['R-squared']
        else:
            trend_coefficient = "N/A"
            category = "N/A"

        st.markdown("""
                <style>
                    .dataframe-container table {
                        width: 100% !important;  /* Forces table to take full width */
                        table-layout: fixed !important; /* Prevents auto-sizing */
                    }
                    .dataframe-container td, .dataframe-container th {
                        overflow: hidden !important;
                        white-space: nowrap !important;
                        text-overflow: ellipsis !important;
                    }
                </style>
            """, unsafe_allow_html=True)
        df_placeholder.data_editor(df_results, hide_index=True, use_container_width=True)


        # Fetch data and perform regression
        data = get_binance_data(symbol_to_visualize, interval, lookback)

        if not data.empty:
            #colx, coly = st.columns(2)

            with colx:
                st.markdown(
                    f"""
                    <div style="background-color: #dddddd; padding: 16px; border-radius: 8px;  text-align: left;">
                        <h6>{symbol_to_visualize}'s Trend</h6>
                        <p style="font-size: 12px; font-weight: bold;">{category} with slope: {trend_coefficient:,.6f}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with coly:
                st.markdown(
                    f"""
                    <div style="background-color: #dddddd; padding: 16px; border-radius: 8px ; text-align: left;">
                        <h6>R-Squared</h6>
                        <p style="font-size: 12px; font-weight: bold;">{rsquared:,.6f}</p>
                        <p style="font-size: 12px; font-weight: bold;"> </p>  <!-- Invisible line -->
                        <p style="font-size: 12px; font-weight: bold;"> </p>  <!-- Invisible line -->
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # ✅ Insert a slight space before showing the chart
            st.markdown("<br>", unsafe_allow_html=True)
            X = sm.add_constant(range(len(data)))
            y = data['Close']
            model = sm.OLS(y, X).fit()
            data['Regression'] = model.predict(X)

            # Ensure Matplotlib runtime works properly inside Streamlit
            plt.style.use("seaborn-v0_8-darkgrid")
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(1, 1, 1)

            # Scatter plot of closing prices
            ax.scatter(data['Open time'], data['Close'], label='Close Prices', color='#1f77b4', edgecolors='black', alpha=0.7, s=30)

            # Regression line
            ax.plot(data['Open time'], data['Regression'], color='red', linestyle='--', linewidth=2, label='Regression Line')

            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())

            # Labels and title
            ax.set_xlabel("Time", fontsize=12, fontweight='bold')
            ax.set_ylabel("Close Price", fontsize=12, fontweight='bold')
            ax.set_title(f"Regression Analysis for {symbol_to_visualize}", fontsize=14, fontweight='bold')

            # Legend styling
            ax.legend(frameon=True, facecolor="white", edgecolor="black", fontsize=11)

            # Rotate x-axis labels
            plt.xticks(rotation=45, ha='right')

            # Display the chart inside the placeholder
            #chart_placeholder.pyplot(fig)
            # ✅ Now render the scatterplot below the rectangles
            chart_placeholder.pyplot(fig)
        else:
            # If no data is available, display a message instead
            chart_placeholder.write("No data available for the selected pair.")

if __name__ == "__main__":
    main()


