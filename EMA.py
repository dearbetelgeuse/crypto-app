import streamlit as st
import pandas as pd
import numpy as np
from binance.client import Client
import plotly.graph_objects as go
from PIL import Image
from fastapi import FastAPI

app = FastAPI()

# Binance API credentials
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')

# Initialize Binance client
client = Client(API_KEY, API_SECRET)

def get_historical_klines(symbol, interval, lookback):
    try:
        klines = client.get_historical_klines(symbol, interval, f"{lookback} day ago UTC")
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 
            'close_time', 'quote_asset_volume', 'number_of_trades', 
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df.astype(float)
        return df
    except Exception as e:
        raise Exception(f"Error fetching data: {e}")

def add_ema(df, periods=[20, 50, 100, 200]):
    for period in periods:
        df[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    return df

def plot_data_with_ema(df):
    fig = go.Figure()

    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Candlesticks'
    ))

    # Define HEX colors avoiding red & green
    ema_colors = {
        20: "#FFD700",  # Gold
        50: "#FF7F50",  # Coral
        100: "#6495ED", # Cornflower Blue
        200: "#800080"  # Purple
    }

    # Add EMAs with specified colors
    for ema_period in [20, 50, 100, 200]:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[f'EMA_{ema_period}'],
            mode='lines',
            name=f'EMA {ema_period}',
            line=dict(color=ema_colors[ema_period], width=2)  # Apply HEX colors
        ))

    # Customize layout
    fig.update_layout(
        title=f"Price Action Chart for {symbol} ({interval})",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

# Loading Image using PIL
im = Image.open('siteicon.png')

st.set_page_config(page_title="CRYPTO ANALYSER", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.sidebar.success("Select a page above to navigate.")
# Sidebar setup
st.sidebar.title("Enter details")

st.markdown("""
    <style>
        /* Remove extra top padding from the main page */
        .block-container {
            padding-top: 3rem !important;
            padding-bottom: 0rem !important;
            padding-left: 32px !important;  /* Reduces left padding */
            padding-right: 32px !important; /* Reduces right padding */
        }

        /* Force sidebar to move to the very top */
        section[data-testid="stSidebar"] {
            padding-top: 0px !important;
            margin-top: 0px !important;
        }
    </style>
""", unsafe_allow_html=True)

symbol = st.sidebar.text_input("Crypto Pair", "BTCUSDT")
interval = st.sidebar.selectbox("Interval", options=["1m", "5m", "15m", "1h", "4h", "1d"], index=3)

# ✅ Replaced text input with a slider bar for lookback (matching binance_reg.py)
lookback = st.sidebar.slider("Lookback (days)", min_value=1, max_value=200, step=1)

# Main page setup - Fit banner properly
image_main = Image.open("banner.png")
st.image(image_main, use_container_width=True)

# Fetch and process data
try:
    df = get_historical_klines(symbol, interval, lookback)
    df = add_ema(df)

    # Get current symbol price and latest EMAs
    current_price = df['close'].iloc[-1]
    ema_values = {p: df[f'EMA_{p}'].iloc[-1] for p in [20, 50, 100, 200]}
    
    # Compute price differences
    ema_differences = {p: current_price - ema_values[p] for p in ema_values}

    # Display metrics in columns
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(
            f"""
            <div style="background-color: #dddddd; padding: 16px; border-radius: 8px; text-align: left;">
                <h6>Last Price</h6>
                <p style="font-size: 12px; font-weight: bold;">${current_price:,.6f}</p>
                <p style="font-size: 10px; ">EMA Spread:</p>  <!-- Invisible line -->

            </div>
            """,
            unsafe_allow_html=True,
        )

    # Loop through EMAs and dynamically create columns
    ema_colors = {
        20: "#FFD700",
        50: "#FF7F50",
        100: "#6495ED",
        200: "#ff99fe"
    }
    
    for i, (ema_period, color) in enumerate(ema_colors.items()):
        with [col2, col3, col4, col5][i]:  # Dynamically assign to columns
            price_diff = ema_differences[ema_period]
            diff_color = "green" if price_diff > 0 else "red"
            diff_arrow = "▲" if price_diff > 0 else "▼"
            
            st.markdown(
                f"""
                <div style="background-color: {color}; padding: 16px; border-radius: 4px; text-align: left;">
                    <h6>EMA {ema_period}</h6>
                    <p style="font-size: 12px; font-weight: bold;">${ema_values[ema_period]:,.6f}</p>
                    <p style="font-size: 10px; color: {diff_color};">
                        {diff_arrow} {price_diff:,.6f} ({price_diff / current_price:.2%})
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Plot chart below metrics
    plot_data_with_ema(df)

except Exception as e:
    st.error(f"Error: {e}")


# streamlit run EMA.py