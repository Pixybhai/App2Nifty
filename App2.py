

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import yfinance as yf
# ... other imports ...


# === AUTO REFRESH EVERY 10 SECONDS ===
st_autorefresh(interval=10000, limit=None, key="niftydata")

st.title("🚀 Nifty 50 AI Price Predictor Dashboard")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load
import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import feedparser
import plotly.graph_objects as go

# ========================= CONFIG =========================
TICKER = "^NSEI"
START_DATE = "2015-01-01"
MODEL_FILE = "nifty_rf_predictor.pkl"
# ========================================================


st.set_page_config(
    page_title="Nifty 50 AI Predictor",
    page_icon="📈",
    layout="wide"
)
st.markdown("**Random Forest + VADER Multi-Source News Sentiment** | Runs offline after first training")

# Cache data fetching
@st.cache_data(ttl=3600)  # Refresh every hour
@st.cache_data(ttl=3600)
def fetch_data():
    data = yf.download(TICKER, start=START_DATE, progress=False)

    # ✅ FIX MULTIINDEX
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data

def get_news_sentiment():
    """Multi-source news: Yahoo Finance + Livemint RSS + VADER"""
    headlines = []
    st.info("🔍 Fetching news from multiple sources...")

    # Source 1: Yahoo Finance
    try:
        ticker = yf.Ticker(TICKER)
        yahoo_news = [item.get('title', '') for item in ticker.news[:10] if item.get('title')]
        headlines.extend(yahoo_news)
    except:
        pass

    # Source 2: Livemint Markets RSS (reliable Indian financial news)
    try:
        feed = feedparser.parse("https://www.livemint.com/rss/markets")
        livemint_headlines = [entry.title for entry in feed.entries[:10]]
        headlines.extend(livemint_headlines)
    except:
        pass

    if not headlines:
        st.warning("⚠️ Could not fetch news. Using price-only prediction.")
        return 0.0, []

    # VADER analysis
    analyzer = SentimentIntensityAnalyzer()
    sentiments = [analyzer.polarity_scores(h)['compound'] for h in headlines]
    avg_sentiment = sum(sentiments) / len(sentiments)

    # Deduplicate headlines
    unique_headlines = list(dict.fromkeys(headlines))[:8]

    sentiment_label = (
        "🟢 Strongly Bullish" if avg_sentiment > 0.15 else
        "🟢 Bullish" if avg_sentiment > 0.05 else
        "🔴 Bearish" if avg_sentiment < -0.05 else
        "🔴 Strongly Bearish"
    )

    return avg_sentiment, unique_headlines, sentiment_label

def prepare_features(data):
    df = data[['Close']].copy()
    
    for i in range(1, 11):
        df[f'lag_{i}'] = df['Close'].shift(i)
    
    df['sma_5']  = df['Close'].rolling(window=5).mean().shift(1)
    df['sma_20'] = df['Close'].rolling(window=20).mean().shift(1)
    df['vol_5']  = df['Close'].rolling(window=5).std(ddof=0).shift(1)
    
    df.dropna(inplace=True)
    
    X = df.drop('Close', axis=1).astype(float)
    y = df['Close'].astype(float)
    
    return X, y, data

def get_model(data):
    """Load existing model or train new one"""
    try:
        model = load(MODEL_FILE)
        st.success("✅ Loaded saved model")
        return model
    except FileNotFoundError:
        with st.spinner("Training new Random Forest model (first time only)..."):
            X, y, _ = prepare_features(data)
            train_size = int(len(X) * 0.8)
            X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
            
            model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            dump(model, MODEL_FILE)
            st.success("✅ New model trained and saved!")
            return model

def predict_next_day(model, data):
    # === SAFER WAY TO GET SCALAR VALUES ===
    last_close = float(data['Close'].iloc[-1].item() if hasattr(data['Close'].iloc[-1], 'item') else data['Close'].iloc[-1])
    
    # Build future features safely
    future_features = {}
    for i in range(1, 11):
        val = data['Close'].iloc[-i]
        future_features[f'lag_{i}'] = float(val.item() if hasattr(val, 'item') else val)
    
    # Technical indicators - force scalar
    sma5 = data['Close'].tail(5).mean()
    sma20 = data['Close'].tail(20).mean()
    vol5 = data['Close'].tail(5).std(ddof=0)   # ddof=0 avoids NaN issues
    
    future_features['sma_5']  = float(sma5.item() if hasattr(sma5, 'item') else sma5)
    future_features['sma_20'] = float(sma20.item() if hasattr(sma20, 'item') else sma20)
    future_features['vol_5']  = float(vol5.item() if hasattr(vol5, 'item') else vol5)
    
    # Create DataFrame and ensure numeric types
    future_df = pd.DataFrame([future_features])
    future_df = future_df.astype(float)
    
    # Extra safety: match the exact feature order the model was trained on
    if hasattr(model, 'feature_names_in_'):
        future_df = future_df.reindex(columns=model.feature_names_in_, fill_value=0.0)
    
    base_prediction = float(model.predict(future_df)[0])
    
    # === News Sentiment Part (unchanged) ===
    sentiment_score, headlines, sentiment_label = get_news_sentiment()
    news_adjustment = sentiment_score * 80
    final_prediction = base_prediction + news_adjustment
    
    return {
        'last_close': last_close,
        'base_prediction': base_prediction,
        'final_prediction': final_prediction,
        'change': final_prediction - last_close,
        'sentiment_score': sentiment_score,
        'sentiment_label': sentiment_label,
        'headlines': headlines
    }

# ========================= MAIN DASHBOARD =========================
data = fetch_data()

if data.empty:
    st.error("Could not fetch market data. Check your internet.")
    st.stop()

model = get_model(data)
prediction = predict_next_day(model, data)

# Top Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Latest Close", f"₹{prediction['last_close']:,.2f}")
with col2:
    st.metric("Predicted Close (Next Day)", f"₹{prediction['final_prediction']:,.2f}", 
              f"{prediction['change']:+,.2f}")
with col3:
    st.metric("News Sentiment", prediction['sentiment_label'], 
              f"{prediction['sentiment_score']:.3f}")
with col4:
    st.metric("Last Updated", datetime.datetime.now().strftime("%H:%M IST"))

# Refresh Button
if st.button("🔄 Refresh Data & Prediction", type="primary", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.divider()

# Interactive Chart - Last 30 days + Next Day Prediction
st.subheader("📊 Nifty 50 Price Chart (Last 30 Days + Tomorrow's Prediction)")
recent = data.tail(30).copy()
next_date = recent.index[-1] + pd.Timedelta(days=1)  # Approximate next trading day
recent.loc[next_date] = np.nan
recent.loc[next_date, 'Close'] = prediction['final_prediction']

fig = go.Figure()
fig.add_trace(go.Scatter(x=recent.index[:-1], y=recent['Close'].iloc[:-1],
                        mode='lines+markers', name='Actual Close', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=[recent.index[-1]], y=[prediction['final_prediction']],
                        mode='markers', name='Predicted Close', 
                        marker=dict(size=14, color='red', symbol='star')))
fig.update_layout(height=450, template="plotly_white", 
                 xaxis_title="Date", yaxis_title="Nifty Close (₹)",
                 hovermode="x unified", legend=dict(yanchor="top", y=0.99))
st.plotly_chart(fig, use_container_width=True)

# News Section
st.subheader("📰 Multi-Source News Headlines & Sentiment")
st.markdown(f"**VADER Average Sentiment:** `{prediction['sentiment_score']:.3f}` → **{prediction['sentiment_label']}**")

with st.expander("View all influencing headlines", expanded=True):
    for i, headline in enumerate(prediction['headlines'], 1):
        st.write(f"{i}. {headline}")

st.caption("Sources: Yahoo Finance + Livemint Markets (updates every run)")

st.divider()
st.markdown("**How to use:** Run `streamlit run Nifty_Dashboard.py` in terminal. Refresh anytime for latest prediction. Model trains only once.")

st.success("✅ Dashboard ready! Your personal Nifty predictor is now in the browser.")
st.write("Data tail:", data.tail(30))
