from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import Attention, Dense, LSTM, Dropout, Input, Bidirectional # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from scipy.optimize import minimize
from datetime import datetime, timedelta
import plotly.graph_objects as go
import yfinance as yf
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

# Constants and settings
RISK_LEVELS = {"Conservative": 0.1, "Moderate": 0.3, "Aggressive": 0.5, "Very Aggressive": 0.7}
TIME_HORIZONS = {"1 day": 1, "7 days": 7, "15 days": 15, "1 month": 30, "3 months": 90, "6 months": 180}
RISK_FREE_RATE = 0.685  # Risk-free rate for Sharpe ratio
CRYPTO_LIST = [
    'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'DOGE-USD',
    'ADA-USD', 'SOL-USD', 'DOT-USD', 'LTC-USD', 'MATIC-USD',
    'SHIB-USD', 'TRX-USD', 'AVAX-USD', 'LINK-USD',
    'ATOM-USD', 'XMR-USD', 'BCH-USD', 'XLM-USD'
]

# Function to fetch cryptocurrency data
def fetch_crypto_data(crypto_list):
    data = []
    for crypto in crypto_list:
        try:
            ticker = yf.Ticker(crypto)
            hist = ticker.history(period="5d")  # Fetch last 5 days of data
            if len(hist) >= 2:  # Ensure sufficient data
                close_today = hist["Close"].iloc[-1]
                close_yesterday = hist["Close"].iloc[-2]
                change_24h = ((close_today - close_yesterday) / close_yesterday) * 100
                trend = hist["Close"].tolist()
                max_trend = max(trend)
                data.append({
                    "symbol": crypto,
                    "current_price": close_today,
                    "change_24h": change_24h,
                    "trend": trend,
                    "max_trend": max_trend
                })
        except Exception as e:
            print(f"Error fetching data for {crypto}: {e}")
    return data


# Example historical crypto data
def load_crypto_data():
    # Replace this with actual data loading logic
    dates = pd.date_range(start="2023-01-01", periods=200)
    data = {crypto: pd.DataFrame({
        "Date": dates,
        "Close": np.random.rand(len(dates)) * 1000
    }).set_index("Date") for crypto in CRYPTO_LIST}
    return data


# Feature Engineering
def create_features(data):
    data['Return'] = data['Close'].pct_change()
    data['7D_MA'] = data['Close'].rolling(window=7).mean()
    data['14D_MA'] = data['Close'].rolling(window=14).mean()
    data['Volatility'] = data['Close'].rolling(window=7).std()
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Upper_BB'] = data['7D_MA'] + (2 * data['Volatility'])
    data['Lower_BB'] = data['7D_MA'] - (2 * data['Volatility'])
    data = data.dropna()
    return data


# Attention-Based LSTM Model
def build_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = Dropout(0.3)(x)
    query = Dense(128)(x)
    attention = Attention()([query, query])
    x = Bidirectional(LSTM(64))(attention)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model


# Prediction Function
def predict_price_lstm(crypto, days_ahead=30):
    try:
        # Fetch historical data
        ticker = yf.Ticker(crypto)
        hist = ticker.history(period="5y")  # Fetch 5 years of data
        if hist.empty or "Close" not in hist.columns or len(hist) < 60:
            print(f"No sufficient data for {crypto}")
            return None, None, None

        # Feature engineering
        hist['RSI'] = compute_rsi(hist['Close'])
        hist['MACD'], hist['Signal'] = compute_macd(hist['Close'])
        hist['Volatility'] = hist['Close'].rolling(window=7).std()
        hist['SMA'] = hist['Close'].rolling(window=30).mean()
        hist.dropna(inplace=True)

        # Prepare data
        features = hist[['Close', 'RSI', 'MACD', 'Signal', 'Volatility', 'SMA']].values
        target = hist[['Close']].values

        # Scale features and target
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
        scaled_features = feature_scaler.fit_transform(features)
        scaled_target = target_scaler.fit_transform(target)

        # Create sequences
        sequence_length = 60
        X, y = [], []
        for i in range(sequence_length, len(scaled_features)):
            X.append(scaled_features[i-sequence_length:i])
            y.append(scaled_target[i, 0])
        X, y = np.array(X), np.array(y)

        # Train-test split
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Build model
        model = tf.keras.Sequential([
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        # Train model
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, callbacks=[early_stopping], verbose=1)

        # Predict future prices
        future_prices = []
        current_sequence = X_test[-1].copy()
        for _ in range(days_ahead):
            next_price = model.predict(current_sequence[np.newaxis, :, :])[0, 0]
            future_prices.append(next_price)

            # Update the sequence
            next_sequence = current_sequence.copy()
            next_sequence[:-1] = current_sequence[1:]
            next_sequence[-1, 0] = next_price  # Predicted price
            current_sequence = next_sequence

        # Reverse scaling
        predicted_prices = target_scaler.inverse_transform(np.array(future_prices).reshape(-1, 1))[:, 0]

        # Generate future dates
        future_dates = [datetime.now() + timedelta(days=i) for i in range(1, days_ahead + 1)]

        return future_dates, predicted_prices, None
    except Exception as e:
        print(f"Error predicting price for {crypto}: {e}")
        return None, None, None
 
 
def prepare_data_and_model(crypto):
    try:
        # Fetch historical data
        ticker = yf.Ticker(crypto)
        hist = ticker.history(period="5y")  # Fetch 5 years of data
        if hist.empty or "Close" not in hist.columns or len(hist) < 60:
            raise ValueError(f"Insufficient data for {crypto}")

        # Feature engineering
        hist['RSI'] = compute_rsi(hist['Close'])
        hist['MACD'], hist['Signal'] = compute_macd(hist['Close'])
        hist['Volatility'] = hist['Close'].rolling(window=7).std()
        hist['SMA'] = hist['Close'].rolling(window=30).mean()
        hist.dropna(inplace=True)

        # Prepare data
        features = hist[['Close', 'RSI', 'MACD', 'Signal', 'Volatility', 'SMA']].values
        target = hist[['Close']].values

        # Scale features and target
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
        scaled_features = feature_scaler.fit_transform(features)
        scaled_target = target_scaler.fit_transform(target)

        # Create sequences
        sequence_length = 60
        X, y = [], []
        for i in range(sequence_length, len(scaled_features)):
            X.append(scaled_features[i-sequence_length:i])
            y.append(scaled_target[i, 0])
        X, y = np.array(X), np.array(y)

        # Train-test split
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Build model
        model = tf.keras.Sequential([
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        # Train model
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, callbacks=[early_stopping], verbose=1)

        # Return all required values
        return X_test, y_test, model, feature_scaler, target_scaler
    except Exception as e:
        raise ValueError(f"Error preparing data and model: {e}")

 
def compute_rsi(series, period=14):    
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series, slow=26, fast=12, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line
 

# Portfolio optimization helper functions
def portfolio_return(weights, mean_returns):
    return np.dot(weights, mean_returns)


def portfolio_risk(weights, covariance_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))


def combined_objective(weights, mean_returns, covariance_matrix, risk_tolerance):
    return -portfolio_return(weights, mean_returns) + risk_tolerance * portfolio_risk(weights, covariance_matrix)


def train_portfolio(mean_returns, cov_matrix, risk_tolerance, min_weight=0.05):
    num_assets = len(mean_returns)
    initial_weights = np.ones(num_assets) / num_assets
    bounds = [(min_weight, 1) for _ in range(num_assets)]
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}

    result = minimize(
        combined_objective,
        initial_weights,
        args=(mean_returns, cov_matrix, risk_tolerance),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return result.x if result.success else None


@app.route("/")
def index():
    crypto_data = fetch_crypto_data(CRYPTO_LIST)
    return render_template("index.html", crypto_data=crypto_data)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            # Get user input
            crypto = request.form.get("crypto")
            days_ahead = request.form.get("days_ahead", 7)  # Default to 7 days if not provided
            days_ahead = int(days_ahead)

            # Validate inputs
            if crypto is None or crypto not in CRYPTO_LIST:
                raise ValueError("Invalid or missing cryptocurrency selection.")
            if days_ahead <= 0:
                raise ValueError("Days Ahead must be a positive number.")

            # Prepare data and model
            X_test, y_test, model, feature_scaler, target_scaler = prepare_data_and_model(crypto)

            # Predict future prices
            future_dates, predicted_prices, _ = predict_price_lstm(crypto, days_ahead)
            if len(future_dates) == 0 or len(predicted_prices) == 0:
                raise ValueError("Unable to generate predictions. Please try again later.")

            # Calculate test error for confidence intervals
            y_pred = model.predict(X_test)
            test_error = np.sqrt(mean_squared_error(y_test, y_pred))

            # Create prediction chart
            prediction_chart = go.Figure()
            prediction_chart.add_trace(go.Scatter(
                x=future_dates,
                y=predicted_prices,
                mode="lines+markers",
                name=f"{crypto} Price Prediction"
            ))
            prediction_chart.add_trace(go.Scatter(
                x=future_dates,
                y=[p + test_error for p in predicted_prices],
                mode="lines",
                line=dict(dash="dot", color="gray"),
                name="Upper Confidence Interval"
            ))
            prediction_chart.add_trace(go.Scatter(
                x=future_dates,
                y=[p - test_error for p in predicted_prices],
                mode="lines",
                line=dict(dash="dot", color="gray"),
                name="Lower Confidence Interval"
            ))
            prediction_chart.update_layout(
                title=f"{crypto} Price Prediction for Next {days_ahead} Days",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                template="plotly_white"
            )

            # Render the prediction chart in the results page
            prediction_chart_html = prediction_chart.to_html(full_html=False)
            return render_template("predict.html", prediction_chart_html=prediction_chart_html)
        except ValueError as ve:
            # Handle user input errors
            print(f"ValueError: {ve}")
            return render_template("predict_form.html", cryptos=CRYPTO_LIST, error=str(ve))
        except Exception as e:
            # Handle unexpected errors
            print(f"Unexpected Error: {e}")
            return render_template("predict_form.html", cryptos=CRYPTO_LIST, error="An unexpected error occurred.")
    else:
        # Render the form with the dropdown options for GET requests
        return render_template("predict_form.html", cryptos=CRYPTO_LIST)
  

@app.route("/portfolio", methods=["GET", "POST"])
def portfolio():
    crypto_data = load_crypto_data()  # Load historical data
    if request.method == "POST":
        try:
            # Retrieve form data
            selected_tickers = request.form.getlist("cryptos")
            investment_amount = float(request.form["investment_amount"])
            risk_profile = request.form["risk_profile"]
            time_horizon = request.form["time_horizon"]

            # Validate inputs
            if not selected_tickers:
                raise ValueError("No cryptocurrencies selected. Please select at least one.")
            if risk_profile not in RISK_LEVELS or time_horizon not in TIME_HORIZONS:
                raise ValueError("Invalid risk profile or time horizon selected.")

            # Map risk tolerance and time horizon
            risk_tolerance = RISK_LEVELS[risk_profile]
            days = TIME_HORIZONS[time_horizon]

            # Process selected crypto data
            historical_data = pd.concat([crypto_data[ticker]["Close"] for ticker in selected_tickers], axis=1)
            historical_data.columns = selected_tickers

            # Ensure valid data by dropping NaNs
            historical_data = historical_data.dropna()

            # Calculate returns and covariance
            mean_returns = historical_data.pct_change().mean()
            cov_matrix = historical_data.pct_change().cov()

            # Optimize portfolio
            weights = train_portfolio(mean_returns, cov_matrix, risk_tolerance)
            if weights is None:
                raise ValueError("Portfolio optimization failed. Try different inputs.")

            # Calculate allocations
            allocations = {ticker: weight * investment_amount for ticker, weight in zip(selected_tickers, weights)}

            # Portfolio metrics
            portfolio_annualized_return = portfolio_return(weights, mean_returns) * 252
            portfolio_annualized_risk = portfolio_risk(weights, cov_matrix) * np.sqrt(252)
            sharpe_ratio = (portfolio_annualized_return - RISK_FREE_RATE) / portfolio_annualized_risk

            # Monte Carlo simulation for investment growth
            simulated_returns = []
            for _ in range(1000):  # Monte Carlo simulations
                simulated_daily_returns = np.random.multivariate_normal(mean_returns, cov_matrix, days)
                simulated_cumulative_returns = np.cumprod(np.dot(simulated_daily_returns, weights) + 1) * investment_amount
                simulated_returns.append(simulated_cumulative_returns)

            # Calculate average cumulative returns
            average_cumulative_returns = np.mean(simulated_returns, axis=0)
            future_dates = [datetime.now() + timedelta(days=i) for i in range(1, days + 1)]

            # Ensure metrics are accurate
            portfolio_annualized_return *= 100  # Convert to percentage
            portfolio_annualized_risk *= 100  # Convert to percentage

            # Investment growth chart
            investment_growth_chart = go.Figure()
            investment_growth_chart.add_trace(go.Scatter(
                x=future_dates,
                y=average_cumulative_returns,
                mode="lines",
                name="Investment Growth"
            ))
            investment_growth_chart.update_layout(
                title="Investment Growth",
                xaxis_title="Date",
                yaxis_title="Portfolio Value",
                template="plotly_white"
            )
            investment_growth_html = investment_growth_chart.to_html(full_html=False)

            # Portfolio allocation pie chart
            pie_chart = go.Figure(data=[go.Pie(
                labels=list(allocations.keys()),
                values=list(allocations.values())
            )])
            pie_chart.update_layout(
                title="Portfolio Allocation",
                template="plotly_white"
            )
            pie_chart_html = pie_chart.to_html(full_html=False)

            return render_template(
                "portfolio.html",
                allocations=allocations,
                pie_chart_html=pie_chart_html,
                investment_growth_html=investment_growth_html,
                portfolio_metrics={
                    "Annualized Return": f"{portfolio_annualized_return:.2f}%",
                    "Annualized Risk": f"{portfolio_annualized_risk:.2f}%",
                    "Sharpe Ratio": f"{sharpe_ratio:.2f}"
                },
                selected_tickers=selected_tickers,
                time_horizon=time_horizon,
                risk_profile=risk_profile,
                investment_amount=investment_amount
            )
        except Exception as e:
            return render_template("portfolio_form.html", cryptos=CRYPTO_LIST, error=str(e))
    return render_template("portfolio_form.html", cryptos=CRYPTO_LIST)

if __name__ == "__main__":
    app.run(debug=True)
