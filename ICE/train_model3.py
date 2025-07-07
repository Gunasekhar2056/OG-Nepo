# train_ice_model.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import plotly.graph_objs as go
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    # Load data
    df = pd.read_csv("global_ice_vehicle_Sales.csv")
    df.columns = df.columns.str.strip()
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['ICE_Sales'] = pd.to_numeric(df['ICE_Sales'], errors='coerce')
    df.dropna(subset=['Year', 'ICE_Sales'], inplace=True)

    # Features and target
    X = df[['Year']]
    y = df['ICE_Sales']

    # Polynomial regression
    degree = 3
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    joblib.dump(model, 'ice_sales_poly_model.pkl')
    joblib.dump(poly, 'ice_poly_transformer.pkl')

    # Metrics
    y_pred = model.predict(X_poly)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    logging.info(f"R²: {r2:.3f} | MSE: {mse:.3f}")

    # Predict future
    future_years = np.arange(df['Year'].min(), 2051).reshape(-1, 1)
    future_poly = poly.transform(future_years)
    future_preds = model.predict(future_poly)

    # Confidence interval
    residuals = y - y_pred
    std_err = np.std(residuals)
    upper = future_preds + 1.96 * std_err
    lower = future_preds - 1.96 * std_err

    # Plot
    fig = go.Figure([
        go.Scatter(x=df['Year'], y=y, mode='lines+markers', name='Actual Sales'),
        go.Scatter(x=future_years.flatten(), y=future_preds, mode='lines', name='Predicted Sales'),
        go.Scatter(x=np.concatenate([future_years.flatten(), future_years.flatten()[::-1]]),
                   y=np.concatenate([upper, lower[::-1]]),
                   fill='toself', fillcolor='rgba(255,165,0,0.2)',
                   line=dict(color='rgba(255,255,255,0)'), name='95% CI')
    ])

    fig.update_layout(title='Global ICE Vehicle Sales Prediction',
                      xaxis_title='Year', yaxis_title='Sales (in Millions)',
                      plot_bgcolor='white')
    fig.show()

except Exception as e:
    logging.error(f"Training failed: {e}")