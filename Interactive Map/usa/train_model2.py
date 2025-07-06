import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import plotly.graph_objects as go
import joblib

# Load and clean data
df = pd.read_excel('ev_sales_yearwise_by_region.csv')
df.columns = df.columns.str.strip()
df_long = df.melt(id_vars='Year', var_name='Region', value_name='EV_Sales')
df_long['Year'] = pd.to_numeric(df_long['Year'], errors='coerce')
df_long['EV_Sales'] = pd.to_numeric(df_long['EV_Sales'], errors='coerce')
df_long.dropna(inplace=True)

# Forecast setup
future_years = np.arange(2025, 2047)
degree = 3

# Step 1: Loop and create individual graphs
for region in df_long['Region'].unique():
    region_data = df_long[df_long['Region'] == region]
    X = region_data[['Year']]
    y = region_data['EV_Sales'] / 1000000

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)

    joblib.dump(model, 'ev_sales_poly_model.pkl')
    joblib.dump(poly, 'poly_transformer.pkl')

    # Predict full range
    all_years = np.concatenate([X['Year'].values, future_years])
    all_years_df = pd.DataFrame({'Year': all_years})
    all_X_poly = poly.transform(all_years_df)
    all_preds = model.predict(all_X_poly)

    # Confidence intervals
    y_pred = model.predict(X_poly)
    residuals = y - y_pred
    std_error = np.std(residuals)
    future_X_poly = poly.transform(pd.DataFrame({'Year': future_years}))
    future_preds = model.predict(future_X_poly)
    ci_upper = future_preds + 1.96 * std_error
    ci_lower = future_preds - 1.96 * std_error

    trace_actual = go.Scatter(x=X['Year'], y=y, mode='lines+markers', name='Actual',
                              line=dict(color='blue'), marker=dict(size=6))
    
    trace_line = go.Scatter(x=all_years, y=all_preds, mode='lines+markers',
                            name='Predicted',
                            line=dict(color='green', width=2,dash='dash'),
                            hovertemplate='Year: %{x}<br>Sales: %{y:.2f}M<extra></extra>')

    trace_ci = go.Scatter(x=np.concatenate([future_years, future_years[::-1]]),
                          y=np.concatenate([ci_upper, ci_lower[::-1]]),
                          fill='toself', fillcolor='rgba(0,255,0,0.1)',
                          line=dict(color='rgba(255,255,255,0)'),
                          name='95% Confidence Interval',
                          hoverinfo='skip')

    layout = go.Layout(
        title=f'EV Sales Forecast - {region}',
        xaxis=dict(title='Year', dtick=1),
        yaxis=dict(title='EV Sales (Millions)'),
        template='plotly_white',
        hovermode='x unified'
    )

    fig = go.Figure(data=[trace_actual, trace_line, trace_ci], layout=layout)
    fig.show()
