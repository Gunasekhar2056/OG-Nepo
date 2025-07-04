import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import plotly.graph_objs as go

# Load data
df = pd.read_csv("IND YEARWISE SALES - Sheet1.csv")
df.columns = df.columns.str.strip()

# Clean and prepare data
df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce')
df['TOTAL EV SALES'] = pd.to_numeric(df['TOTAL EV SALES'], errors='coerce')
df.dropna(subset=['YEAR', 'TOTAL EV SALES'], inplace=True)
df['Sales_Million'] = df['TOTAL EV SALES'] / 1_000_000

# Prepare features and target
x = df[['YEAR']]
y = df['Sales_Million']

# Polynomial transformation (degree = 3)
degree = 3
poly = PolynomialFeatures(degree=degree)
x_poly = poly.fit_transform(x)

# Train model
model = LinearRegression()
model.fit(x_poly, y)

# Save model and transformer
joblib.dump(model, 'ev_sales_poly_model.pkl')
joblib.dump(poly, 'poly_transformer.pkl')

# Accuracy check
y_pred = model.predict(x_poly)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print(f"Model trained and saved.")
print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, R² Score: {r2:.4f}")

# Optional: Visualize prediction with confidence interval
future_years = np.arange(df['YEAR'].min(), 2070).reshape(-1, 1)
future_x_poly = poly.transform(pd.DataFrame(future_years, columns=['YEAR']))
future_preds = model.predict(future_x_poly)
residuals = y - y_pred
std_error = np.std(residuals)
upper = future_preds + 1.96 * std_error
lower = future_preds - 1.96 * std_error

# Plot
trace_actual = go.Scatter(x=df['YEAR'], y=y, mode='markers+lines', name='Actual Sales', line=dict(color='blue'))
trace_predicted = go.Scatter(x=future_years.ravel(), y=future_preds, mode='lines', name='Predicted Sales', line=dict(color='orange', dash='dash'))
trace_ci = go.Scatter(x=np.concatenate([future_years.ravel(), future_years.ravel()[::-1]]),
                      y=np.concatenate([upper, lower[::-1]]), fill='toself',
                      fillcolor='rgba(255,165,0,0.2)', line=dict(color='rgba(255,255,255,0)'),
                      hoverinfo="skip", showlegend=True, name='95% Confidence Interval')

layout = go.Layout(
    title='EV Sales in India with Prediction',
    xaxis=dict(title='Year'), yaxis=dict(title='EV Sales (in Millions)'),
    template='plotly_white'
)

fig = go.Figure(data=[trace_actual, trace_predicted, trace_ci], layout=layout)
fig.show()
