# app.py
from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objs as go

app = Flask(__name__)

model = joblib.load('ice_sales_poly_model.pkl')
poly = joblib.load('ice_poly_transformer.pkl')
df = pd.read_csv('global_ice_vehicle_Sales.csv')
df.columns = df.columns.str.strip()
df['ICE_Sales'] = pd.to_numeric(df['ICE_Sales'], errors='coerce')

@app.route('/')
def home():
    return render_template('index4.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        year = int(request.form['year'])
        if not 2010 <= year <= 2050:
            return render_template('index4.html', error="Year must be between 2010 and 2050.")

        start = df['Year'].min()
        all_years = np.arange(start, year + 1).reshape(-1, 1)
        all_poly = poly.transform(all_years)
        predictions = model.predict(all_poly)
        predicted_value = round(predictions[-1], 2)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Year'], y=df['ICE_Sales'], mode='lines+markers', name='Actual Sales'))
        fig.add_trace(go.Scatter(x=all_years.flatten(), y=predictions, mode='lines', name='Predicted Sales'))
        fig.add_trace(go.Scatter(x=[year], y=[predicted_value],
                                 mode='markers+text', name='Prediction',
                                 text=[f"{predicted_value}"], textposition='top center',
                                 marker=dict(size=10, color='red')))

        fig.update_layout(title='ICE Vehicle Sales Forecast',
                          xaxis_title='Year', yaxis_title='Sales (in Millions)',
                          plot_bgcolor='white')

        graph_html = fig.to_html(full_html=False)

        return render_template('index4.html', prediction=predicted_value,
                               year=year, map_div=graph_html)
    except Exception as e:
        print("Error:", e)
        return render_template('index4.html', error="Error generating prediction.")

if __name__ == '__main__':
    app.run(debug=True)
