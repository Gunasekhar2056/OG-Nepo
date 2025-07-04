from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objs as go

app = Flask(__name__)

# Load model, transformer, and dataset
model = joblib.load('ev_sales_poly_model.pkl')
poly = joblib.load('poly_transformer.pkl')
df = pd.read_csv('IND YEARWISE SALES - Sheet1.csv')
df.columns = df.columns.str.strip()

# Preprocess once
df['EV_Sales'] = pd.to_numeric(df['TOTAL EV SALES'], errors='coerce') / 1_000_000  # Convert to millions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        year = int(request.form['year'])

        # Transform and predict
        year_transformed = poly.transform(np.array([[year]]))
        prediction = float(model.predict(year_transformed)[0])
        prediction = round(prediction, 2)

        # Filter data for actual years and append prediction year
        df_filtered = df[df['YEAR'] <= year].copy()

        fig = go.Figure()

        # Line for actual + predicted
        all_years = list(df_filtered['YEAR']) + [year]
        all_sales = list(df_filtered['EV_Sales']) + [prediction]

        fig.add_trace(go.Scatter(
            x=all_years,
            y=all_sales,
            mode='lines+markers',
            name='Sales (Actual + Prediction)',
            line=dict(color='royalblue', width=3)
        ))

        # Red point only for prediction
        fig.add_trace(go.Scatter(
            x=[year],
            y=[prediction],
            mode='markers+text',
            name='Predicted',
            marker=dict(size=10, color='red'),
            text=[f'{prediction}'],
            textposition='top center'
        ))

        fig.update_layout(
            title='EV Sales Prediction',
            xaxis_title='Year',
            yaxis_title='EV Sales (in Millions)',
            plot_bgcolor='white',
            height=600,
            width=1100,
            margin=dict(l=80, r=80, t=80, b=80),
            hovermode='x unified',
            xaxis=dict(dtick=1, gridcolor='lightgrey'),
            yaxis=dict(dtick=5, gridcolor='lightgrey')
        )

        graph_html = fig.to_html(full_html=False)

        return render_template('index.html',
                               prediction=prediction,
                               year=year,
                               map_div=graph_html)

    except Exception as e:
        print("Plot error:", e)
        return render_template('index.html', error="Error creating plot")

if __name__ == '__main__':
    app.run(debug=True)
