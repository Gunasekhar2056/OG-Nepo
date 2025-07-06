from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import plotly.graph_objs as go
import plotly.io as pio

app = Flask(__name__)

# Load and clean the dataset
df = pd.read_csv("ev_sales_yearwise_by_region.csv")
df.columns = df.columns.str.strip()
print("DEBUG: Columns loaded:", df.columns)

# Prepare models per region
regions = ['China', 'EU27', 'USA']
models = {}
poly = PolynomialFeatures(degree=3)
X = df[['Year']].values
X_poly = poly.fit_transform(X)

for region in regions:
    y = df[region].values / 1_000_000  # convert to millions
    model = LinearRegression()
    model.fit(X_poly, y)
    models[region] = model

def make_prediction(year):
    year_poly = poly.transform([[year]])
    result = {}
    for region, model in models.items():
        pred = model.predict(year_poly)[0]
        result[region] = pred
    return result

def create_plot(region):
    actual_y = df[region].values / 1_000_000
    future_years = np.arange(df['Year'].min(), 2031)
    future_X = poly.transform(future_years.reshape(-1, 1))
    pred_y = models[region].predict(future_X)

    traces = [
        go.Scatter(x=df['Year'], y=actual_y, mode='markers', name=f"{region} Actual"),
        go.Scatter(x=future_years, y=pred_y, mode='lines', name=f"{region} Predicted")
    ]

    layout = go.Layout(
        title=f"{region} EV Sales Prediction (in millions)",
        xaxis=dict(title="Year"),
        yaxis=dict(title="Sales (Millions)"),
        legend=dict(x=0, y=1.2, orientation="h")
    )
    fig = go.Figure(data=traces, layout=layout)
    return pio.to_html(fig, full_html=False)

@app.route("/", methods=["GET"])
def index():
    return render_template("index3.html")

@app.route("/predict", methods=["POST"])
def predict():
    year = int(request.form['year'])
    prediction = make_prediction(year)
    graph_html = {region: create_plot(region) for region in regions}
    return render_template("index3.html", prediction=prediction, year=year, graph_html=graph_html)

if __name__ == "__main__":
    app.run(debug=True,port=8000)
