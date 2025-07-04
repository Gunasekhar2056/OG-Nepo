from flask import Flask, render_template, request
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px

app = Flask(__name__)

# Load your data once
df = pd.read_csv("IND YEARWISE SALES - Sheet1.csv")

@app.route('/')
def index():
    # Create main India map
    fig = px.choropleth(locations=[], locationmode="ISO-3")  # Dummy map here
    map_div = fig.to_html(full_html=False)
    return render_template("index.html", map_div=map_div)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    year = int(data["year"])
    predicted_sales = model.predict([[year]])[0]
    return jsonify({"predicted_sales": predicted_sales})


@app.route('/yearly-graph', methods=['POST'])
def yearly_graph():
    year = request.form['year']
    
    # Filter data for that year
    year_df = df[df['Year'] == int(year)]

    # Example graph (replace with your real logic)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=year_df['State'],
        y=year_df['Sales'],
        name=f"Sales in {year}"
    ))

    plot_div = fig.to_html(full_html=False)
    return render_template("graph.html", plot_div=plot_div, year=year)

if __name__ == '__main__':
    app.run(debug=True)
