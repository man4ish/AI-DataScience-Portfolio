import requests
import pandas as pd
import dash
from dash import dcc, html
import plotly.express as px

# Fetch data from COVID19 API
url = 'https://api.covid19api.com/summary'
response = requests.get(url)
data = response.json()

# Create DataFrame
countries = pd.json_normalize(data['Countries'])

# Start Dash app
app = dash.Dash(__name__)
app.title = "COVID-19 Global Dashboard"

app.layout = html.Div([
    html.H1("COVID-19 Global Dashboard"),
    
    dcc.Dropdown(
        id='country-dropdown',
        options=[{'label': c, 'value': c} for c in countries['Country'].unique()],
        value='United States of America',
        style={'width': '60%'}
    ),
    
    dcc.Graph(id='covid-bar'),
    
    html.P("Data Source: covid19api.com")
])

@app.callback(
    dash.dependencies.Output('covid-bar', 'figure'),
    [dash.dependencies.Input('country-dropdown', 'value')]
)
def update_graph(selected_country):
    df = countries[countries['Country'] == selected_country]
    metrics = {
        'NewConfirmed': df['NewConfirmed'].values[0],
        'TotalConfirmed': df['TotalConfirmed'].values[0],
        'NewDeaths': df['NewDeaths'].values[0],
        'TotalDeaths': df['TotalDeaths'].values[0],
        'NewRecovered': df['NewRecovered'].values[0],
        'TotalRecovered': df['TotalRecovered'].values[0],
    }
    fig = px.bar(
        x=list(metrics.keys()),
        y=list(metrics.values()),
        labels={'x': 'Metric', 'y': 'Count'},
        title=f"COVID-19 Stats for {selected_country}"
    )
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)

