# COVID-19 Dashboard (Interactive Visualization)

This project is an interactive dashboard for visualizing real-time COVID-19 statistics by country using data from [covid19api.com](https://covid19api.com).

## Features

- Dropdown to select a country
- Real-time data visualization of key COVID-19 metrics:
  - New & Total Confirmed Cases
  - New & Total Deaths
  - New & Total Recovered
- Built using Plotly Dash and REST API integration

## Tech Stack

- Plotly Dash
- pandas
- requests (API fetching)
- REST API: covid19api.com

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/covid19_dashboard.git
cd covid19_dashboard
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Run the app:
```bash
python app.py
```

Then open `http://127.0.0.1:8050` in your browser.

## Focus Areas

- API integration
- Real-time data wrangling
- Interactive plotting and dashboards

