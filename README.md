# SmartCart Customer Insights

A polished Flask dashboard that turns the SmartCart notebook into a portfolio-ready web app.

## Features

- Clean, responsive UI with KPI cards and interactive Plotly charts
- CSV upload support for custom customer data
- Built-in demo dataset fallback so the app still runs if the original CSV is missing
- Vercel-ready Python entrypoint

## Local run

Use Python 3.12 for the virtual environment. Python 3.14 will try to build some scientific packages from source on Windows and can fail without Visual Studio build tools.

```bash
"your location" -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

## Deploy to Vercel

Push the repo to GitHub, connect it to Vercel, and deploy with the included `vercel.json`.

## Notes

- The notebook exploration is preserved in `smartcart.ipynb`.
- The web app loads `smartcart_customers.csv` if you add it to the repo, otherwise it uses built-in demo data.