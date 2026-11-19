# Enterprise Revenue Estimator

A Flask application that estimates annual enterprise revenue in USD from company details.

## Run on Windows

1. Make sure `data\G3.csv` is present. The application uses this dataset to train its model at startup.
2. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Start the application:

   ```powershell
   python app.py
   ```

4. Open <http://127.0.0.1:5000> in a browser.

The app trains on positive revenues reported in USD and uses company type, jurisdiction,
employee count, legal-unit count, direct-subsidiary count, hierarchy depth, and standalone
status. It intentionally excludes revenue-year, net income, and market capitalization
from the inputs to avoid relying on target-adjacent or potentially unavailable values.
Missing numeric form values are imputed by the training pipeline.

For a different CSV location, set `REVENUE_CSV` to its path before starting the app.
For deployment, set `FLASK_SECRET_KEY` to a persistent random secret and run with a
production WSGI server rather than Flask's development server.

Predictions are estimates based on historical data, not guaranteed financial results.
