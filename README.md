# MED-PC shared progress dashboard

This package is a read-only DRL and PIT progress website for coworkers and supervisors.

## October 21 update

- DRL-15 and DRL-20 are labeled explicitly on every DRL chart.
- The DRL page has separate **Cohort overview** and **Individual rats** tabs.
- Each rat has press/reward, median-IRT, efficiency, and session-table views.
- Median-IRT plots include the programmed 5/10/15/20-second criterion in red.
- Individual rats use a three-panel scientific overview figure.
- The converter/review page has been removed from the shared website.
- The site uses a light background with high-contrast cards and plots.

## Run locally

```bash
streamlit run medpc_app.py
```

## Deploy privately

Upload this entire folder to the existing private GitHub repository used by the Streamlit app. Set `medpc_app.py` as the app entry point. Keep Streamlit access restricted to approved email addresses.

The dashboard reads:

- `data/drl_progress.xlsx`
- `data/pit_progress.xlsx`

To update what coworkers see, replace those two files with the newest consolidated workbooks and push the change to GitHub. Streamlit will redeploy automatically. Keep the same filenames.

The dashboard is read-only; update the two workbooks in the data folder to refresh it.

