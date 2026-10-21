# MED-PC shared progress dashboard

This package extends the existing MED-PC converter with two pages:

1. **Progress Dashboard** — read-only DRL and PIT group summaries and individual-rat progress.
2. **Analyze New Data** — the existing upload, ID assignment, parsing, preview, and Excel export workflow.

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

The dashboard data are read-only. Uploaded MED-PC files on the **Analyze New Data** page remain in memory for the active session and are not written to the repository.

