# Kokoro 82M Voice Notes (Colab T4)

This tool generates personalized voice notes from CSV (or direct single-test input) using **Kokoro 82M**.

## Input format

CSV must contain these exact columns:
- `Name`
- `No`

Example:

```csv
Name,No
Amit,+919999999999
Sara,9876543210
```

## Features

- Uses `{{Name}}` tag in message template.
- Includes a model health check button in UI to verify Kokoro is installed and generating audio.
- Generates one WAV file per row.
- Output filename is the exact value in `No` plus `.wav` (batch and custom test).
- Supports two storage modes:
   - Google Drive output folder
   - In-memory download (ZIP for batch, WAV for single test)
- Async processing for faster batch generation.
- Simple Gradio UI.

## Colab setup

Run these commands in Google Colab (T4 runtime):

```bash
!apt-get update -y
!apt-get install -y espeak-ng
!pip install -r requirements.txt
```

Then start app:

```bash
!python app.py
```

Gradio will print a local URL in Colab output.

## How to use

1. Click **Check Kokoro Model** to confirm model is installed and working.
2. Select operation mode:
   - **Batch (CSV)**
   - **Custom Test (No CSV)**
3. Write message template with `{{Name}}`, for example:
   - `Hello {{Name}}, this is a reminder call from our team.`
4. Choose storage mode:
   - **Google Drive** for direct saving to drive folder
   - **In-Memory Download** for UI download files
5. For batch run: upload CSV file.
6. For custom test: fill test `Name` and `No` fields.
7. Click **Generate Voice Notes**.
8. Check run summary and per-row result table.

## Notes

- Invalid rows are skipped (missing `Name` or `No`) and shown in result table.
- Voice is fixed to `af_heart` (English).
- If a file write fails for a row, the error appears in `message` column.
