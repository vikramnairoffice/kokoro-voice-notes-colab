# Kokoro 82M Voice Notes (Colab T4)

This tool generates personalized voice notes from a CSV using **Kokoro 82M** and saves `.wav` files to Google Drive.

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
- Generates one WAV file per row.
- Output filename is the exact value in `No` plus `.wav`.
- Saves files to Google Drive folder path you provide.
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

1. Upload your CSV in the UI.
2. Write message template with `{{Name}}`, for example:
   - `Hello {{Name}}, this is a reminder call from our team.`
3. Set output folder (default: `/content/drive/MyDrive/kokoro_voice_notes`).
4. Click **Generate Voice Notes**.
5. Check run summary and per-row result table.

## Notes

- Invalid rows are skipped (missing `Name` or `No`) and shown in result table.
- Voice is fixed to `af_heart` (English).
- If a file write fails for a row, the error appears in `message` column.
