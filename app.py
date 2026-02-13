import asyncio
import os
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import pandas as pd
import soundfile as sf
import torch


REQUIRED_COLUMNS = ("Name", "No")
DEFAULT_OUTPUT_DIR = "/content/drive/MyDrive/kokoro_voice_notes"
DEFAULT_TEMPLATE = "Hello {{Name}}, this is your voice note."
DEFAULT_VOICE = "af_heart"
DEFAULT_LANG_CODE = "a"
DEFAULT_MAX_CONCURRENCY = 4


@dataclass
class RowTask:
    index: int
    name: str
    phone: str
    text: str


@dataclass
class RowResult:
    index: int
    name: str
    phone: str
    status: str
    message: str
    output_path: str = ""


class KokoroVoiceGenerator:
    def __init__(self, lang_code: str = DEFAULT_LANG_CODE, voice: str = DEFAULT_VOICE):
        self.lang_code = lang_code
        self.voice = voice
        self.sample_rate = 24000
        self._pipeline = None

    def preflight(self) -> dict[str, Any]:
        info = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
        return info

    def mount_drive_if_colab(self) -> str:
        if "COLAB_RELEASE_TAG" in os.environ:
            from google.colab import drive  # type: ignore

            drive.mount("/content/drive", force_remount=False)
            return "Google Drive mounted at /content/drive"
        return "Not running in Google Colab; skipped drive mount"

    def ensure_pipeline(self) -> None:
        if self._pipeline is not None:
            return

        try:
            from kokoro import KPipeline
        except Exception as exc:
            raise RuntimeError(
                "Kokoro is not available. Install with `pip install kokoro` in Colab runtime."
            ) from exc

        self._pipeline = KPipeline(lang_code=self.lang_code)

    def synthesize_numpy(self, text: str) -> np.ndarray:
        self.ensure_pipeline()
        assert self._pipeline is not None

        generated = self._pipeline(text, voice=self.voice)

        segments: list[np.ndarray] = []
        for item in generated:
            audio = None
            if isinstance(item, tuple) and len(item) >= 3:
                audio = item[2]
            elif isinstance(item, dict) and "audio" in item:
                audio = item["audio"]
            else:
                audio = item

            if audio is None:
                continue

            array = np.asarray(audio, dtype=np.float32).reshape(-1)
            if array.size > 0:
                segments.append(array)

        if not segments:
            raise RuntimeError("No audio generated for input text")

        combined = np.concatenate(segments)
        return np.clip(combined, -1.0, 1.0)

    def save_wav(self, audio: np.ndarray, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio, self.sample_rate, subtype="PCM_16")


def validate_and_prepare_rows(df: pd.DataFrame, template: str) -> tuple[list[RowTask], list[RowResult]]:
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    tasks: list[RowTask] = []
    invalid_results: list[RowResult] = []

    for index, row in df.iterrows():
        name = "" if pd.isna(row["Name"]) else str(row["Name"]).strip()
        phone = "" if pd.isna(row["No"]) else str(row["No"]).strip()

        if not name:
            invalid_results.append(
                RowResult(
                    index=int(index),
                    name=name,
                    phone=phone,
                    status="skipped",
                    message="Name is empty",
                )
            )
            continue

        if not phone:
            invalid_results.append(
                RowResult(
                    index=int(index),
                    name=name,
                    phone=phone,
                    status="skipped",
                    message="No is empty",
                )
            )
            continue

        text = template.replace("{{Name}}", name).replace("{{NAME}}", name)
        tasks.append(RowTask(index=int(index), name=name, phone=phone, text=text))

    return tasks, invalid_results


async def process_single_row(
    generator: KokoroVoiceGenerator,
    row_task: RowTask,
    output_dir: Path,
    semaphore: asyncio.Semaphore,
) -> RowResult:
    async with semaphore:
        try:
            audio = await asyncio.to_thread(generator.synthesize_numpy, row_task.text)
            output_path = output_dir / f"{row_task.phone}.wav"
            await asyncio.to_thread(generator.save_wav, audio, output_path)
            return RowResult(
                index=row_task.index,
                name=row_task.name,
                phone=row_task.phone,
                status="success",
                message="Generated",
                output_path=str(output_path),
            )
        except Exception as exc:
            return RowResult(
                index=row_task.index,
                name=row_task.name,
                phone=row_task.phone,
                status="failed",
                message=str(exc),
            )


async def process_all_rows_async(
    generator: KokoroVoiceGenerator,
    row_tasks: list[RowTask],
    output_dir: Path,
    max_concurrency: int,
) -> list[RowResult]:
    semaphore = asyncio.Semaphore(max(1, max_concurrency))
    coroutines = [
        process_single_row(generator, task, output_dir, semaphore) for task in row_tasks
    ]
    results = await asyncio.gather(*coroutines)
    return results


def run_async_job(coro: Any) -> Any:
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
    except RuntimeError:
        pass
    return asyncio.run(coro)


def generate_from_csv(
    csv_file_path: str,
    template_text: str,
    drive_output_dir: str,
    max_concurrency: int,
) -> tuple[pd.DataFrame, str]:
    if not csv_file_path:
        empty_df = pd.DataFrame(columns=["index", "name", "phone", "status", "message", "output_path"])
        return empty_df, "Please upload a CSV file."

    generator = KokoroVoiceGenerator()
    preflight = generator.preflight()

    mount_msg = generator.mount_drive_if_colab()

    df = pd.read_csv(csv_file_path, dtype={"Name": "string", "No": "string"})
    row_tasks, invalid_results = validate_and_prepare_rows(df, template_text)

    output_dir = Path(drive_output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_results = run_async_job(
        process_all_rows_async(
            generator=generator,
            row_tasks=row_tasks,
            output_dir=output_dir,
            max_concurrency=max_concurrency,
        )
    )

    all_results = invalid_results + generated_results
    all_results.sort(key=lambda item: item.index)

    result_df = pd.DataFrame(
        [
            {
                "index": result.index,
                "name": result.name,
                "phone": result.phone,
                "status": result.status,
                "message": result.message,
                "output_path": result.output_path,
            }
            for result in all_results
        ]
    )

    success_count = int((result_df["status"] == "success").sum()) if not result_df.empty else 0
    failed_count = int((result_df["status"] == "failed").sum()) if not result_df.empty else 0
    skipped_count = int((result_df["status"] == "skipped").sum()) if not result_df.empty else 0

    summary = (
        f"{mount_msg}\n"
        f"Device: {preflight['device']} (CUDA available: {preflight['cuda_available']})\n"
        f"Rows total: {len(df)} | Success: {success_count} | Failed: {failed_count} | Skipped: {skipped_count}\n"
        f"Output directory: {output_dir}"
    )

    return result_df, summary


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Kokoro 82M Voice Notes") as demo:
        gr.Markdown("## Kokoro 82M Voice Notes Generator")
        gr.Markdown(
            "Upload a CSV with columns **Name** and **No**. Use `{{Name}}` in the message template. "
            "WAV files are saved using phone number as filename."
        )

        with gr.Row():
            csv_file = gr.File(label="CSV File", file_types=[".csv"], type="filepath")
            output_dir = gr.Textbox(
                label="Google Drive Output Folder",
                value=DEFAULT_OUTPUT_DIR,
                lines=1,
            )

        template_text = gr.Textbox(
            label="Message Template",
            value=DEFAULT_TEMPLATE,
            lines=4,
        )

        max_concurrency = gr.Slider(
            label="Async Concurrency",
            minimum=1,
            maximum=16,
            value=DEFAULT_MAX_CONCURRENCY,
            step=1,
        )

        run_button = gr.Button("Generate Voice Notes", variant="primary")

        result_table = gr.Dataframe(
            label="Generation Results",
            headers=["index", "name", "phone", "status", "message", "output_path"],
            wrap=True,
        )
        summary_text = gr.Textbox(label="Run Summary", lines=5)

        run_button.click(
            fn=generate_from_csv,
            inputs=[csv_file, template_text, output_dir, max_concurrency],
            outputs=[result_table, summary_text],
            api_name="generate_voice_notes",
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.queue(default_concurrency_limit=1)
    app.launch(share=True)
