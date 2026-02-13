import asyncio
import io
import os
import tempfile
import zipfile
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
STORAGE_DRIVE = "Google Drive"
STORAGE_MEMORY = "In-Memory Download"
OPERATION_BATCH = "Batch (CSV)"
OPERATION_TEST = "Custom Test (No CSV)"


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

    def _extract_audio_candidate(self, item: Any) -> Any:
        if isinstance(item, tuple) and len(item) >= 3:
            return item[2]
        if isinstance(item, dict):
            if "audio" in item:
                return item["audio"]
            if "wav" in item:
                return item["wav"]
        if hasattr(item, "audio"):
            return getattr(item, "audio")
        return item

    def _flatten_audio_arrays(self, value: Any) -> list[np.ndarray]:
        if value is None:
            return []

        if isinstance(value, torch.Tensor):
            array = value.detach().float().cpu().numpy().reshape(-1)
            return [array] if array.size > 0 else []

        if isinstance(value, np.ndarray):
            array = value.astype(np.float32, copy=False).reshape(-1)
            return [array] if array.size > 0 else []

        if isinstance(value, (list, tuple)):
            arrays: list[np.ndarray] = []
            for part in value:
                arrays.extend(self._flatten_audio_arrays(part))
            return arrays

        try:
            array = np.asarray(value, dtype=np.float32).reshape(-1)
            return [array] if array.size > 0 else []
        except Exception:
            return []

    def synthesize_numpy(self, text: str) -> np.ndarray:
        self.ensure_pipeline()
        assert self._pipeline is not None

        generated = self._pipeline(text, voice=self.voice)

        segments: list[np.ndarray] = []
        for item in generated:
            audio_candidate = self._extract_audio_candidate(item)
            arrays = self._flatten_audio_arrays(audio_candidate)
            if arrays:
                segments.extend(arrays)

        if not segments:
            raise RuntimeError("No audio generated for input text")

        combined = np.concatenate(segments)
        return np.clip(combined, -1.0, 1.0)

    def save_wav(self, audio: np.ndarray, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio, self.sample_rate, subtype="PCM_16")


_GENERATOR: KokoroVoiceGenerator | None = None


def get_generator() -> KokoroVoiceGenerator:
    global _GENERATOR
    if _GENERATOR is None:
        _GENERATOR = KokoroVoiceGenerator()
    return _GENERATOR


def create_zip_from_audio_pairs(
    audio_pairs: list[tuple[RowResult, np.ndarray]],
    sample_rate: int,
) -> str | None:
    if not audio_pairs:
        return None

    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    temp_zip_path = temp_zip.name
    temp_zip.close()

    with zipfile.ZipFile(temp_zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for row_result, audio in audio_pairs:
            wav_bytes = io.BytesIO()
            sf.write(wav_bytes, audio, sample_rate, format="WAV", subtype="PCM_16")
            archive.writestr(f"{row_result.phone}.wav", wav_bytes.getvalue())

    return temp_zip_path


def empty_results_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["index", "name", "phone", "status", "message", "output_path"])


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


async def process_single_row_memory(
    generator: KokoroVoiceGenerator,
    row_task: RowTask,
    semaphore: asyncio.Semaphore,
) -> tuple[RowResult, np.ndarray | None]:
    async with semaphore:
        try:
            audio = await asyncio.to_thread(generator.synthesize_numpy, row_task.text)
            result = RowResult(
                index=row_task.index,
                name=row_task.name,
                phone=row_task.phone,
                status="success",
                message="Generated",
                output_path=f"{row_task.phone}.wav",
            )
            return result, audio
        except Exception as exc:
            result = RowResult(
                index=row_task.index,
                name=row_task.name,
                phone=row_task.phone,
                status="failed",
                message=str(exc),
            )
            return result, None


async def process_all_rows_async_memory(
    generator: KokoroVoiceGenerator,
    row_tasks: list[RowTask],
    max_concurrency: int,
) -> list[tuple[RowResult, np.ndarray | None]]:
    semaphore = asyncio.Semaphore(max(1, max_concurrency))
    coroutines = [
        process_single_row_memory(generator, task, semaphore) for task in row_tasks
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
    storage_mode: str,
    drive_output_dir: str,
    max_concurrency: int,
) -> tuple[pd.DataFrame, str, str | None]:
    if not csv_file_path:
        return empty_results_df(), "Please upload a CSV file.", None

    generator = get_generator()
    preflight = generator.preflight()

    df = pd.read_csv(csv_file_path, dtype={"Name": "string", "No": "string"})
    row_tasks, invalid_results = validate_and_prepare_rows(df, template_text)

    zip_download_path: str | None = None
    if storage_mode == STORAGE_DRIVE:
        mount_msg = generator.mount_drive_if_colab()
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
        output_location = str(output_dir)
    else:
        mount_msg = "Google Drive not used (In-Memory Download mode)."
        memory_results = run_async_job(
            process_all_rows_async_memory(
                generator=generator,
                row_tasks=row_tasks,
                max_concurrency=max_concurrency,
            )
        )
        generated_results = [row_result for row_result, _ in memory_results]
        success_audio_pairs = [
            (row_result, audio)
            for row_result, audio in memory_results
            if row_result.status == "success" and audio is not None
        ]
        zip_download_path = create_zip_from_audio_pairs(success_audio_pairs, generator.sample_rate)
        output_location = "Download ZIP from UI"

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
        f"Storage mode: {storage_mode}\n"
        f"Rows total: {len(df)} | Success: {success_count} | Failed: {failed_count} | Skipped: {skipped_count}\n"
        f"Output: {output_location}"
    )

    return result_df, summary, zip_download_path


def check_model_health() -> str:
    generator = get_generator()
    preflight = generator.preflight()
    try:
        test_audio = generator.synthesize_numpy("Hello, this is a Kokoro model health check.")
        duration_seconds = round(float(test_audio.size) / float(generator.sample_rate), 2)
        return (
            "✅ Kokoro installed and working\n"
            f"Device: {preflight['device']} (CUDA available: {preflight['cuda_available']})\n"
            f"Torch: {preflight['torch_version']}\n"
            f"Test audio generated: {duration_seconds}s"
        )
    except Exception as exc:
        return (
            "❌ Kokoro check failed\n"
            f"Device: {preflight['device']} (CUDA available: {preflight['cuda_available']})\n"
            f"Error: {exc}"
        )


def generate_single_test(
    name: str,
    phone: str,
    template_text: str,
    storage_mode: str,
    drive_output_dir: str,
) -> tuple[tuple[int, np.ndarray] | None, str | None, str]:
    clean_name = (name or "").strip()
    clean_phone = (phone or "").strip()
    if not clean_name:
        return None, None, "Please enter Name for single test generation."
    if not clean_phone:
        return None, None, "Please enter No for single test generation."

    generator = get_generator()
    text = template_text.replace("{{Name}}", clean_name).replace("{{NAME}}", clean_name)

    try:
        audio = generator.synthesize_numpy(text)
        if storage_mode == STORAGE_DRIVE:
            mount_msg = generator.mount_drive_if_colab()
            output_dir = Path(drive_output_dir).expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{clean_phone}.wav"
            generator.save_wav(audio, output_path)
            status = f"{mount_msg}\nSaved to: {output_path}"
            return (generator.sample_rate, audio), str(output_path), status

        temp_dir = Path(tempfile.mkdtemp(prefix="kokoro_test_"))
        temp_wav_path = temp_dir / f"{clean_phone}.wav"
        generator.save_wav(audio, temp_wav_path)
        status = f"Generated in memory mode. Download WAV from UI output: {temp_wav_path.name}"
        return (generator.sample_rate, audio), str(temp_wav_path), status
    except Exception as exc:
        return None, None, f"Failed to generate single test audio: {exc}"


def generate_voice_notes(
    operation_mode: str,
    csv_file_path: str,
    test_name: str,
    test_phone: str,
    template_text: str,
    storage_mode: str,
    drive_output_dir: str,
    max_concurrency: int,
) -> tuple[pd.DataFrame, str, str | None, tuple[int, np.ndarray] | None]:
    if operation_mode == OPERATION_TEST:
        test_audio, test_file_path, test_status = generate_single_test(
            name=test_name,
            phone=test_phone,
            template_text=template_text,
            storage_mode=storage_mode,
            drive_output_dir=drive_output_dir,
        )
        result_status = "success" if test_audio is not None else "failed"
        result_message = "Generated" if test_audio is not None else test_status
        result_output = f"{(test_phone or '').strip()}.wav" if test_audio is not None else ""
        result_df = pd.DataFrame(
            [
                {
                    "index": 0,
                    "name": (test_name or "").strip(),
                    "phone": (test_phone or "").strip(),
                    "status": result_status,
                    "message": result_message,
                    "output_path": result_output,
                }
            ]
        )
        summary = (
            "Custom Test mode\n"
            f"Storage mode: {storage_mode}\n"
            f"Status: {test_status}"
        )
        return result_df, summary, test_file_path, test_audio

    batch_df, batch_summary, batch_zip = generate_from_csv(
        csv_file_path=csv_file_path,
        template_text=template_text,
        storage_mode=storage_mode,
        drive_output_dir=drive_output_dir,
        max_concurrency=max_concurrency,
    )
    return batch_df, batch_summary, batch_zip, None


def update_operation_visibility(operation_mode: str) -> tuple[dict[str, bool], dict[str, bool], dict[str, bool], dict[str, bool]]:
    is_batch = operation_mode == OPERATION_BATCH
    return (
        gr.update(visible=is_batch),
        gr.update(visible=not is_batch),
        gr.update(visible=not is_batch),
        gr.update(visible=is_batch),
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Kokoro 82M Voice Notes") as demo:
        gr.Markdown("## Kokoro 82M Voice Notes Generator")
        gr.Markdown(
            "Use one flow for both modes: select Batch (CSV) or Custom Test (No CSV). "
            "Use `{{Name}}` in message template. Files are generated using `No.wav`."
        )

        health_button = gr.Button("Check Kokoro Model", variant="secondary")
        health_status = gr.Textbox(label="Model Health", lines=4)

        with gr.Row():
            operation_mode = gr.Radio(
                label="Operation Mode",
                choices=[OPERATION_BATCH, OPERATION_TEST],
                value=OPERATION_BATCH,
                scale=1,
            )
            storage_mode = gr.Radio(
                label="Storage Mode",
                choices=[STORAGE_DRIVE, STORAGE_MEMORY],
                value=STORAGE_DRIVE,
                scale=1,
            )
            output_dir = gr.Textbox(
                label="Google Drive Output Folder",
                value=DEFAULT_OUTPUT_DIR,
                lines=1,
                scale=2,
            )

        with gr.Group():
            with gr.Row():
                csv_file = gr.File(label="CSV File (Batch Mode)", file_types=[".csv"], type="filepath", scale=2)
                test_name = gr.Textbox(label="Test Name (Custom Test)", value="Test User", scale=1)
                test_phone = gr.Textbox(label="Test No (Custom Test)", value="0000000000", scale=1)

            template_text = gr.Textbox(
                label="Message Template",
                value=DEFAULT_TEMPLATE,
                lines=3,
            )

            max_concurrency = gr.Slider(
                label="Async Concurrency (Batch Mode)",
                minimum=1,
                maximum=16,
                value=DEFAULT_MAX_CONCURRENCY,
                step=1,
            )

        run_button = gr.Button("Generate Voice Notes", variant="primary")

        with gr.Group():
            summary_text = gr.Textbox(label="Run / Test Summary", lines=5)
            with gr.Row():
                single_audio = gr.Audio(label="Custom Test Audio", type="numpy")
                download_file = gr.File(label="Download Output (Batch ZIP or Test WAV)")

        with gr.Accordion("Generation Results (Batch)", open=False):
            result_table = gr.Dataframe(
                label="Generation Results",
                headers=["index", "name", "phone", "status", "message", "output_path"],
                wrap=True,
            )

        health_button.click(
            fn=check_model_health,
            inputs=[],
            outputs=[health_status],
            api_name="check_model_health",
        )

        run_button.click(
            fn=generate_voice_notes,
            inputs=[operation_mode, csv_file, test_name, test_phone, template_text, storage_mode, output_dir, max_concurrency],
            outputs=[result_table, summary_text, download_file, single_audio],
            api_name="generate_voice_notes",
        )

        operation_mode.change(
            fn=update_operation_visibility,
            inputs=[operation_mode],
            outputs=[csv_file, test_name, test_phone, max_concurrency],
            api_name="update_operation_visibility",
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.queue(default_concurrency_limit=1)
    app.launch(share=True)
