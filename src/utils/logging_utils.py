"""Experiment logging utility.

Appends one row per experiment run to a shared CSV file, following the
experiment log template in the research plan.
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Any
import pandas as pd


class ExperimentLogger:
    """Logs individual experiment configurations and metric outcomes to a CSV file."""

    COLUMNS: list[str] = [
        "run_id",
        "dataset",
        "backbone",
        "method",
        "rank",
        "alpha",
        "target_modules",
        "adapted_layers",
        "trainable_params",
        "total_params",
        "param_pct",
        "peak_gpu_memory_mb",
        "training_time_s",
        "val_auroc",
        "val_accuracy",
        "val_macro_f1",
        "test_auroc",
        "test_accuracy",
        "test_macro_f1",
        "seed",
        "notes",
    ]

    def __init__(self, log_path: str | Path = "experiment_log.csv") -> None:
        """Initialises the logger.

        Creates the CSV file with headers if it does not exist. Appends rows
        to the existing file if it does.

        Args:
            log_path: Path to the target CSV experiment log file.
        """
        self.log_path: Path = Path(log_path)
        if self.log_path.parent != Path("."):
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.log_path.is_file():
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
                writer.writeheader()

    def log(self, **kwargs: Any) -> None:
        """Appends one experiment row to the CSV.

        Accepts keyword arguments matching COLUMNS; missing columns are filled
        with empty string. Logs timestamp automatically.

        Args:
            **kwargs: Arbitrary experiment properties matching COLUMNS keys.
        """
        timestamp_str: str = datetime.now().isoformat()
        existing_notes: str = str(kwargs.get("notes", "")).strip()

        if existing_notes:
            notes_with_time: str = f"[{timestamp_str}] {existing_notes}"
        else:
            notes_with_time = f"[{timestamp_str}]"

        row_dict: dict[str, Any] = {}
        for col in self.COLUMNS:
            if col == "notes":
                row_dict[col] = notes_with_time
            else:
                val: Any = kwargs.get(col, "")
                row_dict[col] = "" if val is None else str(val)

        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writerow(row_dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Reads the full experiment log CSV into a pandas DataFrame.

        Returns empty DataFrame with correct columns if file does not exist.

        Returns:
            DataFrame containing all historical experiment log entries.
        """
        if self.log_path.is_file():
            return pd.read_csv(self.log_path)
        return pd.DataFrame(columns=self.COLUMNS)
