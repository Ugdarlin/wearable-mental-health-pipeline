"""
Local Data Ingestion & Alignment Module
Securely loads de-identified multimodal sensor streams, daily journals, and clinical baseline data.
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

class LocalDataLoader:
    """
    Handles privacy-preserving local ingestion of multimodal patient telemetry.
    Normalizes timezones, parses chronological timestamps, and partitions data
    into standardized 7-day longitudinal weekly observation frames.
    """
    def __init__(self, data_source: Optional[str] = None):
        self.data_source = data_source
        self.raw_data: Dict[str, Any] = {
            "hr_stats": pd.DataFrame(),
            "hr_spikes": pd.DataFrame(),
            "activity": pd.DataFrame(),
            "journal": [],
            "clinical_context": ""
        }

    def load_from_json(self, json_path: str) -> Dict[str, Any]:
        """
        Loads multimodal synthetic / de-identified telemetry from a structured JSON file.
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Input data file not found: {json_path}")

        print(f"[LocalDataLoader] Ingesting secure telemetry from: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        # Parse Heart Rate Stats
        if "daily_heartrate_stats" in payload and payload["daily_heartrate_stats"]:
            df_hr_stats = pd.DataFrame(payload["daily_heartrate_stats"])
            if "date" in df_hr_stats.columns:
                df_hr_stats["date"] = pd.to_datetime(df_hr_stats["date"], utc=True)
            self.raw_data["hr_stats"] = df_hr_stats
        else:
            self.raw_data["hr_stats"] = pd.DataFrame()

        # Parse Heart Rate Spikes / Arousal Epochs
        if "above_average_heartrate" in payload and payload["above_average_heartrate"]:
            df_hr_spikes = pd.DataFrame(payload["above_average_heartrate"])
            if "timestamp" in df_hr_spikes.columns:
                df_hr_spikes["timestamp"] = pd.to_datetime(df_hr_spikes["timestamp"], utc=True)
            self.raw_data["hr_spikes"] = df_hr_spikes
        else:
            self.raw_data["hr_spikes"] = pd.DataFrame()

        # Parse Activity Recognition Streams
        if "activity_logs" in payload and payload["activity_logs"]:
            df_act = pd.DataFrame(payload["activity_logs"])
            if "start_time" in df_act.columns:
                df_act["start_time"] = pd.to_datetime(df_act["start_time"], utc=True)
            if "end_time" in df_act.columns:
                df_act["end_time"] = pd.to_datetime(df_act["end_time"], utc=True)
            self.raw_data["activity"] = df_act
        else:
            self.raw_data["activity"] = pd.DataFrame()

        # Parse Daily Digital Journals
        self.raw_data["journal"] = payload.get("daily_journals", [])
        for entry in self.raw_data["journal"]:
            if "date" in entry:
                entry["date_obj"] = pd.to_datetime(entry["date"], utc=True)

        # Parse Clinical Baseline Context
        self.raw_data["clinical_context"] = payload.get("clinical_profile", {}).get("baseline_anamnesis", "")

        print(f"[LocalDataLoader] Ingested {len(self.raw_data['hr_stats'])} HR stat records, "
              f"{len(self.raw_data['hr_spikes'])} HR spike epochs, "
              f"{len(self.raw_data['activity'])} activity instances, "
              f"{len(self.raw_data['journal'])} journal entries.")
        return self.raw_data

    def get_timeframe(self) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Calculates the comprehensive chronological boundary (start and end) across all streams.
        """
        start_dates = []
        end_dates = []

        if not self.raw_data["hr_stats"].empty and "date" in self.raw_data["hr_stats"].columns:
            start_dates.append(self.raw_data["hr_stats"]["date"].min())
            end_dates.append(self.raw_data["hr_stats"]["date"].max())

        if not self.raw_data["hr_spikes"].empty and "timestamp" in self.raw_data["hr_spikes"].columns:
            start_dates.append(self.raw_data["hr_spikes"]["timestamp"].min())
            end_dates.append(self.raw_data["hr_spikes"]["timestamp"].max())

        if not self.raw_data["activity"].empty and "start_time" in self.raw_data["activity"].columns:
            start_dates.append(self.raw_data["activity"]["start_time"].min())
            end_dates.append(self.raw_data["activity"]["end_time"].max())

        if self.raw_data["journal"]:
            j_dates = [e["date_obj"] for e in self.raw_data["journal"] if "date_obj" in e]
            if j_dates:
                start_dates.append(min(j_dates))
                end_dates.append(max(j_dates))

        if not start_dates:
            return None, None

        return min(start_dates), max(end_dates)

    def get_weekly_slices(self) -> List[Dict[str, Any]]:
        """
        Partitions the ingested continuous longitudinal data streams into discrete 7-day weekly observation frames.
        """
        start_date, end_date = self.get_timeframe()
        if not start_date or not end_date:
            return []

        weekly_slices = []
        current_start = start_date
        week_idx = 1

        while current_start <= end_date:
            current_end = current_start + pd.Timedelta(days=6, hours=23, minutes=59, seconds=59)

            week_frame = {
                "week_num": week_idx,
                "start_date": current_start,
                "end_date": current_end,
                "hr_stats": pd.DataFrame(),
                "hr_spikes": pd.DataFrame(),
                "activity": pd.DataFrame(),
                "journal": [],
                "clinical_context": self.raw_data["clinical_context"]
            }

            if not self.raw_data["hr_stats"].empty:
                week_frame["hr_stats"] = self.raw_data["hr_stats"][
                    (self.raw_data["hr_stats"]["date"] >= current_start) &
                    (self.raw_data["hr_stats"]["date"] <= current_end)
                ].copy()

            if not self.raw_data["hr_spikes"].empty:
                week_frame["hr_spikes"] = self.raw_data["hr_spikes"][
                    (self.raw_data["hr_spikes"]["timestamp"] >= current_start) &
                    (self.raw_data["hr_spikes"]["timestamp"] <= current_end)
                ].copy()

            if not self.raw_data["activity"].empty:
                week_frame["activity"] = self.raw_data["activity"][
                    (self.raw_data["activity"]["start_time"] >= current_start) &
                    (self.raw_data["activity"]["start_time"] <= current_end)
                ].copy()

            if self.raw_data["journal"]:
                week_frame["journal"] = [
                    entry for entry in self.raw_data["journal"]
                    if "date_obj" in entry and current_start <= entry["date_obj"] <= current_end
                ]

            weekly_slices.append(week_frame)
            current_start += pd.Timedelta(days=7)
            week_idx += 1

        return weekly_slices

if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "sample_data/deidentified_inputs.json"
    loader = LocalDataLoader()
    loader.load_from_json(data_path)
    slices = loader.get_weekly_slices()
    print(f"Successfully loaded {len(slices)} weekly observation frames.")
