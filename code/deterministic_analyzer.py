"""
Deterministic Mathematical Feature Extraction Module
Executes exact non-generative biometric and behavioral calculations.
Calibrated thresholds:
  - 120-second physiological arousal inter-spike boundary
  - 300-second psychomotor fidgeting inter-event clustering boundary
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List

class DeterministicAnalyzer:
    """
    Performs deterministic numerical feature extraction on wearable telemetry and journal entries.
    Guarantees strict separation between mathematical calculation and natural language synthesis.
    """
    def __init__(self, hr_bout_threshold_s: float = 120.0, activity_bout_threshold_s: float = 300.0):
        self.hr_bout_threshold_s = hr_bout_threshold_s
        self.activity_bout_threshold_s = activity_bout_threshold_s

    def analyze_week(self, week_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes end-to-end deterministic mathematical analysis for a single weekly frame.
        """
        return {
            "week_num": week_data["week_num"],
            "start_date": week_data["start_date"].strftime("%Y-%m-%d") if hasattr(week_data["start_date"], "strftime") else str(week_data["start_date"]),
            "end_date": week_data["end_date"].strftime("%Y-%m-%d") if hasattr(week_data["end_date"], "strftime") else str(week_data["end_date"]),
            "hr_analysis": self._analyze_heart_rate(week_data["hr_stats"], week_data["hr_spikes"]),
            "activity_analysis": self._analyze_activity(week_data["activity"]),
            "journal_analysis": self._analyze_journal(week_data["journal"])
        }

    def _analyze_heart_rate(self, df_stats: pd.DataFrame, df_spikes: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates heart rate metrics and extracts 120-second physiological arousal bouts.
        """
        if df_stats.empty:
            return {
                "avg_hr": "N/A",
                "max_hr": "N/A",
                "notable_events": "No heart rate telemetry recorded during this period.",
                "raw_avg": None,
                "bout_count": 0,
                "total_arousal_minutes": 0.0
            }

        avg_hr = df_stats["daily_avg_heartrate"].mean()
        max_hr = df_stats["daily_max_heartrate"].max() if "daily_max_heartrate" in df_stats.columns else df_stats["daily_avg_heartrate"].max()

        notable_events = "No significant periods of sustained above-average heart rate recorded."
        num_bouts = 0
        total_duration_s = 0.0
        max_bout_min = 0.0
        peak_hr = max_hr

        if not df_spikes.empty and "timestamp" in df_spikes.columns:
            df_spikes = df_spikes.sort_values("timestamp").copy()
            time_diff = df_spikes["timestamp"].diff().dt.total_seconds()
            
            # 120-second threshold: gaps > 120s demarcate independent arousal events
            bout_starts = time_diff > self.hr_bout_threshold_s
            df_spikes["bout_id"] = bout_starts.cumsum()
            
            # Each spike represents active epoch duration (clipped to 60s max per sample)
            df_spikes["duration_s"] = time_diff.fillna(60.0).clip(upper=60.0)
            bout_durations = df_spikes.groupby("bout_id")["duration_s"].sum()

            num_bouts = len(bout_durations)
            total_duration_s = bout_durations.sum()
            max_bout_min = (bout_durations.max() / 60.0) if num_bouts > 0 else 0.0
            peak_hr = df_spikes["heart_rate"].max() if "heart_rate" in df_spikes.columns else max_hr

            h, rem = divmod(total_duration_s, 3600)
            m, _ = divmod(rem, 60)

            notable_events = (
                f"Elevated heart rate occurred in {num_bouts} distinct bouts, "
                f"totaling {int(h)}h {int(m)}m. "
                f"The longest bout lasted {max_bout_min:.1f} minutes, "
                f"with an overall peak of {peak_hr:.0f} bpm."
            )

        return {
            "avg_hr": f"{avg_hr:.2f} bpm",
            "max_hr": f"{peak_hr:.0f} bpm",
            "notable_events": notable_events,
            "raw_avg": round(float(avg_hr), 2),
            "bout_count": int(num_bouts),
            "total_arousal_minutes": round(total_duration_s / 60.0, 2),
            "longest_bout_min": round(float(max_bout_min), 2)
        }

    def _analyze_activity(self, df_activity: pd.DataFrame) -> Dict[str, Any]:
        """
        Extracts high-confidence psychomotor behaviors and calculates 300-second activity bouts.
        """
        if df_activity.empty:
            return {"summary": "No activity telemetry recorded.", "top_activities": [], "raw_bouts": {}}

        df_act = df_activity.copy()
        if "duration_seconds" not in df_act.columns:
            df_act["duration_seconds"] = (df_act["end_time"] - df_act["start_time"]).dt.total_seconds()

        # Calibration filter: high confidence (>=0.80) and window limit (<=4s)
        if "confidence" in df_act.columns:
            df_act = df_act[(df_act["duration_seconds"] <= 4) & (df_act["confidence"] >= 0.80)]
        else:
            df_act = df_act[df_act["duration_seconds"] <= 4]

        if df_act.empty:
            return {"summary": "No high-confidence activity bouts detected.", "top_activities": [], "raw_bouts": {}}

        df_act.sort_values("start_time", inplace=True)
        df_act["time_diff"] = df_act["start_time"].diff().dt.total_seconds().fillna(0)
        
        # 300-second threshold: activity change or gap > 300s defines new bout
        df_act["bout_start"] = (
            (df_act["predicted_activity"] != df_act["predicted_activity"].shift(1)) |
            (df_act["time_diff"] > self.activity_bout_threshold_s)
        )
        df_act["bout_id"] = df_act["bout_start"].cumsum()

        bout_analysis = df_act.groupby(["predicted_activity", "bout_id"]).agg(
            bout_duration_s=("duration_seconds", "sum"),
            start_time=("start_time", "min")
        ).reset_index()

        total_durations = df_act.groupby("predicted_activity")["duration_seconds"].sum()
        top_activities = total_durations.sort_values(ascending=False).head(4)

        summaries = []
        raw_bouts_dict = {}

        for activity_name, duration_sec in top_activities.items():
            act_bouts = bout_analysis[bout_analysis["predicted_activity"] == activity_name]
            num_bouts = len(act_bouts)
            max_bout_min = (act_bouts["bout_duration_s"].max() / 60.0) if num_bouts > 0 else 0.0

            # Diurnal binning: Late Night (0-6), Morning (6-12), Afternoon (12-18), Evening (18-24)
            act_bouts = act_bouts.copy()
            act_bouts["hour"] = act_bouts["start_time"].dt.hour
            bins = [0, 6, 12, 18, 24]
            labels = ["Late Night (0-6)", "Morning (6-12)", "Afternoon (12-18)", "Evening (18-24)"]
            act_bouts["time_of_day"] = pd.cut(act_bouts["hour"], bins=bins, labels=labels, right=False)
            
            most_freq_time = act_bouts["time_of_day"].value_counts().idxmax() if not act_bouts.empty else "N/A"

            h, rem = divmod(duration_sec, 3600)
            m, _ = divmod(rem, 60)

            clean_label = str(activity_name).replace("_", " ").title()
            summary_str = (
                f"- {clean_label}: Total duration {int(h)}h {int(m)}m. "
                f"Occurred in {num_bouts} distinct bouts. "
                f"Longest bout lasted {max_bout_min:.1f} minutes. "
                f"Most frequent during {most_freq_time}."
            )
            summaries.append(summary_str)
            raw_bouts_dict[str(activity_name)] = int(num_bouts)

        return {
            "summary": "\n".join(summaries),
            "top_activities": list(top_activities.index),
            "raw_bouts": raw_bouts_dict
        }

    def _analyze_journal(self, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Deterministically aggregates subjective patient ratings (mood, nervousness, sleep)
        and parses specific panic attack incidents.
        """
        if not entries:
            return {
                "summary": "No digital journal logs recorded for this period.",
                "avg_mood": "N/A",
                "avg_nervousness": "N/A",
                "avg_sleep": "N/A",
                "panic_attacks": "None reported.",
                "raw_nervousness": 0.0,
                "panic_attack_count": 0
            }

        key_map = {"stimmung": "mood", "nervositaet": "nervousness", "schlafQualitaet": "sleep_quality"}
        averages = {}
        for de_key, res_key in key_map.items():
            vals = [float(e[de_key]) for e in entries if de_key in e and e[de_key] is not None and str(e[de_key]).strip() != ""]
            averages[res_key] = (sum(vals) / len(vals)) if vals else 0.0

        panic_summaries = []
        total_panic_count = 0

        for e in entries:
            count = int(e.get("anzahlPanikattacken", 0)) if str(e.get("anzahlPanikattacken", "0")).isdigit() else 0
            total_panic_count += count
            if count > 0:
                details = e.get("panikattackenDetails", [])
                date_str = str(e.get("date", "Unknown Date"))[:10]
                if details:
                    for d in details:
                        s = (
                            f"On {date_str}: Intensity {d.get('intensitaet', '?')}/100. "
                            f"Symptoms: {', '.join(d.get('symptome', [])) if isinstance(d.get('symptome'), list) else d.get('symptome', 'N/A')}. "
                            f"Situation: {d.get('situation', 'N/A')}."
                        )
                        panic_summaries.append(s)
                else:
                    panic_summaries.append(f"On {date_str}: {count} panic attack(s) logged without detailed questionnaire.")

        return {
            "avg_mood": f"{averages.get('mood', 0.0):.2f}/100",
            "avg_nervousness": f"{averages.get('nervousness', 0.0):.2f}/100",
            "avg_sleep": f"{averages.get('sleep_quality', 0.0):.2f}/100",
            "panic_attacks": "\n".join(panic_summaries) if panic_summaries else "None reported.",
            "raw_nervousness": round(averages.get("nervousness", 0.0), 2),
            "panic_attack_count": total_panic_count
        }
