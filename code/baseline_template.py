"""
Non-LLM Statistical Baseline Script
Generates purely rule-based, template-driven statistical summary reports
for comparative control conditions against the generative multi-persona pipeline.
"""

import os
import sys
import argparse
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from local_loader import LocalDataLoader
from deterministic_analyzer import DeterministicAnalyzer

def generate_flat_statistical_dashboard(analysis_results: dict, clinical_context: str) -> str:
    """Generates non-generative, tabular/template summary report."""
    hr = analysis_results.get("hr_analysis", {})
    act = analysis_results.get("activity_analysis", {})
    jnl = analysis_results.get("journal_analysis", {})

    dashboard = f"""=======================================================
NON-LLM STATISTICAL BASELINE SUMMARY DASHBOARD
=======================================================
Generation Date : {datetime.now().strftime('%Y-%m-%d')}
Observation Period: {analysis_results.get('start_date')} to {analysis_results.get('end_date')} (Week {analysis_results.get('week_num')})

1. CLINICAL BASELINE PROFILE
----------------------------
{clinical_context[:300]}...

2. DETERMINISTIC PHYSIOLOGICAL METRICS
--------------------------------------
Mean Heart Rate (7-day)    : {hr.get('avg_hr', 'N/A')}
Peak Heart Rate Recorded   : {hr.get('max_hr', 'N/A')}
Arousal Bouts (>120s) Count: {hr.get('bout_count', 0)}
Total Arousal Duration     : {hr.get('total_arousal_minutes', 0.0)} minutes
Arousal Event Detail       : {hr.get('notable_events', 'None')}

3. DETERMINISTIC BEHAVIORAL ACTIVITY METRICS
--------------------------------------------
{act.get('summary', 'No activity detected.')}

4. DIGITAL JOURNAL SELF-REPORT RATINGS
--------------------------------------
Mean Subjective Mood (0-100)        : {jnl.get('avg_mood', 'N/A')}
Mean Subjective Nervousness (0-100) : {jnl.get('avg_nervousness', 'N/A')}
Mean Sleep Quality (0-100)          : {jnl.get('avg_sleep', 'N/A')}
Logged Panic Attack Count           : {jnl.get('panic_attack_count', 0)}
Panic Incident Log Details          :
{jnl.get('panic_attacks', 'None')}

=======================================================
DISCLAIMER: Non-LLM baseline dashboard containing raw descriptive telemetry.
Does not perform semantic synthesis, longitudinal trajectory reasoning, or CBT recommendations.
=======================================================
"""
    return dashboard

def main():
    parser = argparse.ArgumentParser(description="Non-LLM Statistical Baseline Generator")
    parser.add_argument("--input_sensor_data", type=str, default="sample_data/deidentified_inputs.json", help="Path to input sensor JSON")
    parser.add_argument("--output_dir", type=str, default="output/baseline/", help="Output directory for baseline dashboards")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    loader = LocalDataLoader()
    loader.load_from_json(args.input_sensor_data)
    analyzer = DeterministicAnalyzer()

    weeks = loader.get_weekly_slices()
    print(f"[Baseline] Generating flat statistical dashboards for {len(weeks)} weeks...")

    for w in weeks:
        week_num = w["week_num"]
        analysis = analyzer.analyze_week(w)
        dashboard_text = generate_flat_statistical_dashboard(analysis, w.get("clinical_context", ""))
        
        out_path = os.path.join(args.output_dir, f"baseline_dashboard_week_{week_num}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(dashboard_text)
        print(f"[Baseline] Saved Week {week_num} dashboard to {out_path}")

    print("[Baseline] Execution completed successfully.")

if __name__ == "__main__":
    main()
