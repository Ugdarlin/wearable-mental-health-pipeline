"""
Central Orchestrator Module
Coordinates the multi-stage pipeline, active recursive context memory loop (N-1 -> N),
and experimental ablation controls.
"""

import os
import sys
import json
import argparse
from typing import Optional

# Add code directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from local_loader import LocalDataLoader
from deterministic_analyzer import DeterministicAnalyzer
from RAG_retrieval import RAGRetriever
from multi_persona_reporter import MultiPersonaReporter

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class CentralOrchestrator:
    """
    Main system coordinator executing sequential pipeline stages:
      1. Ingestion & Alignment
      2. Deterministic Mathematical Analysis
      3. Knowledge Graph & CBT Literature Retrieval
      4. Multi-Persona Report Generation
      5. Recursive Context Memory Update
    """
    def __init__(self, rag_enabled: bool = True, memory_enabled: bool = True, model_path: str = "google/gemma-3-27b-it"):
        self.rag_enabled = rag_enabled
        self.memory_enabled = memory_enabled
        self.loader = LocalDataLoader()
        self.analyzer = DeterministicAnalyzer()
        rag_model = "mock" if str(model_path).lower() in ("mock", "none", "emulate") else "all-MiniLM-L6-v2"
        self.retriever = RAGRetriever(model_name=rag_model) if self.rag_enabled else None
        self.reporter = MultiPersonaReporter(model_id=model_path)

    def run_pipeline(self, input_sensor_data: str, previous_report_path: Optional[str] = None, output_dir: str = "output/"):
        print("=" * 70)
        print("CENTRAL PIPELINE EXECUTION START")
        print(f"RAG Enabled: {self.rag_enabled} | Memory Enabled: {self.memory_enabled}")
        print(f"Input Data: {input_sensor_data}")
        print("=" * 70)

        os.makedirs(output_dir, exist_ok=True)

        # 1. Ingest Data
        self.loader.load_from_json(input_sensor_data)
        weekly_slices = self.loader.get_weekly_slices()

        if not weekly_slices:
            print("[CentralOrchestrator] Error: No data slices available to process.")
            return

        # 2. Ingest Previous Memory State (if provided & memory enabled)
        previous_summary = None
        if self.memory_enabled and previous_report_path and os.path.exists(previous_report_path):
            with open(previous_report_path, "r", encoding="utf-8") as f:
                if previous_report_path.endswith(".json"):
                    prev_json = json.load(f)
                    previous_summary = prev_json.get("summary_text", str(prev_json))
                else:
                    previous_summary = f.read()
            print(f"[CentralOrchestrator] Injected previous recursive memory from: {previous_report_path}")

        # 3. Process Weekly Observation Frames
        for week in weekly_slices:
            week_num = week["week_num"]
            print(f"\n--- Executing Analysis for Week {week_num} ---")

            # Deterministic Feature Extraction
            analysis_results = self.analyzer.analyze_week(week)

            # RAG Retrieval
            rag_context = "RAG retrieval disabled for this ablation condition."
            if self.rag_enabled and self.retriever:
                query = (
                    f"Patient presentation: Heart rate {analysis_results['hr_analysis']['avg_hr']}, "
                    f"Nervousness {analysis_results['journal_analysis']['avg_nervousness']}, "
                    f"Behaviors: {', '.join(analysis_results['activity_analysis'].get('top_activities', []))}"
                )
                rag_context = self.retriever.retrieve_context(query)

            # Context Assembly
            context_data = {
                "clinical": week.get("clinical_context", ""),
                "sensor": {
                    "hr_analysis": analysis_results["hr_analysis"],
                    "activity_analysis": analysis_results["activity_analysis"]
                },
                "journal": {
                    "journal_analysis": analysis_results["journal_analysis"]
                },
                "rag": rag_context,
                "dates": {
                    "start": analysis_results["start_date"],
                    "end": analysis_results["end_date"],
                    "week_num": week_num
                }
            }

            # Generate Reports
            active_memory = previous_summary if self.memory_enabled else None
            reports = self.reporter.generate_all_reports(context_data, previous_report_summary=active_memory)

            # Save Output Reports
            week_out_dir = os.path.join(output_dir, f"week_{week_num}")
            os.makedirs(week_out_dir, exist_ok=True)

            with open(os.path.join(week_out_dir, "clinical_research_log.txt"), "w", encoding="utf-8") as f:
                f.write(reports["clinical_research_log"])
            with open(os.path.join(week_out_dir, "psychologist_summary.txt"), "w", encoding="utf-8") as f:
                f.write(reports["psychologist_summary"])
            with open(os.path.join(week_out_dir, "patient_narrative.txt"), "w", encoding="utf-8") as f:
                f.write(reports["patient_narrative"])

            # Save Structured Summary for Recursive Memory Loop
            summary_payload = {
                "week_num": week_num,
                "start_date": analysis_results["start_date"],
                "end_date": analysis_results["end_date"],
                "summary_text": reports["clinical_research_log"][:1500]
            }
            with open(os.path.join(week_out_dir, "summary_memory_state.json"), "w", encoding="utf-8") as f:
                json.dump(summary_payload, f, indent=2)

            print(f"[CentralOrchestrator] Successfully generated reports for Week {week_num} in: {week_out_dir}")

            # Update recursive memory loop for subsequent week
            if self.memory_enabled:
                previous_summary = reports["clinical_research_log"]

        print("\n" + "=" * 70)
        print("PIPELINE EXECUTION COMPLETE.")
        print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="Bridging the Gap Between Wearable Sensors and Clinical Practice in Agoraphobia and Panic Disorder: Single-Case Formative Evaluation of a Multi-Stage Large Language Model Pipeline")
    parser.add_argument("--input_sensor_data", type=str, default="sample_data/deidentified_inputs.json", help="Path to input sensor JSON")
    parser.add_argument("--previous_report", type=str, default=None, help="Path to previous week summary (Week N-1)")
    parser.add_argument("--output_dir", type=str, default="output/", help="Directory to save generated reports")
    parser.add_argument("--model_path", type=str, default="google/gemma-3-27b-it", help="Model name or local weights path")
    parser.add_argument("--rag_enabled", type=str2bool, default=True, help="Toggle CBT RAG retrieval module")
    parser.add_argument("--memory_enabled", type=str2bool, default=True, help="Toggle recursive longitudinal memory loop")

    args = parser.parse_args()
    orchestrator = CentralOrchestrator(
        rag_enabled=args.rag_enabled,
        memory_enabled=args.memory_enabled,
        model_path=args.model_path
    )
    orchestrator.run_pipeline(
        input_sensor_data=args.input_sensor_data,
        previous_report_path=args.previous_report,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
