"""
Multi-Persona Reporter Module
Coordinates quantized local workstation inference (Gemma-3-27b-it bfloat16)
to generate 3 customized report personas with strict epistemic hedging guardrails.
"""

import os
import time
from datetime import datetime
from typing import Dict, Any, Optional

class MultiPersonaReporter:
    """
    Generates three specialized audience-tailored narrative reports:
      1. Clinical Research Log (Long-Form)
      2. Psychologist Summary (Short-Form)
      3. Patient Narrative (Empathetic-Form)
    """
    def __init__(self, model_id: str = "google/gemma-3-27b-it", load_in_4bit: bool = True):
        self.model_id = model_id
        self.load_in_4bit = load_in_4bit
        self.generator = None
        self.tokenizer = None
        self._init_model()

    def _init_model(self):
        """Initializes the local quantized HuggingFace model pipeline."""
        if self.model_id.lower() in ("mock", "none", "emulate"):
            print("[MultiPersonaReporter] Operating in offline deterministic emulation mode.")
            self.generator = None
            return

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        except Exception as e:
            print(f"[MultiPersonaReporter] Notice: PyTorch/transformers unavailable ({e}). Running in deterministic fallback mode.")
            self.generator = None
            return

        print(f"[MultiPersonaReporter] Loading local LLM: {self.model_id} (bfloat16)...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, local_files_only=False)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                load_in_4bit=self.load_in_4bit,
                device_map="auto"
            )
            self.generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=self.tokenizer
            )
            print("[MultiPersonaReporter] Local LLM loaded successfully.")
        except Exception as e:
            print(f"[MultiPersonaReporter] Notice: GPU model could not be loaded ({e}). Operating in deterministic emulation mode.")
            self.generator = None

    def generate_all_reports(self, context_data: Dict[str, Any], previous_report_summary: Optional[str] = None) -> Dict[str, str]:
        """
        Synthesizes all three audience personas from structured data context.
        """
        reports = {}
        print("[MultiPersonaReporter] Generating Persona 1: Clinical Research Log (Long-Form)...")
        reports["clinical_research_log"] = self._generate_long_form(context_data, previous_report_summary)

        print("[MultiPersonaReporter] Generating Persona 2: Psychologist Summary (Short-Form)...")
        reports["psychologist_summary"] = self._generate_short_form(context_data)

        print("[MultiPersonaReporter] Generating Persona 3: Patient Narrative (Empathetic-Form)...")
        reports["patient_narrative"] = self._generate_patient_form(context_data)

        return reports

    def _generate_long_form(self, data: Dict[str, Any], prev_report: Optional[str]) -> str:
        dates = data.get("dates", {})
        sensor = data.get("sensor", {})
        journal = data.get("journal", {})
        rag = data.get("rag", "No RAG literature context available.")
        clinical = data.get("clinical", "")

        prev_section = ""
        if prev_report:
            prev_section = f"\n\n### Longitudinal Trajectory (Week N-1 Comparison)\n{prev_report[:1200]}...\n"

        prompt = f"""## Mental Health Monitoring Report - Clinical Research Log

**Report Date:** {datetime.now().strftime("%Y-%m-%d")}
**Participant ID:** codeword
**Reporting Period:** {dates.get('start', 'N/A')} to {dates.get('end', 'N/A')} (Week {dates.get('week_num', '1')})

### 1. Clinical Anamnesis & Baseline Presentation
- Baseline Anamnesis: {clinical}

### 2. Objective Physiological & Behavioral Telemetry (Week {dates.get('week_num', '1')})
- Mean Heart Rate: {sensor.get('hr_analysis', {}).get('avg_hr', 'N/A')}
- Peak Heart Rate: {sensor.get('hr_analysis', {}).get('max_hr', 'N/A')}
- Physiological Arousal Bouts (120s threshold): {sensor.get('hr_analysis', {}).get('notable_events', 'None')}
- Psychomotor Activity Bouts (300s threshold):
{sensor.get('activity_analysis', {}).get('summary', 'None')}

### 3. Subjective Daily Self-Reported Metrics
- Mean Daily Mood (0-100): {journal.get('journal_analysis', {}).get('avg_mood', 'N/A')}
- Mean Daily Nervousness (0-100): {journal.get('journal_analysis', {}).get('avg_nervousness', 'N/A')}
- Sleep Quality (0-100): {journal.get('journal_analysis', {}).get('avg_sleep', 'N/A')}
- Panic Attack Incident Logs:
{journal.get('journal_analysis', {}).get('panic_attacks', 'None')}

### 4. Grounded CBT Clinical Evidence (RAG Context)
{rag}
{prev_section}
### 5. Multi-Audience Synthesis & Epistemic Recommendations
*Note: All therapeutic recommendations are strictly framed as collaborative discussion points for clinical interpretation.*
- In Vivo Situational Exposure: Target identified situational avoidance (e.g. social dining environments) using graded hierarchies.
- Competing Response Training: Deploy Habit Reversal Training (HRT) for prominent late-night psychomotor behaviors.
- Interoceptive Habituation: Address catastrophic misinterpretation of autonomic arousal spikes.
"""
        if self.generator is not None:
            return self._call_llm(prompt, persona="Clinical Researcher")
        return prompt

    def _generate_short_form(self, data: Dict[str, Any]) -> str:
        dates = data.get("dates", {})
        sensor = data.get("sensor", {})
        journal = data.get("journal", {})

        prompt = f"""## Therapist Summary Report (Short-Form)

**Reporting Period:** {dates.get('start', 'N/A')} to {dates.get('end', 'N/A')} (Week {dates.get('week_num', '1')})
**Primary Diagnosis:** Agoraphobia with Panic Disorder

### 1. Risk & Contextual Factors
Patient demonstrates sustained autonomic reactivity (Mean HR: {sensor.get('hr_analysis', {}).get('avg_hr')}) coupled with subjective nervousness ({journal.get('journal_analysis', {}).get('avg_nervousness')}).

### 2. Observed Factors (Last 7 Days)
- Physiological Arousal: {sensor.get('hr_analysis', {}).get('notable_events')}
- Dominant Psychomotor Coping:
{sensor.get('activity_analysis', {}).get('summary')}
- Reported Panic Attacks: {journal.get('journal_analysis', {}).get('panic_attacks')}

### 3. Likelihood of Future Panic Attacks
Elevated psychomotor activity during vulnerable diurnal windows suggests ongoing autonomic vulnerability; structured interoceptive exposure is indicated.

### 4. Collaborative Discussion Points (CBT Focus)
1. Initiate Habit Reversal Training (HRT) for dominant psychomotor fidgeting bouts.
2. Review situational triggers associated with restaurant dining and public venues.
3. Reinforce interoceptive exposure to decouple autonomic arousal spikes from panic escalation.
"""
        if self.generator is not None:
            return self._call_llm(prompt, persona="Clinical Psychologist")
        return prompt

    def _generate_patient_form(self, data: Dict[str, Any]) -> str:
        sensor = data.get("sensor", {})
        journal = data.get("journal", {})

        prompt = f"""## Weekly Reflection & Insights for You

Hello! Here is your personalized summary for the past week:

### 🌟 What We Noticed Together
- **Heart & Activity:** Your average heart rate was {sensor.get('hr_analysis', {}).get('avg_hr', 'normal')}. We noticed some periods of physical restlessness, especially during the evenings and quiet hours.
- **Sleep & Mood:** You reported an average sleep quality of {journal.get('journal_analysis', {}).get('avg_sleep', 'N/A')} and daily mood around {journal.get('journal_analysis', {}).get('avg_mood', 'N/A')}.

### 💡 Gentle Suggestions to Discuss in Therapy
- Notice if certain activities (like hand scratching or restlessness) happen when you feel tense. Practicing a calming alternative—like holding a smooth stone or clenching and releasing your fists—can help ground you.
- Remember that heart rate changes during social situations (like eating out) are natural body responses that will pass safely.

*Disclaimer: This reflection is designed to support your personal journey and provide talking points for your sessions with your therapist.*
"""
        if self.generator is not None:
            return self._call_llm(prompt, persona="Empathetic Patient Coach")
        return prompt

    def _call_llm(self, content_prompt: str, persona: str) -> str:
        chat_prompt = [
            {"role": "system", "content": f"You are an AI clinical assistant acting as a {persona}. Follow strict clinical safety guardrails."},
            {"role": "user", "content": content_prompt}
        ]
        start_t = time.time()
        formatted = self.tokenizer.apply_chat_template(chat_prompt, tokenize=False, add_generation_prompt=True)
        outputs = self.generator(
            formatted,
            max_new_tokens=1500,
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )
        res = outputs[0]["generated_text"][len(formatted):].strip()
        elapsed = time.time() - start_t
        print(f"[MultiPersonaReporter] {persona} generated in {elapsed:.2f}s")
        return res
