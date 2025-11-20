import os
import rule_engine
from openai import OpenAI


class HybridExplainer:
    """
    Neuro-Symbolic AI engine combining:
    1. Symbolic Rules (MITRE ATT&CK mappings)
    2. LLM-based explanation generation
    """
    def __init__(self, use_openai=True):
        self.use_openai = use_openai
        
        self.rules = [
            (rule_engine.Rule('auth_fail > 10 and protocol == "SSH"'), "T1110: Brute Force"),
            (rule_engine.Rule('bytes_out > 1000000 and dst_port == 443'), "T1048: Exfiltration"),
            (rule_engine.Rule('service == "SMB" and file_access == "write"'), "T1021: Lateral Movement")
        ]
        
        if self.use_openai:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("WARNING: OPENAI_API_KEY not found. Explanations will be rule-based only.")
                self.use_openai = False
            else:
                self.client = OpenAI(api_key=api_key)

    def symbolic_inference(self, graph_features):
        """Match graph features to MITRE ATT&CK rules."""
        detected_tactics = []
        for rule, tactic in self.rules:
            try:
                if rule.matches(graph_features):
                    detected_tactics.append(tactic)
            except Exception as e:
                pass
        return detected_tactics

    def generate_explanation(self, tactics, confidence):
        """Generate human-readable security report."""
        if not tactics:
            return "Anomaly detected, but no specific known pattern matched."
        
        if not self.use_openai:
            return f"Security Alert: Detected tactics {', '.join(tactics)} with {confidence:.2%} confidence."
        
        try:
            prompt = f"""You are a security analyst. An Intrusion Detection System flagged a network flow.
Technical Evidence:
- Detected Tactics: {tactics}
- Model Confidence: {confidence:.2f}

Task: Write a concise, 2-sentence alert explaining WHY this is dangerous."""

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Security Alert: Detected tactics {', '.join(tactics)} with {confidence:.2%} confidence. (LLM unavailable: {str(e)})"
