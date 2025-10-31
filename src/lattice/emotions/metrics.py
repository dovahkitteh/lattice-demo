# src/lattice/emotions/metrics.py
"""
Calculates and logs metrics for evaluating the performance of the emotional system.
This module is responsible for:
- Calculating metrics like distortion rates, mood diversity, and regulation latency.
- Logging these metrics to a structured file for analysis.
- Providing a diagnostic dump function for debugging.
"""
import logging
import json
from collections import Counter
from typing import List, Dict, Any
from datetime import datetime
import numpy as np

from ..models import EpisodicTrace, EmotionState
from ..config import get_emotion_config

logger = logging.getLogger(__name__)
METRICS_LOG_FILE = "logs/emotional_metrics.jsonl"

class MetricsManager:
    """
    Manages the calculation and state of emotional system metrics.
    """
    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = get_emotion_config().config.get("metrics", {})
        
        self.turn_count = 0
        self.positive_distortions = 0
        self.negative_distortions = 0
        self.mood_counts = Counter()
        self.last_n_traces: List[EpisodicTrace] = []
        self.rolling_window_size = config.get("rolling_window_size", 100)
        
        # For advanced metrics
        self.regulation_loop_latency_samples: List[int] = []
        self.parameter_divergence_samples: Dict[str, List[float]] = {
            "temperature": [],
            "top_p": [],
        }
        self.baseline_params = config.get("baseline_params", {"temperature": 0.7, "top_p": 0.9})


    def update_with_trace(self, trace: EpisodicTrace):
        """Updates running metrics with data from a new episodic trace."""
        self.turn_count += 1
        self.mood_counts[trace.mood_family] += 1
        
        positive_distortion_classes = {"Romanticized Amplification", "Benevolent Over-Attribution"}
        negative_distortion_classes = {"Catastrophizing", "Personalization", "Black/White Splitting", "Meaning Nullification"}

        if trace.distortion_type in positive_distortion_classes:
            self.positive_distortions += 1
        elif trace.distortion_type in negative_distortion_classes:
            self.negative_distortions += 1
            
        self.last_n_traces.append(trace)
        if len(self.last_n_traces) > self.rolling_window_size:
            self.last_n_traces.pop(0)
            
        # Capture parameter divergence
        if trace.param_modulation:
            temp_divergence = trace.param_modulation.get("temperature", self.baseline_params["temperature"]) - self.baseline_params["temperature"]
            top_p_divergence = trace.param_modulation.get("top_p", self.baseline_params["top_p"]) - self.baseline_params["top_p"]
            self.parameter_divergence_samples["temperature"].append(temp_divergence)
            self.parameter_divergence_samples["top_p"].append(top_p_divergence)

    def record_regulation_latency(self, turns_to_trigger: int):
        """Records a sample for the regulation loop latency."""
        self.regulation_loop_latency_samples.append(turns_to_trigger)
        logger.info(f"Recorded regulation latency sample: {turns_to_trigger} turns.")

    def get_distortion_rate(self) -> Dict[str, float]:
        """Calculates the rate of positive and negative distortions."""
        if self.turn_count == 0:
            return {"positive_rate": 0.0, "negative_rate": 0.0, "total_rate": 0.0}
        
        total_distortions = self.positive_distortions + self.negative_distortions
        return {
            "positive_rate": self.positive_distortions / self.turn_count,
            "negative_rate": self.negative_distortions / self.turn_count,
            "total_rate": total_distortions / self.turn_count,
        }

    def get_mood_diversity_entropy(self) -> float:
        """Calculates the Shannon entropy of the mood distribution in the current window."""
        if not self.last_n_traces:
            return 0.0
        
        window_moods = Counter(trace.mood_family for trace in self.last_n_traces)
        total_moods = sum(window_moods.values())
        if total_moods == 0:
            return 0.0
            
        probabilities = [count / total_moods for count in window_moods.values()]
        return -sum(p * np.log2(p) for p in probabilities if p > 0)

    def get_parameter_divergence_stats(self) -> Dict[str, Dict[str, float]]:
        """Calculates statistics on parameter divergence."""
        stats = {}
        for param, samples in self.parameter_divergence_samples.items():
            if not samples:
                stats[param] = {"mean": 0, "std_dev": 0, "abs_mean": 0}
            else:
                stats[param] = {
                    "mean": float(np.mean(samples)),
                    "std_dev": float(np.std(samples)),
                    "abs_mean": float(np.mean(np.abs(samples))),
                }
        return stats
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Compiles all calculated metrics into a single dictionary."""
        return {
            "turn_count": self.turn_count,
            "distortion_rates": self.get_distortion_rate(),
            "mood_diversity_entropy": self.get_mood_diversity_entropy(),
            "parameter_divergence": self.get_parameter_divergence_stats(),
            "regulation_latency_avg": (np.mean(self.regulation_loop_latency_samples) 
                                       if self.regulation_loop_latency_samples else 0),
            "mood_distribution": dict(self.mood_counts)
        }

    def log_metrics_to_file(self):
        """Logs the current metrics snapshot to a structured log file."""
        import os
        log_data = self.get_all_metrics()
        log_data["timestamp"] = datetime.utcnow().isoformat()
        
        try:
            os.makedirs(os.path.dirname(METRICS_LOG_FILE), exist_ok=True)
            with open(METRICS_LOG_FILE, "a") as f:
                f.write(json.dumps(log_data) + "\n")
            logger.info(f"Successfully logged metrics to {METRICS_LOG_FILE}")
        except Exception as e:
            logger.error(f"Failed to log metrics to {METRICS_LOG_FILE}: {e}")

    def generate_diagnostic_dump(self, current_state: EmotionState) -> str:
        """
        Generates a human-readable diagnostic dump of the current state.
        """
        dump_data = {
            "current_emotion_state": json.loads(current_state.json()),
            "compiled_metrics": self.get_all_metrics(),
            "last_3_traces": [json.loads(t.json()) for t in self.last_n_traces[-3:]]
        }
        return json.dumps(dump_data, indent=2, default=str)

# Global instance for managing metrics
# This allows state to be maintained across different modules that import it.
metrics_manager = MetricsManager() 