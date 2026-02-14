"""
Metrics Module
Tracks inference performance and GPU efficiency metrics
"""
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from collections import deque
import statistics


@dataclass
class InferenceMetrics:
    """Data class for inference metrics"""
    inference_id: str
    inference_time: float  # seconds
    throughput: float  # samples/sec
    gpu_memory_used: float  # MB
    gpu_utilization: float  # percentage
    power_efficiency: Optional[float] = None  # inferences/watt


class InferenceTimer:
    """Track inference timing and performance"""
    
    def __init__(self):
        """Initialize inference timer"""
        self.start_time: Optional[float] = None
        self.inference_id: Optional[str] = None
    
    def start(self, inference_id: str):
        """Start timing an inference"""
        self.inference_id = inference_id
        self.start_time = time.time()
    
    def stop(self) -> float:
        """Stop timing and return elapsed time in seconds"""
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        elapsed = time.time() - self.start_time
        self.start_time = None
        return elapsed
    
    def get_elapsed(self) -> float:
        """Get current elapsed time without stopping"""
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        return time.time() - self.start_time


class MetricsCollector:
    """Collect and aggregate inference metrics"""
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize metrics collector
        
        Args:
            max_history: Maximum number of metrics to keep in history
        """
        self.metrics_history: deque = deque(maxlen=max_history)
        self.inference_counter = 0
    
    def record_inference(self, inference_metrics: InferenceMetrics):
        """Record an inference metric"""
        self.metrics_history.append(inference_metrics)
        self.inference_counter += 1
    
    def get_statistics(self) -> Optional[Dict]:
        """Get statistics for all recorded inferences"""
        if not self.metrics_history:
            return None
        
        times = [m.inference_time for m in self.metrics_history]
        throughputs = [m.throughput for m in self.metrics_history]
        memory_used = [m.gpu_memory_used for m in self.metrics_history]
        utilizations = [m.gpu_utilization for m in self.metrics_history]
        
        return {
            'total_inferences': len(self.metrics_history),
            'inference_time': {
                'min': min(times),
                'max': max(times),
                'mean': statistics.mean(times),
                'stdev': statistics.stdev(times) if len(times) > 1 else 0,
            },
            'throughput': {
                'min': min(throughputs),
                'max': max(throughputs),
                'mean': statistics.mean(throughputs),
            },
            'gpu_memory': {
                'min': min(memory_used),
                'max': max(memory_used),
                'mean': statistics.mean(memory_used),
            },
            'gpu_utilization': {
                'min': min(utilizations),
                'max': max(utilizations),
                'mean': statistics.mean(utilizations),
            }
        }
    
    def get_recent_metrics(self, window_size: int = 10) -> List[Dict]:
        """Get recent metrics as dictionaries"""
        recent = list(self.metrics_history)[-window_size:]
        return [asdict(m) for m in recent]
    
    def clear_history(self):
        """Clear metrics history"""
        self.metrics_history.clear()
        self.inference_counter = 0


class EfficiencyAnalyzer:
    """Analyze GPU efficiency and provide recommendations"""
    
    @staticmethod
    def calculate_power_efficiency(inferences_per_second: float, power_watts: float) -> float:
        """Calculate inferences per watt"""
        if power_watts == 0:
            return 0
        return inferences_per_second / power_watts
    
    @staticmethod
    def analyze_utilization(utilization_percent: float) -> Dict:
        """Analyze GPU utilization and provide insights"""
        if utilization_percent < 20:
            status = "UNDERUTILIZED"
            recommendation = "GPU is underutilized. Consider larger batch sizes."
        elif utilization_percent < 50:
            status = "LOW"
            recommendation = "GPU utilization could be improved."
        elif utilization_percent < 80:
            status = "GOOD"
            recommendation = "GPU utilization is good."
        else:
            status = "HIGH"
            recommendation = "GPU is well utilized."
        
        return {
            'status': status,
            'recommendation': recommendation,
            'efficiency_score': min(utilization_percent / 100, 1.0)
        }
    
    @staticmethod
    def analyze_memory(used_mb: float, total_mb: float) -> Dict:
        """Analyze GPU memory usage"""
        memory_percent = (used_mb / total_mb) * 100 if total_mb > 0 else 0
        
        if memory_percent < 30:
            status = "OPTIMAL"
        elif memory_percent < 70:
            status = "GOOD"
        elif memory_percent < 90:
            status = "HIGH"
        else:
            status = "CRITICAL"
        
        return {
            'usage_percent': memory_percent,
            'used_mb': used_mb,
            'total_mb': total_mb,
            'status': status
        }
