"""
GPU Profiler Module
Handles GPU metrics collection and monitoring
"""
import torch
import psutil
import threading
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import deque


@dataclass
class GPUMetrics:
    """Data class to store GPU metrics"""
    timestamp: float
    gpu_utilization: float  # percentage
    gpu_memory_used: float  # MB
    gpu_memory_total: float  # MB
    gpu_power: Optional[float] = None  # Watts
    gpu_temp: Optional[float] = None  # Celsius


class GPUProfiler:
    """Monitor GPU metrics in real-time"""
    
    def __init__(self, polling_interval: float = 0.1):
        """
        Initialize GPU Profiler
        
        Args:
            polling_interval: Polling interval in seconds
        """
        self.polling_interval = polling_interval
        self.is_monitoring = False
        self.metrics_history: deque = deque(maxlen=10000)
        self._lock = threading.Lock()
        self._polling_thread: Optional[threading.Thread] = None
        
        # Check GPU availability
        self.gpu_available = torch.cuda.is_available()
        self.device_name = torch.cuda.get_device_name(0) if self.gpu_available else "CPU"
        
    def start_monitoring(self):
        """Start GPU monitoring thread"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self._polling_thread = threading.Thread(target=self._polling_loop, daemon=True)
        self._polling_thread.start()
    
    def stop_monitoring(self):
        """Stop GPU monitoring thread"""
        self.is_monitoring = False
        if self._polling_thread:
            self._polling_thread.join(timeout=5)
    
    def _polling_loop(self):
        """Main polling loop running in separate thread"""
        while self.is_monitoring:
            metrics = self._collect_metrics()
            with self._lock:
                self.metrics_history.append(metrics)
            time.sleep(self.polling_interval)
    
    def _collect_metrics(self) -> GPUMetrics:
        """Collect current GPU metrics"""
        timestamp = time.time()
        
        if not self.gpu_available:
            return GPUMetrics(
                timestamp=timestamp,
                gpu_utilization=0.0,
                gpu_memory_used=0.0,
                gpu_memory_total=0.0
            )
        
        # Use NVIDIA GPU stats if available
        try:
            props = torch.cuda.get_device_properties(0)
            total_memory = props.total_memory / (1024 ** 2)  # Convert to MB
            allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
            
            # Estimate utilization based on memory usage
            utilization = (allocated / total_memory) * 100
            
            return GPUMetrics(
                timestamp=timestamp,
                gpu_utilization=utilization,
                gpu_memory_used=allocated,
                gpu_memory_total=total_memory
            )
        except Exception as e:
            print(f"Error collecting GPU metrics: {e}")
            return GPUMetrics(
                timestamp=timestamp,
                gpu_utilization=0.0,
                gpu_memory_used=0.0,
                gpu_memory_total=0.0
            )
    
    def get_latest_metrics(self) -> Optional[GPUMetrics]:
        """Get the latest collected metrics"""
        with self._lock:
            if self.metrics_history:
                return self.metrics_history[-1]
        return None
    
    def get_average_metrics(self, window_size: int = 100) -> Optional[Dict]:
        """Get average metrics over a time window"""
        with self._lock:
            if not self.metrics_history:
                return None
            
            recent_metrics = list(self.metrics_history)[-window_size:]
            
            if not recent_metrics:
                return None
            
            avg_utilization = sum(m.gpu_utilization for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.gpu_memory_used for m in recent_metrics) / len(recent_metrics)
            
            return {
                'avg_utilization': avg_utilization,
                'avg_memory_used': avg_memory,
                'max_utilization': max(m.gpu_utilization for m in recent_metrics),
                'max_memory_used': max(m.gpu_memory_used for m in recent_metrics),
                'min_utilization': min(m.gpu_utilization for m in recent_metrics),
                'min_memory_used': min(m.gpu_memory_used for m in recent_metrics),
            }
    
    def get_metrics_summary(self) -> Dict:
        """Get a summary of GPU metrics"""
        latest = self.get_latest_metrics()
        average = self.get_average_metrics()
        
        return {
            'device_name': self.device_name,
            'gpu_available': self.gpu_available,
            'latest': latest,
            'average': average
        }
