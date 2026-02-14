"""
Main Module
Orchestrates GPU profiling, model inference, and metrics collection
"""
import torch
import time
import sys
from typing import Optional

from profiler import GPUProfiler
from model_loader import ModelLoader
from metrics import InferenceTimer, MetricsCollector, InferenceMetrics, EfficiencyAnalyzer


class GreenGPU:
    """Main orchestrator for GPU utilization monitoring and inference"""
    
    def __init__(self, model_name: str = 'resnet18', polling_interval: float = 0.1):
        """
        Initialize GreenGPU
        
        Args:
            model_name: Name of the model to load
            polling_interval: GPU polling interval in seconds
        """
        self.gpu_profiler = GPUProfiler(polling_interval=polling_interval)
        self.model_loader = ModelLoader()
        self.inference_timer = InferenceTimer()
        self.metrics_collector = MetricsCollector()
        self.efficiency_analyzer = EfficiencyAnalyzer()
        self.model_name = model_name
    
    def verify_gpu(self):
        """Verify GPU availability and print info"""
        print("=" * 60)
        print("GPU VERIFICATION")
        print("=" * 60)
        
        if torch.cuda.is_available():
            print(f"✓ CUDA is available")
            print(f"✓ Device Name: {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA Version: {torch.version.cuda}")
            print(f"✓ PyTorch Version: {torch.__version__}")
        else:
            print("✗ CUDA is NOT available - will run on CPU")
            print(f"✓ PyTorch Version: {torch.__version__}")
        
        print("=" * 60)
        print()
    
    def initialize(self):
        """Initialize the profiler and load model"""
        print("Initializing GreenGPU...")
        
        # Start GPU monitoring
        self.gpu_profiler.start_monitoring()
        time.sleep(0.5)  # Let monitoring thread start
        
        # Load model
        try:
            self.model_loader.load_pretrained_model(self.model_name)
            model_info = self.model_loader.get_model_info()
            print(f"Model loaded: {model_info['model_name']}")
            print(f"Total parameters: {model_info['total_parameters']:,}")
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
        
        # Warmup model
        self.model_loader.warmup(num_iterations=5)
        
        return True
    
    def run_inference(self, num_inferences: int = 10, batch_size: int = 1):
        """
        Run multiple inferences and collect metrics
        
        Args:
            num_inferences: Number of inferences to run
            batch_size: Batch size for inference
        """
        print("\n" + "=" * 60)
        print(f"RUNNING {num_inferences} INFERENCES (batch_size={batch_size})")
        print("=" * 60 + "\n")
        
        input_size = (batch_size, 3, 224, 224)
        
        for i in range(num_inferences):
            # Start inference timer
            self.inference_timer.start(f"inference_{i}")
            
            # Prepare input
            input_tensor = self.model_loader.prepare_input(input_size)
            
            # Run inference
            output = self.model_loader.inference(input_tensor)
            
            # Stop timer and collect metrics
            inference_time = self.inference_timer.stop()
            throughput = batch_size / inference_time
            
            # Get GPU metrics
            gpu_metrics = self.gpu_profiler.get_latest_metrics()
            
            if gpu_metrics:
                # Record metrics
                metrics = InferenceMetrics(
                    inference_id=f"inference_{i}",
                    inference_time=inference_time,
                    throughput=throughput,
                    gpu_memory_used=gpu_metrics.gpu_memory_used,
                    gpu_utilization=gpu_metrics.gpu_utilization
                )
                self.metrics_collector.record_inference(metrics)
                
                # Print progress
                print(f"Inference {i+1:3d}: {inference_time*1000:6.2f}ms | "
                      f"Throughput: {throughput:6.2f} samples/s | "
                      f"GPU Memory: {gpu_metrics.gpu_memory_used:6.1f}MB | "
                      f"GPU Util: {gpu_metrics.gpu_utilization:5.1f}%")
        
        print()
    
    def print_report(self):
        """Print comprehensive metrics report"""
        print("=" * 60)
        print("METRICS REPORT")
        print("=" * 60 + "\n")
        
        # Get statistics
        stats = self.metrics_collector.get_statistics()
        
        if not stats:
            print("No metrics collected")
            return
        
        # Print inference statistics
        print("INFERENCE PERFORMANCE:")
        print(f"  Total Inferences: {stats['total_inferences']}")
        print(f"  Inference Time (ms):")
        print(f"    Min: {stats['inference_time']['min']*1000:6.2f}")
        print(f"    Max: {stats['inference_time']['max']*1000:6.2f}")
        print(f"    Mean: {stats['inference_time']['mean']*1000:6.2f}")
        if stats['inference_time']['stdev'] > 0:
            print(f"    StDev: {stats['inference_time']['stdev']*1000:6.2f}")
        
        print(f"\n  Throughput (samples/sec):")
        print(f"    Min: {stats['throughput']['min']:6.2f}")
        print(f"    Max: {stats['throughput']['max']:6.2f}")
        print(f"    Mean: {stats['throughput']['mean']:6.2f}")
        
        # Print GPU statistics
        print(f"\nGPU MEMORY USAGE (MB):")
        print(f"  Min: {stats['gpu_memory']['min']:6.1f}")
        print(f"  Max: {stats['gpu_memory']['max']:6.1f}")
        print(f"  Mean: {stats['gpu_memory']['mean']:6.1f}")
        
        print(f"\nGPU UTILIZATION (%):")
        print(f"  Min: {stats['gpu_utilization']['min']:6.1f}")
        print(f"  Max: {stats['gpu_utilization']['max']:6.1f}")
        print(f"  Mean: {stats['gpu_utilization']['mean']:6.1f}")
        
        # Analyze efficiency
        print(f"\nEFFICIENCY ANALYSIS:")
        avg_memory = stats['gpu_memory']['mean']
        avg_util = stats['gpu_utilization']['mean']
        
        memory_analysis = self.efficiency_analyzer.analyze_memory(avg_memory, 8192)  # Assuming 8GB GPU
        util_analysis = self.efficiency_analyzer.analyze_utilization(avg_util)
        
        print(f"  GPU Utilization: {util_analysis['status']} ({avg_util:.1f}%)")
        print(f"    → {util_analysis['recommendation']}")
        print(f"  Memory Usage: {memory_analysis['status']} ({memory_analysis['usage_percent']:.1f}%)")
        
        print("\n" + "=" * 60)
    
    def shutdown(self):
        """Clean up resources"""
        self.gpu_profiler.stop_monitoring()
        print("GreenGPU shutdown complete")


def main():
    """Main entry point"""
    # Create and run GreenGPU
    greengpu = GreenGPU(model_name='resnet18', polling_interval=0.1)
    
    try:
        # Verify GPU setup
        greengpu.verify_gpu()
        
        # Initialize
        if not greengpu.initialize():
            print("Failed to initialize GreenGPU")
            return
        
        # Run inferences
        greengpu.run_inference(num_inferences=20, batch_size=1)
        
        # Print report
        greengpu.print_report()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        greengpu.shutdown()


if __name__ == '__main__':
    main()
