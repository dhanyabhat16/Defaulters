# GreenGPU - GPU Utilization Profiler

A comprehensive GPU monitoring and profiling toolkit for tracking GPU utilization, memory usage, and inference performance metrics.

## Features

- **GPU Profiling**: Real-time GPU metrics collection with configurable polling intervals
- **Inference Timing**: Precise measurement of model inference latency
- **Metrics Collection**: Track GPU memory, utilization, and throughput
- **Efficiency Analysis**: Automatic recommendations based on GPU usage patterns
- **Model Support**: Easy integration with any PyTorch model
- **Threading**: Non-blocking background GPU monitoring

## Project Structure

```
greengpu/
├── profiler.py      # GPU profiling and metrics collection
├── model_loader.py  # Model loading and inference management
├── metrics.py       # Inference metrics and efficiency analysis
└── main.py         # Main orchestrator and demo
```

## Installation

### 1. Verify GPU Setup (Important!)

First, run this to ensure PyTorch sees your GPU:

```bash
python verify_gpu.py
```

Expected output if GPU is available:
```
✓ CUDA is available
✓ Device Name: NVIDIA GeForce RTX 3060
✓ CUDA Version: 11.8
✓ GPU Count: 1
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `torch` - Deep learning framework
- `torchvision` - Computer vision models
- `psutil` - System monitoring
- `pandas` - Data analysis

## Quick Start

### Run the Demo

```bash
cd greengpu
python main.py
```

This will:
1. Verify GPU availability
2. Load a ResNet18 model
3. Run 20 inferences while monitoring GPU metrics
4. Print a comprehensive performance report

### Example Output

```
============================================================
GPU VERIFICATION
============================================================

✓ CUDA is available
✓ Device Name: NVIDIA GeForce RTX 3060
✓ CUDA Version: 11.8
✓ PyTorch Version: 2.0.0

============================================================

Initializing GreenGPU...
Loading resnet18...
Model loaded on cuda
Total parameters: 11,689,512

Warming up model with 5 iterations...
Warmup complete

============================================================
RUNNING 20 INFERENCES (batch_size=1)
============================================================

Inference   1:   8.23ms | Throughput: 121.51 samples/s | GPU Memory: 1254.5MB | GPU Util:  98.5%
Inference   2:   8.15ms | Throughput: 122.70 samples/s | GPU Memory: 1254.5MB | GPU Util:  98.5%
...

============================================================
METRICS REPORT
============================================================

INFERENCE PERFORMANCE:
  Total Inferences: 20
  Inference Time (ms):
    Min:   8.12
    Max:   8.56
    Mean:   8.31
    StDev:   0.14

  Throughput (samples/sec):
    Min: 116.87
    Max: 123.05
    Mean: 120.37

GPU MEMORY USAGE (MB):
  Min: 1254.5
  Max: 1254.5
  Mean: 1254.5

GPU UTILIZATION (%):
  Min:  98.5
  Max:  98.5
  Mean:  98.5

EFFICIENCY ANALYSIS:
  GPU Utilization: HIGH (98.5%)
    → GPU is well utilized.
  Memory Usage: GOOD (15.3%)

============================================================
```

## Usage Examples

### Basic Usage

```python
from greengpu.main import GreenGPU

# Create profiler
profiler = GreenGPU(model_name='resnet18')

# Verify GPU
profiler.verify_gpu()

# Initialize
profiler.initialize()

# Run inferences
profiler.run_inference(num_inferences=20, batch_size=1)

# Get report
profiler.print_report()

# Cleanup
profiler.shutdown()
```

### Custom Model

```python
from greengpu.model_loader import ModelLoader
from greengpu.profiler import GPUProfiler
from greengpu.metrics import InferenceTimer, MetricsCollector

# Load your model
loader = ModelLoader()
model = loader.load_pretrained_model('resnet50')

# Start GPU profiling
profiler = GPUProfiler()
profiler.start_monitoring()

# Run inference and collect metrics
timer = InferenceTimer()
metrics_collector = MetricsCollector()

# ... your inference code ...

profiler.stop_monitoring()
```

## Module Reference

### profiler.py

**GPUProfiler**
- `start_monitoring()` - Start background GPU monitoring
- `stop_monitoring()` - Stop GPU monitoring
- `get_latest_metrics()` - Get most recent GPU metrics
- `get_average_metrics(window_size)` - Get averaged metrics
- `get_metrics_summary()` - Get GPU summary

**GPUMetrics** (dataclass)
- `timestamp` - Metric collection timestamp
- `gpu_utilization` - GPU utilization percentage
- `gpu_memory_used` - Memory used in MB
- `gpu_memory_total` - Total GPU memory in MB
- `gpu_power` - Power consumption in watts (optional)
- `gpu_temp` - GPU temperature in Celsius (optional)

### model_loader.py

**ModelLoader**
- `load_pretrained_model(model_name)` - Load from torchvision
- `load_from_checkpoint(path)` - Load from checkpoint file
- `get_model_info()` - Get model statistics
- `prepare_input(shape)` - Create dummy input
- `inference(input)` - Run inference
- `warmup(iterations)` - Warmup GPU with dummy inferences

### metrics.py

**InferenceTimer**
- `start(id)` - Start timing
- `stop()` - Stop timing and get elapsed time
- `get_elapsed()` - Get current elapsed time

**MetricsCollector**
- `record_inference(metrics)` - Record inference metrics
- `get_statistics()` - Get aggregated statistics
- `get_recent_metrics(window_size)` - Get recent metrics

**EfficiencyAnalyzer**
- `analyze_utilization()` - Analyze GPU utilization
- `analyze_memory()` - Analyze memory usage
- `calculate_power_efficiency()` - Calculate power efficiency

## Troubleshooting

### GPU Not Detected

1. Check CUDA installation: `nvidia-smi`
2. Verify PyTorch CUDA support: `python verify_gpu.py`
3. Ensure compatible NVIDIA driver

### Out of Memory

- Reduce batch size
- Use smaller model
- Clear GPU cache: `torch.cuda.empty_cache()`

### Slow Inference

- Run warmup iterations first
- Check GPU utilization in report
- Try larger batch sizes to improve utilization

## Advanced Features

### Custom Polling Interval

```python
from greengpu.profiler import GPUProfiler

# Sample GPU metrics every 50ms
profiler = GPUProfiler(polling_interval=0.05)
```

### GPU Memory Tracking

```python
# Get detailed memory breakdown
gpu_metrics = profiler.get_latest_metrics()
print(f"Memory used: {gpu_metrics.gpu_memory_used}MB")
print(f"Memory total: {gpu_metrics.gpu_memory_total}MB")
```

## Performance Tips

1. **Warmup**: Always warmup the model before benchmarking
2. **Batch Size**: Use larger batches to improve GPU utilization
3. **Precision**: Consider using mixed precision (fp16) for faster inference
4. **Model Selection**: Lighter models may have better per-watt performance

## License

MIT

## Support

For issues or questions, please refer to the [project documentation](./README.md).
