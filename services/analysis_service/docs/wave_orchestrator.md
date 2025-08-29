# Wave Orchestrator - Story 9.2 Implementation

## Overview

The Wave Orchestrator implements a multi-stage, progressive processing system for audio analysis pipelines. It enables complex analysis workflows to be broken down into discrete stages that execute in waves, with support for dependencies, quality gates, and resource management.

## Key Features

### 1. Multi-Stage Processing
- **7 Configurable Stages**: Preparation, Initial Analysis, Feature Extraction, Deep Analysis, Aggregation, Post-Processing, and Finalization
- **Stage Dependencies**: Automatic dependency resolution ensures stages execute in the correct order
- **Custom Stage Handlers**: Register custom processing functions for each stage

### 2. Progressive Refinement
- **Quality-Based Termination**: Early termination when quality thresholds are met
- **Partial Results**: Support for partial completion with configurable quality gates
- **Adaptive Processing**: Stages can access results from previous stages for progressive refinement

### 3. Resource Management
- **Parallel Execution**: Process multiple items per stage with configurable concurrency
- **Memory Thresholds**: Configurable memory limits per stage
- **CPU Utilization**: Adaptive scaling based on CPU thresholds
- **Timeout Protection**: Per-stage timeouts prevent runaway processes

### 4. Error Handling & Recovery
- **Retry Logic**: Configurable retries per stage with exponential backoff
- **Graceful Degradation**: Continue processing even when some stages fail
- **Comprehensive Error Tracking**: Detailed error logging and metrics

## Architecture

### Core Components

#### WaveConfig
Configuration class that controls:
- Enabled stages and their order
- Concurrency limits
- Resource thresholds
- Quality gates
- Progressive mode settings

#### WaveContext
Tracks the state of a wave execution:
- Current stage
- Completed and failed stages
- Stage results
- Metrics and errors
- Overall status

#### WaveOrchestrator
Main orchestration engine that:
- Manages stage handlers
- Enforces dependencies
- Executes waves
- Collects metrics
- Handles cancellation

## Usage Examples

### Basic Wave Execution

```python
from wave_orchestrator import WaveOrchestrator, WaveConfig, WaveStage

# Initialize orchestrator
config = WaveConfig(
    max_parallel_items_per_stage=10,
    stage_timeout_seconds=300,
    progressive_mode=True
)
orchestrator = WaveOrchestrator(config=config)

# Register stage handlers
async def analyze_audio(data, context):
    # Perform audio analysis
    return {"bpm": 120, "key": "C major"}

orchestrator.register_stage_handler(
    WaveStage.INITIAL_ANALYSIS,
    analyze_audio
)

# Execute wave
context = await orchestrator.execute_wave(
    wave_id="audio_123",
    input_data={"file_path": "/path/to/audio.mp3"}
)
```

### Progressive Processing with Quality Gates

```python
config = WaveConfig(
    early_termination_on_quality=True,
    quality_threshold=0.95
)

async def quality_handler(data, context):
    result = process_data(data)
    quality_score = calculate_quality(result)
    return {"result": result, "quality": quality_score}

orchestrator.register_stage_handler(
    WaveStage.FEATURE_EXTRACTION,
    quality_handler
)
```

### Parallel Wave Execution

```python
# Process multiple files in parallel
waves = [
    (f"wave_{i}", {"file": f"audio_{i}.mp3"})
    for i in range(100)
]

results = await orchestrator.execute_parallel_waves(
    waves,
    max_concurrent=10
)
```

## Integration with Existing Systems

### AsyncAudioProcessor Integration
The Wave Orchestrator can leverage the existing AsyncAudioProcessor for CPU-bound operations:

```python
async_processor = AsyncAudioProcessor(config)
orchestrator = WaveOrchestrator(
    async_processor=async_processor
)
```

### BatchProcessor Integration
For batch operations within stages:

```python
batch_processor = BatchProcessor(process_func, config)
orchestrator = WaveOrchestrator(
    batch_processor=batch_processor
)
```

## Metrics and Monitoring

The orchestrator provides comprehensive metrics:

```python
stats = orchestrator.get_statistics()
# Returns:
# {
#     "total_waves": 100,
#     "successful_waves": 95,
#     "failed_waves": 5,
#     "success_rate": 95.0,
#     "active_waves": 2
# }
```

Per-wave metrics include:
- Total duration
- Stage durations
- Success rates
- Quality scores
- Error counts

## Configuration Reference

### WaveConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled_stages` | All 7 stages | List of stages to execute |
| `max_parallel_items_per_stage` | 10 | Maximum concurrent items per stage |
| `stage_timeout_seconds` | 300 | Timeout per stage in seconds |
| `max_retries_per_stage` | 2 | Number of retries for failed stages |
| `inter_stage_delay_ms` | 100 | Delay between stages in milliseconds |
| `memory_threshold_mb` | 1000 | Memory limit per stage |
| `cpu_threshold_percent` | 80.0 | CPU utilization threshold |
| `adaptive_scaling` | True | Enable adaptive resource scaling |
| `min_success_rate_per_stage` | 0.8 | Minimum success rate to continue |
| `allow_partial_results` | True | Allow partial wave completion |
| `progressive_mode` | True | Enable progressive refinement |
| `early_termination_on_quality` | True | Terminate early on quality threshold |
| `quality_threshold` | 0.95 | Quality score for early termination |

## Testing

Comprehensive test coverage includes:
- Basic wave execution
- Stage dependency validation
- Error handling and recovery
- Timeout protection
- Parallel execution
- Quality-based termination
- Metrics calculation
- Resource management

Run tests with:
```bash
uv run pytest tests/unit/analysis_service/test_wave_orchestrator.py -v
```

## Performance Considerations

1. **Stage Granularity**: Balance between too many small stages (overhead) and too few large stages (less parallelism)
2. **Concurrency Tuning**: Adjust `max_parallel_items_per_stage` based on system resources
3. **Quality Thresholds**: Set appropriate thresholds to avoid unnecessary processing
4. **Memory Management**: Monitor memory usage and adjust thresholds accordingly
5. **Timeout Configuration**: Set realistic timeouts to prevent resource exhaustion

## Future Enhancements

1. **Dynamic Stage Registration**: Runtime stage addition/removal
2. **Distributed Execution**: Support for distributed wave processing across multiple nodes
3. **Machine Learning Integration**: Adaptive quality thresholds based on historical data
4. **Advanced Scheduling**: Priority-based wave scheduling
5. **Checkpoint/Resume**: Ability to checkpoint and resume long-running waves
6. **Visual Pipeline Builder**: GUI for designing wave pipelines
7. **A/B Testing Support**: Run multiple wave configurations in parallel for comparison

## Conclusion

The Wave Orchestrator provides a robust, scalable solution for complex audio analysis pipelines. Its progressive processing model, combined with comprehensive resource management and quality gates, enables efficient processing of large-scale audio analysis workloads while maintaining high quality standards.
