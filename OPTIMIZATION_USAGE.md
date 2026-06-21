# Fine-Tuning & Optimization Framework - Usage Guide

## Quick Start

### 1. Select Quality Profile

Three pre-configured profiles available:

```bash
# Production Quality (highest accuracy, 3x slower)
export OPTIMIZATION_PROFILE=production_quality

# Balanced (default, good tradeoff)
export OPTIMIZATION_PROFILE=balanced

# Real-Time (fastest, moderate accuracy)
export OPTIMIZATION_PROFILE=real_time
```

### 2. Select Diarization Backend

```bash
# Switch to Sortformer (more accurate, slower)
export DIARIZER=sortformer

# Switch to Pyannote (faster, good accuracy)
export DIARIZER=pyannote
```

### 3. Run Pipeline with Profile

```python
from config.optimization_config import ConfigManager

# Load profile
config = ConfigManager.get("production_quality")

# Use in pipeline
print(f"Backend: {config.backend}")
print(f"Beam Size: {config.get_beam_size()}")
print(f"VAD Threshold: {config.get_vad_threshold()}")

# Get all parameters
params = config.to_dict()
```

---

## Evaluation & Metrics

### Evaluate Transcription

```python
from evaluation.metrics import TranscriptionEvaluator, EvaluationReport

# Calculate WER and CER
reference = "the quick brown fox"
hypothesis = "the quik brown fox"

metrics = TranscriptionEvaluator.word_error_rate(reference, hypothesis)
print(f"WER: {metrics.wer:.1%}")
print(f"CER: {metrics.cer:.1%}")
```

### Evaluate Diarization

```python
from evaluation.metrics import DiarizationEvaluator

reference_timeline = [
    {"start": 0.0, "end": 5.0, "speaker": "Speaker_1"},
    {"start": 2.0, "end": 7.0, "speaker": "Speaker_2"},  # Overlap
]

hypothesis_timeline = [
    {"start": 0.1, "end": 4.9, "speaker": "Speaker_1"},
    {"start": 2.1, "end": 7.1, "speaker": "Speaker_2"},
]

metrics = DiarizationEvaluator.der_from_timeline(
    reference_timeline, hypothesis_timeline
)
print(f"DER: {metrics.der:.2f}%")
print(f"Speaker Attribution: {metrics.speaker_attribution_accuracy:.1f}%")
```

### Combined Evaluation

```python
from evaluation.metrics import CombinedEvaluator, EvaluationResult, EvaluationReport

# Create metrics
trans_metrics = TranscriptionEvaluator.word_error_rate(ref, hyp)
diar_metrics = DiarizationEvaluator.der_from_timeline(ref_timeline, hyp_timeline)

# Calculate overall score
score = CombinedEvaluator.overall_quality_score(trans_metrics, diar_metrics)
print(f"Overall Quality Score: {score:.1f}/100")

# Create report
result = EvaluationResult(
    file_id="test_001",
    transcription=trans_metrics,
    diarization=diar_metrics,
    overall_score=score
)

# Save and print
EvaluationReport.save(result, Path("results/test_001.json"))
print(EvaluationReport.print_summary(result))
```

---

## Configuration Profiles

### Profile Structure

Each profile (YAML) contains:

```yaml
profile_name: "balanced"
description: "Balanced accuracy and speed"

transcription:
  decoding:
    beam_size: 10
    lm_weight: 1.0
    confidence_threshold: 0.5
  preprocessing:
    vad_threshold: 0.6
  segmentation:
    min_segment_duration: 1.0

diarization:
  backend: "sortformer"  # or "pyannote"
  sortformer:
    batch_size: 2
    num_speakers: null
  pyannote:
    segmentation_threshold: 0.5

expected_metrics:
  wer: 0.12
  der: 0.05
```

### Available Profiles

| Profile | Speed | Accuracy | Use Case |
|---------|-------|----------|----------|
| `production_quality` | 3x slower | Highest | Court/legal cases |
| `balanced` | 1.5x normal | Good | Typical processing |
| `real_time` | 1x normal | Moderate | Interactive/live |

---

## Runtime Backend Switching

### Switch Backend Dynamically

```python
from config.optimization_config import ConfigManager

config = ConfigManager.get("balanced")

# Switch from Sortformer to Pyannote
config.switch_backend("pyannote")
print(f"Now using: {config.backend}")

# Or use ConfigManager directly
ConfigManager.switch_backend("sortformer")
```

### API Support

```python
from fastapi import FastAPI

app = FastAPI()

@app.post("/jobs", params={"profile": "balanced", "backend": "sortformer"})
def submit_job(profile: str = "balanced", backend: str = None):
    config = ConfigManager.get(profile)
    if backend:
        config.switch_backend(backend)
    
    return {
        "profile": config.profile_name,
        "backend": config.backend,
        "beam_size": config.get_beam_size(),
    }
```

---

## Hyperparameter Tuning

### Grid Search (Exhaustive)

```python
from optimization.hyperparameter_tuning import GridSearch

params = {
    "beam_size": [5, 10, 20],
    "lm_weight": [0.5, 1.0, 1.5],
    "vad_threshold": [0.5, 0.6, 0.7],
}

search = GridSearch(
    params=params,
    evaluate_fn=evaluate_on_dataset,
    dataset_path="path/to/test_set"
)

results = search.run()
best_config = results.best_config
print(f"Best config: {best_config} with score {results.best_score}")
```

### Bayesian Optimization

```python
from optimization.hyperparameter_tuning import BayesianOptimizer

optimizer = BayesianOptimizer(
    params={
        "beam_size": (1, 50),
        "lm_weight": (0.0, 2.0),
        "vad_threshold": (0.3, 0.8),
    },
    n_iterations=20,
    evaluate_fn=evaluate_on_dataset
)

best_config = optimizer.optimize()
```

---

## Testing

### Run All Tests

```bash
cd /home/ashish-ratna/ACB/Audio_Analyzer

# Run optimization tests
python -m pytest tests/test_optimization.py -v

# Run with coverage
python -m pytest tests/test_optimization.py --cov=evaluation --cov=config
```

### Test Specific Components

```bash
# Test metrics only
pytest tests/test_optimization.py::TestTranscriptionMetrics -v

# Test configuration
pytest tests/test_optimization.py::TestOptimizationConfig -v

# Test backend switching
pytest tests/test_optimization.py::TestBackendEquivalence -v
```

---

## Ground-Truth Dataset Format

### Reference Transcript

```json
{
  "file_id": "test_001",
  "audio_path": "/path/to/audio.wav",
  "reference_transcript": "the quick brown fox",
  "reference_timeline": [
    {"start": 0.0, "end": 5.0, "speaker": "A.O", "text": "the quick"},
    {"start": 2.0, "end": 7.0, "speaker": "Complainant", "text": "brown fox"}
  ],
  "language": "en",
  "metadata": {
    "case_id": "case_001",
    "duration": 10.5,
    "sample_rate": 16000
  }
}
```

### Store in `reference_data/`

```
reference_data/
├── transcripts/
│   ├── test_001_reference.json
│   ├── test_002_reference.json
│   └── ...
├── metrics/
│   ├── test_001_eval.json
│   ├── test_002_eval.json
│   └── ...
└── reports/
    ├── wer_report.html
    └── der_report.html
```

---

## Performance Tracking

### Create Benchmark Report

```python
from evaluation.benchmark import BenchmarkRunner

runner = BenchmarkRunner(
    ground_truth_dir="reference_data/transcripts",
    output_dir="results"
)

# Run evaluation on all files
results = runner.run_all()

# Generate reports
runner.generate_wer_report()
runner.generate_der_report()
runner.generate_html_dashboard()
```

### Compare Backends

```python
from evaluation.benchmark import BackendComparison

comparison = BackendComparison(
    ground_truth_dir="reference_data",
    backends=["sortformer", "pyannote"],
    profiles=["balanced", "production_quality"]
)

results = comparison.run()
comparison.save_comparison_report("results/backend_comparison.html")
```

---

## Integration with Existing Pipeline

### Modify `config.py`

```python
# config.py

from config.optimization_config import ConfigManager, apply_optimization_config

# Load optimization config
opt_config = ConfigManager.get()
apply_optimization_config()

# Use profile settings
ASR_BEAM_SIZE = opt_config.get_beam_size()
ASR_LM_WEIGHT = opt_config.get_lm_weight()
DIARIZER = opt_config.backend
VAD_THRESHOLD = opt_config.get_vad_threshold()
```

### Use in Services

```python
# services/transcription_service.py

from config import ASR_BEAM_SIZE, ASR_LM_WEIGHT

def decode(emissions, encoder_output):
    """Decode with profile-based beam size and LM weight."""
    return ctc_beam_search(
        emissions=emissions,
        beam_size=ASR_BEAM_SIZE,  # From profile
        lm_weight=ASR_LM_WEIGHT,  # From profile
    )
```

---

## Production Deployment

### Docker Support

```dockerfile
# Dockerfile
ENV OPTIMIZATION_PROFILE=production_quality
ENV DIARIZER=sortformer

# Load config at runtime
RUN python -c "from config.optimization_config import apply_optimization_config; apply_optimization_config()"
```

### Kubernetes ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: audio-analyzer-config
data:
  OPTIMIZATION_PROFILE: production_quality
  DIARIZER: sortformer
```

### Environment-based Selection

```python
# Automatically select profile based on environment
import os

ENV = os.getenv("ENVIRONMENT", "production")

PROFILE_MAP = {
    "production": "production_quality",
    "staging": "balanced",
    "development": "real_time",
}

OPTIMIZATION_PROFILE = PROFILE_MAP.get(ENV, "balanced")
```

---

## Metrics Target Checklist

- [ ] WER < 10% (production_quality)
- [ ] CER < 5% (production_quality)
- [ ] DER < 3% (production_quality)
- [ ] Speaker attribution accuracy > 98%
- [ ] Overlap detection > 92%
- [ ] Both backends produce equivalent output format
- [ ] Profile switching works without restarting service
- [ ] All tests pass with >95% coverage

---

## Next Steps

1. **Week 1:** Run tests, validate metrics implementation
2. **Week 2:** Create ground-truth dataset, run initial evaluation
3. **Week 3:** Execute hyperparameter tuning on test set
4. **Week 4:** Generate benchmarking reports, finalize profiles
5. **Week 5:** Deploy to production with monitoring

See `OPTIMIZATION_PLAN.md` for detailed timeline.
