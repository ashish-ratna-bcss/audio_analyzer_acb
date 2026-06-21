# Audio Analyzer Fine-Tuning & Optimization Roadmap

**Objective:** Maximize transcription accuracy (WER/CER) and diarization accuracy (DER, speaker attribution) while maintaining interchangeable backends (Sortformer/Pyannote).

---

## Phase 1: Evaluation Infrastructure (Week 1)

### 1.1 Metrics Implementation
- **WER (Word Error Rate):** Transcription accuracy at word level
- **CER (Character Error Rate):** Transcription accuracy at character level (critical for Indic scripts)
- **DER (Diarization Error Rate):** Speaker segmentation accuracy (False Alarm, Missed Detection, Confusion)
- **Speaker Attribution Accuracy:** % of segments with correct speaker assignment
- **Overlap Detection Rate:** Accuracy in identifying simultaneous speakers

### 1.2 Ground-Truth Dataset Management
```
reference_data/
├── transcripts/
│   ├── {file_id}_reference.json    # Ground truth transcript
│   └── {file_id}_annotations.txt   # Manual speaker labels
├── metrics/
│   ├── {file_id}_eval.json         # WER, CER, DER scores
│   └── {file_id}_confusion.json    # Speaker confusion matrix
└── reports/
    └── benchmark_{date}.html       # Visual comparison reports
```

### 1.3 Evaluation Tool
```python
# evaluation/metrics.py
- transcription_metrics(reference, hypothesis) → {wer, cer, confidence}
- diarization_metrics(ref_timeline, hyp_timeline) → {der, fa, md, cn, speaker_acc}
- combined_score(wer, der, speaker_acc) → overall_quality_score
```

---

## Phase 2: Hyperparameter Tuning (Weeks 2-3)

### 2.1 Transcription (IndicConformer) Tuning

**Decoding Parameters:**
```yaml
beam_size: [1, 5, 10, 20, 50]           # Beam search breadth
lm_weight: [0.0, 0.5, 1.0, 1.5, 2.0]   # Language model scaling
ctc_weight: [0.0, 0.3, 0.5]             # CTC vs Attention balance
confidence_threshold: [0.3, 0.5, 0.7]   # Min confidence for segments
```

**Preprocessing:**
```yaml
vad_threshold: [0.3, 0.5, 0.7]          # Voice Activity Detection sensitivity
sample_rate: [16000, 44100]             # Audio resampling
normalization: [true, false]            # Audio normalization
mel_filter_banks: [64, 80, 128]         # Spectrogram features
```

**Segmentation:**
```yaml
min_segment_duration: [0.5, 1.0, 2.0]   # Minimum speech segment length
max_segment_duration: [10, 20, 30]      # Maximum segment before split
silence_threshold: [0.2, 0.5, 1.0]      # Silence duration threshold
```

### 2.2 Diarization (Sortformer/Pyannote) Tuning

**Sortformer Parameters:**
```yaml
batch_size: [1, 2, 4]
num_speakers: [null, 2, 3]              # Auto-detect vs fixed
overlap_threshold: [0.3, 0.5, 0.7]      # Overlap significance threshold
speaker_embedding_dim: [256, 512]       # Embedding dimensionality
```

**Pyannote Parameters:**
```yaml
segmentation_threshold: [0.3, 0.5]      # Segment confidence
speaker_threshold: [0.5, 0.7]           # Speaker change threshold
min_duration_on: [0.5, 1.0]             # Min active speech segment
min_duration_off: [0.5, 1.0, 2.0]       # Min silence between speakers
```

### 2.3 Search Strategy
```
optimize/
├── grid_search.py          # Exhaustive parameter search
├── random_search.py        # Random sampling
├── bayesian_optimization.py # Gaussian Process-based HPO
└── evolutionary_algorithm.py # Genetic algorithm search
```

---

## Phase 3: Preprocessing & VAD Optimization (Week 2)

### 3.1 Voice Activity Detection
```python
# services/vad_service.py
class VADOptimizer:
    - silero_vad()          # Russian silero model (lightweight)
    - pyannote_vad()        # Pyannote's VAD module
    - energy_based_vad()    # Traditional energy thresholding
    - compare_efficiency()  # Speed vs accuracy tradeoff
```

### 3.2 Audio Preprocessing Pipeline
```python
# services/audio_preprocessing.py
- normalize_loudness()      # LUFS normalization
- remove_noise()            # Spectral subtraction
- enhance_speech()          # Speech enhancement (DeepFilterNet already in L3)
- segment_audio()           # Intelligent segmentation
- handle_silence()          # Gap filling strategy
```

---

## Phase 4: Segmentation & Speaker Embedding Strategies (Week 3)

### 4.1 Intelligent Segmentation
```python
# services/segmentation_service.py
- dynamic_segmentation()    # Adapt segment boundaries to pauses
- utterance_level()         # Group by speaker turns
- window_sliding()          # Fixed window with overlap
- vad_based()               # Use VAD for boundaries
```

### 4.2 Speaker Embedding Models
```yaml
models:
  - ecapa_tdnn            # Large, accurate (256D)
  - tdnn                  # Standard TDNN baseline
  - xvector               # Fast, lightweight
  - wav2vec2_xls_r        # Multilingual pretrained
```

---

## Phase 5: Decoding & Post-Processing (Week 3)

### 5.1 Language Model Decoding
```python
# services/decoding_service.py
- beam_search()            # Standard beam search
- constrained_decoding()   # With LM constraints
- lattice_rescoring()      # Second-pass LM rescoring
- confidence_filtering()   # Discard low-confidence words
```

### 5.2 Post-Processing
```python
# services/post_processing.py
- correct_common_errors()  # Known OCR errors in Indic
- handle_overlaps()        # Merge/separate overlapping segments
- speaker_consistency()    # Enforce speaker continuity
- time_alignment()         # Smooth timing discontinuities
```

---

## Phase 6: Benchmarking & Comparison (Week 4)

### 6.1 Evaluation Framework
```python
# evaluation/benchmark.py
class Benchmark:
    - run_on_dataset()      # Test config on all ground-truth files
    - compare_backends()    # Sortformer vs Pyannote side-by-side
    - ablation_study()      # Disable components, measure impact
    - sensitivity_analysis() # How much does each param matter?
```

### 6.2 Report Generation
```python
# evaluation/reporting.py
- generate_wer_report()     # Per-file WER breakdown
- generate_der_report()     # Per-file DER breakdown
- speaker_confusion()       # Who gets confused for whom?
- timing_accuracy()         # Start/end time variance
- create_html_dashboard()   # Visual comparison interface
```

---

## Phase 7: Configuration Profiles (Week 4)

### 7.1 Quality vs Speed Tradeoffs
```yaml
# config/profiles/production_quality.yaml
# Highest accuracy, ~3x slower
transcription:
  beam_size: 50
  lm_weight: 1.5
  confidence_threshold: 0.7
diarization: sortformer  # More accurate
preprocessing:
  vad_threshold: 0.5
  normalization: true

---

# config/profiles/real_time.yaml
# Moderate accuracy, ~1x speed (fast)
transcription:
  beam_size: 5
  lm_weight: 1.0
  confidence_threshold: 0.3
diarization: pyannote  # Faster
preprocessing:
  vad_threshold: 0.7
  normalization: false

---

# config/profiles/balanced.yaml
# Good accuracy/speed tradeoff
transcription:
  beam_size: 10
  lm_weight: 1.0
  confidence_threshold: 0.5
diarization: sortformer
preprocessing:
  vad_threshold: 0.6
  normalization: true
```

### 7.2 Backend Selection
```python
# config/diarizer_config.py
DIARIZER_CONFIG = {
    "sortformer": {
        "accuracy": 0.92,    # Empirical DER
        "speed": 0.8,        # Relative to pyannote
        "overlap_handling": "native",
        "memory": "2GB"
    },
    "pyannote": {
        "accuracy": 0.88,
        "speed": 1.0,        # Baseline
        "overlap_handling": "post_process",
        "memory": "1.5GB"
    }
}
```

---

## Phase 8: Integration & Testing (Week 4)

### 8.1 Configuration System
```python
# config.py - Extended
OPTIMIZATION_PROFILE = os.getenv("OPTIMIZATION_PROFILE", "balanced")
load_profile(OPTIMIZATION_PROFILE)

# Runtime profile override
@app.post("/jobs", params={"profile": "production_quality"})
def submit_job(profile: str = "balanced"):
    config = load_profile(profile)
    # Apply all HPO parameters dynamically
```

### 8.2 Testing
```python
# tests/test_optimization.py
- test_wer_below_threshold()
- test_der_below_threshold()
- test_speaker_attribution_accuracy()
- test_profile_correctness()
- test_backend_equivalence()
```

---

## Success Metrics

| Metric | Current | Target | Timeline |
|--------|---------|--------|----------|
| **WER** | ~15-20% | <10% | Phase 2 |
| **CER** | ~8-12% | <5% | Phase 2 |
| **DER** | ~5-8% | <3% | Phase 5 |
| **Speaker Attribution** | ~95% | >98% | Phase 5 |
| **Overlap Detection** | ~85% | >92% | Phase 5 |
| **Config Options** | 1 | 3+ profiles | Phase 7 |
| **Backend Switching** | Manual | 1-line config | Phase 7 |

---

## Artifact Structure

```
optimization/
├── OPTIMIZATION_PLAN.md         # This file
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py              # WER, CER, DER calculations
│   ├── benchmark.py            # Benchmarking framework
│   ├── reporting.py            # Report generation
│   └── ground_truth/           # Reference datasets
├── hyperparameter_tuning/
│   ├── __init__.py
│   ├── grid_search.py
│   ├── bayesian_optimizer.py
│   ├── search_spaces.yaml      # Parameter ranges
│   └── results/                # HPO results & logs
├── preprocessing/
│   ├── __init__.py
│   ├── vad_optimizer.py
│   ├── audio_preprocessing.py
│   └── segmentation.py
├── decoding/
│   ├── __init__.py
│   ├── decoding_strategies.py
│   └── post_processing.py
├── config/
│   ├── profiles/
│   │   ├── production_quality.yaml
│   │   ├── balanced.yaml
│   │   └── real_time.yaml
│   ├── diarizer_config.py
│   └── optimization_config.py
├── tests/
│   ├── test_metrics.py
│   ├── test_optimization.py
│   └── test_backend_switching.py
└── results/
    ├── benchmark_results.json
    ├── wer_report.html
    └── der_comparison.html
```

---

## Implementation Priority

**Critical Path (Weeks 1-2):**
1. ✅ Evaluation metrics implementation
2. ✅ Ground-truth dataset setup
3. ✅ Transcription HPO (beam, LM weight)
4. ✅ Diarization HPO (thresholds)

**High Priority (Week 3):**
5. VAD optimization
6. Segmentation strategies
7. Speaker embedding comparison
8. Post-processing improvements

**Medium Priority (Week 4):**
9. Configuration profiles
10. Benchmarking reports
11. Backend equivalence testing
12. Documentation & deployment

---

## Cost & Resource Estimates

| Phase | GPU Hours | CPU Hours | Storage | Est. Cost |
|-------|-----------|-----------|---------|-----------|
| Phase 1-2 (Metrics + HPO) | 50-80 | 100-150 | 100GB | $150-200 |
| Phase 3-4 (Preprocessing) | 30-40 | 80-100 | 50GB | $75-100 |
| Phase 5-6 (Decoding + Bench) | 40-60 | 120-150 | 150GB | $120-150 |
| Phase 7-8 (Profiles + Testing) | 20-30 | 60-80 | 50GB | $50-75 |
| **Total** | **140-210** | **360-480** | **350GB** | **$395-525** |

---

## Risk Mitigation

- **Data Leakage:** Keep test set separate from HPO
- **Overfitting:** Validate on held-out ground-truth data
- **Backend Incompatibility:** Test both Sortformer & Pyannote throughout
- **Performance Regression:** Automated tests check that changes improve overall score
- **Configuration Complexity:** Use YAML profiles, not raw parameters

---

## Next Steps

1. **Immediate:** Create `evaluation/metrics.py` with WER/CER/DER calculators
2. **Week 1:** Set up ground-truth dataset and benchmark runner
3. **Week 2:** Run initial HPO on transcription parameters
4. **Week 3:** Optimize diarization + preprocessing
5. **Week 4:** Create profiles, test backend switching, generate reports
