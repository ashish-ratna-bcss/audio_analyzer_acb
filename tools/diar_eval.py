import csv
import sys
import os
import argparse
from typing import List, Dict, Tuple

try:
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate
    from pyannote.metrics.segmentation import SegmentationPurity, SegmentationCoverage
except ImportError:
    print("Warning: pyannote.core or pyannote.metrics not installed. Run: pip install pyannote.metrics")
    Annotation = None

def load_ground_truth(csv_path: str) -> List[Dict]:
    """
    Load ground truth from a CSV file.
    Expected to have columns: start, end, speaker (or time, person, etc.)
    For this specific use case, we might need to parse 'MM.SS' timestamps.
    """
    segments = []
    with open(csv_path, "r", encoding="utf-8") as f:
        # Simple dialect sniffing or assume standard header
        reader = csv.DictReader(f)
        for row in reader:
            # handle case variations
            row_lower = {k.lower().strip(): v for k, v in row.items()}
            
            # Extract start and end. If there's only 'time', it's a point in time (we need ranges)
            # In MM.SS format, we might need to convert it.
            start_val = row_lower.get('start') or row_lower.get('time')
            end_val = row_lower.get('end')
            speaker = row_lower.get('speaker') or row_lower.get('person')
            
            if not start_val or not speaker:
                continue
                
            def parse_time(t_str):
                if not t_str: return 0.0
                if ':' in t_str:
                    parts = t_str.split(':')
                    return float(parts[0]) * 60 + float(parts[1])
                elif '.' in t_str:
                    parts = t_str.split('.')
                    return float(parts[0]) * 60 + float(parts[1])
                return float(t_str)
                
            start = parse_time(start_val)
            end = parse_time(end_val) if end_val else start + 3.0 # Fallback 3s duration if missing
            
            segments.append({
                "start": start,
                "end": end,
                "speaker": speaker
            })
    return segments

def evaluate_diarization(ref_segments: List[Dict], hyp_segments: List[Dict]):
    if Annotation is None:
        print("Cannot compute exact DER without pyannote.metrics.")
        return
        
    reference = Annotation()
    for s in ref_segments:
        reference[Segment(s['start'], s['end'])] = s['speaker']
        
    hypothesis = Annotation()
    for s in hyp_segments:
        hypothesis[Segment(s['start'], s['end'])] = s['speaker']
        
    der_metric = DiarizationErrorRate()
    der = der_metric(reference, hypothesis)
    
    print(f"Diarization Error Rate (DER): {der * 100:.2f}%")
    
    # Speaker count accuracy
    ref_speakers = set(s['speaker'] for s in ref_segments)
    hyp_speakers = set(s['speaker'] for s in hyp_segments)
    
    print(f"Reference Speakers: {len(ref_speakers)}")
    print(f"Hypothesis Speakers: {len(hyp_speakers)}")
    
    # Detailed components
    der_detail = der_metric(reference, hypothesis, detailed=True)
    if isinstance(der_detail, dict):
        print(f"Missed Detection: {der_detail.get('missed detection', 0)}")
        print(f"False Alarm: {der_detail.get('false alarm', 0)}")
        print(f"Confusion: {der_detail.get('confusion', 0)}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate diarization performance against ground truth")
    parser.add_argument("--ref", required=True, help="Path to ground truth CSV")
    parser.add_argument("--audio", required=True, help="Path to audio file to test")
    parser.add_argument("--diarizer", choices=["pyannote", "sortformer"], default="pyannote")
    args = parser.parse_args()
    
    # Force the diarizer choice
    os.environ["DIARIZER"] = args.diarizer
    
    print(f"Loading ground truth from {args.ref}")
    ref_segments = load_ground_truth(args.ref)
    print(f"Loaded {len(ref_segments)} reference segments.")
    
    print(f"Running {args.diarizer} diarization on {args.audio}...")
    from services.diarization_service import diarize_with_overlap
    hyp_segments = diarize_with_overlap(args.audio)
    print(f"Generated {len(hyp_segments)} hypothesis segments.")
    
    print("\n--- Evaluation Results ---")
    evaluate_diarization(ref_segments, hyp_segments)

if __name__ == "__main__":
    main()
