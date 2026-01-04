"""
v0 Baseline Submission - Wunder Challenge
==========================================

This is our first submission - a direct copy of the provided baseline.

Model: Vanilla GRU
Score (local validation): 0.2595
- t0: 0.3884
- t1: 0.1306

Architecture:
- Input: (1, 100, 32) - last 100 steps, 32 features
- GRU layers
- Output: (1, 2) - predictions for t0 and t1

Local Testing:
    cd /Users/thomasmyles/dev/wunder-predictorium/submissions/v0_baseline
    source ../../venv/bin/activate
    python solution.py

Packaging for Submission:
    cd /Users/thomasmyles/dev/wunder-predictorium/submissions/v0_baseline
    zip -r ../v0_submission.zip .
    # Creates v0_submission.zip (253 KB) with solution.py and baseline.onnx

Submission:
    1. Go to: https://predictorium.wundernn.io/submit
    2. Upload: submissions/v0_submission.zip
    3. Wait for scoring (max 90 minutes)
    4. Check leaderboard: https://predictorium.wundernn.io/leaderboard

Expected Performance:
    - Local validation: 0.2595
    - Expected test: ~0.25-0.27 (may vary)
"""

import os
import numpy as np
import onnxruntime as ort


class PredictionModel:
    """
    Baseline Solution (Vanilla GRU).
    
    Uses a simple GRU model trained on 32 features to predict t0 and t1.
    Model is pre-trained and exported to ONNX for fast inference.
    """

    def __init__(self):
        self.current_seq_ix = None
        self.sequence_history = []
        
        # Determine model path (same directory as this file)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        onnx_path = os.path.join(base_dir, "baseline.onnx")
        
        # Initialize ONNX Runtime Session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.ort_session = None
        
        try:
            # Load ONNX Model
            self.ort_session = ort.InferenceSession(
                onnx_path, 
                sess_options, 
                providers=['CPUExecutionProvider']
            )
            self.input_name = self.ort_session.get_inputs()[0].name
            self.output_name = self.ort_session.get_outputs()[0].name
            print(f"✓ Loaded ONNX model from {onnx_path}")
                
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            self.ort_session = None

    def predict(self, data_point) -> np.ndarray:
        """
        Generate predictions for the current data point.
        
        Args:
            data_point: DataPoint object with attributes:
                - seq_ix: sequence ID
                - step_in_seq: step number (0-999)
                - need_prediction: whether prediction is needed
                - state: numpy array of 32 features
        
        Returns:
            None if prediction not needed, otherwise numpy array of shape (2,)
            containing [t0_prediction, t1_prediction]
        """
        # Reset state on new sequence
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.sequence_history = []

        # Update history with current state
        self.sequence_history.append(data_point.state.copy())

        # Return None if prediction not needed (warm-up period)
        if not data_point.need_prediction:
            return None
            
        # Fallback if model failed to load
        if self.ort_session is None:
            return np.zeros(2, dtype=np.float32)

        # Prepare input: last 100 steps
        history_window = self.sequence_history[-100:]
        
        # Pad with zeros if needed (shouldn't happen after step 99)
        if len(history_window) < 100:
            padding = [np.zeros_like(history_window[0])] * (100 - len(history_window))
            history_window = padding + history_window

        # Convert to numpy array with shape (1, 100, 32)
        data_arr = np.asarray(history_window, dtype=np.float32)
        data_tensor = np.expand_dims(data_arr, axis=0)
        
        # Run ONNX inference
        ort_inputs = {self.input_name: data_tensor}
        output = self.ort_session.run([self.output_name], ort_inputs)[0]
        
        # Handle different output shapes
        if len(output.shape) == 3:
            # Shape: (Batch, Seq, Features) -> take last timestep
            prediction = output[0, -1, :]
        else:
            # Shape: (Batch, Features)
            prediction = output[0]
            
        return prediction


# Local testing code
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
    
    from wnn_predictorium_starterpack.utils import ScorerStepByStep
    
    # Test on validation set
    valid_path = "../../wnn_predictorium_starterpack/datasets/valid.parquet"
    
    if os.path.exists(valid_path):
        print("=" * 60)
        print("Testing v0 Baseline Submission")
        print("=" * 60)
        
        model = PredictionModel()
        scorer = ScorerStepByStep(valid_path)
        
        results = scorer.score(model)
        
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Mean Weighted Pearson: {results['weighted_pearson']:.6f}")
        for target, score in results.items():
            if target != 'weighted_pearson':
                print(f"  {target}: {score:.6f}")
        print("=" * 60)
    else:
        print(f"Validation file not found: {valid_path}")

