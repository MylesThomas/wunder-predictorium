"""
Export v1 PyTorch model to ONNX

Usage:
    cd /Users/thomasmyles/dev/wunder-predictorium
    source venv/bin/activate
    python src/export/export_v1_to_onnx.py
"""

import os
import sys
import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.models.improved_gru import ImprovedGRU


def export_to_onnx(model_path='models/v1_improved_gru.pt',
                   output_path='submissions/v1_improved_arch/v1_model.onnx'):
    
    print("=" * 60)
    print("Exporting v1 Model to ONNX")
    print("=" * 60)
    print()
    
    # Check input exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return False
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load model
    print(f"Loading PyTorch model from {model_path}...")
    device = torch.device('cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    model = ImprovedGRU()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Valid loss: {checkpoint['val_loss']:.6f}")
    print()
    
    # Create dummy input
    dummy_input = torch.randn(1, 100, 32)
    
    # Export to ONNX
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        opset_version=11
    )
    
    print(f"✓ ONNX export complete")
    print()
    
    # Verify ONNX model
    print("Verifying ONNX model...")
    import onnxruntime as ort
    
    session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
    
    # Test inference
    test_input = np.random.randn(1, 100, 32).astype(np.float32)
    onnx_output = session.run(['output'], {'input': test_input})[0]
    
    # Compare with PyTorch
    with torch.no_grad():
        torch_output = model(torch.FloatTensor(test_input)).numpy()
    
    diff = np.abs(onnx_output - torch_output).max()
    print(f"✓ ONNX model verified")
    print(f"  Max difference vs PyTorch: {diff:.6e}")
    print()
    
    # File size
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ ONNX model saved")
    print(f"  Path: {output_path}")
    print(f"  Size: {file_size:.2f} MB")
    print()
    
    print("=" * 60)
    print("Next steps:")
    print("1. Check: ls -lh submissions/v1_improved_arch/")
    print("2. Create submission: cd submissions/v1_improved_arch && zip -r ../v1_submission.zip .")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = export_to_onnx()
    sys.exit(0 if success else 1)


