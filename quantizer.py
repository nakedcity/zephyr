from onnxruntime.quantization import quantize_dynamic, QuantType
import os

def quantize_model(input_path: str, output_path: str):
    """
    Quantize an ONNX model to INT8 using dynamic quantization.
    """
    print(f"Quantizing model {input_path} to {output_path}...")
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8
    )
    print("Quantization complete.")
