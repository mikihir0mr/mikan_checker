import tensorflow as tf
import numpy as np
import os

model_path = r"c:\Users\stemw\Downloads\orangechecker\tflite_models-20251205T022915Z-3-001\tflite_models\mikan_classifier.tflite"

if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    exit(1)

try:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Input Details:")
    for i, detail in enumerate(input_details):
        print(f"Input {i}: Name: {detail['name']}, Shape: {detail['shape']}, Type: {detail['dtype']}, Index: {detail['index']}")

    print("\nOutput Details:")
    for i, detail in enumerate(output_details):
        print(f"Output {i}: Name: {detail['name']}, Shape: {detail['shape']}, Type: {detail['dtype']}, Index: {detail['index']}")

except Exception as e:
    print(f"Error inspecting model: {e}")
