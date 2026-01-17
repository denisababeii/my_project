# import torch
# import torchvision

# model = torchvision.models.resnet18(weights=None)
# model.eval()

# dummy_input = torch.randn(1, 3, 224, 224)
# torch.onnx.export(
#     model,
#     dummy_input,
#     "resnet18.onnx",
#     input_names=["input"],
#     output_names=["output"],
#     dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
# )

# import onnx
# model = onnx.load("resnet18.onnx")
# onnx.checker.check_model(model)
# print(onnx.printer.to_text(model.graph))

# import onnxruntime as rt
# import numpy as np

# ort_session = rt.InferenceSession("/Volumes/MASTER/Year1/MLOps/my_project/resnet18.onnx")
# input_names = [i.name for i in ort_session.get_inputs()]
# output_names = [i.name for i in ort_session.get_outputs()]
# batch = {input_names[0]: np.random.randn(1, 3, 224, 224).astype(np.float32)}
# out = ort_session.run(output_names, batch)

# import sys
# import time
# from statistics import mean, stdev

# import onnxruntime as ort
# import torch
# import torchvision


# def timing_decorator(func, function_repeat: int = 10, timing_repeat: int = 5):
#     """Decorator that times the execution of a function."""

#     def wrapper(*args, **kwargs):
#         timing_results = []
#         for _ in range(timing_repeat):
#             start_time = time.time()
#             for _ in range(function_repeat):
#                 result = func(*args, **kwargs)
#             end_time = time.time()
#             elapsed_time = end_time - start_time
#             timing_results.append(elapsed_time)
#         print(f"Avg +- Stddev: {mean(timing_results):0.3f} +- {stdev(timing_results):0.3f} seconds")
#         return result

#     return wrapper


# model = torchvision.models.resnet18()
# model.eval()

# dummy_input = torch.randn(1, 3, 224, 224)
# torch.onnx.export(
#     model,
#     dummy_input,
#     "resnet18.onnx",
#     input_names=["input.1"],
#     dynamic_axes={"input.1": {0: "batch_size", 2: "height", 3: "width"}},
# )

# ort_session = ort.InferenceSession("resnet18.onnx")


# @timing_decorator
# def torch_predict(image) -> None:
#     """Predict using PyTorch model."""
#     model(image)


# @timing_decorator
# def onnx_predict(image) -> None:
#     """Predict using ONNX model."""
#     ort_session.run(None, {"input.1": image.numpy()})


# if __name__ == "__main__":
#     for size in [224, 448, 896]:
#         dummy_input = torch.randn(1, 3, size, size)
#         print(f"Image size: {size}")
#         torch_predict(dummy_input)
#         onnx_predict(dummy_input)


import onnxruntime as rt
sess_options = rt.SessionOptions()

# Set graph optimization level
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

# To enable model serialization after graph optimization set this
sess_options.optimized_model_filepath = "/Volumes/MASTER/Year1/MLOps/my_project/resnet18.onnx"

session = rt.InferenceSession("/Volumes/MASTER/Year1/MLOps/my_project/resnet18.onnx", sess_options)