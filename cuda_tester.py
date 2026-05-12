import tensorflow as tf
import sys
import os
import ctypes

print(tf.config.list_physical_devices("GPU"))
print(tf.__version__)

print(tf.test.is_built_with_cuda())


print("cudnn64_8.dll:", os.path.exists(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cudnn64_8.dll"))
print("cudnn_ops_infer64_8.dll:", os.path.exists(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cudnn_ops_infer64_8.dll"))
print("cudnn_cnn_infer64_8.dll:", os.path.exists(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cudnn_cnn_infer64_8.dll"))
print("cudnn_adv_infer64_8.dll:", os.path.exists(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cudnn_adv_infer64_8.dll"))


print("cudart64_110.dll:", os.path.exists(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cudart64_110.dll"))
print("cublas64_11.dll:", os.path.exists(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cublas64_11.dll"))
print("cublasLt64_11.dll:", os.path.exists(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cublasLt64_11.dll"))



print("CUDA in PATH:", any("CUDA\\v11.8\\bin" in p for p in os.environ["PATH"].split(";")))


print(sys.version)

import struct
print(struct.calcsize("P") * 8)



dlls = [
    "cudart64_110.dll",
    "cublas64_11.dll",
    "cudnn64_8.dll",
    "cudnn_ops_infer64_8.dll",
    "cudnn_cnn_infer64_8.dll",
    "cudnn_adv_infer64_8.dll"
]

for dll in dlls:
    try:
        ctypes.WinDLL(dll)
        print(dll, "LOADED")
    except Exception as e:
        print(dll, "FAILED:", e)


#even with everything on the line, despite blood sweat and tears, it wasn't meant to happen.
