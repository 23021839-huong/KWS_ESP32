"""
export_to_c.py
Chạy sau khi train xong:
    python export_to_c.py

Output: model_weights.h  (nhúng vào firmware Arduino)
"""

import torch
import numpy as np
from model_tiny import KWS_SNN_Tiny
from config_tiny import Config

def quantize_int8(tensor):
    """Float32 → int8, trả về (data_int8, scale, zero_point)"""
    t = tensor.numpy().flatten()
    t_min, t_max = t.min(), t.max()
    scale = (t_max - t_min) / 255.0 + 1e-9
    zero_point = int(-t_min / scale)
    zero_point = max(0, min(255, zero_point))
    q = np.clip(np.round(t / scale + zero_point), 0, 255).astype(np.uint8)
    return q, scale, zero_point

def tensor_to_c_array(name, tensor, f):
    data, scale, zp = quantize_int8(tensor)
    size = len(data)
    f.write(f"// {name}: shape={list(tensor.shape)}, scale={scale:.8f}, zp={zp}\n")
    f.write(f"const float {name}_scale = {scale:.8f}f;\n")
    f.write(f"const int   {name}_zp    = {zp};\n")
    f.write(f"const int   {name}_size  = {size};\n")
    f.write(f"const uint8_t {name}[{size}] PROGMEM = {{\n  ")
    for i, v in enumerate(data):
        f.write(f"0x{v:02X}")
        if i < size - 1:
            f.write(", ")
            if (i + 1) % 16 == 0:
                f.write("\n  ")
    f.write("\n};\n\n")

def export():
    model = KWS_SNN_Tiny()
    model.load_state_dict(torch.load("checkpoints/tiny_best.pth", map_location="cpu"))
    model.eval()

    sd = model.state_dict()

    # In thông tin kích thước
    total_bytes = 0
    for k, v in sd.items():
        size = v.numel()
        total_bytes += size
        print(f"  {k:40s} {list(v.shape)}  {size*4/1024:.1f} KB (f32)")
    print(f"\nTổng float32: {total_bytes*4/1024:.1f} KB")
    print(f"Ước tính int8: {total_bytes/1024:.1f} KB")

    with open("model_weights.h", "w") as f:
        f.write("#pragma once\n")
        f.write("#include <Arduino.h>\n\n")
        f.write(f"// KWS SNN Tiny — tự động tạo bởi export_to_c.py\n")
        f.write(f"// Keywords: {Config.KEYWORDS}\n")
        f.write(f"// TIME_STEPS={Config.TIME_STEPS}, MAX_LEN={Config.MAX_LEN}, N_MELS={Config.N_MELS}\n\n")

        f.write(f"#define KWS_NUM_CLASSES {Config.NUM_CLASSES}\n")
        f.write(f"#define KWS_TIME_STEPS  {Config.TIME_STEPS}\n")
        f.write(f"#define KWS_MAX_LEN     {Config.MAX_LEN}\n")
        f.write(f"#define KWS_N_MELS      {Config.N_MELS}\n\n")

        f.write('const char* KWS_LABELS[] = {')
        f.write(', '.join(f'"{k}"' for k in Config.KEYWORDS))
        f.write('};\n\n')

        # Xuất từng layer
        tensor_to_c_array("conv1_weight",  sd["conv1.weight"],  f)
        tensor_to_c_array("conv1_bias",    sd["conv1.bias"],    f)
        tensor_to_c_array("bn1_weight",    sd["bn1.weight"],    f)
        tensor_to_c_array("bn1_bias",      sd["bn1.bias"],      f)
        tensor_to_c_array("bn1_mean",      sd["bn1.running_mean"], f)
        tensor_to_c_array("bn1_var",       sd["bn1.running_var"],  f)
        tensor_to_c_array("fc1_weight",    sd["fc1.weight"],    f)
        tensor_to_c_array("fc1_bias",      sd["fc1.bias"],      f)
        tensor_to_c_array("fc2_weight",    sd["fc2.weight"],    f)
        tensor_to_c_array("fc2_bias",      sd["fc2.bias"],      f)
        # LIF beta
        f.write(f"const float LIF_BETA = {Config.BETA}f;\n")

    print("\nXuất xong: model_weights.h")
    print("Copy file này vào thư mục sketch Arduino.")

if __name__ == "__main__":
    export()
