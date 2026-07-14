"""
generate_full_requirements.py
自动检测当前环境信息（Python/CUDA/cuDNN/操作系统）
并生成项目所需的 requirements.txt
"""

import sys
import platform
import subprocess
import importlib.metadata as metadata

OUTPUT_FILE = "requirements.txt"
ENV_INFO_FILE = "environment_info.txt"

# 你项目实际用到的包列表（按你之前提供的清单 + 补充 ultralytics）
PACKAGE_LIST = [
    "ultralytics",
    "torch",
    "torchvision",
    "Pillow",
    "opencv-python",
    "lap",
    "matplotlib",
    "mmcv",
    "mmengine",
    "numpy",
    "pandas",
    "psutil",
    "pytest",
    "requests",
    "scipy",
    "thop",
    "timm",
    "tqdm",
    "PyYAML",
]


def get_version(pkg_name):
    try:
        return metadata.version(pkg_name)
    except metadata.PackageNotFoundError:
        return None


def get_cuda_info():
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda
        cudnn_version = torch.backends.cudnn.version()
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
        return cuda_available, cuda_version, cudnn_version, gpu_name
    except Exception as e:
        return False, None, None, f"检测失败: {e}"


def get_nvidia_smi():
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
        return result.stdout
    except Exception:
        return "nvidia-smi 不可用（可能未安装NVIDIA驱动或非GPU环境）"


def main():
    print("========== 环境信息检测 ==========\n")

    py_version = platform.python_version()
    os_info = platform.platform()
    print(f"Python 版本: {py_version}")
    print(f"操作系统: {os_info}")

    cuda_available, cuda_version, cudnn_version, gpu_name = get_cuda_info()
    print(f"CUDA 是否可用: {cuda_available}")
    print(f"CUDA 版本: {cuda_version}")
    print(f"cuDNN 版本: {cudnn_version}")
    print(f"GPU 型号: {gpu_name}")

    nvidia_smi_output = get_nvidia_smi()

    print("\n========== 依赖包版本检测 ==========\n")
    lines = []
    for pkg in PACKAGE_LIST:
        version = get_version(pkg)
        if version:
            print(f"  {pkg}=={version}  (已安装)")
            lines.append(f"{pkg}=={version}")
        else:
            print(f"  {pkg}  (未安装，请自行确认所需版本)")
            lines.append(f"{pkg}  # 未安装，请确认版本")

    # 写入 requirements.txt
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n已生成 {OUTPUT_FILE}")

    # 写入环境信息文件（方便直接复制进README）
    with open(ENV_INFO_FILE, "w", encoding="utf-8") as f:
        f.write("========== Environment Information ==========\n\n")
        f.write(f"Python version: {py_version}\n")
        f.write(f"Operating System: {os_info}\n")
        f.write(f"CUDA available: {cuda_available}\n")
        f.write(f"CUDA version: {cuda_version}\n")
        f.write(f"cuDNN version: {cudnn_version}\n")
        f.write(f"GPU: {gpu_name}\n\n")
        f.write("========== nvidia-smi output ==========\n")
        f.write(nvidia_smi_output)

    print(f"已生成 {ENV_INFO_FILE}，可直接复制进README的Requirements部分")


if __name__ == "__main__":
    main()