Here is **Appendix B** of "Modeling Financial Chaos."

This section is the "Mechanic's Manual."
Mamba is not a standard Python library like `pandas`. It relies on custom CUDA kernels that must be compiled against specific versions of the NVIDIA Toolkit. If your environment is not perfectly aligned, `pip install` will fail with cryptic C++ errors.

This appendix provides a reproducible, battle-tested setup guide for Linux and Windows (WSL2), along with a Troubleshooting Matrix for the most common build failures.

-----

# Appendix B: Mamba Installation and CUDA Troubleshooting

## B.1 The "Golden Environment" Reference

Deep Learning libraries move fast. To avoid "Dependency Hell," we recommend pinning your environment to this specific configuration known to work with Mamba 1.2+ and Mamba 2.0.

| Component        | Version Requirement                     | Notes                                                                           |
| :--------------- | :-------------------------------------- | :------------------------------------------------------------------------------ |
| **OS**           | Linux (Ubuntu 20.04/22.04) or WSL2      | **Do not use native Windows.** Mamba kernels do not support Windows I/O easily. |
| **GPU**          | NVIDIA Pascal or newer (Compute \> 6.0) | GTX 1080 Ti, RTX 30/40 Series, A100, H100.                                      |
| **Python**       | 3.10 (Recommended) or 3.11              | Python 3.12 support is experimental.                                            |
| **PyTorch**      | 2.1.0 or newer                          | Must be the **CUDA** version, not CPU.                                          |
| **CUDA Toolkit** | 11.8 or 12.1                            | Must match the version PyTorch was built with.                                  |
| **Packaging**    | `ninja`, `packaging`, `wheel`           | Required for Just-In-Time (JIT) compilation.                                    |

-----

## B.2 Installation Walkthrough (Linux / WSL2)

### Step 1: Clean Environment

Do not install this into your system Python. Create a dedicated Conda environment.

```bash
conda create -n mamba_chaos python=3.10
conda activate mamba_chaos
```

### Step 2: Install PyTorch (The Critical Step)

You **must** ensure PyTorch matches your system's CUDA driver.
Check your driver version:

```bash
nvidia-smi
# Look for "CUDA Version: 12.x" in the top right corner.
```

If you have CUDA 12.x:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

If you have CUDA 11.x:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Verification:** Run python and check `torch.version.cuda`. If it says "None" or "cpu", stop. You cannot proceed.

### Step 3: Install Build Tools

Mamba compiles C++ code on the fly during installation. It needs the Ninja build system.

```bash
pip install packaging ninja wheel
```

### Step 4: Install Mamba-SSM and Causal-Conv1d

These two libraries are inseparable partners. `causal-conv1d` provides the optimized convolution kernel used in the Mamba block.

```bash
# Install the convolution kernel first
pip install causal-conv1d>=1.2.0

# Install the main Mamba library
pip install mamba-ssm
```

*Note: This step might take 5-10 minutes. You will see "Building wheel..." This is normal. It is compiling CUDA code.*

-----

## B.3 Troubleshooting Matrix

If `pip install` fails, consult this table. 90% of errors are due to **Version Mismatches**.

| Error Message                                                                 | Root Cause                                                                                        | The Fix                                                                                                                                                                |
| :---------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------ | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `RuntimeError: No CUDA GPUs are available`                                    | You installed the CPU version of PyTorch.                                                         | Uninstall torch. Reinstall using the `--index-url .../cu118` flag.                                                                                                     |
| `nvcc not found` or `CUDA_HOME environment variable is not set`               | The compiler cannot find the CUDA Toolkit on your disk.                                           | Install the **CUDA Toolkit** (not just drivers) via `conda install -c nvidia cuda-toolkit` or `apt-get install cuda-toolkit`. Then `export CUDA_HOME=/usr/local/cuda`. |
| `limit argument must be an int, not float` (in `selective_scan_interface.py`) | Version conflict between PyTorch 2.3+ and older Mamba versions.                                   | Upgrade Mamba: `pip install mamba-ssm --upgrade --no-cache-dir`.                                                                                                       |
| `undefined symbol: ...` (at runtime)                                          | Binary incompatibility. You compiled Mamba with CUDA 11.8 but are running PyTorch with CUDA 12.1. | Nuke the environment. Reinstall PyTorch and Mamba ensuring versions match exactly.                                                                                     |
| `Ninja is required to load C++ extensions`                                    | Missing Ninja build tool.                                                                         | `pip install ninja`. Ensure it is in your `$PATH`.                                                                                                                     |
| `Out of Memory (OOM)` during installation                                     | Compiling kernels takes a lot of RAM.                                                             | Use `MAX_JOBS=4 pip install mamba-ssm` to limit parallel compilation threads.                                                                                          |

-----

## B.4 Docker Production Setup

For the Software Architect deploying to AWS/GCP, do not rely on manual installation. Use this Dockerfile to guarantee a working build.

```dockerfile
# Start with the official NVIDIA PyTorch image (Pre-loaded with CUDA/CuDNN)
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Set environment variables to force CUDA build
ENV CUDA_HOME="/usr/local/cuda"
ENV MAMBA_FORCE_BUILD="TRUE"

# Install dependencies
RUN pip install --upgrade pip
RUN pip install packaging ninja

# Install Mamba-SSM
# We clone and build from source for maximum stability in Docker
RUN git clone https://github.com/state-spaces/mamba.git \
    && cd mamba \
    && pip install .

# Copy your Chaos Trader Code
WORKDIR /app
COPY . /app

# Default command
CMD ["python", "deploy/inference_service.py"]
```

-----

## B.5 Verification Script

Run this Python script to prove your environment is ready for the Heston training loop.

```python
import torch
import time

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device Name: {torch.cuda.get_device_name(0)}")

try:
    from mamba_ssm import Mamba
    print("✅ Mamba Library Imported Successfully")
    
    # Test a forward pass on the GPU
    batch, length, dim = 2, 128, 64
    x = torch.randn(batch, length, dim).cuda()
    model = Mamba(
        d_model=dim, 
        d_state=16, 
        d_conv=4, 
        expand=2
    ).cuda()
    
    start = time.time()
    y = model(x)
    end = time.time()
    
    print(f"✅ Forward Pass Successful. Output Shape: {y.shape}")
    print(f"⏱️ Inference Time (128 seq): {(end-start)*1000:.2f}ms")
    
except ImportError as e:
    print(f"❌ Mamba Import Failed: {e}")
except Exception as e:
    print(f"❌ Runtime Error: {e}")
```

If you see the **Green Checkmarks**, you are ready to model Chaos.