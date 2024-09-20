## Install
- python -m venv .venv
- pip install -r requirements.txt

## Examples
- python simple_img.py
- python sd3_controlnet_depth.py

## Debug 
### CUDA memory usage
```text
>_ nvidia-smi
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A10                     On  | 00000000:08:00.0 Off |                    0 |
|  0%   33C    P8              15W / 150W |      7MiB / 23028MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```

## Trubleshoting
### Apple Silicon
If you will receive aan error about being out of memory, you can try tweak PYTORCH_MPS_HIGH_WATERMARK_RATIO giving a value between 0.0 and 1.0. The lower the value, the more it will take, because it will use more swapping space on the disk to load the models.

Example:
```
>_ PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6 python3 sd3_controlnet_depth.py
```
