## Install
- python -m venv .venv
- source .venv/bin/activate
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

## Run

The majority of the step necessary to test build and deploy the are explained here https://replicate.com/docs/guides/push-a-model

### Install

#### Cog

```
>_: cd ..
>_: sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
>_: sudo chmod +x /usr/local/bin/cog
>_: cd serve-diffuse
```

#### Go

```
>_: wget https://go.dev/dl/go1.23.1.linux-amd64.tar.gz
>_: sudo tar -C /usr/local -xzf go1.23.1.linux-amd64.tar.gz
>_: echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
>_: source ~/.bashrc
```

### Debug

```
>_: sudo cog predict -i prompt="monkey scuba diving"
```

### Build

```
>_: sudo cog build -t <your-model-name>
```

### Test

Or, you can use docket images (need the docket deamon running):

```
# If your model uses a CPU:
docker run -d -p 5001:5000 <your-model-name>

# If your model uses a GPU:
docker run -d -p 5001:5000 --gpus all <your-model-name>

# If you're on an M1 Mac:
docker run -d -p 5001:5000 --platform=linux/amd64 <your-model-name>
```

and then you can run the model with:

```
>_: curl http://localhost:5001/predictions -X POST \
    --header "Content-Type: application/json" \
    --data '{"input": {"prompt": "a fury dragon" }}'
```

### Deploy

```
>_: cog login
>_: cog push r8.im/<replicate-username>/<your-model-name>
```
