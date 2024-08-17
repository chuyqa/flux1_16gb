# Flux.1 16gb 4060ti 

Flux.1 typically requires a 24GB GPU, but you can run w/ 8bit precision and achieve ~ 20 sec/image using 14GB.

Check out run_lite.py for a lightweight version, or run a Gradio app as shown below.


---
### Environment Setup
Using Conda:

```bash
Copy code
conda env create -f flux-env.yaml
```
Or with pip:

```bash
Copy code
pip install -r reqs.txt
```

### Running the Project
```
# ~30GB required - export to custom disk if needed. 
# export HF_HOME=/data/nvme1/workspace/
# export HUGGINGFACE_HUB_CACHE=/data/nvme1/workspace/models/

conda activate flux-env
cd /data/nvme1/workspace/flux1_16gb
python run_lite.py
```

---

### Performace
~ 20 sec / image

<img src=flux_003.png width=30%><img src=flux_002.png width=30%><img src=flux_001.png width=30%>



---
### Memory usage
```
### 512 x 512 , 4 steps
Every 1.0s: nvidia-smi                       lana: Fri Aug 16 23:45:30 2024

Fri Aug 16 23:45:30 2024
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.12             Driver Version: 535.104.12   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 4060 Ti     Off | 00000000:1F:00.0 Off |                  N/A |
| 30%   72C    P2             145W / 165W |  13169MiB / 16380MiB |    100%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

### 1024 x 1024 , 4 steps
Every 1.0s: nvidia-smi                       lana: Fri Aug 16 23:47:06 2024

Fri Aug 16 23:47:06 2024
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.12             Driver Version: 535.104.12   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 4060 Ti     Off | 00000000:1F:00.0 Off |                  N/A |
| 34%   61C    P2             132W / 165W |  13671MiB / 16380MiB |    100%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

### 1024 x 1024 , 8 steps
Every 1.0s: nvidia-smi                       lana: Fri Aug 16 23:47:47 2024

Fri Aug 16 23:47:47 2024
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.12             Driver Version: 535.104.12   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 4060 Ti     Off | 00000000:1F:00.0 Off |                  N/A |
| 34%   66C    P2             137W / 165W |  13701MiB / 16380MiB |    100%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
