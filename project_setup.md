# Device prepare
GPU device: A100
cuda version: 11.6
python: 3.10

# ✅step 1: set up the environment
ensure cuda version is 11.6
```bash
module load cuda/11.6
NVCC_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | cut -d',' -f1)
if [ "$NVCC_VERSION" == "11.6" ]; then
    echo "✅ [PASS] Detected NVCC version 11.6"
else
    echo "❌ [ERROR] Version mismatch! Expected 11.6, but found $NVCC_VERSION"
    echo "Please run 'module load cuda/11.6' or check your PATH."
    exit 1
fi
```

```bash
conda create -n gaussianart python=3.10 -y
conda activate gaussianart

pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

# verify cuda and torch
python -c "import torch; print('CUDA version:', torch.version.cuda); print('Torch version:', torch.__version__)"
```

## install dependencies: submodules
```bash
pip install -r requirements.txt
git submodule update --init --recursive
pip install --no-build-isolation submodules/simple-knn 
pip install --no-build-isolation submodules/art-diff-gaussian-rasterization
```

## install pytorch3d from 
source [here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)


```bash
cd ..
wget https://github.com/facebookresearch/pytorch3d/archive/refs/tags/v0.7.5.tar.gz
tar -xzvf v0.7.5.tar.gz
cd pytorch3d-0.7.5
pip install --no-build-isolation .
```


## reinstall submodules
```bash
cd /data/enalisn1/ylu174/research/GaussianArt/submodules/simple-knn
rm -rf build/ dist/ *.egg-info
pip install --no-build-isolation .

cd /data/enalisn1/ylu174/research/GaussianArt/submodules/art-diff-gaussian-rasterization
rm -rf build/ dist/ *.egg-info
pip install .

python -c "import torch; import diff_gaussian_rasterization; print('Rasterizer is OK')"
cd $(git rev-parse --show-toplevel)
cd ..
rm -rf v0.7.5.tar.gz pytorch3d-0.7.5

```

# ✅step 2: prepare dataset
```bash
cd $(git rev-parse --show-toplevel)
pip install -U "huggingface_hub[cli]"
if [[ ":$PATH:" != *":$CONDA_PREFIX/bin:"* ]]; then
    export PATH="$CONDA_PREFIX/bin:$PATH"
fi
# check whether hf is here
hf --help > /dev/null 2>&1; if [ $? -eq 0 ]; then echo "pass"; else echo "huggingface-cli not found"; fi
hf download LiCheng23/MPArt-90 --repo-type dataset --local-dir ./data
cd data
for f in *.zip; do UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -o "$f" && rm "$f"; done
```

## ✅step 3: train
```bash
cd $(git rev-parse --show-toplevel)
python run.py --model_id Box_102379
python eval_axis.py -m output/MPArt90/Box_102379
python render_video.py -m output/MPArt90/Box_102379
```
run in HPC cluster with slurm:
```bash
cd SLURM_execution/SLURM_script
sbatch baseline.sh
sbatch --dependency=afterok:<job_id> evaluation.sh
sbatch --dependency=afterok:<job_id> render.sh
```

alternatively
```bash
cd SLURM_execution/SLURM_script
sbatch pipeline.sh
```