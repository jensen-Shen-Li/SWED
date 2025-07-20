## Requirements

To install all the required dependencies:

1. Install MuJoCo engine, which can be downloaded from [here](https://mujoco.org/download).

2. Install Python packages listed in `requirements.txt` using `pip`. You should specify the versions of `mujoco_py` and `dm_control` in `requirements.txt` depending on the version of MuJoCo engine you have installed as follows:
    - MuJoCo 2.0: `mujoco-py<2.1,>=2.0`, `dm_control==0.0.364896371`
    - MuJoCo 2.1.0: `mujoco-py<2.2,>=2.1`, `dm_control==0.0.403778684`
    - MuJoCo 2.1.1: not supported

3. Manually download and install `d4rl` package from [here](https://github.com/rail-berkeley/d4rl). You should remove lines including `dm_control` in `setup.py`.

Here is an example of how to install all the dependencies on Ubuntu:
  
```bash
conda create -n edac python=3.7
conda activate edac
# Specify versions of mujoco-py and dm_control in requirements.txt
pip install --no-cache-dir -r requirements.txt

cd .
git clone https://github.com/rail-berkeley/d4rl.git

cd d4rl
# Remove lines including 'dm_control' in setup.py
pip install -e .
```

## Reproducing the results

### Gym

To reproduce SAC-N results for MuJoCo Gym, run:

```bash
python -m scripts.sac --env_name [ENVIRONMENT] --num_qs [N]
```

To reproduce EDAC results for MuJoCo Gym, run:

```bash
python -m scripts.sac --env_name [ENVIRONMENT] --num_qs [N] --eta [ETA]
```
python -m scripts.sac --env_name halfcheetah-medium-v2 --num_qs 6 --q_samples 4 --win_step 2 --eta 1.0
python -m scripts.sac --env_name halfcheetah-medium-v2 --num_qs 10 --q_samples 6 --win_step 4 --eta 1.0
python -m scripts.sac --env_name halfcheetah-medium-replay-v2 --num_qs 10 --q_samples 6 --win_step 4 --eta 1.0
python -m scripts.sac --env_name halfcheetah-medium-expert-v2 --num_qs 10 --q_samples 6 --win_step 4 --eta 5.0
python -m scripts.sac --env_name halfcheetah-expert-v2 --num_qs 10 --q_samples 6 --win_step 4 --eta 10.0
python -m scripts.sac --env_name hopper-medium-v2 --num_qs 10 --q_samples 6 --eta 1.0
python -m scripts.sac --env_name halfcheetah-expert-v2 --num_qs 10 --q_samples 8 --win_step 6 --eta 1.0
python -m scripts.sac --env_name hopper-medium-v2 --num_qs 40 --q_samples 32 --win_step 16 --eta 1.0
python -m scripts.sac --env_name hopper-medium-replay-v2 --num_qs 40 --q_samples 32 --win_step 16 --eta 1.0
python -m scripts.sac --env_name hopper-medium-expert-v2 --num_qs 40 --q_samples 32 --win_step 16 --eta 1.0
### Adroit

On Adroit tasks, we apply reward normalization for further training stability. For example, to reproduce the EDAC results for pen-human, run:

```bash
python -m scripts.sac --env_name pen-human-v1 --epoch 200 --num_qs 20 --plr 3e-5 --eta 1000 --reward_mean --reward_std
```

To reproduce the EDAC results for pen-cloned, run:

```bash
python -m scripts.sac --env_name pen-human-v1 --epoch 200 --num_qs 20 --plr 3e-5 --eta 10 --max_q_backup --reward_mean --reward_std
```

