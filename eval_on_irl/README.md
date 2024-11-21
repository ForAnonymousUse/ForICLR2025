# Source code, training logs, evaluation screenshots, checkpoints of two IRL algorithms

* Source code: AIRL_impl.py
* training logs: logs/
* evaluation screenshots: screenshots/
* checkpoints: checkpoints/

## train scripts

1. Density Kernel

    ```bash
    cd eval_on_irl
    python DK_impl.py --config-file-name envs/vvcgym_config.json --exp-name DK_seed1 --train-env-num 64 --eval-env-num 32 --eval-freq 3000 --eval-episodes 96 --total-timesteps 200000000 --train-seed 42 --eval-seed 12 --rollouts-path demos_forIRL.npy
    ```

2. AIRL

    ```bash
    cd eval_on_irl
    python AIRL_impl.py --config-file-name envs/vvcgym_config.json --exp-name AIRL_seed1 --train-env-num 64 --eval-env-num 32 --eval-freq 150 --eval-episodes 96 --total-timesteps 10000000 --train-seed 333 --eval-seed 31 --rollouts-path demos_forIRL.npy
    ```
