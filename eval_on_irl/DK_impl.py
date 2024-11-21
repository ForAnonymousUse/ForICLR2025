import argparse
import os
import pprint
import sys
from pathlib import Path
import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.logger import configure, Logger

from imitation.util.util import make_vec_env
from imitation.algorithms import density as db
from imitation.data import serialize
from imitation.util import util
from imitation.util.logger import HierarchicalLogger

PROJECT_ROOT_DIR = Path(__file__).parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

import envs.env
from envs.env import ConcatObsWrapper
from utils_my.sb3.my_eval_callback import MyEvalCallback
from utils_my.sb3.my_wrappers import ScaledObservationWrapper, ScaledActionWrapper
from utils_my.sb3.my_evaluate_policy import evaluate_policy_with_success_rate


def train(exp_name, env_config_path, total_timesteps, train_env_num,eval_env_num, eval_freq, eval_episodes,train_seed, eval_seed, rollouts_path, sb3_logger):
    config_path = env_config_path

    venv = make_vec_env(
        'VVCGym-v0',
        rng=np.random.default_rng(train_seed),
        n_envs=train_env_num,
        env_make_kwargs={
            "config_file": config_path
        },
        post_wrappers=[lambda env, _: ConcatObsWrapper(ScaledActionWrapper(ScaledObservationWrapper(env)))]
    )
    eval_env = make_vec_env(
        'VVCGym-v0',
        rng=np.random.default_rng(eval_seed),
        n_envs=eval_env_num,
        env_make_kwargs={
            "config_file": config_path
        },
        post_wrappers=[lambda env, _: ConcatObsWrapper(ScaledActionWrapper(ScaledObservationWrapper(env)))]
    )
    rollouts = np.load(rollouts_path, allow_pickle=True)
    rollouts = rollouts.tolist()

    imitation_trainer = PPO(
        ActorCriticPolicy, venv, learning_rate=3e-4, gamma=0.95, ent_coef=1e-4, n_steps=2048
    )
    imitation_trainer.set_logger(sb3_logger)

    density_trainer = db.DensityAlgorithm(
        venv=venv,
        rng=np.random.default_rng(train_seed),
        demonstrations=rollouts,
        rl_algo=imitation_trainer,
        density_type=db.DensityType.STATE_ACTION_DENSITY,
        is_stationary=True,
        kernel="gaussian",
        kernel_bandwidth=0.4,
        standardise_inputs=True,
        allow_variable_horizon=True,
        custom_logger=HierarchicalLogger(sb3_logger)
    )
    density_trainer.train()

    def print_stats(density_trainer, n_trajectories):
        stats = density_trainer.test_policy(n_trajectories=n_trajectories)
        print("True reward function stats:")
        pprint.pprint(stats)
        stats_im = density_trainer.test_policy(true_reward=False, n_trajectories=n_trajectories)
        print("Imitation reward function stats:")
        pprint.pprint(stats_im)

    print("Stats before training:")
    print_stats(density_trainer, 1)

    save_dir = str((PROJECT_ROOT_DIR / "checkpoints" / "DK" / exp_name).absolute())
    os.makedirs(save_dir, exist_ok=True)

    density_trainer.train_policy(
        n_timesteps=total_timesteps,
    )

    print("Stats after training:")
    print_stats(density_trainer, 1)

    reward, _, success_rate = evaluate_policy_with_success_rate(imitation_trainer, eval_env, n_eval_episodes=1000)
    sb3_logger.info("Reward after DK: ", reward)
    sb3_logger.info("Success rate: ", success_rate)
    final_policy_path = str(PROJECT_ROOT_DIR / "checkpoints" / exp_name / "final_policy")
    os.makedirs(final_policy_path, exist_ok=True)
    imitation_trainer.save(final_policy_path)


if __name__ == "__main__":

 
    parser = argparse.ArgumentParser(description="experiment_setting")
    parser.add_argument("--config-file-name", type=str, help="config_file", default="envs/vvcgym_config.json")
    parser.add_argument("--exp-name", type=str, default="DK_seed3_train_55_eval_96")
    parser.add_argument("--train-env-num", type=int, default=64, help="")
    parser.add_argument("--eval-env-num", type=int, default=32, help="")
    parser.add_argument("--eval-freq", type=int, default=3000, help="")
    parser.add_argument("--eval-episodes", type=int, default=96, help="")
    parser.add_argument("--total-timesteps", type=int, default=int(1e7), help="")
    parser.add_argument("--train-seed", type=int, default=55, help="")
    parser.add_argument("--eval-seed", type=int, default=96, help="")
    parser.add_argument("--rollouts-path", type=str, default="/data/forIRL_not_dict_new.npy", help="")

    args = parser.parse_args()

    sb3_logger: Logger = configure(folder=str((PROJECT_ROOT_DIR / "logs" / args.exp_name).absolute()),
                                   format_strings=['stdout', 'log', 'csv', 'tensorboard'])
    train(
        exp_name=args.exp_name,
        env_config_path = Path(__file__).absolute().parent /args.config_file_name,
        total_timesteps = args.total_timesteps,
        train_env_num = args.train_env_num,
        eval_env_num = args.eval_env_num,
        eval_freq = args.eval_freq,
        eval_episodes = args.eval_episodes,
        train_seed = args.train_seed,
        eval_seed=args.eval_seed,
        rollouts_path = args.rollouts_path,
        sb3_logger = sb3_logger
    )
