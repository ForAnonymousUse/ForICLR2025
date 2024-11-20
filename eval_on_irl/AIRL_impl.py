import argparse
import os
import sys
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TransformObservation

from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.logger import configure, Logger

from imitation.algorithms.density import DensityAlgorithm
from imitation.algorithms.adversarial.airl import AIRL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data.types import TransitionsMinimal
from imitation.policies.serialize import load_policy
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from imitation.util.logger import HierarchicalLogger

import vvcgym
from envs.env import ConcatObsWrapper
from my_wrappers import ScaledActionWrapper, ScaledObservationWrapper
import envs.env
from utils_my.sb3.my_evaluate_policy import evaluate_policy_with_success_rate

PROJECT_ROOT_DIR = Path(__file__).parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

def on_best_success_rate_save(algo, eval_env, eval_freq, n_eval_episodes, exp_name, policy_save_name):
    max_success_rate = -1.0  

    def calc_func(index_cnt):
        if index_cnt % eval_freq == 0:
            algo.policy.set_training_mode(mode=False)

            nonlocal max_success_rate

            tmp_reward, _, tmp_success_rate = evaluate_policy_with_success_rate(algo, eval_env, n_eval_episodes, deterministic=True)

            if tmp_success_rate > max_success_rate:
                sb3_logger.info(f"update success rate from {max_success_rate} to {tmp_success_rate}!")
                max_success_rate = tmp_success_rate

                # save policy
                checkpoint_save_dir = PROJECT_ROOT_DIR / "checkpoints" / "IRL" / exp_name
                os.makedirs(checkpoint_save_dir, exist_ok=True)

                algo.save(checkpoint_save_dir)

            algo.policy.set_training_mode(mode=True)

    return calc_func


def train(exp_name, env_config_path, total_timesteps, train_env_num,eval_env_num, eval_freq, eval_episodes, train_seed, eval_seed, rollouts_path, sb3_logger):
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

    learner = PPO(
        env=venv,
        policy=MlpPolicy,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0005,
        gamma=0.95,
        clip_range=0.1,
        vf_coef=0.1,
        n_epochs=5,
        seed=train_seed,
    )
    reward_net = BasicShapedRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=RunningNorm,
    )
    airl_trainer = AIRL(
        demonstrations=rollouts,
        demo_batch_size=2048,
        gen_replay_buffer_capacity=512,
        n_disc_updates_per_round=16,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True,
        custom_logger=HierarchicalLogger(sb3_logger)
    )

    airl_trainer.train(
        total_timesteps=total_timesteps,
        callback=on_best_success_rate_save(learner, eval_env, eval_freq, eval_episodes, exp_name, "best_policy")
    )

    reward, _, success_rate = evaluate_policy_with_success_rate(learner, eval_env, n_eval_episodes=1000)
    sb3_logger.info("Reward after AIRL: ", reward)
    sb3_logger.info("Success rate: ", success_rate)
    final_policy_path = str(PROJECT_ROOT_DIR / "checkpoints" / exp_name / "final_policy")
    os.makedirs(final_policy_path, exist_ok=True)
    learner.save(final_policy_path)

    learner_rewards_after_training, _ = evaluate_policy(
        learner, eval_env, 100, return_episode_rewards=True,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="experiment_setting")
    parser.add_argument("--config-file-name", type=str, help="config_file", default="envs/vvcgym_config.json")
    parser.add_argument("--exp-name", type=str, default="AIRL_seed1_train_154_eval_78")
    parser.add_argument("--train-env-num", type=int, default=64, help="")
    parser.add_argument("--eval-env-num", type=int, default=32, help="")
    parser.add_argument("--eval-freq", type=int, default=150, help="")
    parser.add_argument("--eval-episodes", type=int, default=96, help="")
    parser.add_argument("--total-timesteps", type=int, default=int(1e7), help="")
    parser.add_argument("--train-seed", type=int, default=154, help="")
    parser.add_argument("--eval-seed", type=int, default=78, help="")
    parser.add_argument("--rollouts-path", type=str, default="demos_forIRL.npy", help="")
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