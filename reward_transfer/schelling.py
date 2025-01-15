"""
Runs the analysis required to plot a Schelling diagram for an environment given
checkpoints with policies said to implement cooperation and defection
"""

import argparse
import logging
import os

import json
import ray
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.utils.checkpoints import get_checkpoint_info
from ray.rllib.policy.policy import Policy
from ray.tune.registry import register_env
from ray.tune.analysis.experiment_analysis import ExperimentAnalysis

from examples.rllib import utils
from meltingpot import substrate


def main():
  print("Start")
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--num_cpus", type=int, required=True, help="number of CPUs to use")
  parser.add_argument(
    "--n_episodes", type=int, required=True, help="number of episodes to evaluate over")
  parser.add_argument(
      "--experiment_state",
      type=str,
      required=True,
      help="ray.tune experiment_state to load. The default setting will load"
      " the last training run created by self_play_train.py. If you want to use"
      " a specific run, provide a path, expected to be of the format "
      " ~/ray_results/PPO/experiment_state-DATETIME.json")
  parser.add_argument(
      "--cooperation_checkpoint",
      type=str,
      required=True,
      help="Cooperative policies")
  parser.add_argument(
      "--defection_checkpoint",
      type=str,
      required=True,
      help="Defect policies")

  args = parser.parse_args()

  print("Calling init")
  ray.init(address="local",
           num_cpus=args.num_cpus,
           logging_level=logging.ERROR)

  register_env("meltingpot", utils.env_creator)

  experiment = ExperimentAnalysis(
      args.experiment_state,
      default_metric="env_runners/episode_reward_mean",
      default_mode="max")

  config = PPOConfig.from_dict(experiment.best_config)

  substrate_config = substrate.get_config(config["env_config"]["substrate"])

  config["env_config"]["self-interest"] = 1
  config["env_config"]["roles"] = substrate_config.default_player_roles

  base_env = utils.env_creator(config["env_config"])
  aids = base_env._ordered_agent_ids


  config = config.resources(num_gpus=0)
  config = config.env_runners(
      num_env_runners=0,
      num_envs_per_env_runner=1,
  )
  config = config.evaluation(
      evaluation_duration=args.n_episodes,
      evaluation_num_env_runners=args.num_cpus - 1,
      evaluation_interval=1,
  )

  print("Building PPO instance")
  ppo = config.build()

  # load all the defect policies...
  for aid in aids:
    policy_path = os.path.join(args.defection_checkpoint, "policies", aid)
    print(f"Update policy {aid} to defect using path {policy_path}")
    policy = Policy.from_checkpoint(policy_path)
    ppo.get_policy(aid).set_weights(policy.get_weights())

  print("Running evaluate()")
  results = ppo.evaluate()
  print(results)

  i = 0
  with open(os.path.join(os.path.dirname(args.experiment_state), f"n_c_{i}.json"), mode="w", encoding="utf8") as f:
    json.dump(results, f)

  # sweep over the possible policy pairings
  for aid in aids:
    # load a cooperative policy
    policy_path = os.path.join(args.cooperation_checkpoint, "policies", aid)
    print(f"Update policy {aid} to cooperate using path {policy_path}")
    policy = Policy.from_checkpoint(policy_path)
    ppo.get_policy(aid).set_weights(policy.get_weights())

    print("Running evaluate()")
    results = ppo.evaluate()
    print(results)

    i += 1
    with open(os.path.join(os.path.dirname(args.experiment_state), f"n_c_{i}.json"), mode="w", encoding="utf8") as f:
      json.dump(results, f)


if __name__ == "__main__":
  main()
