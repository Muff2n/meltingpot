"""Run experiments"""

import argparse
from collections import defaultdict
from ml_collections.config_dict import ConfigDict
import ray
from ray.air import CheckpointConfig, RunConfig
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env

from examples.rllib import utils
from meltingpot import substrate
from reward_transfer.callbacks import MyCallbacks

# Use Tuner.fit() to gridsearch over exchange values
# Thus I need to stick a custom parameter in the config and hope I can access this in the callback
# worry about loading pre-training later

SUBSTRATE_NAME = "coins"
# SUBSTRATE_NAME = "allelopathic_harvest__open"
NUM_GPUS = 0
LOGGING_LEVEL = "WARN"
VERBOSE = 1
KEEP_CHECKPOINTS_NUM = 1  # Default None
CHECKPOINT_FREQ = 10  # Default 0

NUM_WORKERS = 3
NUM_ENVS_PER_WORKER = 1
NUM_EPISODES_PER_WORKER = 1
SGD_MINIBATCH_SIZE = 4096  # 256 = minimum for efficient CPU training
LR = 2e-4
VF_CLIP_PARAM = 2.0
NUM_SGD_ITER = 10
EXPLORE_EVAL = False
ENTROPY_COEFF = 0.003
# TODO: Fix evaluation at end of training
EVAL_DURATION = 80

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--n_iterations",
      type=int,
      required=True,
      help="number of training iterations to use")
  parser.add_argument(
      "--num_cpus", type=int, required=True, help="number of CPUs to use")
  parser.add_argument(
      "--num_gpus", type=int, default=0, help="number of GPUs to use")
  parser.add_argument(
      "--local_dir",
      type=str,
      required=True,
      help="The path the results will be saved to")
  parser.add_argument(
      "--tmp_dir",
      type=str,
      default=None,
      help="Custom tmp location for temporary ray logs")
  args = parser.parse_args()

  ray.init(
      address="local",
      num_cpus=args.num_cpus,
      num_gpus=args.num_gpus,
      logging_level=LOGGING_LEVEL,
      _temp_dir=args.tmp_dir)

  register_env("meltingpot", utils.env_creator)

  # TODO: Fix if multiple roles

  substrate_config = substrate.get_config(SUBSTRATE_NAME)
  player_roles = substrate_config.default_player_roles
  num_players = len(player_roles)
  unique_roles = defaultdict(list)
  for i, role in enumerate(player_roles):
    unique_roles[role].append(f"player_{i}")

    # 1. import the module.
    # 2. call build
  import importlib

  env_module = importlib.import_module(f"meltingpot.configs.substrates.{SUBSTRATE_NAME}")
  substrate_definition = env_module.build(player_roles, substrate_config)
  horizon = substrate_definition["maxEpisodeLengthFrames"]
  sprite_size = substrate_definition["spriteSize"]

  def policy_mapping_fn(aid, *args, **kwargs):
    for role, pids in unique_roles.items():
      if aid in pids:
        return role
    assert False

  # TODO: SPRITE_SIZE
  env_config = ConfigDict({
      "substrate": SUBSTRATE_NAME,
      "substrate_config": substrate_config,
      "roles": player_roles,
      "scaled": 1
  })

  base_env = utils.env_creator(env_config)
  policies = {}
  for i, role in enumerate(unique_roles):
    rgb_shape = base_env.observation_space[f"player_{i}"]["RGB"].shape
    sprite_x = rgb_shape[0]
    sprite_y = rgb_shape[1]

    policies[role] = PolicySpec(
        # policy_class=None,  # use default policy
        observation_space=base_env.observation_space[f"player_{i}"],
        action_space=base_env.action_space[f"player_{i}"],
        # TODO: FIX to have an actual convolution for spacial purposes
        config={
        # config={
        #     "model": {
        #         "conv_filters": [[16, [8, 8], 8],
        #                          [128, [sprite_x, sprite_y], 1]],
        #                          # [128, [11, 11], 1]],
        #     },
        })

  if sprite_size == 8:
    conv_filters = [[16, [8, 8], 8],
                    [32, [4, 4], 1],
                    [64, [sprite_x // sprite_size, sprite_y // sprite_size], 1]]
  elif sprite_size == 1:
    conv_filters = [[16, [3, 3], 1],
                    [32, [3, 3], 1],
                    [64, [sprite_x, sprite_y], 1]]
  else:
    assert False, "Unknown sprite_size of {sprite_size}"

  DEFAULT_MODEL = {
      "conv_filters": conv_filters,
      "conv_activation": "relu",
      "post_fcnet_hiddens": [64, 64],
      "post_fcnet_activation": "relu",
      "vf_share_layers": True,
      "use_lstm": True,
      "lstm_use_prev_action": False,
      # "lstm_use_prev_action": True,
      "lstm_use_prev_reward": False,
      "lstm_cell_size": 128,
  }

  # TODO: Get maxEpisodeLengthFrames from substrate definition
  train_batch_size = max(
      1, NUM_WORKERS) * NUM_ENVS_PER_WORKER * NUM_EPISODES_PER_WORKER * horizon

  config = PPOConfig().training(
      model=DEFAULT_MODEL,
      lr=LR,
      train_batch_size=train_batch_size,
      lambda_=0.80,
      vf_loss_coeff=0.5,
      entropy_coeff=ENTROPY_COEFF,
      clip_param=0.2,
      vf_clip_param=VF_CLIP_PARAM,
      sgd_minibatch_size=min(SGD_MINIBATCH_SIZE, train_batch_size),
      num_sgd_iter=NUM_SGD_ITER,
  ).rollouts(
      batch_mode="complete_episodes",
      num_rollout_workers=NUM_WORKERS,
      rollout_fragment_length=100,
      num_envs_per_worker=NUM_ENVS_PER_WORKER,
  ).multi_agent(
      policies=policies,
      policy_mapping_fn=policy_mapping_fn,
  ).fault_tolerance(
      recreate_failed_workers=True,
      num_consecutive_worker_failures_tolerance=3,
  ).environment(
      env="meltingpot",
      env_config=env_config,
  ).debugging(
      log_level=LOGGING_LEVEL,
  ).resources(
      num_gpus=args.num_gpus,
      num_cpus_per_worker=1,
      num_gpus_per_worker=0,
      num_cpus_for_local_worker=1,
      num_learner_workers=0,
  ).framework(
      framework="tf",
  ).reporting(
      metrics_num_episodes_for_smoothing=1,
  ).evaluation(
      evaluation_interval=None,  # don't evaluate unless we call evaluation()
      evaluation_config={
          "explore": EXPLORE_EVAL,
      },
      evaluation_duration=EVAL_DURATION,
  ).experimental(
    _disable_preprocessor_api=False  # will be set to true in future versions of Ray, was True in baselines
  )

  # TODO: MyCallbacks are putting the reward as a list or scalar, and that is the opposite to what is required
  # MyCallbacks.set_transfer_map({f"policy_{i}": 1 - i/5 for i in range(n)})
  # config = config.callbacks(MyCallbacks)

  tune_config = tune.TuneConfig(reuse_actors=False)

  checkpoint_config = CheckpointConfig(
      num_to_keep=KEEP_CHECKPOINTS_NUM,
      checkpoint_frequency=CHECKPOINT_FREQ,
      checkpoint_at_end=True)

  experiment = tune.run(
      run_or_experiment="PPO",
      name=SUBSTRATE_NAME,
      metric="episode_reward_mean",
      mode="max",
      stop={"training_iteration": args.n_iterations},
      config=config,
      checkpoint_config=checkpoint_config,
      verbose=VERBOSE,
      log_to_file=False,
      local_dir=args.local_dir,
      # storage_path=args.local_dir
    )

  # run_config = RunConfig(
  #     name=SUBSTRATE_NAME,
  #     local_dir=args.local_dir,
  #     stop={"training_iteration": args.n_iterations},
  #     checkpoint_config=checkpoint_config,
  #     verbose=VERBOSE)

  # tuner = tune.Tuner(
  #     "PPO", param_space=config, tune_config=tune_config, run_config=run_config)
  # results = tuner.fit()

  # best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
  # print(best_result)

  ray.shutdown()
