# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Runs the bots trained in self_play_train.py and renders in pygame.

You must provide experiment_state, expected to be
~/ray_results/PPO/experiment_state_YOUR_RUN_ID.json
"""

import argparse

import dm_env
from dmlab2d.ui_renderer import pygame
import numpy as np
from ray.rllib.agents.registry import get_trainer_class
from ray.tune.analysis.experiment_analysis import ExperimentAnalysis
from ray.tune.registry import register_env

from examples.rllib import utils
from meltingpot.python.human_players import level_playing_utils


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--experiment_state",
      type=str,
      default="~/ray_results/PPO",
      help="ray.tune experiment_state to load. The default setting will load"
      " the last training run created by self_play_train.py. If you want to use"
      " a specific run, provide a path, expected to be of the format "
      " ~/ray_results/PPO/experiment_state-DATETIME.json")
  parser.add_argument(
      "--human",
      action="store_true",
      help="human talks the place of one of the bots")

  args = parser.parse_args()

  agent_algorithm = "PPO"

  register_env("meltingpot", utils.env_creator)

  experiment = ExperimentAnalysis(
      args.experiment_state,
      default_metric="episode_reward_mean",
      default_mode="max")

  best_trial = experiment.get_best_trial(scope="last")
  config = best_trial.config
  checkpoint_path = best_trial.checkpoint.value
  # checkpoint_path = experiment.get_trial_checkpoints_paths(best_trial)[-1][0]
  # config = experiment.get_best_config()
  # checkpoint_path = experiment.get_best_checkpoint()
  # TODO: Do I need a serious evaluation during these passes? Would PBT then use that?

  config["explore"] = False
  config["in_evaluation"] = True

  trainer = get_trainer_class(agent_algorithm)(config=config)
  trainer.restore(checkpoint_path)

  # Create a new environment to visualise
  env = utils.env_creator(config["env_config"]).get_dmlab2d_env()

  num_bots = config["env_config"]["num_players"]
  if args.human:
    num_bots = num_bots - 1
  bots = [
      utils.RayModelPolicy(
          trainer, config["env_config"]["individual_observation_names"], "av")
  ] * num_bots

  timestep = env.reset()
  states = [bot.initial_state() for bot in bots]
  actions = [0] * len(bots)

  # Configure the pygame display
  scale = 4
  fps = 8

  pygame.init()
  clock = pygame.time.Clock()
  pygame.display.set_caption("DM Lab2d")
  obs_spec = env.observation_spec()
  shape = obs_spec[0]["WORLD.RGB"].shape
  game_display = pygame.display.set_mode(
      (int(shape[1] * scale), int(shape[0] * scale)))

  total_rewards = np.zeros(config["env_config"]["num_players"])

  for _ in range(config["horizon"]):
    obs = timestep.observation[0]["WORLD.RGB"]
    obs = np.transpose(obs, (1, 0, 2))
    surface = pygame.surfarray.make_surface(obs)
    rect = surface.get_rect()
    surf = pygame.transform.scale(surface,
                                  (int(rect[2] * scale), int(rect[3] * scale)))

    game_display.blit(surf, dest=(0, 0))
    pygame.display.update()
    clock.tick(fps)

    if args.human:
      while True:
        a = 0
        for event in pygame.event.get():
          if event.type == pygame.KEYDOWN:
            a = level_playing_utils.get_direction_pressed()
            break
          # TODO: fix bug where two quick presses, e.g. 1,2, are counted as 1,1

        if a != 0:
          break

      human_action = [a]
    else:
      human_action = []

    for i, bot in enumerate(bots):
      timestep_bot = dm_env.TimeStep(
          step_type=timestep.step_type,
          reward=timestep.reward[i],
          discount=timestep.discount,
          observation=timestep.observation[i])

      actions[i], states[i] = bot.step(timestep_bot, states[i])

    timestep = env.step(actions + human_action)
    print(actions + human_action, timestep.reward)
    total_rewards = total_rewards + timestep.reward

  print("Total rewards: {}".format(total_rewards))


if __name__ == "__main__":
  main()
