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
"""

import argparse

import dm_env
from dmlab2d.ui_renderer import pygame
from ml_collections.config_dict import ConfigDict
import numpy as np
from ray.rllib.algorithms.registry import _get_algorithm_class
from ray.tune.analysis.experiment_analysis import ExperimentAnalysis
from ray.tune.registry import register_env

from meltingpot import substrate
from examples.rllib.utils import env_creator, RayModelPolicy


class RandomBot:
  def __init__(self, action_space):
    self.action_space = action_space

  def initial_state(self):
    return None

  def step(self, *args):
    return self.action_space.sample(), None


def get_human_action():
  a = None
  for event in pygame.event.get():
    if event.type == pygame.KEYDOWN:
      key_pressed = pygame.key.get_pressed()
      if key_pressed[pygame.K_SPACE]:
        a = 0
      if key_pressed[pygame.K_UP]:
        a = 1
      if key_pressed[pygame.K_DOWN]:
        a = 2
      if key_pressed[pygame.K_LEFT]:
        a = 3
      if key_pressed[pygame.K_RIGHT]:
        a = 4
      if key_pressed[pygame.K_z]:
        a = 5
      if key_pressed[pygame.K_x]:
        a = 6
      break  # removing this did not solve the bug
  return a

def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--experiment_state",
      type=str,
      required=True,
      help="ray.tune experiment_state to load. The default setting will load"
      " the last training run created by self_play_train.py. If you want to use"
      " a specific run, provide a path, expected to be of the format "
      " ~/ray_results/PPO/experiment_state-DATETIME.json")
  parser.add_argument(
      "--checkpoint",
      type=str,
      default=None,
      help="If provided, use this checkpoint instead of the last checkpoint")
  parser.add_argument(
      "--human", action="store_true", help="a human controls one of the bots")
  parser.add_argument(
      "--fps",
      type=int,
      default=8,
      help="Frames per second (default 8)")
  parser.add_argument(
      "--substrate",
      type=str,
      default=None,
      help="Use this substrate instead of the original")

  args = parser.parse_args()

  agent_algorithm = "PPO"

  register_env("meltingpot", env_creator)

  experiment = ExperimentAnalysis(
      args.experiment_state,
      default_metric="episode_reward_mean",
      default_mode="max")

  if args.checkpoint:
    checkpoint_path = args.checkpoint
  else:
    checkpoint_path = experiment.best_checkpoint

  config = experiment.best_config

  config["explore"] = False
  config["num_rollout_workers"] = 0

  trainer = _get_algorithm_class(agent_algorithm)(config=config)
  trainer.load_checkpoint(checkpoint_path)

  # Create a new environment to visualise
  if args.substrate:
    substrate_config = substrate.get_config(args.substrate)

    env_config = ConfigDict({
        "substrate": args.substrate,
        "substrate_config": substrate_config,
        "roles": substrate_config["default_player_roles"],
        "scaled": 1
    })
  else:
    env_config = config["env_config"]

  env = env_creator(env_config)
  action_space = env.action_space["player_0"]
  env = env.get_dmlab2d_env()
  obs_spec = env.observation_spec()
  shape = obs_spec[0]["WORLD.RGB"].shape

  # Setup the players
  roles = env_config["roles"]
  num_players = len(roles)
  bots = [
      RayModelPolicy(
        trainer,
        env_config["substrate_config"]["individual_observation_names"],
        role)
    if role != "random" else
      RandomBot(action_space)
    for role in roles
  ]
  bots = bots[1:] if args.human else bots

  # Configure the pygame display
  pygame.init()
  scale = 1000 // max(int(shape[0]), int(shape[1]))
  fps = args.fps
  game_display = pygame.display.set_mode(
      (int(shape[1] * scale), int(shape[0] * scale)))
  clock = pygame.time.Clock()
  pygame.display.set_caption("DM Lab2d")

  total_rewards = np.zeros(num_players)
  timestep = env.reset()
  states = [bot.initial_state() for bot in bots]
  actions = [0] * len(bots)

  for _ in range(500):
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
        a = get_human_action()
        # TODO: fix bug where two quick presses, e.g. 1,2, are counted as 1,1

        if a is not None:
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
