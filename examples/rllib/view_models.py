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
import importlib
import logging

import cv2
import dm_env
from dmlab2d.ui_renderer import pygame
from ml_collections.config_dict import ConfigDict
import numpy as np
import ray
from ray.rllib.algorithms import PPO, PPOConfig
from ray.tune.analysis.experiment_analysis import ExperimentAnalysis
from ray.tune.registry import register_env

from meltingpot import substrate
from examples.rllib.utils import env_creator, RayModelPolicy

# Configure logging
logging.basicConfig(
    filename="view_models.log", filemode="w", level=logging.INFO)
logger = logging.getLogger(__name__)


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
      "--fps", type=int, default=8, help="Frames per second (default 8)")
  parser.add_argument(
      "--timesteps",
      type=int,
      default=500,
      help="Number of timesteps to run the epsiode for")
  parser.add_argument(
      "--video",
      type=str,
      default=None,
      help="Save the recording at this filepath")
  parser.add_argument(
      "--training",
      type=str,
      default="self-play",
      choices=["self-play", "independent"],
      help="""self-play: all players share the same policy
    independent: use n policies""")
  parser.add_argument(
      "--substrate", type=str, default=None, help="Only use if you know what you are doing")
  parser.add_argument(
      "--num_players", type=int, default=None, help="Only use if you know what you are doing")

  args = parser.parse_args()

  ray.init(
      address="local",
      num_gpus=0)

  register_env("meltingpot", env_creator)

  experiment = ExperimentAnalysis(
      args.experiment_state,
      default_metric="env_runners/episode_reward_mean",
      default_mode="max")

  checkpoint_path = args.checkpoint if args.checkpoint is not None else experiment.best_checkpoint.path

  config = PPOConfig.from_dict(experiment.best_config)

  config = config.env_runners(num_env_runners=0).resources(num_gpus=0)

  trainer = PPO(config=config)

  trainer.load_checkpoint(checkpoint_path)

  if args.substrate:
    substrate_config = substrate.get_config(args.substrate)

    env_module = importlib.import_module(
        f"meltingpot.configs.substrates.{args.substrate}")

    num_players = len(substrate_config.default_player_roles)
    if args.num_players:
      num_players = min(args.num_players, num_players)

    roles = substrate_config.default_player_roles[0:num_players]

    env_config = ConfigDict({
        "substrate": args.substrate,
        "substrate_config": substrate_config,
        "roles": roles,
        "scaled": 1
    })
  else:
    env_config = config["env_config"]
    num_players = len(env_config["roles"])

  env = env_creator(env_config)

  if args.training == "independent":
    policies = env._ordered_agent_ids
  else:
    policies = env_config["roles"]

  bots = [
      RayModelPolicy(
          trainer,
          env_config["substrate_config"]["individual_observation_names"],
          policy) for policy in policies
  ]
  bots = bots[1:] if args.human else bots

  env = env.get_dmlab2d_env()
  obs_spec = env.observation_spec()
  shape = obs_spec[0]["WORLD.RGB"].shape

  # Configure the pygame display
  pygame.init()
  scale = 1000 // max(int(shape[0]), int(shape[1]))
  fps = args.fps
  frame_size = (int(shape[1] * scale), int(shape[0] * scale))
  game_display = pygame.display.set_mode(frame_size)
  clock = pygame.time.Clock()
  pygame.display.set_caption("DM Lab2d")

  if args.video:
    # Video recording setup
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(args.video, fourcc, fps, frame_size)

  total_rewards = np.zeros(num_players)
  timestep = env.reset()
  states = [bot.initial_state() for bot in bots]
  actions = [0] * len(bots)

  for _ in range(args.timesteps):
    obs = timestep.observation[0]["WORLD.RGB"]
    obs = np.transpose(obs, (1, 0, 2))
    surface = pygame.surfarray.make_surface(obs)
    rect = surface.get_rect()
    surf = pygame.transform.scale(surface,
                                  (int(rect[2] * scale), int(rect[3] * scale)))

    game_display.blit(surf, dest=(0, 0))
    pygame.display.update()
    # pgyame.image.save(game_display, "image_name.jpeg")

    if args.video:
      # Capture the frame for recording
      frame = pygame.surfarray.array3d(game_display)
      frame = cv2.transpose(frame)
      # frame = cv2.flip(frame, 0)
      frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
      out.write(frame)

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

    timestep = env.step(human_action + actions)
    print(human_action + actions, timestep.reward)
    total_rewards = total_rewards + timestep.reward

  print(f"Total rewards: {total_rewards}")

  pygame.quit()
  if args.video:
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
  main()
