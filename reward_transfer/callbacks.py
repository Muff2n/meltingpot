"""Custom rllib callbacks to load policy weights and save episode results"""

import json
import logging
import os

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID

logging.basicConfig(filename="callbacks.log", level=logging.INFO)
logger = logging.getLogger(__name__)


class LoadPolicyCallback(DefaultCallbacks):

  def __init__(self):
    super().__init__()

  def on_create_policy(self, *, policy_id: PolicyID, policy: Policy) -> None:
    """Callback run when a new policy is added to an algorithm.

    Args:
        policy_id: ID of the newly created policy.
        policy: The policy just created.
    """
    policy_checkpoint = policy.config.get("policy_checkpoint")

    if policy_checkpoint is not None:
      pretrained_path = os.path.join(policy_checkpoint, "policies", policy_id)

      if not os.path.isdir(pretrained_path) and policy.config.get(
          "training-mode") == "independent":
        # There are two situations where we could be running in independent mode
        # and the policy does not exist under the expected policy_id
        test_path = os.path.join(policy_checkpoint, "policies", "default")

        # Case 1: we are training from a policy returned from the hyperparameter
        # optimisation, which is called "default"
        if os.path.isdir(test_path):
          pretrained_path = test_path

        # Case 2: we are pre-training and, the additional player will not have a
        # prior policy to start from, so we use "player-0"
        else:
          pretrained_path = os.path.join(policy_checkpoint, "policies",
                                         "player_0")

      if os.path.isdir(pretrained_path):
        logger.info(
            "on_create_policy::Process %s:Load pretrained policy from %s for policy %s",
            os.getpid(), pretrained_path, policy_id)
        pretrained_policy = Policy.from_checkpoint(pretrained_path)
        pretrained_weights = pretrained_policy.get_weights()
        policy.set_weights(pretrained_weights)
      else:
        logger.warn(
            "on_create_policy::Process %s:Pretrained policy %s does not exist",
            os.getpid(), pretrained_path)


class SaveResultsCallback(DefaultCallbacks):

  def __init__(self):
    super().__init__()

  def on_train_result(self, *, algorithm, metrics_logger, result, **kwargs) -> None:
    """Callback run after a training iteration has finished."""

    results_filepath = os.path.join(algorithm.config["working_folder"],
                                    "results.json")

    info = {}
    info["training_iteration"] = result["training_iteration"]
    self_interest = algorithm.config.env_config.get("self-interest")
    info["self-interest"] = 1 if self_interest is None else self_interest
    info["num_players"] = len(algorithm.config.env_config["roles"])
    info["training-mode"] = algorithm.config.get("training-mode")
    info.update(result["env_runners"]["hist_stats"])

    with open(results_filepath, mode="a", encoding="utf8") as f:
      json.dump(info, f)
      f.write("\n")
      logger.debug("on_train_result::%s", info)


  def on_sample_end(
        self,
        *,
        samples,
        **kwargs,
    ) -> None:
      logger.debug("on_sample_end::env_steps=%s, agent_steps=%s", samples.env_steps(), samples.agent_steps())
