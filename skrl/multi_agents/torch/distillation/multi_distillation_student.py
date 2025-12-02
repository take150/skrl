from typing import Any, Mapping, Optional, Sequence, Union

import copy
import itertools
import gymnasium
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence

from skrl import config, logger
from skrl.multi_agents.torch import MultiAgent
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.resources.schedulers.torch import KLAdaptiveLR


# fmt: off
# [start-config-dict-torch]
DISTILLATION_DEFAULT_CONFIG = {
    "rollouts": 16,                 # number of rollouts before updating
    "learning_epochs": 8,           # number of learning epochs during each update
    "mini_batches": 2,              # number of mini batches during each learning epoch

    "learning_rate": 1e-3,                  # learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs

    "state_preprocessor": None,             # state preprocessor class
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0.5,                  # clipping coefficient for the norm of the gradients

    "mixed_precision": False,       # enable automatic mixed precision

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)
        
        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately
        
        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs
    }
}
# [end-config-dict-torch]
# fmt: on


class MultiDistillationStudent(MultiAgent):
    def __init__(
        self,
        possible_agents: Sequence[str],
        models: Mapping[str, Model],
        memories: Optional[Mapping[str, Memory]] = None,
        observation_spaces: Optional[Union[Mapping[str, int], Mapping[str, gymnasium.Space]]] = None,
        action_spaces: Optional[Union[Mapping[str, int], Mapping[str, gymnasium.Space]]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        """
        Distillation Agent (Student mimics Teacher via KL Divergence)
        
        :param models: Dictionary containing "policy" (Student) and "teacher" (Teacher) models.
        """
        _cfg = copy.deepcopy(DISTILLATION_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(
            possible_agents=possible_agents,
            models=models,
            memories=memories,
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            device=device,
            cfg=_cfg,
        )

        # models
        self.policies = {uid: self.models[uid].get("policy", None) for uid in self.possible_agents}
        
        for uid in self.possible_agents:
            # checkpoint models
            self.checkpoint_modules[uid]["policy"] = self.policies[uid]

            # broadcast models' parameters in distributed runs
            if config.torch.is_distributed:
                logger.info(f"Broadcasting models' parameters")
                if self.policy is not None:
                    self.policy.broadcast_parameters()

        # configuration
        self._learning_epochs = self._as_dict(self.cfg["learning_epochs"])
        self._mini_batches = self._as_dict(self.cfg["mini_batches"])
        self._rollouts = self.cfg["rollouts"]
        self._rollout = 0

        self._grad_norm_clip = self._as_dict(self.cfg["grad_norm_clip"])
        
        self._learning_rate = self._as_dict(self.cfg["learning_rate"])
        self._learning_rate_scheduler = self._as_dict(self.cfg["learning_rate_scheduler"])
        self._learning_rate_scheduler_kwargs = self._as_dict(self.cfg["learning_rate_scheduler_kwargs"])

        self._state_preprocessor = self._as_dict(self.cfg["state_preprocessor"])
        self._state_preprocessor_kwargs = self._as_dict(self.cfg["state_preprocessor_kwargs"])

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]
        
        self._mixed_precision = self.cfg["mixed_precision"]

        # set up automatic mixed precision
        self._device_type = torch.device(device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self._mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._mixed_precision)

        # set up optimizer (Only for Student Policy)
        self.optimizers = {}
        self.schedulers = {}

        for uid in self.possible_agents:
            policy = self.policies[uid]
            if policy is not None:
                optimizer = torch.optim.Adam(policy.parameters(), lr=self._learning_rate[uid])
                self.optimizers[uid] = optimizer
                if self._learning_rate_scheduler[uid] is not None:
                    self.schedulers[uid] = self._learning_rate_scheduler[uid](
                        optimizer, **self._learning_rate_scheduler_kwargs[uid]
                    )

            self.checkpoint_modules[uid]["optimizer"] = self.optimizers[uid]

            # set up preprocessors
            if self._state_preprocessor[uid] is not None:
                self._state_preprocessor[uid] = self._state_preprocessor[uid](**self._state_preprocessor_kwargs[uid])
                self.checkpoint_modules[uid]["state_preprocessor"] = self._state_preprocessor[uid]
            else:
                self._state_preprocessor[uid] = self._empty_preprocessor

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent"""
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # create tensors in memory
        if self.memories:
            for uid in self.possible_agents:
                self.memories[uid].create_tensor(name="states", size=self.observation_spaces[uid], dtype=torch.float32)
                self.memories[uid].create_tensor(name="actions", size=self.action_spaces[uid], dtype=torch.float32)
                self.memories[uid].create_tensor(name="teacher_actions", size=self.action_spaces[uid], dtype=torch.float32)
                self.memories[uid].create_tensor(name="rewards", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="terminated", size=1, dtype=torch.bool)
                self.memories[uid].create_tensor(name="truncated", size=1, dtype=torch.bool)

            # tensors sampled during training
            self._tensors_names = ["states", "actions", "teacher_actions"]

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the STUDENT policy"""
        
        # if timestep < self._random_timesteps:
        #     return self.policy.random_act({"states": self._state_preprocessor(states)}, role="policy")
        
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            data = [
                self.policies[uid].act({"states": self._state_preprocessor[uid](states[uid])}, role="policy")
                for uid in self.possible_agents
            ]

            actions = {uid: d[0] for uid, d in zip(self.possible_agents, data)}
            outputs = {uid: d[2] for uid, d in zip(self.possible_agents, data)}

        return actions, outputs

    def record_transition(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record an environment transition in memory"""
        super().record_transition(
            states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
        )

        if self.memories:
            teacher_actions = infos.get("teacher_actions", None)
            for uid in self.possible_agents:
                # storage transition in memory
                # Note: We don't need value computation here for Distillation
                self.memories[uid].add_samples(
                    states=states[uid],
                    actions=actions[uid],
                    teacher_actions=teacher_actions[uid],
                    rewards=rewards[uid],
                    terminated=terminated[uid],
                    truncated=truncated[uid],
                )

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        pass

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        self._rollout += 1
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step (Distillation / Supervised Learning)"""
        
        # sample mini-batches from memory
        for uid in self.possible_agents:
            policy = self.policies[uid]
            memory = self.memories[uid]

            sampled_batches = memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches[uid])

            cumulative_loss = 0

            # learning epochs
            for epoch in range(self._learning_epochs[uid]):
                
                # mini-batches loop
                for (
                    sampled_states,
                    sampled_actions,
                    sampled_teacher_actions,
                ) in sampled_batches:

                    with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                        
                        # Preprocess states
                        sampled_states = self._state_preprocessor[uid](sampled_states, train=not epoch)
                        
                        predicted_actions, _, _ = policy.act({"states": sampled_states}, role="policy")
                    
                        loss = F.mse_loss(predicted_actions, sampled_teacher_actions)

                    # optimization step
                    self.optimizers[uid].zero_grad()
                    self.scaler.scale(loss).backward()
                    
                    if self._grad_norm_clip[uid] > 0:
                        self.scaler.unscale_(self.optimizers[uid])
                        nn.utils.clip_grad_norm_(policy.parameters(), self._grad_norm_clip[uid])

                    self.scaler.step(self.optimizers[uid])
                    self.scaler.update()

                    # update cumulative losses
                    cumulative_loss += loss.item()

                # update learning rate
                if self._learning_rate_scheduler[uid]:
                    self.schedulers[uid].step()

            # record data
            self.track_data(
                f"Loss / MSE ({uid})",
                cumulative_loss / (self._learning_epochs[uid] * self._mini_batches[uid]),
            )

            if self._learning_rate_scheduler[uid]:
                self.track_data(f"Learning / Learning rate ({uid})", self.schedulers[uid].get_last_lr()[0])
