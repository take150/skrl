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
    "state_preprocessor": None,             # state preprocessor class
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "mixed_precision": False,       # enable automatic mixed precision
    }

# [end-config-dict-torch]
# fmt: on


class MultiDistillationTeacher(MultiAgent):
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

        for uid in self.possible_agents:
            # set up preprocessors
            if self._state_preprocessor[uid] is not None:
                self._state_preprocessor[uid] = self._state_preprocessor[uid](**self._state_preprocessor_kwargs[uid])
                self.checkpoint_modules[uid]["state_preprocessor"] = self._state_preprocessor[uid]
            else:
                self._state_preprocessor[uid] = self._empty_preprocessor

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        self.set_mode("eval")

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