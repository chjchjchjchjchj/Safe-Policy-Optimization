from __future__ import annotations
import random
from typing import Any, ClassVar
import torch
from gymnasium import spaces
import math
import numpy as np
import gym
import ipdb

class CalculateInferencetime:
    def __init__(self, Ci, Co, Wo, Ho, Kc, Kp, Sc, Sp, Pc, Pp, L, R, TC, hierarchy, BW, DataSize):

        self.Ci = [3, 96, 256, 384, 384]
        self.Co = [96, 256, 384, 384, 256]
        self.Wo = [55, 27, 13, 13, 13]
        self.Ho = [55, 27, 13, 13, 13]
        self.Kc = [11, 5, 3, 3, 3]
        self.Kp = [3, 3, 1, 1, 3]
        self.Sc = [4, 1, 1, 1, 1]
        self.Sp = [2, 2, 1, 1, 2]
        self.Pc = [3, 3, 1, 1, 3]
        self.Pp = [0, 0, 0, 0, 0]    

        """
        [tot, 0, 0, 0, 0]
        [x_1, tot-x_t, 0, 0, 0]
        [x_1, tot-x_t, 0, 0, 0]
        """

        self.L = 5  # Number of layers
        self.R = [2] * self.L  # Duplication for each layer
        self.TC= [5] * self.L
        self.hierarchy = [[0] for _ in range(self.L)]  # Hierarchy levels per layer
        self.BW = [1] * self.L  # Bandwidth at each hierarchy level
        self.DataSize = [[100] * len(self.hierarchy[i]) for i in range(self.L)]  # Data size per layer and hierarchy
    
    def PreOpInterval(self, i, k, Wp):
        Product = self.R[i]

        for j in range(k, i+1):
            row = math.ceil(Product / self.Wo[j])
            col = Product - (row - 1) * self.Wo[j]
            prow = (row - 1) * self.Sc[j] + self.Kc[j] - self.Pc[j]
            pcol = min((col - 1) * self.Sc[j] + self.Kc[j] - self.Pc[j], Wp[j - 1])
            crow = self.Kp[j - 1] + self.Sp[j - 1] * (prow - 1) - self.Pp[j - 1]
            ccol = min(self.Kp[j - 1] + self.Sp[j - 1] * (pcol - 1) - self.Pp[j - 1], self.Wo[j - 1])
            Product = math.ceil(((crow - 1) * self.Wo[j - 1] + ccol) / self.R[j - 1]) * self.R[j - 1]
        
        pre = math.ceil((crow - 1) * self.Wo[k] + ccol) / self.R[k] - 1
        return pre

    def InferenceTime(self, R):
        Op = [0] * self.L
        PreOp = [0] * self.L
        T_step = [0] * self.L
        Op[0] = math.ceil(self.Wo[0] * self.Ho[0] / R[0])
        Wp = [(self.Wo[l] - self.Kp[l] + 2 * self.Pp[l]) / self.Sp[l] + 1 for l in range(self.L)]
        for i in range(1, self.L):
            NormalOp = math.ceil(self.Wo[i] * self.Ho[i] / R[i])
            tmp = -1
            for k in range(i):
                tmp = max(tmp, self.PreOpInterval(i, k, Wp) + PreOp[k])
            PreOp[i] = tmp
            Tail = math.ceil(self.Wo[i] * (self.Pc[i] // self.Sc[i]) / R[i])
            Op[i] = max(NormalOp + PreOp[i], Op[i - 1] + Tail)
            TA = sum(self.DataSize[i][h] / self.BW[h] for h in self.hierarchy[i])
            T_step[i] = max(TA, self.TC[i])
        max_T_step = max(T_step)
        return Op[-1] * max_T_step

    def cal(self, R):
        return self.InferenceTime(R)        


# class CustomExampleEnv(gym.Env):
class CrossBarEnv(gym.Env):

    def __init__(self, env_id: str, **kwargs: dict[str, Any]) -> None:
        self.L = 5  # Number of layers
        self._count = 0
        self._num_envs = 1
        self._observation_space = spaces.Box(low=0, high=1000, shape=(self.L,))
        self._action_space = spaces.MultiDiscrete([3] * self.L)
        self.action_mapping = np.array([-1, 0, 1])

        self.num_steps = 10000

        self.Ci = np.array([3, 96, 256, 384, 384])
        self.Co = np.array([96, 256, 384, 384, 256])
        self.Wo = np.array([55, 27, 13, 13, 13])
        self.Ho = np.array([55, 27, 13, 13, 13])
        self.Kc = np.array([11, 5, 3, 3, 3])
        self.Kp = np.array([3, 3, 1, 1, 3])
        self.Sc = np.array([4, 1, 1, 1, 1])
        self.Sp = np.array([2, 2, 1, 1, 2])
        self.Pc = np.array([3, 3, 1, 1, 3])
        self.Pp = np.array([0, 0, 0, 0, 0])  

        self.M = 16
        self.N = 16
        self.cost = -100
        self.total = 100000

        self.R = np.zeros(self.L)
        self.TC = np.array([5] * self.L)
        self.hierarchy = np.zeros(self.L)
        self.BW = np.ones(self.L)
        self.DataSize = np.full((self.L, self.L), 100)
        self.set = np.ceil(self.Kc * self.Kc * self.Ci / self.M) * np.ceil(self.Co / self.N)

        self.calculator = CalculateInferencetime(
            Ci=self.Ci, 
            Co=self.Co, 
            Wo=self.Wo, 
            Ho=self.Ho, 
            Kc=self.Kc, 
            Kp=self.Kp, 
            Sc=self.Sc, 
            Sp=self.Sp, 
            Pc=self.Pc, 
            Pp=self.Pp, 
            L=self.L, 
            R=self.R, 
            TC=self.TC, 
            hierarchy=self.hierarchy, 
            BW=self.BW, 
            DataSize=self.DataSize
        )
    
    def cal_cost(self):
        x = self.R * self.set
        if np.sum(x) > self.total:
            return self.cost
        
        if np.any(self.R < 1) or np.any(self.R > self.Wo * self.Ho):
            return self.cost

        return 0

    def PreOpInterval(self, i, k, Wp):
        Product = self.R[i]

        for j in range(k, i+1):
            row = np.ceil(Product / self.Wo[j])
            col = Product - (row - 1) * self.Wo[j]
            prow = (row - 1) * self.Sc[j] + self.Kc[j] - self.Pc[j]
            pcol = np.minimum((col - 1) * self.Sc[j] + self.Kc[j] - self.Pc[j], Wp[j - 1])
            crow = self.Kp[j - 1] + self.Sp[j - 1] * (prow - 1) - self.Pp[j - 1]
            ccol = np.minimum(self.Kp[j - 1] + self.Sp[j - 1] * (pcol - 1) - self.Pp[j - 1], self.Wo[j - 1])
            Product = np.ceil(((crow - 1) * self.Wo[j - 1] + ccol) / self.R[j - 1]) * self.R[j - 1]

        pre = np.ceil((crow - 1) * self.Wo[k] + ccol) / self.R[k] - 1
        return pre

    def InferenceTime(self, R):
        Op = np.zeros(self.L)
        PreOp = np.zeros(self.L)
        T_step = np.zeros(self.L)
        Op[0] = np.ceil(self.Wo[0] * self.Ho[0] / R[0])
        Wp = (self.Wo - self.Kp + 2 * self.Pp) / self.Sp + 1
        for i in range(1, self.L):
            NormalOp = np.ceil(self.Wo[i] * self.Ho[i] / R[i])
            tmp = -1
            for k in range(i):
                tmp = np.maximum(tmp, self.PreOpInterval(i, k, Wp) + PreOp[k])
            PreOp[i] = tmp
            Tail = np.ceil(self.Wo[i] * (self.Pc[i] // self.Sc[i]) / R[i])
            Op[i] = np.maximum(NormalOp + PreOp[i], Op[i - 1] + Tail)
            TA = np.sum(self.DataSize[i] / self.BW[self.hierarchy[i]])
            T_step[i] = np.maximum(TA, self.TC[i])
        max_T_step = np.max(T_step)
        return Op[-1] * max_T_step

    def cal(self, R):
        return self.InferenceTime(R)       

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        self._count += 1
        action = self.action_mapping[action]
        self.state = self.state + action.cpu().numpy()
        reward = -self.InferenceTime(self.state)
        truncated = torch.as_tensor(self._count > self.max_episode_steps)
        terminated = torch.as_tensor(self._count > self.max_episode_steps)
        cost = torch.as_tensor(self.cal_cost())
        return self.state, reward, cost, terminated, truncated, {'final_observation': self.state}

    @property
    def max_episode_steps(self) -> int:
        """The max steps per episode."""
        return 1000

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict]:
        self.set_seed(seed)
        obs = torch.as_tensor(self._observation_space.sample())
        # obs = torch.as_tensor(np.zeros(self._observation_space.shape))
        self.state = torch.zeros(self._observation_space.shape, dtype=torch.int32)
        self._count = 0
        return obs, {}
        # return self.state, {}

    def set_seed(self, seed: int) -> None:
        random.seed(seed)

    def close(self) -> None:
        pass

    def render(self) -> Any:
        pass
