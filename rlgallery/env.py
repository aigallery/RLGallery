import gym
import yaml
from .registry import ENVS


@ENVS.register_module()
class Envs:
    def __init__(self, env_name="CartPole-v0", action_dim=1) -> None:
        self.env_name = env_name
        self.action_dim = action_dim
    
    def create_true_instance(self):
        self._gym = gym.make(self.env_name)
        return self

    @property
    def gym_env(self):
        self._gym.is_continus = isinstance(
            self._gym.action_space, gym.spaces.Box)

        self._gym.state_dim = self._gym.observation_space.shape[0]
        self._gym.action_dim = self.action_dim if self._gym.is_continus else self._gym.action_space.n
        self._gym.action_bound = self._gym.action_space.high[0]
        self._gym.target_entropy = -self._gym.action_space.shape[0]
        return self._gym

    def dis_to_con(self, discrete_action, env, action_dim):  # 离散动作转回连续的函数
        action_lowbound = env.action_space.low[0]  # 连续动作的最小值
        action_upbound = env.action_space.high[0]  # 连续动作的最大值
        return action_lowbound + (discrete_action / (action_dim - 1)) * (action_upbound - action_lowbound)
