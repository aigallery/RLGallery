import random
from tqdm import tqdm
import torch
import numpy as np
from .registry import RUNNERS, ENVS, MODELS, UTILS, LOGGRS, ALGORITHMS


@RUNNERS.register_module()
class Runner:
    '''
    train and visualize
    '''

    def __init__(self, algo="VanillaDQN", num_episodes=500, batch_size=64, minimal_size=100, device="cpu", seed=0) -> None:

        self.algo = algo
        self.num_episodes = num_episodes
        self.minimal_size = minimal_size
        self.batch_size = batch_size
        self.device = device
        self.seed = seed

    def create_true_instance(self):
        self.device = torch.device(self.device)
        # logger
        self.logger = LOGGRS.build({'type': 'Logger'})
        self.save_dir = self.logger.save_dir
        # env
        self.em = ENVS.build({'type':'Envs'})
        self.env = self.em.gym_env
        # seed
        self.env.seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # replay buffer
        self.replay_buffer = UTILS.build({"type": "ReplayBuffer"})
        if self.algo == 'VPG' or self.algo == 'REINFORCE':
            self.agent = ALGORITHMS.build({'type': 'VPG', 'state_dim': self.env.state_dim, 'action_dim': self.env.action_dim, 'device': self.device})
        elif self.algo in ['VanillaDQN', 'DoubleDQN', 'DuelingDQN']:
            self.agent = ALGORITHMS.build(
                {'type': 'DQN', 'state_dim': self.env.state_dim, 'action_dim': self.env.action_dim, 'algo': self.algo, 'device': self.device})
        elif self.algo == "ActorCritic":
            self.agent = ALGORITHMS.build({"type":"ActorCritic", 'state_dim': self.env.state_dim, 'action_dim': self.env.action_dim, 'device': self.device})
        elif self.algo == "TRPO":
            self.agent = ALGORITHMS.build({"type":"TRPO", 'state_dim': self.env.state_dim, 'action_dim': self.env.action_dim, 'device': self.device})
        elif self.algo == "PPO":
            self.agent = ALGORITHMS.build({"type":"PPO", 'state_dim': self.env.state_dim, 'action_dim': self.env.action_dim, 'device': self.device})
        elif self.algo == "PPOContinuous":
            self.agent = ALGORITHMS.build({"type":"PPOContinuous", 'state_dim': self.env.state_dim, 'action_dim': self.env.action_dim, 'device': self.device})
        elif self.algo == "DDPG":
            self.agent = ALGORITHMS.build({"type":"DDPG", 'state_dim': self.env.state_dim, 'action_dim': self.env.action_dim, 'action_bound': self.env.action_bound, 'device': self.device})
        elif self.algo == "SACContinuous":
            self.agent = ALGORITHMS.build({"type":"SACContinuous", 'state_dim': self.env.state_dim, 'action_dim': self.env.action_dim, 'action_bound': self.env.action_bound, 'target_entropy': self.env.target_entropy, 'device': self.device})

        else:
            print(f"not support {self.algo} currently!")
            exit()
        
        return self

    def train_off_policy_agent(self):
        for _ in tqdm(range(self.num_episodes), desc='Processing'):
            episode_return = 0
            state = self.env.reset()
            done = False
            while not done:
                action = self.agent.take_action(state)
                # if self.env.is_continus:
                #     action_continuous = self.em.dis_to_con(
                #         action, self.env, self.agent.action_dim)
                #     next_state, reward, done, _ = self.env.step(
                #         [action_continuous])
                # else:
                next_state, reward, done, _ = self.env.step(action)

                self.replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward
                # 当buffer 数据的数量超过一定值后， 才进行Q网络训练, off-policy
                if self.replay_buffer.size() > self.minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = self.replay_buffer.sample(
                        self.batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    self.agent.update(transition_dict)

            info_dict = {
                f'{self.em.env_name}::{self.algo}::Return_of_Episode': episode_return}
            self.logger.log_info(info_dict)

    def train_on_policy_agent(self):
        for _ in tqdm(range(self.num_episodes), desc='Processing'):
            episode_return = 0
            transition_dict = {
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': []
            }
            state = self.env.reset()
            done = False
            while not done:
                action = self.agent.take_action(state)
                # if self.env.is_continus:
                #     action_continuous = self.em.dis_to_con(
                #         action, self.env, self.agent.action_dim)
                #     next_state, reward, done, _ = self.env.step(
                #         [action_continuous])
                # else:
                next_state, reward, done, _ = self.env.step(action)
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                state = next_state
                episode_return += reward
            self.agent.update(transition_dict)
            info_dict = {
                f'{self.em.env_name}::{self.algo}::Return_of_Episode': episode_return}
            self.logger.log_info(info_dict)



if __name__ == "__main__":
    run = RUNNERS.build({'type': "Runner"})
    if run is None:
        print("Failed to create the runner object.")
    if run.algo in ["VPG", "REINFORCE"]:
        run.train_on_policy_agent()
    else:
        run.train_off_policy_agent()