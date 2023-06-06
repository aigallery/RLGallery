import torch
import torch.nn.functional as F
from torch.distributions import Normal
from .registry import MODELS


@MODELS.register_module()
class Qnet(torch.nn.Module):
    '''
    只有一层隐藏层的Q网络
    '''

    def __init__(self, state_dim=10, hidden_dim=128, action_dim=11):
        super(Qnet, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

    def create_true_instance(self):
        self.fc1 = torch.nn.Linear(self.state_dim, self.hidden_dim)
        self.fc2 = torch.nn.Linear(self.hidden_dim, self.action_dim)
        return self
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


@MODELS.register_module()
class VAnet(torch.nn.Module):
    ''' 只有一层隐藏层的A网络和V网络 '''

    def __init__(self, state_dim=10, hidden_dim=128, action_dim=11):
        super(VAnet, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

    def create_true_instance(self):
        self.fc1 = torch.nn.Linear(self.state_dim, self.hidden_dim)  # 共享网络部分
        self.fc_A = torch.nn.Linear(self.hidden_dim, self.action_dim)
        self.fc_V = torch.nn.Linear(self.hidden_dim, 1)
        return self
    
    def forward(self, x):
        A = self.fc_A(F.relu(self.fc1(x)))
        V = self.fc_V(F.relu(self.fc1(x)))
        Q = V + A - A.mean(1).view(-1, 1)  # Q值由V值和A值计算得到
        return Q


@MODELS.register_module()
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim=10, hidden_dim=128, action_dim=11):
        super(PolicyNet, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

    def create_true_instance(self):
        self.fc1 = torch.nn.Linear(self.state_dim, self.hidden_dim)
        self.fc2 = torch.nn.Linear(self.hidden_dim, self.action_dim)
        return self
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

@MODELS.register_module()
class ValueNet(torch.nn.Module):
    """input is a state, output is the coresponding value.
    """
    def __init__(self, state_dim=10, hidden_dim=128):
        super(ValueNet, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

    def create_true_instance(self):
        self.fc1 = torch.nn.Linear(self.state_dim, self.hidden_dim)
        self.fc2 = torch.nn.Linear(self.hidden_dim, 1)
        return self

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

@MODELS.register_module()
class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim=10, hidden_dim=128, action_dim=11):
        super(PolicyNetContinuous, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

    def create_true_instance(self):
        self.fc1 = torch.nn.Linear(self.state_dim, self.hidden_dim)
        self.fc_mu = torch.nn.Linear(self.hidden_dim, self.action_dim)
        self.fc_std = torch.nn.Linear(self.hidden_dim, self.action_dim)
        return self

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = 2.0 * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std
    
@MODELS.register_module()
class PolicyNetDDPG(torch.nn.Module):
    def __init__(self, state_dim=10, hidden_dim=128, action_dim=11, action_bound=1):
        super(PolicyNetDDPG, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.action_bound = action_bound  # action_bound是环境可以接受的动作最大值
    
    def create_true_instance(self):
        self.fc1 = torch.nn.Linear(self.state_dim, self.hidden_dim)
        self.fc2 = torch.nn.Linear(self.hidden_dim, self.action_dim)
        return self
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)) * self.action_bound

@MODELS.register_module()
class QValueNet(torch.nn.Module):
    def __init__(self, state_dim=10, hidden_dim=128, action_dim=11):
        super(QValueNet, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
    
    def create_true_instance(self):
        self.fc1 = torch.nn.Linear(self.state_dim + self.action_dim, self.hidden_dim)
        self.fc2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_out = torch.nn.Linear(self.hidden_dim, 1)
        return self

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1) # 拼接状态和动作
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)
    
@MODELS.register_module()
class PolicyNetContinuousSAC(torch.nn.Module):
    def __init__(self, state_dim=10, hidden_dim=128, action_dim=11, action_bound=1):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
    
    def create_true_instance(self):
        super(PolicyNetContinuousSAC, self).__init__()
        self.fc1 = torch.nn.Linear(self.state_dim, self.hidden_dim)
        self.fc_mu = torch.nn.Linear(self.hidden_dim, self.action_dim)
        self.fc_std = torch.nn.Linear(self.hidden_dim, self.action_dim)
        return self

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        normal_sample = dist.rsample()  # rsample()是重参数化采样
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        # 计算tanh_normal分布的对数概率密度
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        action = action * self.action_bound
        return action, log_prob

@MODELS.register_module()
class QValueNetContinuousSAC(torch.nn.Module):
    def __init__(self, state_dim=10, hidden_dim=128, action_dim=11):
        super(QValueNetContinuousSAC, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

    def create_true_instance(self):
        self.fc1 = torch.nn.Linear(self.state_dim + self.action_dim, self.hidden_dim)
        self.fc2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_out = torch.nn.Linear(self.hidden_dim, 1)
    
        return self

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)