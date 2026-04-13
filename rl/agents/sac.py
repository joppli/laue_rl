import torch
import math
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch import distributions as pyd

import rl.utils as utils

# Original SAC source: https://github.com/denisyarats/pytorch_sac

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)

class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))

class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

class NatureEncoder(nn.Module):
    def __init__(self, obs_shape, filters=32, border_padding=True):
        super().__init__()
        assert len(obs_shape) == 3 and obs_shape[1:] == (84, 84)
        in_ch = obs_shape[0]
        padding = 1 if border_padding else 0
        self.repr_dim = filters * 21 * 21 if border_padding else filters * 16 * 16

        # Modified NatureCNN (one more downsampling stage in the second conv layer)
        self.convnet = nn.Sequential(
            nn.Conv2d(in_ch, filters, 3, stride=2, padding=padding), nn.ReLU(),
            nn.Conv2d(filters, filters, 3, stride=2, padding=padding), nn.ReLU(),
            nn.Conv2d(filters, filters, 3, stride=1, padding=padding), nn.ReLU(),
            nn.Conv2d(filters, filters, 3, stride=1, padding=padding), nn.ReLU()
        )

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h
    
class Actor(nn.Module):
    """
    DiagGaussianActor: torch.distributions implementation of an diagonal Gaussian policy.
    Using same networks as DrM.
    """
    def __init__(self, obs_dim, action_dim, feature_dim, hidden_arch, log_std_bounds):
        super().__init__()
        self.log_std_bounds = log_std_bounds
        # self.trunk = mlp(obs_dim, hidden_dim, 2 * action_shape[0], hidden_depth)
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, feature_dim),
            nn.LayerNorm(feature_dim), 
            nn.Tanh()
        )

        _policy_layers = []
        in_features = feature_dim
        for out_features in hidden_arch:
            _policy_layers.extend([nn.Linear(in_features, out_features), nn.ReLU(inplace=True)])
            in_features = out_features
        _policy_layers.append(nn.Linear(out_features, 2*action_dim))
        self.policy = nn.Sequential(*_policy_layers)

        self.apply(utils.weight_init)

    def forward(self, obs):
        h = self.trunk(obs['pixels'])
        mu, log_std = self.policy(h).chunk(2, dim=-1)
        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()
        dist = SquashedNormal(mu, std)
        return dist


class Critic(nn.Module):
    """
    DoubleQCritic: critic network, employes double Q-learning.
    """
    def __init__(self, repr_dim, action_dim, feature_dim, hidden_arch):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.LayerNorm(feature_dim), 
            nn.Tanh()
        )

        _q1_layers = []
        _q2_layers = []
        in_features = feature_dim + action_dim
        for out_features in hidden_arch:
            _q1_layers.extend([nn.Linear(in_features, out_features), nn.ReLU(inplace=True)])
            _q2_layers.extend([nn.Linear(in_features, out_features), nn.ReLU(inplace=True)])
            in_features = out_features
        _q1_layers.append(nn.Linear(out_features, 1))
        _q2_layers.append(nn.Linear(out_features, 1))
        self.Q1 = nn.Sequential(*_q1_layers)
        self.Q2 = nn.Sequential(*_q2_layers)   

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs['pixels'])
        obs_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)
        return q1, q2

class SACPixelAgent:
    """
    SAC algorithm for pixel-based observations.
    """
    def __init__(self, encoder_kwargs, obs_shape, action_dim, action_range, device,
                 lr, feature_dim, hidden_arch, critic_target_tau, 
                 init_temperature, learnable_temperature, log_std_bounds, 
                 num_expl_steps, dormant_threshold, aug_pad=0, use_tb=True):

        self.action_range = action_range
        self.device = torch.device(device)
        self.critic_target_tau = critic_target_tau
        self.learnable_temperature = learnable_temperature
        self.num_expl_steps = num_expl_steps
        self.dormant_threshold = dormant_threshold
        self.aug_pad = aug_pad
        self.use_tb = use_tb

        # Creating init args for automated restoration of checkpoints
        self._init_args = {
            "encoder_kwargs": encoder_kwargs,
            "obs_shape": obs_shape,
            "action_dim": action_dim,
            "action_range": action_range,
            "device": device,
            "lr": lr,
            "feature_dim": feature_dim,
            "hidden_arch": hidden_arch,
            "critic_target_tau": critic_target_tau,
            "init_temperature": init_temperature,
            "learnable_temperature": learnable_temperature,
            "log_std_bounds": log_std_bounds,
            "num_expl_steps": num_expl_steps,
            "dormant_threshold": dormant_threshold,
            "aug_pad": aug_pad,
            "use_tb": use_tb
        }

        # models
        self.encoder = NatureEncoder(obs_shape, **encoder_kwargs).to(device)
        self.actor = Actor(self.encoder.repr_dim, action_dim, feature_dim, hidden_arch, log_std_bounds).to(device)
        self.critic = Critic(self.encoder.repr_dim, action_dim, feature_dim, hidden_arch).to(self.device)
        self.critic_target = Critic(self.encoder.repr_dim, action_dim, feature_dim, hidden_arch).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -action_dim # set target entropy to -|A|

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.log_alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=self.aug_pad)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def set_num_expl_steps(self, num_expl_steps):
        self.num_expl_steps = num_expl_steps

    def act(self, obs, step, eval_mode):
        img_obs = self.encoder(torch.as_tensor(obs['pixels'], device=self.device).unsqueeze(0))
        obs = {'pixels': img_obs, 'state': torch.as_tensor(obs['state'], device=self.device).unsqueeze(0)}
        dist = self.actor(obs)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample()
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        action = action.clamp(*self.action_range)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            dist = self.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + (discount * target_V)

        # get current Q estimates
        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # Optimize the critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    # def update_actor_and_alpha(self, obs, logger, step):
    def update_actor(self, obs, step):
        metrics = dict()
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)
        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_entropy'] = -log_prob.mean().item()

        # optimize the actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.learnable_temperature:
            alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
            if self.use_tb:
                metrics['alpha_loss'] = alpha_loss.item()
                metrics['alpha_value'] = self.alpha.item()
            self.log_alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.log_alpha_opt.step()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        batch = next(replay_iter)
        obs_dict, action, reward, discount, next_obs_dict = batch
        
        # Conversion to tensors
        action = torch.as_tensor(action, device=self.device)
        reward = torch.as_tensor(reward, device=self.device)
        discount = torch.as_tensor(discount, device=self.device)

        # augment (not done in original SAC pixel algorithm)
        pixel_obs = self.aug(torch.as_tensor(obs_dict['pixels'], device=self.device).float())
        next_pixel_obs = self.aug(torch.as_tensor(next_obs_dict['pixels'], device=self.device).float())

        # encode
        pixel_obs = self.encoder(pixel_obs)
        with torch.no_grad():
            next_pixel_obs = self.encoder(next_pixel_obs)

        obs = {'pixels': pixel_obs, 'state': torch.as_tensor(obs_dict['state'], device=self.device)}
        next_obs = {'pixels': next_pixel_obs, 'state': torch.as_tensor(next_obs_dict['state'], device=self.device)}

        # calculate dormant ratio
        self.dormant_ratio = utils.cal_dormant_ratio(self.actor, *[{k: v.detach() for k, v in obs.items()}], percentage=self.dormant_threshold)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()
            metrics['actor_dormant_ratio'] = self.dormant_ratio

        # update critic
        metrics.update(self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor({k: v.detach() for k, v in obs.items()}, step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

        return metrics
    
    def save(self, savename, **meta_kwargs):
        """
        Saves class module state dictionaries and meta data to separate files. 
        """
        with open(f"{savename}_modules.pt", 'wb') as fid:
            modules_state_dict = {
                'encoder': self.encoder.state_dict(),
                'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
                'critic_target': self.critic_target.state_dict(),
                'encoder_opt': self.encoder_opt.state_dict(),
                'actor_opt': self.actor_opt.state_dict(),
                'critic_opt': self.critic_opt.state_dict(),
            }
            torch.save(modules_state_dict, fid)

        with open(f"{savename}_meta.pkl", 'wb') as fid:
            meta_dict = {
                k: v for k, v in self.__dict__.items() if not isinstance(v, torch.nn.Module) and not isinstance(v, torch.optim.Optimizer)
            }
            meta_dict['_init_args'] = self._init_args

            for k, v in meta_kwargs.items():
                meta_dict[k] = v

            pickle.dump(meta_dict, fid)

    @classmethod
    def load(cls, savename, map_location='cpu'):
        """
        Loads class module state dictionaries and meta data from separate files.
        """
        with open(f"{savename}_meta.pkl", 'rb') as fid:
            meta_dict = pickle.load(fid)

        init_args = meta_dict.pop("_init_args")
        init_args['device'] = map_location

        agent = cls(**init_args)

        ckpt = torch.load(f"{savename}_modules.pt", map_location=agent.device)
        agent.encoder.load_state_dict(ckpt['encoder'])
        agent.actor.load_state_dict(ckpt['actor'])
        agent.critic.load_state_dict(ckpt['critic'])
        agent.critic_target.load_state_dict(ckpt['critic_target'])

        agent.encoder_opt.load_state_dict(ckpt['encoder_opt'])
        agent.actor_opt.load_state_dict(ckpt['actor_opt'])
        agent.critic_opt.load_state_dict(ckpt['critic_opt'])

        for k, v in meta_dict.items():
            if k != 'device':
                setattr(agent, k, v)

        return agent, meta_dict

if __name__ == '__main__':
    from torchinfo import summary

    enc = NatureEncoder(obs_shape=(1,84,84), filters=32, border_padding=True)
    summary(enc, input_size=(1,1,84,84), col_names=['input_size', 'output_size', 'num_params'])
