import torch
import math
import pickle
import torch.nn as nn
import torch.nn.functional as F

import rl.utils as utils

# Original DrM source: https://github.com/XuGW-Kevin/DrM

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def set_pad(self, pad):
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
    def __init__(self, repr_dim, state_dim, action_dim, feature_dim, state_feature_dim, state_arch, hidden_arch):
        super().__init__()
        
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.LayerNorm(feature_dim), 
            nn.Tanh()
        )

        self.state_obs = bool(state_dim)
        
        if self.state_obs:
            _state_layers = []
            in_features = state_dim
            if len(state_arch) > 0:
                for out_features in state_arch:
                    _state_layers.extend([nn.Linear(in_features, out_features), nn.ReLU(inplace=True)])
                    in_features = out_features
            else:
                out_features = in_features

            if state_feature_dim > 0:
                self.state_mlp = nn.Sequential(
                    *_state_layers,
                    nn.Linear(out_features, state_feature_dim),
                    nn.LayerNorm(state_feature_dim),
                    nn.Tanh()
                )
            else:
                self.state_mlp = nn.Identity()
                state_feature_dim = state_dim # update state feature dim

        _policy_layers = []
        in_features = feature_dim + state_feature_dim*self.state_obs
        for out_features in hidden_arch:
            _policy_layers.extend([nn.Linear(in_features, out_features), nn.ReLU(inplace=True)])
            in_features = out_features
        _policy_layers.append(nn.Linear(out_features, action_dim))
        self.policy = nn.Sequential(*_policy_layers)

        self.apply(utils.weight_init)

    def forward(self, obs, std=0):
        h = self.trunk(obs['pixels'])
        if self.state_obs:
            h = torch.cat([h, self.state_mlp(obs['state'])], dim=-1)
        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std
        dist = utils.TruncatedNormal(mu, std)
        return dist

class Critic(nn.Module):
    def __init__(self, repr_dim, state_dim, action_dim, feature_dim, state_feature_dim, state_arch, hidden_arch):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.LayerNorm(feature_dim), 
            nn.Tanh()
        )

        self.state_obs = bool(state_dim)

        if self.state_obs:
            _state_layers = []
            in_features = state_dim
            if len(state_arch) > 0:
                for out_features in state_arch:
                    _state_layers.extend([nn.Linear(in_features, out_features), nn.ReLU(inplace=True)])
                    in_features = out_features
            else:
                out_features = in_features

            if state_feature_dim > 0:
                self.state_mlp = nn.Sequential(
                    *_state_layers,
                    nn.Linear(out_features, state_feature_dim),
                    nn.LayerNorm(state_feature_dim),
                    nn.Tanh()
                )
            else:
                self.state_mlp = nn.Identity()
                state_feature_dim = state_dim # update state feature dim

        _q1_layers = []
        _q2_layers = []
        in_features = feature_dim + state_feature_dim*self.state_obs + action_dim
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
        if self.state_obs:
            h = torch.cat([h, self.state_mlp(obs['state'])], dim=-1)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)
        return q1, q2

class VNetwork(nn.Module):
    def __init__(self, repr_dim, state_dim, feature_dim, state_feature_dim, state_arch, hidden_arch):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.LayerNorm(feature_dim), 
            nn.Tanh()
        )

        self.state_obs = bool(state_dim)

        if self.state_obs:
            _state_layers = []
            in_features = state_dim
            if len(state_arch) > 0:
                for out_features in state_arch:
                    _state_layers.extend([nn.Linear(in_features, out_features), nn.ReLU(inplace=True)])
                    in_features = out_features
            else:
                out_features = in_features

            if state_feature_dim > 0:
                self.state_mlp = nn.Sequential(
                    *_state_layers,
                    nn.Linear(out_features, state_feature_dim),
                    nn.LayerNorm(state_feature_dim),
                    nn.Tanh()
                )
            else:
                self.state_mlp = nn.Identity()
                state_feature_dim = state_dim # update state feature dim

        _v_layers = []
        in_features = feature_dim + state_feature_dim*self.state_obs
        for out_features in hidden_arch:
            _v_layers.extend([nn.Linear(in_features, out_features), nn.ReLU(inplace=True)])
            in_features = out_features
        _v_layers.append(nn.Linear(out_features, 1))
        self.V = nn.Sequential(*_v_layers)

        self.apply(utils.weight_init)

    def forward(self, obs):
        h = self.trunk(obs['pixels'])
        if self.state_obs:
            h = torch.cat([h, self.state_mlp(obs['state'])], dim=-1)
        v = self.V(h)
        return v

class DrMAgent:
    """
    DrM algorithm for pixel-based observations.
    Experimental feature of multi-modal observations (image + state).
    """
    def __init__(self, encoder_kwargs, obs_shape, state_dim, action_dim, device, lr, 
                 feature_dim, state_feature_dim, state_arch, hidden_arch, critic_target_tau, 
                 dormant_threshold, target_dormant_ratio, dormant_temp, target_lambda,
                 lambda_temp, dormant_perturb_interval, min_perturb_factor,
                 max_perturb_factor, perturb_rate, num_expl_steps, stddev_type,
                 stddev_schedule, stddev_clip, expectile, aug_pad=4, use_tb=True):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_type = stddev_type
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.dormant_threshold = dormant_threshold
        self.target_dormant_ratio = target_dormant_ratio
        self.dormant_temp = dormant_temp
        self.target_lambda = target_lambda
        self.lambda_temp = lambda_temp
        self.dormant_ratio = 1
        self.dormant_perturb_interval = dormant_perturb_interval
        self.min_perturb_factor = min_perturb_factor
        self.max_perturb_factor = max_perturb_factor
        self.perturb_rate = perturb_rate
        self.expectile = expectile
        self.aug_pad = aug_pad
        self.awaken_step = None

        # Creating init args for automated restoration of checkpoints
        self._init_args = {
            "encoder_kwargs": encoder_kwargs,
            "obs_shape": obs_shape,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "device": device,
            "lr": lr,
            "feature_dim": feature_dim,
            "state_feature_dim": state_feature_dim,
            "state_arch": state_arch,
            "hidden_arch": hidden_arch,
            "critic_target_tau": critic_target_tau,
            "dormant_threshold": dormant_threshold,
            "target_dormant_ratio": target_dormant_ratio,
            "dormant_temp": dormant_temp,
            "target_lambda": target_lambda,
            "lambda_temp": lambda_temp,
            "dormant_perturb_interval": dormant_perturb_interval,
            "min_perturb_factor": min_perturb_factor,
            "max_perturb_factor": max_perturb_factor,
            "perturb_rate": perturb_rate,
            "num_expl_steps": num_expl_steps,
            "stddev_type": stddev_type,
            "stddev_schedule": stddev_schedule,
            "stddev_clip": stddev_clip,
            "expectile": expectile,
            "aug_pad": aug_pad,
            "use_tb": use_tb
        }

        # models
        self.encoder = NatureEncoder(obs_shape, **encoder_kwargs).to(device)
        self.actor = Actor(self.encoder.repr_dim, state_dim, action_dim, feature_dim, state_feature_dim, state_arch, hidden_arch).to(device)
        self.value_predictor = VNetwork(self.encoder.repr_dim, state_dim, feature_dim, state_feature_dim, state_arch, hidden_arch).to(device)
        self.critic = Critic(self.encoder.repr_dim, state_dim, action_dim, feature_dim, state_feature_dim, state_arch, hidden_arch).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, state_dim, action_dim, feature_dim, state_feature_dim, state_arch, hidden_arch).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.predictor_opt = torch.optim.Adam(self.value_predictor.parameters(), lr=lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=self.aug_pad)

        self.train()
        self.critic_target.train()

    @property
    def dormant_stddev(self):
        return 1 / (1 + math.exp(-self.dormant_temp * (self.dormant_ratio - self.target_dormant_ratio)))

    def stddev(self, step):
        if self.stddev_type == "max":
            return max(utils.schedule(self.stddev_schedule, step), self.stddev)
        elif self.stddev_type == "dormant":
            return self.dormant_stddev
        elif self.stddev_type == "awake":
            if self.awaken_step == None:
                return self.dormant_stddev
            else:
                return max(self.dormant_stddev, utils.schedule(self.stddev_schedule, step - self.awaken_step))
        else:
            raise NotImplementedError(self.stddev_type)

    def perturb_factor(self, step):
        return min(max(self.min_perturb_factor, 1 - self.perturb_rate * self.dormant_ratio), self.max_perturb_factor)

    @property
    def lambda_(self):
        return self.target_lambda / (1 + math.exp(self.lambda_temp * (self.dormant_ratio - self.target_dormant_ratio)))

    def set_num_expl_steps(self, num_expl_steps):
        self.num_expl_steps = num_expl_steps

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
        self.value_predictor.train(training)

    def act(self, obs, step, eval_mode):
        img_obs = self.encoder(torch.as_tensor(obs['pixels'], device=self.device).unsqueeze(0))
        obs = {'pixels': img_obs, 'state': torch.as_tensor(obs['state'], device=self.device).unsqueeze(0)}
        dist = self.actor(obs, self.stddev(step))
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_predictor(self, obs, action):
        metrics = dict()
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)
        V = self.value_predictor(obs)
        vf_err = V - Q
        vf_sign = (vf_err > 0).float()
        vf_weight = (1 - vf_sign) * self.expectile + vf_sign * (1 - self.expectile)
        predictor_loss = (vf_weight * (vf_err**2)).mean()

        if self.use_tb:
            metrics['predictor_loss'] = predictor_loss.item()

        self.predictor_opt.zero_grad(set_to_none=True)
        predictor_loss.backward()
        self.predictor_opt.step()

        return metrics

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            dist = self.actor(next_obs, self.stddev(step))
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V_explore = torch.min(target_Q1, target_Q2)
            target_V_exploit = self.value_predictor(next_obs)
            target_V = self.lambda_ * target_V_exploit + (1 - self.lambda_) * target_V_explore
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()
        noise = self.stddev(step)
        dist = self.actor(obs, noise)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
            metrics['actor_noise'] = noise
            metrics['actor_exploitation'] = self.lambda_

        return metrics

    def perturb(self, step):
        utils.perturb(self.actor, self.actor_opt, self.perturb_factor(step))
        utils.perturb(self.critic, self.critic_opt, self.perturb_factor(step))
        utils.perturb(self.critic_target, self.critic_opt,
                      self.perturb_factor(step))
        utils.perturb(self.encoder, self.encoder_opt,
                      self.perturb_factor(step))
        utils.perturb(self.value_predictor, self.predictor_opt,
                      self.perturb_factor(step))

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.dormant_perturb_interval == 0:
            self.perturb(step)

        batch = next(replay_iter)
        obs_dict, action, reward, discount, next_obs_dict = batch

        # Conversion to tensors
        action = torch.as_tensor(action, device=self.device)
        reward = torch.as_tensor(reward, device=self.device)
        discount = torch.as_tensor(discount, device=self.device)

        # obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)

        # augment
        pixel_obs = self.aug(torch.as_tensor(obs_dict['pixels'], device=self.device).float())
        next_pixel_obs = self.aug(torch.as_tensor(next_obs_dict['pixels'], device=self.device).float())

        pixel_obs = self.encoder(pixel_obs)
        with torch.no_grad():
            next_pixel_obs = self.encoder(next_pixel_obs)

        obs = {'pixels': pixel_obs, 'state': torch.as_tensor(obs_dict['state'], device=self.device)}
        next_obs = {'pixels': next_pixel_obs, 'state': torch.as_tensor(next_obs_dict['state'], device=self.device)}

        # calculate dormant ratio
        self.dormant_ratio = utils.cal_dormant_ratio(
            self.actor, *[{k: v.detach() for k, v in obs.items()}, 0], percentage=self.dormant_threshold)

        if self.awaken_step is None and step > self.num_expl_steps and self.dormant_ratio < self.target_dormant_ratio:
            self.awaken_step = step

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()
            metrics['actor_dormant_ratio'] = self.dormant_ratio

        # update predictor
        metrics.update(self.update_predictor({k: v.detach() for k, v in obs.items()}, action))

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
                'value_predictor': self.value_predictor.state_dict(),
                'encoder_opt': self.encoder_opt.state_dict(),
                'actor_opt': self.actor_opt.state_dict(),
                'critic_opt': self.critic_opt.state_dict(),
                'predictor_opt': self.predictor_opt.state_dict()
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
        agent.value_predictor.load_state_dict(ckpt['value_predictor'])

        agent.encoder_opt.load_state_dict(ckpt['encoder_opt'])
        agent.actor_opt.load_state_dict(ckpt['actor_opt'])
        agent.critic_opt.load_state_dict(ckpt['critic_opt'])
        agent.predictor_opt.load_state_dict(ckpt['predictor_opt'])

        for k, v in meta_dict.items():
            if k != 'device':
                setattr(agent, k, v)

        return agent, meta_dict

if __name__ == '__main__':
    from torchinfo import summary

    enc = NatureEncoder(obs_shape=(1,84,84), filters=32, border_padding=True)
    summary(enc, input_size=(1,1,84,84), col_names=['input_size', 'output_size', 'num_params'])
