import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
	def __init__(self, task_repr_vec, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.task_repr_vec = task_repr_vec
		self.task_lin_layer = nn.Linear(self.task_repr_vec.shape[-1], 256)
		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 256)
		self.l4 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		task_repr = F.leaky_relu(self.task_lin_layer(self.task_repr_vec))
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		a = task_repr * a
		a = F.relu(self.l3(a))
		return self.max_action * torch.tanh(self.l4(a))


class Critic(nn.Module):
	def __init__(self, task_repr_vec, state_dim, action_dim):
		super(Critic, self).__init__()

		self.task_repr_vec = task_repr_vec
		self.task_lin_layer = nn.Linear(self.task_repr_vec.shape[-1], 256)

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 256)
		self.l4 = nn.Linear(256, 1)

		# Q2 architecture
		self.l5 = nn.Linear(state_dim + action_dim, 256)
		self.l6 = nn.Linear(256, 256)
		self.l7 = nn.Linear(256, 256)
		self.l8 = nn.Linear(256, 1)


	def forward(self, state, action):
		task_repr = F.leaky_relu(self.task_lin_layer(self.task_repr_vec))

		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = task_repr * q1
		q1 = F.relu(self.l3(q1))
		q1 = self.l4(q1)

		q2 = F.relu(self.l5(sa))
		q2 = F.relu(self.l6(q2))
		q2 = task_repr * q2
		q2 = F.relu(self.l7(q2))
		q2 = self.l8(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)
		task_repr = F.leaky_relu(self.task_lin_layer(self.task_repr_vec))

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = task_repr * q1
		q1 = F.relu(self.l3(q1))
		q1 = self.l4(q1)
		return q1


class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		task_repr_vec,
		load_filename,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
	):

		task_repr_vec = torch.FloatTensor(task_repr_vec).to(device)
		self.old_actor = Actor(task_repr_vec, state_dim, action_dim, max_action).to(device)
		self.old_actor.load_state_dict(torch.load(load_filename + "_actor"))
		for p in self.old_actor.parameters():
			p.requires_grad = False
		self.actor = Actor(task_repr_vec, state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.old_critic = Critic(task_repr_vec, state_dim, action_dim).to(device)
		self.old_critic.load_state_dict(torch.load(load_filename + "_critic"))
		for p in self.old_critic.parameters():
			p.requires_grad = False
		self.critic = Critic(task_repr_vec, state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1

		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		weight_diff = 0.
		for p, p_old in zip(self.critic.parameters(), self.old_critic.parameters()):
			weight_diff += (p - p_old).norm()

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) + weight_diff/100

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:
			weight_diff = 0.
			for p, p_old in zip(self.actor.parameters(), self.old_actor.parameters()):
				weight_diff += (p - p_old).norm()
			# Compute actor losse
			actor_loss = -self.critic.Q1(state, self.actor(state)).mean() + weight_diff
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)
		