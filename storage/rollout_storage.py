# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import h5py
import torch
from tensordict import TensorDict

from rsl_rl.utils import split_and_pad_trajectories


class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.actions = None
            self.privileged_actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        obs,
        actions_shape,
        device="cpu",
        imitation_dataset_path: str | None = None,
    ):
        # store inputs
        self.training_type = training_type
        self.device = device
        self.imitation_dataset_path = imitation_dataset_path
        self.obs_structure = obs  # Store the observation structure for later use
        print("original transitions per env", num_transitions_per_env)
        print("num_envs from environment", num_envs)
        
        # Load imitation dataset if provided to determine num_transitions_per_env and shapes
        if imitation_dataset_path is not None:
            print("Loading imitation expert dataset provided, loading from:", imitation_dataset_path)
            dataset_info = self._get_dataset_info(imitation_dataset_path)
            num_demos = dataset_info['num_demos']
            num_transitions_per_env = dataset_info['num_transitions']
            print("num_transitions_per_env:", dataset_info['num_transitions'])
            print("num_envs (num of demos):", dataset_info['num_demos'])
            
            # Update obs to have first dimension as num_demos
            updated_obs = {}

            for key, value in obs.items():
                # Create a new tensor with first dimension = num_demos
                # Original shape: (original_dim1, dim2, ...)
                # New shape: (num_demos, dim2, ...)
                original_shape = value.shape
                new_shape = (num_demos,) + original_shape[1:]
                updated_obs[key] = torch.zeros(*new_shape, device=device)

            obs = updated_obs
            num_envs = num_demos  # Update num_envs to be num_demos
        
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.actions_shape = actions_shape

        # Core
        self.observations = TensorDict(
            {key: torch.zeros(num_transitions_per_env, *value.shape, device=device) for key, value in obs.items()},
            batch_size=[num_transitions_per_env, num_envs],
            device=self.device,
        )

        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # for distillation
        if training_type == "distillation":
            self.privileged_actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        # for reinforcement learning
        if training_type == "rl":
            self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        # For RNN networks
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        # counter for the number of transitions stored
        self.step = 0
        
        # Load imitation dataset if provided
        if imitation_dataset_path is not None:
            self._load_imitation_dataset(imitation_dataset_path)

    def _get_dataset_info(self, dataset_path: str) -> dict:
        """Get dataset information including max size and shapes from all demos."""
        dataset_path = os.path.expanduser(dataset_path)
        with h5py.File(dataset_path, 'r') as f:
            # Recursively find all paths in the HDF5 file
            def get_all_paths(group, prefix=''):
                paths = []
                for key in group.keys():
                    path = f"{prefix}/{key}" if prefix else key
                    if isinstance(group[key], h5py.Group):
                        paths.extend(get_all_paths(group[key], path))
                    else:
                        paths.append(path)
                return paths
            
            all_paths = get_all_paths(f)
            
            # Find all demos and track their sizes
            demo_sizes = {}
            first_demo_id = None
            actions_shape = None
            
            for path in all_paths:
                if '/demo_' in path:
                    parts = path.split('/')
                    demo_id = None
                    for part in parts:
                        if part.startswith('demo_'):
                            demo_id = part
                            break
                    
                    if demo_id is None:
                        continue
                    
                    # Track the first demo we encounter
                    if first_demo_id is None:
                        first_demo_id = demo_id
                    
                    if '/actions' in path:
                        actions_data = f[path]
                        demo_sizes[demo_id] = actions_data.shape[0]
                        # Get action shape from first demo
                        if actions_shape is None:
                            actions_shape = actions_data.shape[1:]
            
            if actions_shape is None:
                raise ValueError(f"No actions found in dataset: {dataset_path}")
            
            # Use the minimum number of transitions across all demos
            min_transitions = min(demo_sizes.values())
            num_demos = len(demo_sizes)
            
            # Calculate total observation size for "policy" key from first demo
            total_obs_size = 0
            for path in all_paths:
                if '/demo_' in path and first_demo_id in path:
                    if '/obs/' in path and 'table_cam' not in path and 'wrist_cam' not in path:
                        obs_data = f[path]
                        obs_shape = obs_data.shape[1:]
                        # Calculate flattened size
                        flattened_size = 1
                        for dim in obs_shape:
                            flattened_size *= dim
                        total_obs_size += flattened_size
            
            # Create the obs_dict with "policy" key having the total observation size
            obs_dict = {'policy': torch.zeros(min_transitions, total_obs_size, device=self.device)}
            
            return {
                'num_transitions': min_transitions,
                'actions_shape': actions_shape,
                'obs_dict': obs_dict,
                'demo_sizes': demo_sizes,
                'num_demos': num_demos
            }

    def _load_imitation_dataset(self, dataset_path: str):
        """Load expert demonstrations from HDF5 file and add them as transitions."""
        dataset_path = os.path.expanduser(dataset_path)
        with h5py.File(dataset_path, 'r') as f:
            # Recursively find all paths in the HDF5 file
            def get_all_paths(group, prefix=''):
                paths = []
                for key in group.keys():
                    path = f"{prefix}/{key}" if prefix else key
                    if isinstance(group[key], h5py.Group):
                        paths.extend(get_all_paths(group[key], path))
                    else:
                        paths.append(path)
                return paths
            
            all_paths = get_all_paths(f)
            
            # Group paths by demo (format: /data/demo_0/actions or /data/demo_0/obs/...)
            demos = {}
            for path in all_paths:
                # Extract demo identifier from path (e.g., "demo_0" from "/data/demo_0/actions")
                if '/demo_' in path:
                    # Find the demo_X part
                    parts = path.split('/')
                    demo_id = None
                    for part in parts:
                        if part.startswith('demo_'):
                            demo_id = part
                            break
                    
                    if demo_id is None:
                        continue
                    
                    if demo_id not in demos:
                        demos[demo_id] = {'actions': None, 'observations': {}}
                    
                    # Check if this is an action or observation path
                    if '/obs/' in path and 'table_cam' not in path and 'wrist_cam' not in path:
                        # Load observations - extract the observation key from the path
                        # e.g., from "/data/demo_0/obs/joint_pos" extract "joint_pos"
                        obs_key = path.split('/obs/')[-1]
                        obs_data = torch.from_numpy(f[path][:])
                        demos[demo_id]['observations'][obs_key] = obs_data
                    elif '/actions' in path:
                        # Load actions
                        actions_data = torch.from_numpy(f[path][:])
                        demos[demo_id]['actions'] = actions_data
            
            # Collect all demos' observations and actions
            # Shape will be: [num_demos, num_transitions_per_env, obs_dim/action_dim]
            all_demos_obs = []
            all_demos_actions = []
            
            for demo_id, demo_data in demos.items():
                if demo_data['actions'] is None or len(demo_data['observations']) == 0:
                    continue
                
                # Get the number of steps from the actions
                num_steps = demo_data['actions'].shape[0]
                
                # Use num_transitions_per_env from dataset info
                num_steps = min(num_steps, self.num_transitions_per_env)
                
                # Flatten all observations into a single tensor
                all_obs = []
                for obs_key, obs_data in demo_data['observations'].items():
                    # Take only first num_transitions_per_env transitions
                    obs_data = obs_data[:num_steps]
                    # Flatten each observation (keep first dimension as time steps)
                    if obs_data.dim() > 2:
                        obs_data = obs_data.flatten(start_dim=1)
                    all_obs.append(obs_data)
                
                # Concatenate all flattened observations along the last dimension
                # Shape: [num_steps, flattened_obs_dim]
                flattened_obs = torch.cat(all_obs, dim=-1).to(self.device)
                
                # Pad if this demo is shorter than num_transitions_per_env
                if num_steps < self.num_transitions_per_env:
                    pad_size = self.num_transitions_per_env - num_steps
                    obs_padding = torch.zeros(pad_size, flattened_obs.shape[1], device=self.device)
                    flattened_obs = torch.cat([flattened_obs, obs_padding], dim=0)
                
                # Get actions and pad if needed
                actions = demo_data['actions'][:num_steps].to(self.device)
                if num_steps < self.num_transitions_per_env:
                    pad_size = self.num_transitions_per_env - num_steps
                    action_padding = torch.zeros(pad_size, *actions.shape[1:], device=self.device)
                    actions = torch.cat([actions, action_padding], dim=0)
                
                all_demos_obs.append(flattened_obs)
                all_demos_actions.append(actions)
            
            # Stack all demos: [num_demos, num_transitions_per_env, dim]
            all_demos_obs = torch.stack(all_demos_obs, dim=0)
            all_demos_actions = torch.stack(all_demos_actions, dim=0)
            
            # Transpose to get: [num_transitions_per_env, num_demos, dim]
            all_demos_obs = all_demos_obs.transpose(0, 1)
            all_demos_actions = all_demos_actions.transpose(0, 1)
            
            print(f"Collected observations shape: {all_demos_obs.shape}")
            print(f"Collected actions shape: {all_demos_actions.shape}")
            
            # Now iterate through num_transitions_per_env and add each transition
            for step in range(self.num_transitions_per_env):
                # Create a TensorDict with observations for all demos at this timestep
                # Shape: [num_demos, flattened_obs_dim]
                obs_tensordict = TensorDict(
                    {"policy": all_demos_obs[step]},
                    batch_size=[len(demos)],
                    device=self.device,
                )
                
                # Create a Transition object for this timestep across all demos
                transition = self.Transition()
                transition.actions = all_demos_actions[step]  # Shape: [num_demos, action_dim]
                transition.observations = obs_tensordict
                
                # Set default values for other required fields
                transition.rewards = torch.zeros(len(demos), 1, device=self.device)
                transition.dones = torch.zeros(len(demos), 1, device=self.device).byte()
                
                if self.training_type == "distillation":
                    transition.privileged_actions = all_demos_actions[step].clone()
                
                if self.training_type == "rl":
                    transition.values = torch.zeros(len(demos), 1, device=self.device)
                    transition.actions_log_prob = torch.zeros(len(demos), 1, device=self.device)
                    transition.action_mean = all_demos_actions[step].clone()
                    transition.action_sigma = torch.ones_like(all_demos_actions[step])
                
                transition.hidden_states = None
                
                # Add the transition for all demos at this timestep
                self.add_transitions(transition)
                
                

    def add_transitions(self, transition: Transition):
        # check if the transition is valid
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        # Core
        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        # for distillation
        if self.training_type == "distillation":
            self.privileged_actions[self.step].copy_(transition.privileged_actions)

        # for reinforcement learning
        if self.training_type == "rl":
            self.values[self.step].copy_(transition.values)
            self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
            self.mu[self.step].copy_(transition.action_mean)
            self.sigma[self.step].copy_(transition.action_sigma)

        # For RNN networks
        self._save_hidden_states(transition.hidden_states)

        # increment the counter
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states == (None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)
        # initialize if needed
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))
            ]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam, normalize_advantage: bool = True):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            # if we are at the last step, bootstrap the return value
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - self.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            self.returns[step] = advantage + self.values[step]

        # Compute the advantages
        self.advantages = self.returns - self.values
        # Normalize the advantages if flag is set
        # This is to prevent double normalization (i.e. if per minibatch normalization is used)
        if normalize_advantage:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    # for distillation
    def generator(self):
        if self.training_type != "distillation":
            raise ValueError("This function is only available for distillation training.")

        for i in range(self.num_transitions_per_env):
            yield self.observations[i], self.actions[i], self.privileged_actions[i], self.dones[i]

    # for reinforcement learning with feedforward networks
    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        # Core
        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)

        # For PPO
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                # Select the indices for the mini-batch
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                # Create the mini-batch
                # -- Core
                obs_batch = observations[batch_idx]
                actions_batch = actions[batch_idx]

                # -- For PPO
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                # yield the mini-batch
                yield obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    None,
                    None,
                ), None

    # for reinfrocement learning with recurrent networks
    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
                # then take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_a
                ]
                hid_c_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_c
                ]
                # remove the tuple for GRU
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else hid_c_batch

                yield obs_batch, actions_batch, values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    hid_a_batch,
                    hid_c_batch,
                ), masks_batch

                first_traj = last_traj
