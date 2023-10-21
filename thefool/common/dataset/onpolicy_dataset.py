import numpy as np
from thefool.common.uilt.generalized_advantage_estimator import compute_gae, compute_returns, calcAdv


class Rollout:
    def __init__(self,
                 state_shape,
                 time_horizon):
        self.time_horizon = time_horizon
        self.state_shape = state_shape
        self.index = 0
        self.flush()

    def add(self, obs_t, action_t, reward_tp1, value_t, log_prob_t, terminal_tp1):
        if self.index < self.time_horizon:
            self.obs_t[self.index] = obs_t
            self.actions_t[self.index] = action_t
            self.rewards_tp1[self.index] = reward_tp1
            self.values_t[self.index] = value_t
            self.log_probs_t[self.index] = log_prob_t
            self.terminals_tp1[self.index] = terminal_tp1
        self.index += 1

    def flush(self):
        self.index = 0
        self.obs_t = np.zeros([self.time_horizon] + self.state_shape, dtype=np.float32)
        self.actions_t = np.zeros([self.time_horizon], dtype=np.int32)
        self.rewards_tp1 = np.zeros([self.time_horizon], dtype=np.float32)
        self.values_t = np.zeros([self.time_horizon], dtype=np.float32)
        self.log_probs_t = np.zeros([self.time_horizon], dtype=np.float32)
        self.terminals_tp1 = np.zeros([self.time_horizon], dtype=np.float32)


class OnPolicyDataset:
    def __init__(self,
                 nenv,
                 state_shape,
                 time_horizon):
        self.nenv = nenv
        self.rollouts = [Rollout(state_shape, time_horizon) for _ in range(nenv)]

    def flush(self):
        for rollout in self.rollouts:
            rollout.flush()

    def add(self, states, actions, rewards, values, log_probs, dones):
        for i in range(self.nenv):
            self.rollouts[i].add(
                        obs_t=states[i],
                        reward_tp1=rewards[i],
                        action_t=actions[i],
                        value_t=values[i],
                        log_prob_t=log_probs[i],
                        terminal_tp1=1.0 if dones[i] else 0.0,
                    )

    def rollout_trajectories(self, bootstrap_values, gamma, lam):
        obs_t = []
        actions_t = []
        rewards_tp1 = []
        values_t = []
        log_probs_t = []
        terminals_tp1 = []
        for rollout in self.rollouts:
            obs_t.append(rollout.obs_t)
            actions_t.append(rollout.actions_t)
            rewards_tp1.append(rollout.rewards_tp1)
            values_t.append(rollout.values_t)
            log_probs_t.append(rollout.log_probs_t)
            terminals_tp1.append(rollout.terminals_tp1)

        obs_t = np.array(obs_t, dtype=np.float32)
        actions_t = np.array(actions_t, dtype=np.int32)
        rewards_tp1 = np.array(rewards_tp1, dtype=np.float32)
        values_t = np.array(values_t, dtype=np.float32)
        log_probs_t = np.array(log_probs_t, dtype=np.float32)
        terminals_tp1 = np.array(terminals_tp1, dtype=np.float32)

        #returns_t = compute_returns(rewards_tp1, bootstrap_values, terminals_tp1, gamma)
        #advs_t = compute_gae(rewards_tp1, values_t, bootstrap_values, terminals_tp1, gamma, lam)

        advs_t, returns_t = calcAdv(rewards_tp1, values_t, bootstrap_values, terminals_tp1, gamma, lam)

        advs_t = (advs_t - np.mean(advs_t)) / np.std(advs_t)

        return obs_t, actions_t, log_probs_t, values_t, returns_t, advs_t
