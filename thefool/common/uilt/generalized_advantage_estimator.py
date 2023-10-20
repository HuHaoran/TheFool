import numpy as np

def compute_returns(rewards, bootstrap_value, terminals, gamma):
    # (N, T) -> (T, N)
    rewards = np.transpose(rewards, [1, 0])
    terminals = np.transpose(terminals, [1, 0])
    returns = []
    R = bootstrap_value
    for i in reversed(range(rewards.shape[0])):
        R = rewards[i] + (1.0 - terminals[i]) * gamma * R
        returns.append(R)
    returns = reversed(returns)
    # (T, N) -> (N, T)
    returns = np.transpose(list(returns), [1, 0])
    return returns

def compute_gae(rewards, values, bootstrap_values, terminals, gamma, lam):
    # (N, T) -> (T, N)
    rewards = np.transpose(rewards, [1, 0])
    values = np.transpose(values, [1, 0])
    values = np.vstack((values, [bootstrap_values]))
    terminals = np.transpose(terminals, [1, 0])
    # compute delta
    deltas = []
    for i in reversed(range(rewards.shape[0])):
        V = rewards[i] + (1.0 - terminals[i]) * gamma * values[i + 1]
        delta = V - values[i]
        deltas.append(delta)
    deltas = np.array(list(reversed(deltas)))
    # compute gae
    A = deltas[-1,:]
    advantages = [A]
    for i in reversed(range(deltas.shape[0] - 1)):
        A = deltas[i] + (1.0 - terminals[i]) * gamma * lam * A
        advantages.append(A)
    advantages = reversed(advantages)
    # (T, N) -> (N, T)
    advantages = np.transpose(list(advantages), [1, 0])
    return advantages

def calcAdv(rewards, values, value_inits, terminals, gamma, lam):
    last_gae_lam = 0
    rewards = np.transpose(rewards, [1, 0])
    values = np.transpose(values, [1, 0])
    values = np.vstack((values, [value_inits]))
    terminals = np.transpose(terminals, [1, 0])
    advantages = np.zeros(rewards.shape, dtype=np.float32)
    td_values = np.zeros(rewards.shape, dtype=np.float32)
    for i in reversed(range(rewards.shape[0])):
        delta = rewards[i] + (1.0 - terminals[i]) * gamma * values[i + 1] - values[i]
        last_gae_lam = delta + (1.0 - terminals[i]) * last_gae_lam * gamma * lam
        advantages[i] = last_gae_lam
        td_values[i] = advantages[i] + values[i]

    advantages = np.transpose(list(advantages), [1, 0])
    td_values = np.transpose(list(td_values), [1, 0])

    return advantages, td_values
