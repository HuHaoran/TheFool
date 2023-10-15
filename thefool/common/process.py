from tqdm import tqdm
import numpy as np

def learn_process(train_env, test_env, act_func, eval_func, total_steps):
    pbar = tqdm(total=total_steps, dynamic_ncols=True)
    episode = 0
    global_step = 0
    last_step = 0
    train_nums = train_env.env_nums
    rewards = [0 for _ in range(train_nums)]
    dones = [False for _ in range(train_nums)]
    episode_reward = [0 for _ in range(train_nums)]
    local_step = [0 for _ in range(train_nums)]
    states = train_env.reset()
    while True:
        prev_dones = dones
        action = act_func(states, rewards, prev_dones)
        states, rewards, dones, infos = train_env.step(action)
        global_step += train_nums
        for i, (state, reward, done) in enumerate(zip(states, rewards, dones)):
            episode_reward[i] += reward
            local_step[i] += 1
            if done:
                raw_reward = episode_reward[i]
                episode += 1
                gs = global_step - (train_env.env_nums - i - 1)
                msg = 'step: {}, episode: {}, reward: {}'
                pbar.update(local_step[i])
                pbar.set_description(
                    msg.format(gs, episode, raw_reward))
                local_step[i] = 0
                episode_reward[i] = 0

        if global_step - last_step > 100000:
            eval_process(test_env, eval_func, 50000)
            last_step = global_step


def eval_process(test_env, eval_func, total_step):

    local_step = 0
    test_nums = test_env.env_nums
    episode_reward = [0 for _ in range(test_nums)]
    states = test_env.reset()
    sum_reward = []

    while local_step < total_step:

        action = eval_func(states)
        states, rewards, dones, infos = test_env.step(action)
        local_step += test_nums
        for i, (state, reward, done) in enumerate(zip(states, rewards, dones)):
            episode_reward[i] += reward
            if done:
                raw_reward = episode_reward[i]
                episode_reward[i] = 0
                sum_reward.append(raw_reward)

    print("average reward:", np.average(sum_reward), "max reward:", np.max(sum_reward))