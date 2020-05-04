from safe_rl.utils.load_utils import load_policy
import matplotlib.pyplot as plt
from scripts import plot as plt_util

import gym
import numpy as np
import os.path as osp
import time

from gym.wrappers import Monitor
from recovery_rl.env import wrappers
import moviepy.editor as mpy
import pickle
from scipy.interpolate import make_interp_spline, BSpline

experiment_map = {
    "maze": {
        "algs": {
            "cpo": ['2020-04-28_cpo-maze/2020-04-28_00-33-48-cpo-maze_s0'],
            "trpo": ['2020-04-30_trpo-maze/2020-04-30_03-44-32-trpo-maze_s0'],
            "ppo": ['2020-04-30_ppo-maze/2020-04-30_00-50-30-ppo-maze_s0'],
        },
        "outfile": "maze_plot.png"
    },
    "pointbot0": {
        "algs": {
            "cpo": ['2020-04-28_cpo-pointbot0/2020-04-28_00-31-23-cpo-pointbot0_s0'],
            "trpo": ['2020-04-30_trpo-pointbot0/2020-04-30_03-44-05-trpo-pointbot0_s0'],
            "ppo": ['2020-04-30_ppo-pointbot0/2020-04-30_00-51-04-ppo-pointbot0_s0'],
        },
        "outfile": "pointbot0_plot.png"
    },
    "pointbot1": {
        "algs": {
            "cpo": ['2020-04-28_cpo-pointbot1/2020-04-28_00-32-24-cpo-pointbot1_s0'],
            "trpo": ['2020-04-30_trpo-pointbot1/2020-04-30_03-43-41-trpo-pointbot1_s0'],
            "ppo": ['2020-04-30_ppo-pointbot1/2020-04-30_00-51-28-ppo-pointbot1_s0'],
        },
        "outfile": "pointbot1_plot.png"
    },
}

names = {
    "cpo": "CPO",
    "ppo": "PPO",
    "trpo": "TRPO"
}


colors = {
    "cpo": "g",
    "ppo": "b",
    "trpo": "r"
}

def npy_to_gif(im_list, filename, fps=4):
    clip = mpy.ImageSequenceClip(im_list, fps=fps)
    clip.write_gif(filename + '.gif')

def run_policy(env, savedir, get_action, num_episodes=100, render=False):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :("

    o, r, d, ep_ret, ep_cost, ep_len, n = env.reset(), 0, False, 0, 0, 0, 0
    im_list = [env.render()]
    episodes, states = [], [o]
    while n < num_episodes:
        if render:
            im_list.append(env.render())
            # time.sleep(1e-3)

        a = get_action(o)
        a = np.clip(a, env.action_space.low, env.action_space.high)
        o, r, d, info = env.step(a)
        ep_ret += r
        ep_cost += info.get('cost', 0)
        ep_len += 1
        states.append(o)
        if d:
            # print("HERE", im_list)
            npy_to_gif(im_list, osp.join(savedir, "ep_{}".format(n)))
            o, r, d, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0
            im_list = [env.render()]
            episodes.append(states)
            states = [o]
            n += 1
            
    return episodes

def get_stats(data):
    mu = np.mean(data, axis=0)
    lb = mu - np.std(data, axis=0)
    ub = mu + np.std(data, axis=0)
    return mu, lb, ub


def plot_experiment(experiment, max_eps=20000):

    if experiment == 'maze':
        fig, axs = plt.subplots(3, figsize=(16, 19))

        axs[0].set_title("Cumulative Constraint Violations vs. Episode", fontsize=20)
        axs[0].set_ylim(-0.1, max_eps+1)
        axs[0].set_xlabel("Episode", fontsize=16)
        axs[0].set_ylabel("Cumulative Constraint Violations", fontsize=16)
        axs[0].tick_params(axis='both', which='major', labelsize=14)

        axs[1].set_title("Reward vs. Episode", fontsize=20)
        axs[1].set_ylim(-0.45, 0)
        axs[1].set_xlabel("Episode", fontsize=16)
        axs[1].set_ylabel("Final Reward", fontsize=16)
        axs[1].tick_params(axis='both', which='major', labelsize=14)

        axs[2].set_title("Cumulative Task Successes vs. Episode", fontsize=20)
        axs[2].set_ylim(0, max_eps+1)
        axs[2].set_xlabel("Episode", fontsize=16)
        axs[2].set_ylabel("Cumulative Task Successes", fontsize=16)
        axs[2].tick_params(axis='both', which='major', labelsize=14)

    elif experiment.startswith('pointbot'):
        fig, axs = plt.subplots(2, figsize=(16, 19))

        axs[0].set_title("Cumulative Constraint Violations vs. Episode", fontsize=20)
        axs[0].set_ylim(-0.1, max_eps+1)
        axs[0].set_xlabel("Episode", fontsize=16)
        axs[0].set_ylabel("Cumulative Constraint Violations", fontsize=16)
        axs[0].tick_params(axis='both', which='major', labelsize=14)

        axs[1].set_title("Reward vs. Episode", fontsize=20)
        if experiment == 'pointbot0':
            axs[1].set_ylim(-10000, 0)
        else:
            axs[1].set_ylim(-4000, -1000)
        axs[1].set_xlabel("Episode", fontsize=16)
        axs[1].set_ylabel("Reward", fontsize=16)
        axs[1].tick_params(axis='both', which='major', labelsize=14)

    else:
        assert(False) 

    for alg in experiment_map[experiment]["algs"]:
        exp_dirs = experiment_map[experiment]["algs"][alg]
        fnames = [osp.join('./data', exp_dir, "run_stats.pkl") for exp_dir in exp_dirs]

        task_successes_list = []
        train_rewards_list = []
        train_violations_list = []

        for fname in fnames:
            print("FNAME: ", fname)
            with open(fname, "rb") as f:
                data = pickle.load(f)
            train_stats = data['train_stats']

            train_violations = []
            train_rewards = []
            last_rewards = []
            for traj_stats in train_stats:
                train_violations.append([])
                train_rewards.append(0)
                last_reward = 0
                for step_stats in traj_stats:
                    train_violations[-1].append(step_stats['constraint'])
                    train_rewards[-1] += step_stats['reward']
                    last_reward = step_stats['reward']
                last_rewards.append(last_reward)

            # print("TRAIN VIOLATIONS", train_violations)
            ep_lengths = np.array([len(t) for t in train_violations])[:max_eps]
            train_violations = np.array([np.sum(t) > 0 for t in train_violations])[:max_eps] # TODO: can just show total constraint viols too...
            train_violations = np.cumsum(train_violations)
            train_rewards = np.array(train_rewards)[:max_eps]
            last_rewards = np.array(last_rewards)[:max_eps]

            # for i in range(len(train_rewards)):
            #     if ep_lengths[i] != 50:
            #         diff = 50 - ep_lengths[i]
            #         train_rewards[i] += diff * last_rewards[i]

            task_successes = (-last_rewards < 0.03).astype(int)
            task_successes = np.cumsum(task_successes)
            print("TASK SUCCESSES", len(task_successes))
            print("TRAIN REWARDS", len(train_rewards))
            print("TRAIN VIOLS", len(train_violations))

            x = np.arange(len(last_rewards))
            xnew = np.linspace(x.min(), x.max(), 100)
            spl = make_interp_spline(x,last_rewards, k=3)
            last_rewards_smooth = spl(xnew)

            x = np.arange(len(train_rewards))
            xnew = np.linspace(x.min(), x.max(), 100)
            spl = make_interp_spline(x,train_rewards, k=3)
            train_rewards_smooth = spl(xnew)

            task_successes_list.append(task_successes)
            if experiment == 'maze':
                train_rewards_list.append(last_rewards_smooth)
            else:
                train_rewards_list.append(train_rewards_smooth)

            train_violations_list.append(train_violations)

        task_successes_list = np.array(task_successes_list)
        train_rewards_list = np.array(train_rewards_list)
        train_violations_list = np.array(train_violations_list)

        ts_mean, ts_lb, ts_ub = get_stats(task_successes_list)
        tr_mean, tr_lb, tr_ub = get_stats(train_rewards_list)
        tv_mean, tv_lb, tv_ub = get_stats(train_violations_list)

        axs[0].fill_between(range(tv_mean.shape[0]), tv_ub, tv_lb,
                     color=colors[alg], alpha=.5, label=names[alg])
        axs[0].plot(tv_mean, colors[alg])
        axs[1].fill_between(xnew, tr_ub, tr_lb,
                     color=colors[alg], alpha=.5, label=names[alg])
        axs[1].plot(xnew, tr_mean, colors[alg])

        if experiment == 'maze':
            axs[2].fill_between(range(ts_mean.shape[0]), ts_ub, ts_lb,
                         color=colors[alg], alpha=.5)
            axs[2].plot(ts_mean, colors[alg], label=names[alg])

    axs[0].legend(loc="lower right")
    axs[1].legend(loc="lower right")
    if experiment == 'maze':
        axs[2].legend(loc="lower right")
    plt.savefig(experiment_map[experiment]["outfile"])


if __name__ == '__main__':
    # load_dir = osp.join('./data', '2020-04-26_cpo-maze2/2020-04-26_03-03-02-cpo-maze2_s0')
    # data = plt_util.get_all_datasets([load_dir])
    # plt_util.plot_data(data[0], savedir=load_dir, title='', xaxis='TotalEnvInteracts', value='CumulativeCost')
    # plt_util.plot_data(data, savedir=load_dir, title='', xaxis='TotalEnvInteracts')
    experiments = ['pointbot0', 'pointbot1', 'maze']

    # env, get_action, sess = load_policy(load_dir, 'last', True)
    # vid_path = osp.join(load_dir, 'videos')
    # env = Monitor(env, vid_path, force=True)

    # num_episodes = 4
    # episodes = run_policy(env, load_dir, get_action, num_episodes, render=True)
    # env.close()

    for exp in experiments:
        plot_experiment(exp)