import pickle
import os
import shutil

import matplotlib
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from dataclasses import dataclass

import yaml

# Open and load the YAML file
with open('examples/mpc/config_overrides/quadrotor_3D/quadrotor_3D_tracking.yaml', 'r') as file:
    config_file_data = yaml.safe_load(file)

breakpoint()
def latexify_plot() -> None:
    text_usetex = True if shutil.which('latex') else False
    params = {
            'text.latex.preamble': r"\usepackage{gensymb} \usepackage{amsmath}",
            'axes.labelsize': 9,
            'axes.titlesize': 9,
            'legend.fontsize': 9,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'text.usetex': text_usetex,
            'font.family': 'serif'
    }

    matplotlib.rcParams.update(params)
    return
latexify_plot()


@dataclass
class Results:
    reward: list
    t_wall: list
    cns_viol: list
    success: list
    max_cns_viol: list
    cost_cns_viol: list
    name: str
    state_list: list
    control_list: list


DATA_DIR = 'examples/mpc/temp-data'
FIGURE_DIR = 'figures/results'
DATE = '13_11'

os.makedirs(FIGURE_DIR, exist_ok=True)


def get_main_stats(name, METHOD, RTI, CONDESING, RL=False, snd_phase=False, **kwargs):
    mydict = {}
    for k, v in kwargs.items():
        mydict.update({k: v})

    if RL:
        FILENAME = f'{DATE}_trajscal_{mydict["traj_scaling"]}_rl_ppo_timed_l4casadi_data_quadrotor_traj_tracking.pkl'
    else:
        FILENAME = f'{DATE}_trajscal_{mydict["traj_scaling"]}_{METHOD}_N{mydict["N"]}_RTI_{RTI}_init_{mydict["initialization"]}_with_warm_starting_data_quadrotor_traj_tracking.pkl'
    if snd_phase:
        FILENAME = f'{DATE}_trajscal_{mydict["traj_scaling"]}_{METHOD}_M{mydict["M"]}N{mydict["N"]}_RTI_{RTI}_sndphase_{snd_phase}_init_{mydict["initialization"]}_initpept_{mydict["initialization_pept"]}_barrier_{mydict["barrier_parameter"]}_condensed_{CONDESING}_data_quadrotor_traj_tracking.pkl'

    path = os.path.join(DATA_DIR, FILENAME)

    with open(path, "rb") as f:
        results = pickle.load(f)

    reward_list = results['trajs_data']['reward']
    LEN_EPISODE = max([len(r) for r in reward_list])

    t_wall_list = results['trajs_data']['controller_data'][0]['t_wall']

    state_list = results['trajs_data']['state']
    control_list = results['trajs_data']['action']

    constraint_violation_list = []
    max_violation_list = []
    success_list = []
    constraint_penalization_list = []
    weight = 100  # 20 times higher than max weight in cost function, 0.02 is delta_T
    for idx, sim in enumerate(results['trajs_data']['info']):
        success = True
        if "solver_failed" in sim[-1].keys():
            if sim[-1]["solver_failed"]:
                success = False
        elif sim[-1]["current_step"] != LEN_EPISODE:
            success = False
        success_list.append(success)
        if success:
            viol = 0
            max_viol = -np.inf
            cost_constr_viol = []
            for sim_i in sim:
                try:
                    viol += sim_i['constraint_violation']
                    if sim_i['constraint_violation']:
                        max_viol = max(max_viol, max(sim_i['constraint_values'][sim_i['constraint_values'] >0]))
                except:
                    viol += any(sim_i['constraint_values'] >0) * 1
                if any(sim_i['constraint_values'] >0):
                    cost_constr_viol.append(weight * sum((sim_i['constraint_values'][sim_i['constraint_values'] > 1e-6])))

            constraint_violation_list.append(viol)
            max_violation_list.append(max_viol)
            constraint_penalization_list.append(np.array(cost_constr_viol))
        else:
            viol = 0
            max_viol = -np.inf
            cost_constr_viol = []
            for sim_i in sim[:-1]:
                try:
                    viol += sim_i['constraint_violation']
                    if sim_i['constraint_violation']:
                        max_viol = max(max_viol, max(sim_i['constraint_values'][sim_i['constraint_values'] >0]))
                except:
                    viol += any(sim_i['constraint_values'] >0) * 1
                if any(sim_i['constraint_values'] >0):
                    cost_constr_viol.append(weight * sum((sim_i['constraint_values'][sim_i['constraint_values'] > 1e-6])))

            constraint_violation_list.append(viol)
            max_violation_list.append(max_viol)
            constraint_penalization_list.append(np.array(cost_constr_viol))

    res = Results(reward=reward_list, t_wall=t_wall_list,
                  success=success_list, cns_viol=constraint_violation_list,
                  cost_cns_viol=constraint_penalization_list,
                  max_cns_viol=max_violation_list, name=name,
                  state_list=state_list, control_list=control_list)

    return res

if __name__ == "__main__":

    TRAJ_SCALING = 1  # 0.8, 1
    N = 20

    results = []
    kwargs = {}
    kwargs.update({"traj_scaling": TRAJ_SCALING, "N": N,})
    kwargs.update({"initialization": "tracking_goal"})

    kwargs.update({"N": 40})
    results.append(get_main_stats("SQP-40", "acados", False, False, **kwargs))
    results.append(get_main_stats("RTI-40", "acados", True, False, **kwargs))
    kwargs.update({"initialization": "policy"})
    results.append(get_main_stats("Riccati-RL-40", "acados", True, False, **kwargs))
    results.append(get_main_stats("PPO-RL", "rl", False, False, RL=True, **kwargs))
    state_ref = np.load(f"{DATA_DIR}" + f"/state_ref_scaling_{TRAJ_SCALING}.npy")


    LEN_EPISODE = len(results[-1].reward[0])
    N_SIM = len(results[-1].reward)

    get_set_idx_success = lambda x: set(x.success * np.arange(1, N_SIM+1) - 1) - set([-1])
    is_success_fn = lambda success_idx, total_episodes, : [True if i in success_idx else False for i in range(total_episodes)]

    idx_all_success = set.intersection(*[get_set_idx_success(l) for l in results])
    is_success = is_success_fn(idx_all_success, N_SIM)

    t_sel_tot = lambda x : [t[:, 0] for t, s in zip(x.t_wall, is_success) if s == True]
    t_sel_prep = lambda x : [t[:, 1] for t, s in zip(x.t_wall, is_success) if s == True]
    t_sel_feed = lambda x : [t[:, 2] for t, s in zip(x.t_wall, is_success) if s == True]
    rew_sel_sum = lambda x : [sum(r) for r, s in zip(x.reward, is_success) if s == True]
    cost_cns_sel_sum = lambda x : [sum(c) for c, s in zip(x.cost_cns_viol, is_success) if s == True]
    cns_viol_ratio = lambda x : [c/LEN_EPISODE for c, s in zip(x.cns_viol, is_success) if s == True]
    max_cns_viol = lambda x : [c for c, s in zip(x.max_cns_viol, is_success) if s == True]

    labels = [l.name for l in results]
    plot_labels = labels


    # Print average runtime
    print("Average total runtime in milliseconds:")
    [print(l.name, np.round(np.array(t_sel_tot(l)).mean()*1e3, 2)) for l in (results)]
    print("Average feedback runtime in milliseconds:")
    [print(l.name, np.round(np.array(t_sel_feed(l)).mean()*1e3, 2)) for l in (results)]

    # # Plot state trajectories (many trajectories shadowed)
    total_reward = lambda x : [-np.nanmean(r) if s == True else 9999 for r, s in zip(x.reward, is_success)]
    total_const = lambda x : [np.nanmean(r) if s == True else 9999 for r, s in zip(x.cost_cns_viol, is_success)]

    idx_max_viol_clc = np.argmin(total_reward(results[0]))
    # [print(f"reward {r.name} : {total_reward(r)[idx_max_viol_clc]}") for r in results]
    state_list = [r.state_list[idx_max_viol_clc] for r in results]
    control_list = [r.control_list[idx_max_viol_clc] for r in results]

    nx = 12
    nu = 4
    state_lb = config_file_data['task_config']['constraints'][0]['lower_bounds']
    state_ub = config_file_data['task_config']['constraints'][0]['upper_bounds']
    control_lb = config_file_data['task_config']['constraints'][1]['lower_bounds']
    control_ub = config_file_data['task_config']['constraints'][1]['upper_bounds']
    control_ref = np.array([0.06615, 0.06615, 0.06615, 0.06615])
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    def plot_position_projections(axs, state_traj, title, ylabel=False):
        # Position-Velocity Projection XZ, YZ
        axs[0].plot(np.array(state_traj).T[0, :-1], np.array(state_traj).T[4, :-1], marker="", alpha=0.9, linewidth=2)
        axs[0].plot(np.array(state_ref).T[0, :-1], np.array(state_ref).T[4, :-1], color='k', linestyle=':', alpha=0.3,)
        if ylabel:
            axs[0].set_ylabel('$p^\mathrm{z}$ (m)', fontsize=8)
        axs[0].set_xlabel('$p^\mathrm{x}$ (m)', fontsize=8, labelpad=0.5)
        axs[0].set_xlim(-1.3, 1.3)

        axs[1].plot(np.array(state_traj).T[2, :-1], np.array(state_traj).T[4, :-1], marker="", alpha=0.9, linewidth=2)
        axs[1].plot(np.array(state_ref).T[2, :-1], np.array(state_ref).T[4, :-1], color='k', linestyle=':', alpha=0.3,)
        if ylabel:
            axs[1].set_ylabel('$p^\mathrm{z}$ (m)', fontsize=8)
        axs[1].set_xlabel('$p^\mathrm{y}$ (m)', fontsize=8, labelpad=0.5)
        axs[1].set_xlim(-0.6, 0.1)
        fig.align_ylabels(axs)
        axs[0].set_title(title, fontsize=8, pad=0.5)
        # legend = fig.legend(plot_labels, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.01),
                            # frameon=False, fontsize=7, handlelength=0.8)
        # fig.subplots_adjust(left=0.24, bottom=0.18, hspace=0.6)
        # fig.tight_layout()

    fig, axs = plt.subplots(2, 4, figsize=(8, 2.3), sharey=True)
    plot_position_projections(axs[:, 0], state_list[0], plot_labels[0], ylabel=True)
    plot_position_projections(axs[:, 1], state_list[1], plot_labels[1])
    plot_position_projections(axs[:, 2], state_list[2], plot_labels[2])
    plot_position_projections(axs[:, 3], state_list[3], plot_labels[3])

    for a in axs.flatten():
        a.tick_params(axis="y",direction="in",)
        a.tick_params(axis="x",direction="in",)

    # fig.tight_layout(w_pad=0.0)
    fig.subplots_adjust(wspace=0, hspace=0.45, bottom=0.15, top=0.9)
    fig.savefig(os.path.join(FIGURE_DIR, f"pos_projections.pdf"), bbox_inches="tight", pad_inches=0.05)

    def plot_velocity_projections(axs, state_traj, title, ylabel=False):
        dt = 0.05
        time_array = np.arange(0, state_traj.shape[0]-1) * dt
        # Position-Velocity Projection XZ, YZ
        axs[0].plot(time_array, np.array(state_traj).T[1, :-1], marker="", alpha=0.9, linewidth=2)
        axs[0].plot(time_array, np.array(state_ref).T[1, :-1], color='k', linestyle=':', alpha=0.3,)
        if ylabel:
            axs[0].set_ylabel('$v^\mathrm{x}$ (m)', fontsize=8)
        axs[0].axhline(state_lb[1], color='r', linestyle="--")
        axs[0].axhline(state_ub[1], color='r', linestyle="--")
        # axs[0].set_xlabel('$t$ (sec)', fontsize=8, labelpad=0.5)
        # axs[0].set_xlim(-1.3, 1.3)

        axs[1].plot(time_array, np.array(state_traj).T[5, :-1], marker="", alpha=0.9, linewidth=2)
        axs[1].plot(time_array, np.array(state_ref).T[5, :-1], color='k', linestyle=':', alpha=0.3,)
        if ylabel:
            axs[1].set_ylabel('$v^\mathrm{z}$ (m)', fontsize=8)
        axs[1].set_xlabel('$t$ (sec)', fontsize=8, labelpad=0.5)
        axs[1].set_xlim(0, time_array[-1])
        axs[1].axhline(state_lb[5], color='r', linestyle="--")
        axs[1].axhline(state_ub[5], color='r', linestyle="--")

        fig.align_ylabels(axs)
        axs[0].set_title(title, fontsize=8, pad=0.5)
        # legend = fig.legend(plot_labels, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.01),
                            # frameon=False, fontsize=7, handlelength=0.8)
        # fig.subplots_adjust(left=0.24, bottom=0.18, hspace=0.6)
        # fig.tight_layout()

    fig, axs = plt.subplots(2, 4, figsize=(8, 2.3), sharey=True, sharex=True)
    plot_velocity_projections(axs[:, 0], state_list[0], plot_labels[0], ylabel=True)
    plot_velocity_projections(axs[:, 1], state_list[1], plot_labels[1])
    plot_velocity_projections(axs[:, 2], state_list[2], plot_labels[2])
    plot_velocity_projections(axs[:, 3], state_list[3], plot_labels[3])

    for a in axs.flatten():
        a.tick_params(axis="y",direction="in",)
        a.tick_params(axis="x",direction="in",)

    # fig.tight_layout(w_pad=0.0)
    fig.subplots_adjust(wspace=0, hspace=0, bottom=0.15, top=0.9)
    fig.savefig(os.path.join(FIGURE_DIR, f"vel_x_z_subplots.pdf"), bbox_inches="tight", pad_inches=0.05)




    plt.show()
