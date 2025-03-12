import argparse

import numpy as np
import matplotlib.pyplot as plt
import json

from mushroom_rl.core import Agent



def plot_results(
):


    #
    # experiments = ["RARL_05", "RARL_04", "RARL_03", "RARL_02", "RARL_01", "RARL_00"]
    # seeds = ["S1", "S2", "S3", "S4", "S5"]
    # model_id = "RARL_alpha"
    # alphas = [0.5, 0.4, 0.3,0.2,0.1, 0.0]
    # runs = [0, 1, 2, 3, 4]
    # levels_a = [0,1,2]
    # velocities = [1, 0.5]
    # protagonist_ids =[ f"{model_id}_{alpha}" for alpha in alphas]
    # values = np.zeros((len(experiments),len(seeds))) # Random values for demonstration
    # values_acc = np.zeros((len(experiments),10))
    # exp_id = 0
    # for level_a in levels_a:
    #     for vel in velocities:
    #         for i,protagonist_id in enumerate(protagonist_ids):
    #
    #             for run in runs:
    #                 file_path = f"results_exp/{protagonist_id}_seed_{run}_antagonist_level_{level_a}_velocity_{vel}.json"
    #                 with open(file_path, 'r') as f:
    #                     loaded_data = json.load(f)
    #                     values[i, run] = loaded_data["success_rate"]
    #             values_acc[i,exp_id] = np.mean(values[i, :])
    #
    #         exp_id += 1
    #
    # for level_a in [0]:
    #     for vel in [0.0]:
    #         for i,protagonist_id in enumerate(protagonist_ids):
    #
    #             for run in runs:
    #                 file_path = f"results_exp/{protagonist_id}_seed_{run}_antagonist_level_{level_a}_velocity_{vel}.json"
    #                 with open(file_path, 'r') as f:
    #                     loaded_data = json.load(f)
    #                     values[i, run] = loaded_data["success_rate"]
    #             values_acc[i,exp_id] = np.mean(values[i, :])
    #
    #         exp_id += 1
    #
    # exp_id = 6
    model_id = "RARL_alpha"
    alphas = [0.5, 0.4, 0.3,0.2,0.1, 0.0]

    experiments = ["DORAEMON", "UNIFORM"]
    seeds = ["S1", "S2", "S3", "S4", "S5"]
    protagonist_ids_fixed =[ f"{model_id}_{alpha}" for alpha in alphas]

    rationalities = [True, False]
    protagonist_ids = []
    for protagonist in experiments:
        for rationality in rationalities:
            model_id = f"{protagonist}_aware_rationality_{rationality}"
            protagonist_ids.append(model_id)
    protagonist_ids.extend(protagonist_ids_fixed)
    runs = [0,1,2,3,4]

    from itertools import product

    levels_a = [0, 1, 2]
    velocities = [1., 0.5]

    level_velocities = list(product(levels_a, velocities))
    level_velocities.append((0,0.))

    values = np.zeros((len(protagonist_ids),len(seeds)))
    results = np.zeros((len(protagonist_ids),7))
    rows = protagonist_ids
    columns = [f"antagonist_level_{level_vel[0]}_velocity_{level_vel[1]}"  for level_vel in level_velocities ]
    for i, protagonist_id in enumerate(protagonist_ids):
        exp_id = 0
        for level_vel in level_velocities:
            vel = level_vel[1]
            level = level_vel[0]
            if vel==1. and "RARL" in protagonist_id:
                vel=1
            for run in runs:
                file_path = f"results_exp/{protagonist_id}_seed_{run}_antagonist_level_{level}_velocity_{vel}.json"
                with open(file_path, 'r') as f:
                    loaded_data = json.load(f)
                    values[i, run] = loaded_data["success_rate"]
            results[i,exp_id] = np.mean(values[i, :])
            exp_id += 1

    import seaborn as sns
    import pandas as pd

    df = pd.DataFrame(results[:,[6,1,3,5,0,2,4]], index=rows, columns=np.array(columns)[[6,1,3,5,0,2,4]])

        # Plot heatmap
    # Plot heatmap with adjusted font sizes and centering
    plt.figure(figsize=(12, 6))

    # Create heatmap
    ax = sns.heatmap(df, annot=True, cmap="coolwarm", fmt=".3f", linewidths=0.5,
                     annot_kws={"size": 8}, cbar_kws={"shrink": 0.75})

    # Adjust labels
    plt.title("Performance Comparison Between Models", fontsize=14, pad=20)
    plt.xlabel("Antagonist Levels & Velocities", fontsize=12, labelpad=10)
    plt.ylabel("Models", fontsize=12, labelpad=10)

    # Reduce font size of tick labels and adjust rotation
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8, rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8, rotation=0)

    # Adjust heatmap position
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)

    plt.show()







if __name__ == "__main__":
     plot_results()



