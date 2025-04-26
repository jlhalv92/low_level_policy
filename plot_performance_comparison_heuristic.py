import matplotlib.pyplot as plt
import numpy as np
import os



model_names = [
               'UNIFORM',
                'RARL',
                'SAC_STATIC',
               'SAC_HEURISTIC',
                'RAMEM']
velocity = 1.5
performances = {}
files = os.listdir("results_heuristic")

for idx,data_addr in enumerate(files):
    loaded_array = np.load(f"results_heuristic/{data_addr}")
    print(idx, model_names[idx])
    performances[model_names[idx]] = loaded_array




x = 0.1
total_models = len(model_names)
bar_width = 0.2

plt.figure(figsize=(12, 6))

# Plot each model with an offset
for i, model in enumerate(model_names):
    offset = (i - total_models / 2) * bar_width + bar_width / 2
    plt.bar(model, performances[model], width=bar_width, label=model, alpha=0.6)

# Labels and title
# plt.xlabel("Temperature")
plt.ylabel("Success Rate")
plt.title(f"Obstacle Navigation Performance, Heuristic, max_vel=1.5")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tick_params(axis='x', bottom=False)

plt.legend(title="Models")
# Add caption
plt.xlabel('''for each of the cases we evaluated the model on 50 runs, the model were trained on 5 seeds each ''')

plt.tight_layout()
plt.show()