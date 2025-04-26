import matplotlib.pyplot as plt
import numpy as np
import os

# Mock data
temperatures = np.round(np.linspace(0.5, 0, num=5),2)
files = os.listdir("results")

model_names = [
               'UNIFORM',
                'SAC_STATIC',
                'RARL',
               'SAC_HEURISTIC',
                'RAMEM']
velocity = 1.5
performances = {}
files = os.listdir("results")

for idx,data_addr in enumerate(files):
    loaded_array = np.load(f"results/{data_addr}")
    print(idx, model_names[idx])
    performances[model_names[idx]] = loaded_array


x = np.arange(len(temperatures))
total_models = len(model_names)
bar_width = 0.1

plt.figure(figsize=(12, 6))

# Plot each model with an offset
for i, model in enumerate(model_names):
    offset = (i - total_models / 2) * bar_width + bar_width / 2
    plt.bar(x + offset, performances[model], width=bar_width, label=model, alpha=0.6)

# Labels and title
plt.xlabel("Temperature")
plt.ylabel("Success Rate")
plt.title(f"Obstacle Navigation Performance, Antagonist max velocity = {velocity}")
plt.xticks(x, temperatures)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend(title="Models")
# Add caption
plt.xlabel('''for each of the cases we evaluated the model on 50 runs, the model were trained on 5 seeds each ''')
# plt.figtext(
#     2, -3,
#     "This plot shows the success rate of 4 models across different temperatures during obstacle navigation tasks. "
#     "Each group of bars corresponds to a temperature setting.",
#     wrap=True, horizontalalignment='center', fontsize=10
# )

# Layout and show
plt.tight_layout()
plt.show()