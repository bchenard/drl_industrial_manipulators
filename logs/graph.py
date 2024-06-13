import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
data = pd.read_csv('/home/bchenard/Documents/drl_env/stage/pybullet/ur3e/logs_training/graph.csv')

# Filter out values greater than 100
data = data[(data['sac gpu'] <= 500) & (data['generalised ppo'] != -float('inf'))]

# Plot the data
plt.figure(figsize=(10, 6))
# plt.plot(data['nb_steps'], data['PPO dense bs 256'], label='PPO dense 256 bs', color='orange')
# plt.plot(data['nb_steps'], data['PPO dense bs 64'], label='PPO dense bs 64', color='red')
# plt.plot(data['nb_steps'], data['SAC lr 0.001'], label='SAC lr 0.001', color='pink')
# plt.plot(data['nb_steps'], data['SAC lr0.0003 bs 256'], label='SAC lr 0.0003 bs 256', color='purple')
# plt.plot(data['nb_steps'], data['SAC lr0.0003 bs 64'], label='SAC lr 0.0003 bs 64', color='blue')
# plt.plot(data['nb_steps'], data['generalised ppo'], label='Generalised PPO bs 256', color='orange')
plt.plot(data['nb_steps'], data['sac gpu'], label='Generalised PPO bs 256', color='orange')


# Add labels and legend
plt.xlabel('Number of Steps')
plt.ylabel('Performance')
plt.title('Performance Comparison of Different Algorithms and Configurations')
plt.legend()
plt.grid(True)
plt.savefig('graph.png')
# Show the plot
plt.show()
