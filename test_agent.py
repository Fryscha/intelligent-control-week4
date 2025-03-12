import gym
import numpy as np
from dqn_agent import DQNAgent

env = gym.make('MountainCar-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Inisialisasi agen (gunakan model terlatih jika tersedia)
agent = DQNAgent(state_size, action_size)
agent.epsilon = 0.01  # Minimalkan eksplorasi saat testing

for e in range(5):  # Uji selama 5 episode
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(500):
        env.render()  # Menampilkan lingkungan
        action = agent.act(state)  # Ambil tindakan berdasarkan model
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = np.reshape(next_state, [1, state_size])
        if done:
            print(f"Test Episode: {e+1}, Score: {time}")
            break
env.close()
