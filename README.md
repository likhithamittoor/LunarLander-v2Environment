# Lunar Lander Deep Reinforcement Learning Agent
This code lunar_lander.ipynb implements a deep reinforcement learning agent for the Lunar Lander environment from OpenAI Gym (https://gymnasium.farama.org/environments/box2d/lunar_lander/). The agent uses a Deep Q-Network (DQN) to learn the optimal policy for landing the lunar lander safely on the landing pad.
# Implementation Details
The DQN agent is implemented using TensorFlow and Keras. The Q-Network consists of an input layer, two hidden layers with 128 neurons each and ReLU activation, and an output layer with linear activation. The agent uses experience replay and target network techniques to stabilize the learning process.
# The agent is trained for 1000 episodes with the following hyperparameters:

Discount factor (GAMMA): 0.99
Replay buffer size (BUFFER_SIZE): 10000
Batch size (BATCH_SIZE): 128
Learning rate (LEARNING_RATE): 0.001
Update frequency (UPDATE_EVERY): 10
Exploration rate start (EXPLORATION_RATE_START): 1.0
Exploration rate end (EXPLORATION_RATE_END): 0.01
Exploration decay rate (EXPLORATION_DECAY_RATE): 0.995
Epsilon-greedy steps (EPSILON_GREEDY_STEPS): 1000000

# Results
The learning progress of the agent is plotted over 1000 episodes. Two figures are generated:

Episode Rewards: Shows the reward obtained in each episode.
Episode Losses: Shows the training loss in each episode.

After training, the agent's performance is demonstrated by rendering videos of the last 30 episodes. The videos are saved in the videos directory.
Usage

# Install the required dependencies:

TensorFlow
Keras
OpenAI Gym
NumPy
Matplotlib
OpenCV (cv2)
imageio


# Run the script: lunar_lander.ipynb
The script will train the DQN agent for 1000 episodes, plot the learning progress, save the trained model, and render videos of the last 30 episodes.

# Discussion
The DQN agent successfully learns to land the lunar lander safely on the landing pad. The episode rewards plot shows an increasing trend, indicating that the agent improves its performance over time. The episode losses plot shows a decreasing trend, suggesting that the agent's predictions become more accurate as training progresses.
The rendered videos demonstrate the agent's ability to control the lunar lander and make successful landings. The agent learns to adjust the thrust and orientation of the lander to minimize the landing velocity and avoid crashing.
However, there is still room for improvement. The agent's performance can be further enhanced by fine-tuning the hyperparameters, exploring different network architectures, and incorporating advanced techniques such as prioritized experience replay or dueling networks.
Source Code
The source code for the DQN agent and the Lunar Lander environment is provided in the lunar_lander.ipynb file.