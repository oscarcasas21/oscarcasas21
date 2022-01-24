import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL.Image
import pyvirtualdisplay
import collections
import gym

import tensorflow as tf
import tf_agents
from tf_agents.environments import suite_atari

initial_collect_steps = 200
gamma = 0.99 #discount factor

batch_size =   32
log_interval =   5000

num_eval_episodes = 1
eval_interval = 25000  

max_episode_frames=108000
ATARI_FRAME_SKIP = 4

class ObservationCollector(gym.Wrapper):
    
  def __init__(self, env):
    super(ObservationCollector, self).__init__(env)
    self._observations = collections.deque(maxlen=50000)
    
  def step(self, action):
    observation, accumulated_reward, is_terminal, info = self.env.step(action)
    self._observations.append(observation) 
    return observation, accumulated_reward, is_terminal, info
  
  def reset(self):
    observation = self.env.reset()
    self._observations.clear()
    self._observations.append(observation)
    return observation
  
  def return_observations(self):
    return self._observations

class Normalizer(tf_agents.environments.wrappers.PyEnvironmentBaseWrapper):
    
  def __init__(self, env):
    super(Normalizer, self).__init__(env)
    self._env = env
    self._observation_spec = tf_agents.specs.BoundedArraySpec(
        shape = env.observation_spec().shape,
        dtype = np.float32,
        minimum = 0.0,
        maximum = 1.0,
        name = env.observation_spec().name)
    
  def _step(self, action):
    time_step = self._env.step(action)  
    observation = time_step.observation.astype('float32')
    time_step = time_step._replace(observation = observation/255.0)
    return time_step

  def observation_spec(self):
    return self._observation_spec
  
  def _reset(self):
    time_step = self._env.reset()
    observation = time_step.observation.astype('float32')
    time_step = time_step._replace(observation = observation/255.0)
    return time_step

# which environment or game we want to play
environment_name = "MsPacman-v0"

train_py_env = suite_atari.load(
    environment_name, 
    gym_env_wrappers = suite_atari.DEFAULT_ATARI_GYM_WRAPPERS_WITH_STACKING, 
    env_wrappers=(Normalizer,))

test_py_env = suite_atari.load(
    environment_name,
    gym_env_wrappers = (ObservationCollector,) + suite_atari.DEFAULT_ATARI_GYM_WRAPPERS_WITH_STACKING, 
    env_wrappers = (Normalizer,))

train_tf_env = tf_agents.environments.tf_py_environment.TFPyEnvironment(train_py_env)
test_tf_env = tf_agents.environments.tf_py_environment.TFPyEnvironment(test_py_env)

# here we define the agent which is DQN in this case
global_step = tf.compat.v1.train.get_or_create_global_step()

q_net = tf_agents.networks.q_network.QNetwork(
    input_tensor_spec = train_tf_env.observation_spec(),
    action_spec = train_tf_env.action_spec(),
    conv_layer_params = ((32, 8, 4), (64, 4, 2), (64, 3, 1)), 
    fc_layer_params = (512,))

agent = tf_agents.agents.DqnAgent(
    train_tf_env.time_step_spec(),
    train_tf_env.action_spec(),
    q_network = q_net,
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0003),
    epsilon_greedy = 0.03,
    n_step_update = 2,
    target_update_tau = 0.005,
    td_errors_loss_fn = tf_agents.utils.common.element_wise_huber_loss,
    gamma = 0.99,
    train_step_counter = global_step)

agent.initialize()

# replay buffer to store the observations which will agent uses to learn the environmet
replay_buffer = tf_agents.replay_buffers.tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec = agent.collect_data_spec,
    batch_size = train_tf_env.batch_size,
    max_length = 10000)

dataset = replay_buffer.as_dataset(sample_batch_size = 64, num_steps = 3, num_parallel_calls = 3).prefetch(3)
dataset = iter(dataset)

number_episodes_metric = tf_agents.metrics.tf_metrics.NumberOfEpisodes()
average_return_metric = tf_agents.metrics.tf_metrics.AverageReturnMetric()
random_policy = tf_agents.policies.random_tf_policy.RandomTFPolicy(train_tf_env.time_step_spec(), train_tf_env.action_spec())

train_driver = tf_agents.drivers.dynamic_step_driver.DynamicStepDriver(
    env = train_tf_env,
    policy = agent.collect_policy,
    observers = [replay_buffer.add_batch, number_episodes_metric],
    num_steps = 10)

test_driver = tf_agents.drivers.dynamic_episode_driver.DynamicEpisodeDriver(
    env = test_tf_env,
    policy = agent.policy,
    observers = [average_return_metric],
    num_episodes = 2)

random_driver = tf_agents.drivers.dynamic_step_driver.DynamicStepDriver(
    env = train_tf_env,
    policy = random_policy,
    observers = [replay_buffer.add_batch],
    num_steps = 1000)

checkpoint_dir = os.path.join(os.getcwd(), 'checkpoint')

train_checkpointer = tf_agents.utils.common.Checkpointer(
    ckpt_dir = checkpoint_dir,
    max_to_keep = 1,
    agent = agent,
    policy = agent.policy,
    replay_buffer = replay_buffer,
    global_step = global_step)

train_checkpointer.initialize_or_restore()
global_step = tf.compat.v1.train.get_global_step()

# helper function to store videos
def capture_episodes(video_filename, num_episodes = 5):
    with imageio.get_writer(video_filename, fps = 30) as video:
      for _ in range(num_episodes):
        time_step = test_tf_env.reset()
        while not time_step.is_last():
          policy_step = agent.policy.action(time_step)
          time_step = test_tf_env.step(policy_step.action)

        for observation in test_py_env.return_observations():
          video.append_data(observation)  
if global_step == 0:
  random_driver.run()
  os.makedirs('videos')   
  capture_episodes('videos/no_training.mp4')
  
step = global_step
time_step = train_tf_env.reset()
agent.train = tf_agents.utils.common.function(agent.train)

# running an agent
# note that this will take quite a long time.
for epoch in range(step + 1, step + 100001):
  time_step, _ = train_driver.run(time_step)
  experience, _ = next(dataset)
  loss, _ = agent.train(experience)

  if epoch % 1000  == 0:
    test_driver.run()
    num_episodes = number_episodes_metric.result().numpy()
    test_score = average_return_metric.result().numpy() 
    average_return_metric.reset()
    capture_episodes(f'videos/epoch_{epoch}_episode_{num_episodes+6848}_score_{test_score}.mp4')
    train_checkpointer.save(global_step)
    step = epoch
    print(loss)