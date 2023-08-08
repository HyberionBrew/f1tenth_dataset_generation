# import gym

import f110_gym
import gym
from argparse import Namespace
from f110_gym.envs.base_classes import Integrator
import yaml
import numpy as np
# from numba import njit

from ftg_agents.agents import *
import pickle as pkl
from absl import flags, app
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_integer('timesteps',10000, 'Number of timesteps to run for')
flags.DEFINE_integer('sub_sample',10, 'Number of lidar rays to subsample')
flags.DEFINE_string('agent', 'StochasticFTGAgent', 'Name of agent to use')
flags.DEFINE_string('map_config', 'config.yaml' , 'Name of map config file')
flags.DEFINE_string('dataset_name', 'dataset.pkl', 'Name of dataset file')
flags.DEFINE_bool('record', False, 'Whether to record the run')
flags.DEFINE_bool('render', True, 'Whether to render the run')
flags.DEFINE_float('speed', 1.0, 'Mean speed of the car')

agents = {
    'DeterministicFTGAgent': DeterministicFTGAgent,
    'StochasticFTGAgent': StochasticFTGAgent,
'StochasticFTGAgentRandomSpeed': StochasticFTGAgentRandomSpeed, 
'StochasticFTGAgentDynamicSpeed': StochasticFTGAgentDynamicSpeed} #, 'FTGAgent': FTGAgent, 'PurePursuitAgent': PurePursuitAgent}

def main(argv):

    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 1.375}

    with open(FLAGS.map_config) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1, timestep=0.01, integrator=Integrator.RK4)
    # set maximum timesteps for env
    env.max_steps = FLAGS.timesteps

    agent = agents[FLAGS.agent](env, sub_sample=FLAGS.sub_sample, speed=FLAGS.speed)
    
    
    timestep = 0
    pbar = tqdm(total=FLAGS.timesteps, desc="Processing", position=0, leave=True)

    while timestep < FLAGS.timesteps:
    #for timestep in range(FLAGS.timesteps):
        obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
        if FLAGS.render:
            env.render()   
        while not done and timestep < FLAGS.timesteps:
            if timestep % 5 == 0: # corresponds to an action rate of 20Hz
                speed, steering = agent.get_action(obs["scans"][0])
                steering = float(steering)
                speed = float(speed)
            obs, step_reward, done, info = env.step(np.array([[steering, speed]]))
            

            if (timestep % 5 == 0 or done) and FLAGS.record:
                with open("raw_datasets/"+FLAGS.dataset_name, 'ab') as f:
                    pkl.dump((speed, steering, obs, step_reward, done, info, timestep, (FLAGS.agent,FLAGS.speed)), f)
            #if timestep % 300 == 0:
            #    print(timestep)
            timestep += 1
            # Update the progress bar by 1
            pbar.update(1)
            if FLAGS.render:
                env.render()
        
if __name__ == '__main__':
    app.run(main)
