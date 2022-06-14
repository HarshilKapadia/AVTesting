import sys

sys.path.append('/home/e5_5044/Desktop/carla/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg')
sys.path.append('/home/e5_5044/Desktop/VerifAI/src/verifai/simulators/carla/agents/')
sys.path.append('/home/e5_5044/Desktop/carla/PythonAPI/carla/')
from verifai.simulators.carla.client_carla import *
from verifai.simulators.carla.carla_world import *
from verifai.simulators.carla.carla_task import *
from verifai.simulators.carla.carla_scenic_task import *

import numpy as np
from dotmap import DotMap

import carla

# Falsifier (not CARLA) params
PORT = 8888
BUFSIZE = 4096

class sim_task(scenic_sampler_task):
    def __init__(self,
                 n_sim_steps=500,
                 display_dim=(1280,720),
                 carla_host='127.0.0.1',
                 carla_port=2000,
                 carla_timeout=4.0,
                 world_map='Town05'):
        super().__init__(n_sim_steps=n_sim_steps,
                 display_dim=display_dim,
            carla_host=carla_host,
            carla_port=carla_port,
            carla_timeout=carla_timeout,
            world_map=world_map)

    def use_sample(self, sample):
        super().use_sample(sample)
    def trajectory_definition(self):
        safe = True

        traj = {
            'safe': safe
        }
        return traj 


simulation_data = DotMap()
simulation_data.port = PORT
simulation_data.bufsize = BUFSIZE
# Note: The world_map param below should correspond to the MapPath 
# specified in the scenic file. E.g., if world_map is 'Town01',
# the MapPath in the scenic file should be the path to Town01.xodr.
simulation_data.task = scenic_sampler_task(world_map='Town05')



client_task = ClientCarla(simulation_data)
while client_task.run_client():
    pass
print('End of all simulations.')




