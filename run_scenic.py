
from dotmap import DotMap
import sys
sys.path.append('/home/e5_5044/Desktop/carla/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg')
sys.path.append('/home/e5_5044/Desktop/VerifAI/src/verifai/simulators/carla/agents/')

sys.path.append('/home/e5_5044/Desktop/carla/PythonAPI/carla/')
from verifai.samplers.scenic_sampler import ScenicSampler
from verifai.scenic_server import ScenicServer
from verifai.falsifier import generic_falsifier, mtl_falsifier
import os
from scenic.core.vectors import Vector
import math
from verifai.monitor import specification_monitor, mtl_specification
from verifai.simulators.carla.client_carla import *
from verifai.simulators.carla.carla_world import *
from verifai.simulators.carla.carla_task import *
from verifai.simulators.carla.carla_scenic_task import *


import carla
import numpy




# from utils import sampleWithFeedback, checkSaveRestore

## Dynamic scenarios
path_dir = '/home/e5_5044/Desktop/Scenic/'
path = os.path.join(path_dir, 'examples/carla/Carla_Challenge/carlaChallenge9.scenic')
sampler = ScenicSampler.fromScenario(path)
PORT = 8888
BUFSIZE = 4096


class MyMonitor(specification_monitor):
    def __init__(self):
        self.specification = mtl_specification(['G safe'])
        super().__init__(self.specification)

    def evaluate(self, traj):
        print("traj", traj)
        eval_dictionary = {'safe' : [[index, self.compute_dist(traj[index])-5] for index in range(len(traj))]}
        return self.specification.evaluate(eval_dictionary)

    def compute_dist(self, coords):
        vector0 = coords[0]
        vector1 = coords[1]

        x0, y0 = vector0[0], vector0[1]
        x1, y1 = vector1[0], vector1[1]

        return math.sqrt(math.pow(x0-x1,2) +  math.pow(y0-y1,2))


falsifier_params = DotMap(
    n_iters=5,
    save_error_table=True,
    save_safe_table=True,
    error_table_path='error_table.csv',
    safe_table_path='safe_table.csv'
)
server_options = DotMap(port=PORT, bufsize=BUFSIZE, maxreqs=5)
falsifier = generic_falsifier(sampler=sampler,sampler_type='scenic',
                              monitor = MyMonitor(),
                              falsifier_params=falsifier_params,
                              server_options=server_options)
falsifier.run_falsifier()

print('end of test')

print("error_table: ", falsifier.error_table.table)
print("safe_table: ", falsifier.safe_table.table)

