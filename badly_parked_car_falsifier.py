from verifai.samplers.scenic_sampler import ScenicSampler
from dotmap import DotMap
from verifai.falsifier import mtl_falsifier
import csv

path_to_scenic_file = 'oas_scenario_06.scenic'
sampler = ScenicSampler.fromScenario(path_to_scenic_file)

MAX_ITERS = 75
PORT = 8888
MAXREQS = 5
BUFSIZE = 4096
specification = ['G(~(collision))']

falsifier_params = DotMap(
    n_iters=MAX_ITERS,
    save_error_table=True,
    save_safe_table=True,
    error_table_path='error_table_carla_realistic_scenario_python_api.csv',
    safe_table_path='safe_table.csv'
)

# Uncomment the following code segment when running for test Agent to record what distances are recorded by the Object ddetection method
# It will also calculate the error rate where there is a car/vehicle but OD is not able to get it - False Negative rate
# distance_file = open('distance_data_carla_realistic_scenario.csv', 'w')
# writer = csv.writer(distance_file)
# writer.writerow(["Computed", "Carla", "FalseNegative"])
# distance_file.close()

server_options = DotMap(port=PORT, bufsize=BUFSIZE, maxreqs=MAXREQS)

falsifier = mtl_falsifier(sampler=sampler, sampler_type='scenic',
                              falsifier_params=falsifier_params,specification=specification,
                              server_options=server_options)

falsifier.run_falsifier()


print("error_table: ", falsifier.error_table.table)
print("safe_table: ", falsifier.safe_table.table)

