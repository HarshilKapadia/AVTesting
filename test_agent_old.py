import carla
import sys
from agents.navigation.agent import *
from agents.navigation.controller import VehiclePIDController
from agents.tools.misc import distance_vehicle, draw_waypoints
import time, random
from pid_agent import *
from pid_follow_controller import *
import math
import numpy as np
from enum import IntEnum

#Imports related to OD Model
import sys
sys.path.append('/home/e5_5044/Desktop/yolo-v4-tf')
from models import Yolov4
import keras
import os
FOLDER_PATH = '/home/e5_5044/Desktop/yolo-v4-tf/data/train/'
class_name_path = '/home/e5_5044/Desktop/yolo-v4-tf/data/classes.txt'
saved_model_path = '/home/e5_5044/Desktop/yolo-v4-tf/checkpoints/carla_od_yolo4_model_epoch_200.hdf5'









class LaneChange(IntEnum):
    LEFT = -1
    RIGHT = 1
'''This is a Test Agent'''
class TestAgent(PIDAgent):
    def __init__(self, vehicle, opt_dict=None):
        super().__init__(vehicle, opt_dict)
        # Distance to maintain from other vehicles.
        self.clear_dist = 10.0
        self.current_image = None
        self.location_pid_dict = {
            'K_P': 1.0,
            'K_D': 0.05,
            'K_I': 1,
            'dt': 0.05
        }
        if opt_dict:
            if 'target_speed' in opt_dict:
                self.location_pid_dict['dt'] = 1.0 / self.target_speed
            if 'clear_dist' in opt_dict:
                self.clear_dist = opt_dict['clear_dist']
            if 'location_pid_dict' in opt_dict:
                self.location_pid_dict = opt_dict['location_pid_dict']

        self.controller = PIDFollowController(
            vehicle,
            clear_dist=self.clear_dist,
            args_lateral=self.lateral_pid_dict,
            args_longitudinal=self.longitudinal_pid_dict)
        # Magnitude of lane_state is how far vehicle is from its
        # desired lane. Negative/positive means to the left/right
        # of desired lane, respectively.
        self.lane_state = 0
        self.is_changing_lane = False
        self.current_image = None

        self.actor = self._world.get_actor(vehicle.id)
        print(self.actor)
        #Add rgb and depth sensors to the Ego here
        #Adding RGB and Depth Camera on Ego
        self.fov = None
        cam_bp = self._world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute("image_size_x",str(1080))
        cam_bp.set_attribute("image_size_y",str(1080))
        cam_bp.set_attribute("fov",str(110))
        cam_bp.set_attribute('sensor_tick', '0.2')
        cam_location = carla.Location(2,0,1)
        cam_rotation = carla.Rotation(0,0,0)
        cam_transform = carla.Transform(cam_location,cam_rotation)
        self.rgb_camera = self._world.try_spawn_actor(cam_bp,cam_transform,self.actor,carla.AttachmentType.Rigid)
        
        depth_sensor_bp = self._world.get_blueprint_library().find('sensor.camera.depth')
        depth_sensor_bp.set_attribute("image_size_x",str(1080))
        depth_sensor_bp.set_attribute("image_size_y",str(1080))
        depth_sensor_bp.set_attribute("fov",str(110))
        depth_sensor_bp.set_attribute('sensor_tick', '0.2')
        depth_location = carla.Location(2,0,1)
        depth_rotation = carla.Rotation(0,0,0)
        depth_transform = carla.Transform(depth_location,depth_rotation)
        self.depth_sensor = self._world.spawn_actor(depth_sensor_bp,depth_transform,self.actor, attachment_type=carla.AttachmentType.Rigid)
        
        
        self.rgb_image = None
        self.depth_image = None


        #Load model for object detection
        self.od_model = Yolov4(weight_path=None, class_name_path=class_name_path)
        self.od_model.load_model(saved_model_path)

        self.rgb_camera.listen(lambda image: self.process_rgb_sensor_data(image))
        self.depth_sensor.listen(lambda image: self.process_depth_sensor_data(image))

    
    def get_lane_change_w(self, cur_w, lane_change):
        # Return waypoint corresponding to LANE_CHANGE (either LEFT or RIGHT)
        # from waypoint CUR_W. If lane change illegal or unsafe, returns None.
        next_w = None
        lane_safe = lambda w: self.lane_clear(w.transform.location)[0] and \
            self.lane_clear(w.transform.location,
                            min_dist=10.0*self.clear_dist,
                            forward=w.transform.get_forward_vector())[0]
        if lane_change is LaneChange.LEFT and cur_w.lane_change & carla.LaneChange.Left:
            if lane_safe(cur_w.get_left_lane()):
                next_w = cur_w.get_left_lane()
        elif cur_w.lane_change & carla.LaneChange.Right:
            if lane_safe(cur_w.get_right_lane()):
                next_w = cur_w.get_right_lane()
        return next_w


    def change_lane(self, lane_change=None):
        ''' By default, picks left or right at random. If no possible lane change,
        does nothing.'''
        if lane_change:
            potential_change = [lane_change]
        else:
            potential_change = [LaneChange.LEFT, LaneChange.RIGHT]
        potential_w = []
        cur_w = self.waypoints[0]
        for change in potential_change:
            next_w = self.get_lane_change_w(cur_w, change)
            if next_w:
                potential_w.append((change, next_w))
        if potential_w:
            lane_change, next_w = random.choice(potential_w)
            self.is_changing_lane = True
            self.lane_state += lane_change
            self.waypoints = [cur_w, next_w]


    def lane_clear(self, location, min_dist=None, forward=None):
        '''
        Check that the lane LOCATION is in is clear of vehicles.
        Lane is clear if no vehicles within MAX_DIST away.
        If not clear, return tuple of False and blocking vehicle.
        If FORWARD, only check vehicles along direction of FORWARD.
        '''
        def norm(vec):
            return np.sqrt(vec.x ** 2 + vec.y ** 2 + vec.z ** 2)

        def norm_dot(a, b):
            a /= norm(a)
            b /= norm(b)
            dot = a.x * b.x + a.y * b.y + a.z * b.z
            return dot
        
        if not min_dist:
            min_dist = self.clear_dist
            
        lane_id = self._map.get_waypoint(location).lane_id
        vehicles = self._world.get_actors().filter('*vehicle*')
        
        
        output = self.get_detection(self.rgb_image)
        #print("Model output")
        # Columns in 'output' Dataframe: [x1, y1, x2, y2, class_name, score, w, h]
        if not (output.empty):
            for row in output.iterrows():
                if row[1]['class_name'] == 'passenger_car':
                    centre_x, centre_y = math.floor(abs(row[1]['x1'] + row[1]['x2'])/2), math.floor(abs(row[1]['y1'] + row[1]['y2'])/2)
                    print("Distance from sensor: "+str(self.compute_distance_from_depth_image(centre_x, centre_y)))
                    
                    

        



        for v in vehicles:
            # Check if v is self.
            if v.id == self._vehicle.id:
                continue

            # Check if v is on same lane as self.
            v_loc = v.get_location()
            #print("CARLA compued vehicle distance"+str(v_loc.distance(location)))
            v_w = self._map.get_waypoint(v_loc)
            if lane_id != v_w.lane_id:
                continue

            if forward and norm_dot(forward, v_loc - location) < 0.0:
                continue

            if v_loc.distance(location) < min_dist:
                return (False, v)
        return (True, None)
    
    
    def run_step(self):
        self_loc = self._vehicle.get_location()
        self_forward = self._vehicle.get_transform().get_forward_vector()
        is_clear, blocker = self.lane_clear(self_loc,
                                            min_dist=2.0*self.clear_dist,
                                            forward=self_forward)
        super().run_step()
        
        cur_w = self.waypoints[0]
        if self.lane_state != 0:
            speed = self.target_speed * 1.5
        else:
            speed = self.target_speed

        if not is_clear:
            if not self.is_changing_lane:
                self.change_lane()
                    
            return self.controller.run_step(speed,
                                            self.waypoints[0],
                                            blocker.get_location())
        else:
            self.is_changing_lane = False
            if self.lane_state != 0:
                lane_change = LaneChange(-np.sign(self.lane_state))
                self.change_lane(lane_change=lane_change)
                
            return self.controller.run_step(speed,
                                            self.waypoints[0])

    def process_rgb_sensor_data(self, image):
        img = np.array(image.raw_data)
        img = img.reshape(1080,1080,4)
        img = img[:,:,:3]
        self.rgb_image = img
        image.save_to_disk('/home/e5_5044/Desktop/temp/%d.jpg' % image.frame)

    def process_depth_sensor_data(self, image):
        self.fov = image.fov
        img = np.array(image.raw_data)
        img = img.reshape(1080, 1080,4)
        img = img[:,:,:3]
        self.depth_image = img

    def get_detection(self, image):
        return self.od_model.predict_img(image)

    def compute_distance_from_depth_image(self, x, y):

        dist = 1000*(self.depth_image[x][y][0] + (self.depth_image[x][y][1]*256) + (self.depth_image[x][y][2]*256*256))/(256*256*256-1)
        fov = self.fov
        center = 1080 / 2
        focal_length = (center)/(math.tan(fov/2))
        diagDist = math.sqrt(pow((x-center-0.5),2)+pow(y-center-0.5,2)+pow(focal_length,2))
        true_depth = dist * (diagDist/focal_length)
        return true_depth
         
        