import pybullet as p
import pybullet_data
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import random 
import math
import csv
import os.path as osp
import pybullet_utils.bullet_client as bc
import argparse
import pandas as pd
from scipy.spatial.transform import Rotation as R

parser = argparse.ArgumentParser()
parser.add_argument('--env', default = 1)
parser.add_argument('--width',default= 1000)
parser.add_argument('--height',default =1000)
args = parser.parse_args()

######################################______Ground_Truth_For_Max_Reward_______##################################

def ground_truth():
	segmentation= np.asarray(pd.read_csv('out.csv',header = None))
	value, count = np.unique(segmentation,return_counts=True)

	ground_truth = {}
	segmentation_weight = {}

	for i,v in enumerate(value):
		if v ==-1:
			ground_truth[v]=0
		else:
			ground_truth[v]= count[i]

	total_segmentation_score = sum(ground_truth.values())
	normalize_ground_truth = {}
	normalize_ground_truth.update((x, round(y/total_segmentation_score,4)) for x, y in ground_truth.items())
	segmentation_weight.update((x, 0) if x==-1 else (x,1-y) for x, y in normalize_ground_truth.items())
	Max_reward = sum([a*b for a,b in zip(list(normalize_ground_truth.values()),list(segmentation_weight.values()))])
	return round(Max_reward,4), segmentation_weight,normalize_ground_truth,total_segmentation_score

################################_____Align End_Effector_Axis_With_Bin___#######################################

def transform_points_from_local_frame_to_world(points_in_local_frame,
local_pos_in_world_frame, local_orn_in_world_frame, scale = [1.0, 1.0, 1.0]):
    "points_in_local_frame np.array of shape (n,3)"
    "Transform points from local frame to world frame"

    points_in_local_frame = np.array(points_in_local_frame)
    points_in_world_frame = np.zeros(points_in_local_frame.shape)
    data = p.getMatrixFromQuaternion(local_orn_in_world_frame)
    rotation_matrix_wrt_world = np.array([[data[0], data[1], data[2]], [data[3],
    data[4], data[5]], [data[6], data[7], data[8]]])
    trans_mat_wrt_world = np.eye(4)
    trans_mat_wrt_world[:3,:3] = rotation_matrix_wrt_world
    trans_mat_wrt_world[0:3,3] = local_pos_in_world_frame

    for i in range(points_in_local_frame.shape[0]):
        homo_pnt_wrt_local_frame = np.ones([4,1])
        homo_pnt_wrt_local_frame[0, 0] = scale[0]*points_in_local_frame[i, 0]
        homo_pnt_wrt_local_frame[1, 0] = scale[1]*points_in_local_frame[i, 1]
        homo_pnt_wrt_local_frame[2, 0] = scale[2]*points_in_local_frame[i, 2]

        transformed_point = np.matmul(trans_mat_wrt_world, homo_pnt_wrt_local_frame)

        points_in_world_frame[i,0] = transformed_point[0]/transformed_point[3]
        points_in_world_frame[i,1] = transformed_point[1]/transformed_point[3]
        points_in_world_frame[i,2] = transformed_point[2]/transformed_point[3]

    return points_in_world_frame


def cross_product_axb(a, b):
    c = np.zeros([3,])
    a = np.array(a)
    b = np.array(b)
    c[0] = a[1]*b[2] - a[2]*b[1]
    c[1] = a[2]*b[0] - a[0]*b[2]
    c[2] = a[0]*b[1] - a[1]*b[0]

    dis = np.linalg.norm(c)
    c[0] /= dis
    c[1] /= dis
    c[2] /= dis
    return c

def visualize_frame(pos, orn, frame_length=1):
    points_wrt_local_frame = [[0,0,0], [frame_length,0,0], [0,frame_length,0], [0,0,frame_length]]
    pnts_wrt_world = transform_points_from_local_frame_to_world(points_wrt_local_frame, pos, orn)

    point_in_world_frame_origin = pnts_wrt_world[0,:]
    point_in_world_frame_x_axis = pnts_wrt_world[1,:]
    point_in_world_frame_y_axis = pnts_wrt_world[2,:]
    point_in_world_frame_z_axis = pnts_wrt_world[3,:]

    start_point = point_in_world_frame_origin[:3]
    end_point = point_in_world_frame_x_axis[:3]
    p.addUserDebugLine(start_point, end_point, [1,0,0])

    end_point = point_in_world_frame_y_axis[:3]
    p.addUserDebugLine(start_point, end_point, [0,1,0])

    end_point = point_in_world_frame_z_axis[:3]
    p.addUserDebugLine(start_point, end_point, [0,0,1])

def align_gripper_axis_along_a_specific_axis(new_pos,new_orn, gripper_axis_to_be_aligned, target_axis_wrt_world):
    orn = new_orn
    pos = new_pos	
    r = R.from_quat(orn)
    current_transformation = np.eye(4)
    current_transformation[:3,:3] = r.as_matrix()
    current_transformation[0:3,3] = pos
    points_wrt_gripper_frame = [[0,0,0], [1.0,0,0], [0,1.0,0], [0,0,1.0]]
    gripper_points_world_frame = transform_points_from_local_frame_to_world(points_wrt_gripper_frame, pos, orn)

    a = [gripper_points_world_frame[1,0] - gripper_points_world_frame[0,0], gripper_points_world_frame[1,1] - gripper_points_world_frame[0,1], gripper_points_world_frame[1,2] - gripper_points_world_frame[0,2]]

    if gripper_axis_to_be_aligned == 1:
        a = [gripper_points_world_frame[2,0] - gripper_points_world_frame[0,0], gripper_points_world_frame[2,1] - gripper_points_world_frame[0,1], gripper_points_world_frame[2,2] - gripper_points_world_frame[0,2]]
   
    if gripper_axis_to_be_aligned == 2:
        a = [gripper_points_world_frame[3,0] - gripper_points_world_frame[0,0], gripper_points_world_frame[3,1] - gripper_points_world_frame[0,1], gripper_points_world_frame[3,2] - gripper_points_world_frame[0,2]]
       
    b = target_axis_wrt_world
    c = cross_product_axb(a, b)
    angle = np.arccos(a[0]*b[0] + a[1]*b[1] + a[2]*b[2])
    r = R.from_rotvec(angle * np.array(c))
    local_transformation = np.eye(4)
    local_transformation[:3,:3] = r.as_matrix()
    local_transformation[:3,3] = pos
    final_transformation = local_transformation@current_transformation
    r = R.from_matrix(final_transformation[:3,:3])
    target_orn = r.as_quat()
    orn = target_orn
    return pos,orn

##############################################_____Reward_generator_____################################################

def get_reward(args,segmentation,total_segmentation_score,segmentation_weight):
	Envs_reward = []

	value, count = np.unique(segmentation,return_counts=True)
		# print('value:',value,'\n','count:',count)
		# print('segmentation_weight:', segmentation_weight)

	segmentation_truth = {}
	normalize_segmentation_truth= {}
	reward = 0
	for j, v in enumerate(value):
		if v==-1:
			segmentation_truth[v] = 0
		else:
			segmentation_truth[v]=count[j]
	normalize_segmentation_truth.update((x, round(y/total_segmentation_score,4)) for x, y in segmentation_truth.items())
	for key in segmentation_weight.keys():
		if key in normalize_segmentation_truth.keys():
			reward = reward + normalize_segmentation_truth[key]*segmentation_weight[key]
		else:
			reward = reward - 0.1 * segmentation_weight[key]
	reward = round(reward,4)
		# print('reward:',reward)
	Envs_reward = reward
	return Envs_reward

#################################_____PyBullet_Enviornmet_setup______##########################################



#####################################_____Camera_setup_____###########################################

def set_camera(args,pos,orn):
	fov = 60
	aspect =  1
	near = 0.02
	far = 1
	projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
		# num_joints = p.getNumJoints(ur10)
	joint_pos, joint_orn, = pos,orn # p.getLinkState(ur10,num_joints-1,computeForwardKinematics=True)
	rot_matrix = p.getMatrixFromQuaternion(joint_orn)
	rot_matrix = np.array(rot_matrix).reshape(3, 3)

	init_camera_vector = (1,0,0) # directly looking downward toward table
	init_up_vector = (0,0,1) # top of the camera along x axis

	camera_vector = rot_matrix.dot(init_camera_vector)
	up_vector = rot_matrix.dot(init_up_vector)

	view_matrix = p.computeViewMatrix(joint_pos, joint_pos+0.01*camera_vector, up_vector)
	width,height,rgb,depth,segmentation =  p.getCameraImage(args.width, args.height, view_matrix, projection_matrix)
	segmentation[segmentation<3]=-1

	return segmentation, rgb
####################################______Exploratory_WorkSpace_Setup_For_RL_____################################

def Act_on_Env(args,home_pos,home_orn,Envs_action,Max_score,segmen_weight):
	pi = 3.141
	Theta = pi
	Phi = 2*pi
	Beta = pi		


	theta = Theta*Envs_action[0]
	phi = pi - Phi*Envs_action[1]
	beta = Beta*Envs_action[2]
		# num_joints = p.getNumJoints(ur10)
		# joint_pos, joint_orn, *_=  p.getLinkState(ur10,num_joints-1,computeForwardKinematics=True)
	radius = home_pos[2]
	new_pos = [home_pos[0]+radius*math.sin(theta)*math.sin(phi),home_pos[1]+radius*math.sin(theta)*math.cos(phi),radius*math.cos(theta)]
	new_orn = p.getQuaternionFromEuler([-pi/2-theta,0,beta])

	final_pos = [home_pos[0],home_pos[1],0.625]

	dir_x = final_pos[0] - new_pos[0]
	dir_y = final_pos[1] - new_pos[1]
	dir_z = final_pos[2] - new_pos[2]
			
	dis = np.sqrt(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z)
	dir_x /= dis
	dir_y /= dis
	dir_z /= dis

	pos, orn = align_gripper_axis_along_a_specific_axis(new_pos,new_orn, 0, [dir_x, dir_y, dir_z])
		# joint_values = p.calculateInverseKinematics(ur10,num_joints-1,pos,orn)
		# if n== args.env-1:
		# 	visualize_frame(p,pos, orn, frame_length=0.625)
		# force = 0.01  
		# tolerance = 0.0001
		# while True: 
		# 	for i in range(6):
		# 		p.setJointMotorControl2(ur10, i, p.POSITION_CONTROL, joint_values[i], force)

		# 	p.stepSimulation()
		# 	current_joint_values = []
		# 	for i in range (6):
		# 		jointPosition, *_ = p.getJointState(ur10, i)
		# 		current_joint_values.append(jointPosition)
 
		# 	err = (np.abs(np.array(current_joint_values) - np.array(joint_values))).sum()
		# 	if err < tolerance:
		# 		break
		# if p.setTimeOut(1):
		# 	reset_env()

	segmentation,rgb =  set_camera(args,pos,orn)
	reward = get_reward(args,segmentation,Max_score,segmen_weight)
	return reward,rgb

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)   # everything should fall down
p.setTimeStep(0.0001)       # this slows everything down, but let's be accurate...
p.setRealTimeSimulation(0)  # we want to be faster than real time :)

planeId = p.loadURDF("plane.urdf", [0, 0, 0])

pos = [0.4,0,0.0] ## Red line is x axis , Green line is y axis, Blue line is z axis 
orn = [0,0,0,1]
fix_plane = p.loadURDF("table/table.urdf", basePosition=pos, baseOrientation=orn, useFixedBase=1)

		# pos = [0,0,0.625]
		# orn = [0,0,0,1]
		# ur10 = p.loadURDF("./ur10.urdf", basePosition=pos, baseOrientation=orn, useFixedBase=1)
		# num_joints = p.getNumJoints(ur10)
		# home_position_angles = [0,-1.57079, 1.57079, -1.57079,-1.57079,0] ##-0.529

		# force = 0.01  
		# tolerance = 0.0001

		# while True: 
		# 	for i in range(6):
		# 		p.setJointMotorControl2(ur10, i, p.POSITION_CONTROL, home_position_angles[i], force)

		# 	p.stepSimulation()
		# 	current_joint_values = []
		# 	for i in range (6):
		# 		jointPosition, *_ = p.getJointState(ur10, i)
		# 		current_joint_values.append(jointPosition)
 
		# 	err = (np.abs(np.array(current_joint_values) - np.array(home_position_angles))).sum()
		# 	if err < tolerance:
		# 		break
home_pos= (0.6880142924334216, 0.16391079010116585, 1.272083995434558) 
home_orn = (2.5575477154815562e-05, 0.7071200906333197, -1.4091309770647433e-05, 0.7070934708863147)

pos = [home_pos[0],home_pos[1],0.625]
orn = [0,0,0,1]
p.loadURDF("./pr_assets-master/data/objects/block_bin.urdf", basePosition=pos, baseOrientation=orn, useFixedBase=1)

objects = ['Banana','Pear','TomatoSoupCan','Scissors','ChipsCan','Strawberry','TennisBall','CrackerBox']
objects_id = {}

for i, Object in enumerate(objects):
	path = './ycb_objects/'+'Ycb'+Object+'/model.urdf'
	pos = [home_pos[0],home_pos[1],0.85]
	orn = [random.random(),random.random(),random.random(),random.random()]
	object_id = p.loadURDF(path, basePosition=pos, baseOrientation=orn, useFixedBase=0)
	objects_id[object_id] = Object
			# print(objects_id)
	for i in range(20000):
		p.stepSimulation()

Max_R, segmen_weight,normalize_gt,Max_score = ground_truth()
segmentation, rgb = set_camera(args,home_pos,home_orn)
Envs_reward = get_reward(args,segmentation,Max_score,segmen_weight)
count =0
# Reward_1 = sum(Envs_reward)/len(Envs_reward)
for i in range(100):
	Envs_action = [random.random(),random.random(),random.random()]
	Reward_1,rgb_1=  Act_on_Env(args,home_pos,home_orn,Envs_action,Max_score,segmen_weight)
	# print('Reward_at_start:',round(Reward_1.mean(),4))
	Envs_action = [random.random(),random.random(),random.random()]
			
	Reward_2,rgb_2=  Act_on_Env(args,home_pos,home_orn,Envs_action,Max_score,segmen_weight)
	if Reward_2>Reward_1+0.1:
		count +=1
		print(count,Reward_1,Reward_2)
	# print("Reward_after_action:", round(Reward_2.mean(),4))	
	# plt.subplot(121)
	# plt.imshow(rgb_1)
	# plt.xlabel('start') 
	# plt.title("reward:"+str(round(Reward_1,4)))
	# plt.subplot(122)
	# plt.imshow(rgb_2)
	# plt.xlabel('itr_1')
	# plt.title("reward:"+str(round(Reward_2,4)))
	# # plt.show()
	# name = 'roll_eval_' +str(i+1)
	# plt.savefig(name)