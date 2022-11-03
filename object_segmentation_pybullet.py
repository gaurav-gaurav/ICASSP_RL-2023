import pybullet
import pybullet_data
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import random 
import math
import csv
from scipy.spatial.transform import Rotation as R
from collections import Counter
import pandas as pd
import torch
import torch.nn as nn 
from torchvision import models
from torchsummary import summary
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torchvision import transforms as T
import pybullet_utils.bullet_client as bc
import argparse
import os.path as osp
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument('--env', default = 1)
parser.add_argument('--width',default= 1000)
parser.add_argument('--height',default =1000)
parser.add_argument('--epoch', type=int, default=40, help="maximum epoch for training")
parser.add_argument('--episode', type=int, default=1000, help="maximum episode per epoch")
parser.add_argument('--save-dir', type=str, default='log/baseline', help="path to save output (default: 'log/')")
parser.add_argument('--resume', type=str, default='', help="path to resume file")
parser.add_argument('--eval', type=str, default='', help="path to resume file")
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

def transform_points_from_local_frame_to_world(p,points_in_local_frame,
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

def visualize_frame(p,pos, orn, frame_length=1):
    points_wrt_local_frame = [[0,0,0], [frame_length,0,0], [0,frame_length,0], [0,0,frame_length]]
    pnts_wrt_world = transform_points_from_local_frame_to_world(p,points_wrt_local_frame, pos, orn)

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

def align_gripper_axis_along_a_specific_axis(p,new_pos,new_orn, gripper_axis_to_be_aligned, target_axis_wrt_world):
    orn = new_orn
    pos = new_pos	
    r = R.from_quat(orn)
    current_transformation = np.eye(4)
    current_transformation[:3,:3] = r.as_matrix()
    current_transformation[0:3,3] = pos
    points_wrt_gripper_frame = [[0,0,0], [1.0,0,0], [0,1.0,0], [0,0,1.0]]
    gripper_points_world_frame = transform_points_from_local_frame_to_world(p,points_wrt_gripper_frame, pos, orn)

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

	for i in range(segmentation.shape[0]):
		value, count = np.unique(segmentation[i,...],return_counts=True)
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
		Envs_reward.append(reward)
	return Envs_reward

###################################______Get_Features_From_RGB_Image_____#######################################

def get_features(args,Envs_rgb):
	Envs_features = []
	model = models.resnet50(pretrained=True)
	newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))
	newmodel.to('cuda:0')
	newmodel.eval()
	preprocess = T.Compose([T.ToTensor()])
	for i in range(Envs_rgb.shape[0]):
		img = Envs_rgb[i,:,:,:]
		Img = Image.fromarray(img,mode ='RGBA').convert('RGB')
		Img = preprocess(Img)
		Img = torch.unsqueeze(Img, 0).to('cuda:0')

		features = newmodel(Img)
		features = torch.flatten(features).cpu().tolist()
		Envs_features.append(features)
	return Envs_features

#################################_____PyBullet_Enviornmet_setup______##########################################

def make_bullet_enviornmet(args,Envs,pybullet):
	for n, p in Envs.items():
		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		p.setGravity(0, 0, -9.81)   # everything should fall down
		p.setTimeStep(0.0001)       # this slows everything down, but let's be accurate...
		p.setRealTimeSimulation(0)  # we want to be faster than real time :)

		planeId = p.loadURDF("plane.urdf", [0, 0, 0])

		pos = [0.4,0,0.0] ## Red line is x axis , Green line is y axis, Blue line is z axis 
		orn = [0,0,0,1]
		fix_plane = p.loadURDF("table/table.urdf", basePosition=pos, baseOrientation=orn, useFixedBase=1)

		pos = [0,0,0.625]
		orn = [0,0,0,1]
		ur10 = p.loadURDF("./ur10.urdf", basePosition=pos, baseOrientation=orn, useFixedBase=1)
		num_joints = p.getNumJoints(ur10)
		home_position_angles = [0,-1.57079, 1.57079, -1.57079,-1.57079,0] ##-0.529

		force = 0.01  
		tolerance = 0.0001

		while True: 
			for i in range(6):
				p.setJointMotorControl2(ur10, i, p.POSITION_CONTROL, home_position_angles[i], force)

			p.stepSimulation()
			current_joint_values = []
			for i in range (6):
				jointPosition, *_ = p.getJointState(ur10, i)
				current_joint_values.append(jointPosition)
 
			err = (np.abs(np.array(current_joint_values) - np.array(home_position_angles))).sum()
			if err < tolerance:
				break

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
	return Envs,home_pos,home_orn

#####################################_____Camera_setup_____###########################################

def set_camera(args,Envs,pos,orn):
	Envs_segmentation = np.empty((args.env,args.width,args.height))
	Envs_rgb = np.empty((args.env,args.width,args.height,4))
	for n,p in enumerate(Envs.values()):
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

		plt.subplot(121)
		plt.imshow(rgb)
		plt.title("RGB Image of Clutter")
		plt.subplot(122)
		plt.imshow(segmentation)
		plt.title("Masked Image of Clutter")
		plt.show()

		segmentation[segmentation<3]=-1
		Envs_segmentation[n,...] = segmentation
		Envs_rgb[n,...] = rgb
	return Envs_segmentation, Envs_rgb

##############################################______RL_Agent_____##############################################

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

####################################______Exploratory_WorkSpace_Setup_For_RL_____################################

def Act_on_Env(args,Envs,home_pos,home_orn,Envs_action,Max_score,segmen_weight):
	pi = 3.141
	Theta = pi/4
	Phi = -pi
	Beta = pi		
	Envs_action = Envs_action.detach().squeeze()
	# print(Envs_action)
	for n,p in enumerate(Envs.values()):

		theta = Theta - Theta*Envs_action[n,0]
		phi = Phi*Envs_action[n,1]
		beta = Beta*Envs_action[n,2]
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

		pos, orn = align_gripper_axis_along_a_specific_axis(p,new_pos,new_orn, 0, [dir_x, dir_y, dir_z])
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

	Envs_segmentation,Envs_rgb =  set_camera(args,Envs,pos,orn)
	Envs_reward = get_reward(args,Envs_segmentation,Max_score,segmen_weight)
	Envs_reward = sum(Envs_reward)/len(Envs_reward)
	return Envs_reward
##########################################_____Reset_Enviornment_____################################

def reset_env(Envs):	
	for n,p in Envs.items():
		p.resetSimulation()

###########################################_____MAIN_MODULE_____#######################################
def main(args):
	bullet_env = {}

	for i in range(args.env):
		p = str('p')+str(i)
		if i== args.env-1:
			bullet_env[p] = bc.BulletClient(connection_mode=pybullet.GUI)
		else:
			bullet_env[p] = bc.BulletClient(connection_mode=pybullet.DIRECT)

	Max_R, segmen_weight,normalize_gt,Max_score = ground_truth()
	model_itr =0
	
	model = Net()
	model.to('cuda:0')
	optimizer = optim.Adam(model.parameters(), lr=0.001)
	writer = SummaryWriter('log/baseline')
	baseline =0

	if args.resume:
		PATH = args.resume
		checkpoint = torch.load(PATH)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		epoch = checkpoint['epoch']
		loss = checkpoint['loss']
		model.train()

	if args.eval:
		PATH = args.eval
		checkpoint = torch.load(PATH)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		model_itr = checkpoint['epoch']
		loss = checkpoint['loss']
		model.eval()
	
	for i in range(args.epoch):
		Envs,home_pos,home_orn = make_bullet_enviornmet(args,bullet_env,pybullet)
		Envs_segmentation, Envs_rgb = set_camera(args,Envs,home_pos,home_orn)

		Envs_features = get_features(args,Envs_rgb)
		Envs_features = torch.FloatTensor(Envs_features).unsqueeze(0).to('cuda:0')

		for j in range(args.episode):
			Envs_action_mean = model(Envs_features)
			Envs_action_dist = Normal(loc = Envs_action_mean, scale= torch.ones([1,3]).cuda(0))

			Envs_action = Envs_action_dist.sample()
			Envs_action = torch.clamp(Envs_action, min =0, max =1)
			Envs_log_probs = Envs_action_dist.log_prob(Envs_action).to('cuda:0')
			
			Reward =  Act_on_Env(args,Envs,home_pos,home_orn,Envs_action,Max_score,segmen_weight)
			expected_reward = Envs_log_probs.mean()*(Reward.mean()-baseline)
			baseline = 0.9 * baselines + 0.1 * Reward.mean()
			cost = -expected_reward

			optimizer.zero_grad()
			cost.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
			optimizer.step()
			model_itr+=1
			print('Mean_rewad for itr:{}'.format(model_itr), round(Reward.mean(),4))

			writer.add_scalar("Loss/train",Reward.mean,model_itr)

			if model_itr%100==0:
				model_save_path = osp.join(args.save_dir, 'model_itr' + str(model_itr) + '.pt')
				torch.save({
            		'epoch': model_itr,
            		'model_state_dict': model.state_dict(),
            		'optimizer_state_dict': optimizer.state_dict(),
            		'loss': Reward.mean(),
            		}, model_save_path)
			print('\n','Model_save_after itr:',model_itr)

		reset_env(Envs)
		print('Enviornment_Reset')
	writer.flush()
	writer.close()

if __name__ == '__main__':
    main(args)
