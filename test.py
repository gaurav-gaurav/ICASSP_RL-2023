import pybullet_utils.bullet_client as bc
import pybullet
import pybullet_data
bullet_env = {}


p0 = bc.BulletClient(connection_mode=pybullet.DIRECT)
p0.setAdditionalSearchPath(pybullet_data.getDataPath())

p1 = bc.BulletClient(connection_mode=pybullet.DIRECT)
p1.setAdditionalSearchPath(pybullet_data.getDataPath())

#can also connect using different modes, GUI, SHARED_MEMORY, TCP, UDP, SHARED_MEMORY_SERVER, GUI_SERVER
pgui = bc.BulletClient(connection_mode=pybullet.GUI)

p0.loadURDF("r2d2.urdf")
p1.loadSDF("stadium.sdf")
print(p0._client)
print(p1._client)
print("p0.getNumBodies()=",p0.getNumBodies())
print("p1.getNumBodies()=",p1.getNumBodies())
while (1):
        p0.stepSimulation()
        p1.stepSimulation()
        pgui.stepSimulation()
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
r2d2 = p.loadURDF("r2d2.urdf", [0, 0, 1])
for l in range(p.getNumJoints(r2d2)):
  print(p.getJointInfo(r2d2, l))

p.loadURDF("r2d2.urdf", [2, 0, 1])
p.loadURDF("r2d2.urdf", [4, 0, 1])

# p.getCameraImage(320, 200, flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
# segLinkIndex = 1
# verbose = 0

# while (1):
#   keys = p.getKeyboardEvents()
#   #for k in keys:
#   # print("key=",k,"state=", keys[k])
#   if ord('1') in keys:
#     state = keys[ord('1')]
#     if (state & p.KEY_WAS_RELEASED):
#       verbose = 1 - verbose
#   if ord('s') in keys:
#     state = keys[ord('s')]
#     if (state & p.KEY_WAS_RELEASED):
#       segLinkIndex = 1 - segLinkIndex
#       #print("segLinkIndex=",segLinkIndex)
#   flags = 0
#   if (segLinkIndex):
#     flags = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX

#   img = p.getCameraImage(320, 200, flags=flags)
#   #print(img[0],img[1])
#   seg = img[4]
#   if (verbose):
#     for pixel in seg:
#       if (pixel >= 0):
#         obUid = pixel & ((1 << 24) - 1)
#         linkIndex = (pixel >> 24) - 1
#         print("obUid=", obUid, "linkIndex=", linkIndex)

#   p.stepSimulation()
# import numpy as np
# import pybullet as p
# import pybullet_data

# p.connect(p.GUI)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())

# p.setGravity(0, 0, -10)
# p.setTimeStep(0.01)

# # Add plane
# plane_id = p.loadURDF("plane.urdf")

# # Add kuka bot
# start_pos = [0, 0, 0.001]
# start_orientation = p.getQuaternionFromEuler([0, 0, 0])
# kuka_id = p.loadURDF("./kuka_iiwa/model.urdf", start_pos, start_orientation)

# fov, aspect, nearplane, farplane = 60, 1.0, 0.02, 1
# projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, nearplane, farplane)
# def kuka_camera():
#     # Center of mass position and orientation (of link-7)
#     com_p, com_o, _, _, _, _ = p.getLinkState(kuka_id, 6)
#     rot_matrix = p.getMatrixFromQuaternion(com_o)
#     rot_matrix = np.array(rot_matrix).reshape(3, 3)
#     # Initial vectors
#     init_camera_vector = (0, 0, 1) # z-axis
#     init_up_vector = (0, 1, 0) # y-axis
#     # Rotated vectors
#     camera_vector = rot_matrix.dot(init_camera_vector)
#     up_vector = rot_matrix.dot(init_up_vector)
#     view_matrix = p.computeViewMatrix(com_p, com_p + 0.2 * camera_vector, up_vector)
#     img = p.getCameraImage(1000, 1000, view_matrix, projection_matrix)
#     return img

# # Main loop
# while True:
#     p.stepSimulation()
#     kuka_camera()