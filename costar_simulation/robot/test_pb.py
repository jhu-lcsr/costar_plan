import pybullet as p
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setGravity(0,0,-10)
planeId = p.loadURDF("jaco_robot.urdf")

