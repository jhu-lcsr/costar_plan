import task_tree_search.road_world as rw
import numpy as np

'''
Used for road world scenarios: add extra cars
'''
def hz_rw_add_other_cars(w,T,rx=220,lateral=False,traj=False):
  if not lateral:
    sc2 = rw.RoadState(w, rx+40,100,0,np.pi/2)
    sc3 = rw.RoadState(w, rx+30,30,0,np.pi/2)
    sc4 = rw.RoadState(w, rx+50,50,6,np.pi/2)
    if traj:
      sc2 = rw.planning.RoadTrajectory([sc2])
      sc3 = rw.planning.RoadTrajectory([sc3])
      sc4 = rw.planning.RoadTrajectory([sc4])
      policy = rw.planning.PlanningRoadSpeedPolicy(speed=w.speed_limit,
        max_acc=1.0,
        noise=0.5)
    else:
      policy=rw.core.RoadWorldRandomAccPolicy()
  else:
    sc2 = rw.core.LateralState(w,x=rx+40,y=100,v=0,w=np.pi/2,hz=False)
    sc3 = rw.core.LateralState(w,x=rx+30,y=30,v=0,w=np.pi/2,hz=False)
    sc4 = rw.core.LateralState(w,x=rx+50,y=50,v=6,w=np.pi/2,hz=False)
    if traj:
      sc2 = rw.planning.LateralTrajectory([sc2])
      sc3 = rw.planning.LateralTrajectory([sc3])
      sc4 = rw.planning.LateralTrajectory([sc4])
      policy = rw.planning.PlanningLateralSpeedPolicy(speed=w.speed_limit,
        max_acc=1.0,
        noise=0.5)
    else:
      policy = rw.core.LateralRandomAccPolicy()
  car2 = rw.core.RoadActor(sc2,policy,T)
  car3 = rw.core.RoadActor(sc3,policy,T)
  car4 = rw.core.RoadActor(sc4,policy,T)
  car2.color = (0,0,255)
  car3.color = (0,0,255)
  car4.color = (0,0,255)
  w.addActor(car2);
  w.addActor(car3);
  w.addActor(car4);
