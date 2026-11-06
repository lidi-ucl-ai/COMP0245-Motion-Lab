"""Offline teacher calibration to the existing Bullet physics; no physics edits."""
import numpy as np
import pinocchio as pin

def align_missing_inertias(env):
 if getattr(env,'_missing_inertias_aligned',False):return
 names={'robot_base_footprint','robot_base_docking_contact_link','robot_arm_base_link','robot_j2s7s300_end_effector'};rows=[]
 links={env.client.getBodyInfo(env.rid)[0].decode():-1}
 for jid in range(env.client.getNumJoints(env.rid)):links[env.client.getJointInfo(env.rid,jid)[12].decode()]=jid
 for name in sorted(names):
  assert name in links,name
  d=env.client.getDynamicsInfo(env.rid,links[name]);frame=env.dyn.pin_model.frames[env.dyn.pin_model.getFrameId(name)]
  rotation=np.asarray(env.client.getMatrixFromQuaternion(d[4])).reshape(3,3);inertia=rotation@np.diag(d[2])@rotation.T
  body=pin.Inertia(d[0],np.asarray(d[3]),inertia)
  env.dyn.pin_model.appendBodyToJoint(frame.parentJoint,body,frame.placement)
  rows.append(dict(link=name,mass=d[0],inertia_diagonal=list(d[2])))
 env.dyn.pin_data=env.dyn.pin_model.createData();env._missing_inertias_aligned=True;env._inertia_alignment=rows
 return rows
