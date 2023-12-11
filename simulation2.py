import math
import time
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

"""
How to get started:
- install mujoco as a python package. i think its just "pip install mujoco"
- there might be a few other dependencies to insstall. i dont remember anything directly related to the sim being super annoying


Notes on what to do:
- the classes should be broken into different files and a decent file structure should be implemented, i was lazy and wrote this in a night
- my dumb radians shit should probably use math.rad() or whatever that function is
- Leg.go_to() and Leg.get_position() need to be fixed/ implemented. these are the forward and inverse kinematic functions. 
    They should give positions in 'leg space' aka with the 0,0,0 at the abductor joint of the leg.
    The Quad class should probably then implement a function that adds the offsets so that the legs can be positioned in cartesian from the 0,0,0 of the dog (body's center)

"""


class Servo:
    def __init__(self, id):
        self.id = id

    def set_target_angle(self, pos_deg, use_rad=False):
        """
        set position in degrees
        """
        factor = 1 if use_rad else np.pi / 180
        data.ctrl[self.id] = pos_deg * factor

    def get_angle(self, use_rad=True):
        """
        get position in degrees
        """
        factor = 1 if use_rad else 180 / np.pi
        return data.qpos[self.id + joint_data_offset] * factor

    def get_torque(self):
        """
        get torque in Nm (I think its Nm?)
        """
        return data.actuator_force[self.id]


class Leg:

    """
    Legs have a zero position at the point where the abductor pivot line intersects with the hip pivot line

    Body Dim: 300 x 240 x 130 mm
    Body to Leg: 65 x 160 x 0 mm
    """

    ABDUCTOR_TO_HIP_LEN = 107.95
    HIP_TO_UPPER_LEG_LEN = 93.6625
    UPPER_LEG_LEN = 222.25
    LOWER_LEG_LEN = 244

    def __init__(
        self, name, abductor_id, hip_id, knee_id, z_dir_forward, y_dir_to_right
    ):
        self.name = name
        self.abductor = Servo(abductor_id)
        self.hip = Servo(hip_id)
        self.knee = Servo(knee_id)
        self.offset_forward = z_dir_forward
        self.offset_right = y_dir_to_right

    def go_to(self, position, retRad=False):
        # go to a position in leg space
        # leg segment lengths
        r1 = self.ABDUCTOR_TO_HIP_LEN
        r2 = self.HIP_TO_UPPER_LEG_LEN
        r3 = self.UPPER_LEG_LEN
        r4 = self.LOWER_LEG_LEN

        x = position[0] + 1 * 10**-5
        y = position[1] + 1 * 10**-5
        z = position[2] + 1 * 10**-5

        b = np.sqrt(x * x + y * y - r2 * r2)
        c = np.sqrt(b * b + (z - r1) * (z - r1))
        t1 = np.arctan(np.sqrt(b * b) / r2) + np.arctan(x / y)

        t2 = np.arctan((z - r1) / b) - np.arccos(
            np.clip((r4 * r4 - c * c - r3 * r3) / (-2 * c * r3), -1, 1)
        )
        t3 = np.pi - np.arccos(
            np.clip((c * c - r3 * r3 - r4 * r4) / (-2 * r3 * r4), -1, 1)
        )
        self.setServos([t1, t2, t3], True)

        if retRad:
            return [t1, t2, t3]
        else:
            return [180 / np.pi * t1, 180 / np.pi * t2, 180 / np.pi * t3]

    def get_position(self):
        # joint angles in radians
        [t1, t2, t3] = self.getServos(True)
        print(t1, t2, t3)
        # leg segment lengths
        r1 = self.ABDUCTOR_TO_HIP_LEN
        r2 = self.HIP_TO_UPPER_LEG_LEN
        r3 = self.UPPER_LEG_LEN
        r4 = self.LOWER_LEG_LEN
        # Calculate leg segment positions
        x = -1 * (
            r2 * np.sin(t1)
            + r3 * np.cos(t1) * np.cos(t2)
            + r4 * np.cos(t1) * np.cos(t2 + t3)
        )
        if self.name == "FL" or self.name == "BR":
            y = (
                -1 * r2 * np.cos(t1)
                - r3 * np.sin(t1) * np.cos(t2)
                - r4 * np.sin(t1) * np.cos(t2 + t3)
            )
        else:
            y = (
                r2 * np.cos(t1)
                - r3 * np.sin(t1) * np.cos(t2)
                - r4 * np.sin(t1) * np.cos(t2 + t3)
            )
        z = r1 + r3 * np.sin(t2) + r4 * np.sin(t2 + t3)
        return x, y, z

    def setServos(self, angles, use_rad=False):
        self.abductor.set_target_angle(angles[0], use_rad)
        self.hip.set_target_angle(angles[1], use_rad)
        self.knee.set_target_angle(angles[2], use_rad)

    def getServos(self, use_rad=True):
        t1 = self.abductor.get_angle(use_rad)  # abductor_rad
        t2 = self.hip.get_angle(use_rad)  # hip_rad
        t3 = self.knee.get_angle(use_rad)  # knee_rad
        return [t1, t2, t3]

    def is_grounded(self):
        # determines if a leg is grounded based on position and torques

        return


class Quad:
    def __init__(self):
        # create the leg objects
        self.FR_leg = Leg("FR", 3, 4, 5, True, True)
        self.FL_leg = Leg("FL", 0, 1, 2, True, False)
        self.BR_leg = Leg("BR", 9, 10, 11, False, True)
        self.BL_leg = Leg("BL", 6, 7, 8, False, False)

        self.legs = [self.FR_leg, self.FL_leg, self.BR_leg, self.BL_leg]
        self.pos = [0, 0, 401.25]
        self.vel = [0, 0, 0]
        self.rot = [0, 0, 0]
        return

    def setBodyPos(
        self, newPos, newRot=[0, 0, 0], centerOfRot=[0, 0, 0], isDegrees=True
    ):
        print(newRot)
        # rot = [gamma ie roll (x->z), beta ie pitch (y), alpha ie yaw (z->x)]
        alpha = newRot[2]
        beta = newRot[1]
        gamma = newRot[0]
        r = R.from_euler("xyz", [alpha, beta, gamma], degrees=isDegrees)
        r_mat = r.as_matrix()
        center = np.matrix((centerOfRot[0], centerOfRot[1], centerOfRot[2])).reshape(
            3, 1
        )
        footZero = [-466.25, 93.6625, 107.95]
        legMatrix = np.array(
            [[0, 65, 160], [0, -65, 160], [0, 65, -160], [0, -65, -160]]
        ).reshape((4, 3))
        for i, leg in enumerate(self.legs):
            [x, y, z] = leg.get_position()
            # target = np.matmul(np.linalg.inv(r_mat),legMatrix[i].reshape(3,1) - center)
            # footPos = target - legMatrix[i].reshape(3,1) + center
            # newPos = footPos.reshape(1,3) + pos
            goToPos = [-1 * newPos[0], newPos[1] - y, z - newPos[2]]
            print(goToPos)
            if i > 1:
                print(leg.name)
                [t1, t2, t3] = leg.go_to(goToPos)
            else:
                print(i)
                [t1, t2, t3] = leg.go_to(goToPos)
        return r_mat

    def setLegs(self, arr, set_angles=False, use_rad=True):
        # Set all legs to same value, either set angles or position
        for i, limb in enumerate(self.legs):
            if set_angles:
                limb.setServos(arr, use_rad)
            else:
                [t1, t2, t3] = limb.go_to(arr)
                print("%.4f, %.4f, %.4f" % (t1, t2, t3))
        # input()
        return

    def getLegAngles(self, retRad=False):
        angles = np.array([])
        for i, limb in enumerate(self.legs):
            limbAngles = np.array(limb.getServos(retRad)).reshape(3, 1)
            angles = np.append(angles, limbAngles)
        return np.array(angles).reshape(4, 3)

    def controller(self, model, data):
        # TODO: build some sort of controller here. takes in the model and the data and then can control all the legs through the Leg objects
        return


# setup quad
quadruped = Quad()
runSim = True
printOutput = True
# setup scene
file_path = "scene.xml"
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)
# mujoco.set_mjcb_control(quadruped.controller)
joint_data_offset = (
    7  # there are 19 data points, 12 are the joints, first 7 aren't joints
)
if runSim:
    viewer = mujoco.viewer.launch_passive(model, data)
    # quadruped.setBodyPos([300,0,0],[0,0,0],[0,0,0],True)

    # input()
    while viewer.is_running():
        mujoco.mj_step(model, data)

        # quadruped.setLegs([-198.8935, 93.6625, 350.8295],False,False)
        quadruped.setLegs([0, 0, 90], True, False)
        # quadruped.setLegs([-222.25,93.6625,315.95],False,False)
        if printOutput:
            angleArr = quadruped.getLegAngles()
            print(angleArr)
            # print("%.4f, %.4f, %.4f" % (t1, t2, t3))
            [x, y, z] = quadruped.legs[0].get_position()
            print("FR %.4f, %.4f, %.4f" % (x, y, z))
            [x, y, z] = quadruped.legs[1].get_position()
            print("FL %.4f, %.4f, %.4f" % (x, y, z))
            [x, y, z] = quadruped.legs[2].get_position()
            print("BR %.4f, %.4f, %.4f" % (x, y, z))
            [x, y, z] = quadruped.legs[3].get_position()
            print("BL %.4f, %.4f, %.4f" % (x, y, z))
        viewer.sync()

else:
    print("No Sim")

    quadruped.setLegs([-466.25, 93.6625, 107.95], False, False)


""""



Test Cases 

[-222.25,93.6625,315.95]

[0,90,0]: [0.0,93.6625,574.2]
[-173.1017,442.94, 107.95]
[-392, -267.88, 107.95]
"""
