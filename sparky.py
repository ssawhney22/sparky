import sys
from pynput import keyboard
import math
import mujoco
import mujoco.viewer
import numpy as np
import asyncio
import time as tm
import matplotlib.pyplot as plt

# import moteus
# import moteus_pi3hat


class Servo:
    def __init__(self, id):
        self.id = id
        self.defaultAngles = np.array([0, -30, 70])
        #self.set_target_angle(self.defaultAngles[self.id%3],False)
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


class Moteus:
    def __init__(self):
        print("Moteus Setup")
        self.inputAngles = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.moteusAngles = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.defaultVelocities = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        self.velocities = self.defaultVelocities
        self.settingAngle = False
        # self.transport = moteus_pi3hat.Pi3HatRouter()
        self.controllerArray = []

    # for i in range(0,5):
    # self.controllerArray.append(moteus.Controller(id=i+1,transport=self.transport))
    # command = "sudo moteus_tool -t 1 --dump-config > configfile.cfg"
    # result = subprocess.run(command, shell=True, check =True, text=True)
    # self.edit_cfg_file("configfile.cfg", 1389, "id.id 1")
    # self.loop = asyncio.get_event_loop()
    # self.task = self.loop.create_task(self.sendAngle())
    # self.loop.run_until_complete(self.task)

    def getInputAngles(self, legID):
        if legID == 1:
            return self.inputAngles[:3]
        else:
            return self.inputAngles[-3:]

    def setAngle(self, targetAngles, legID, velocity=np.nan, use_rad=False):
        targetAnglesAdjusted = [0, 0, 0]
        for i, angle in enumerate(targetAngles):
            factor = 180 / np.pi if use_rad else 1
            targetAnglesAdjusted[i] = (angle * factor) * 10 % 360.0
            if i == 2:
                targetAnglesAdjusted[i] = (angle * factor) * 10 * 22 / 32 % 360.0

        if velocity.any() != np.nan:
            print("velocity")
            if legID == 1:
                self.velocities[:3] = velocity
            else:
                self.velocities[-3:] = velocity
        else:
            if not np.equal(self.velocities, self.defaultVelocities):
                if legID == 1:
                    self.velocities[:3] = self.defaultVelocities[:3]
                else:
                    self.velocities[-3:] = self.defaultVelocities[-3:]
        if legID == 1:
            self.inputAngles[:3] = targetAngles
            self.moteusAngles[:3] = targetAnglesAdjusted
        else:
            self.inputAngles[-3:] = targetAngles
            self.moteusAngles[-3:] = targetAnglesAdjusted

        return self.inputAngles

    def edit_cfg_file(self, filename, line_number, new_content):
        try:
            with open(filename, "r") as file:
                lines = file.readlines()

            # Check if the line number is valid
            if 1 <= line_number <= len(lines):
                lines[line_number - 1] = new_content + "\n"

                with open(filename, "w") as file:
                    file.writelines(lines)

                print(f"Successfully edited line {line_number} in {filename}")
            else:
                print(
                    f"Error: Line number {line_number} is out of range for {filename}"
                )

        except FileNotFoundError:
            print(f"Error: File '{filename}' not found")
        except Exception as e:
            print(f"An error occurred: {e}")

    # async def sendAngle(self):
    #     """
    #     set position in degrees
    #     """
    #     for controller in self.controllerArray:
    #         await controller.set_stop()
    #         while not self.settingAngle:
    #             state = await controller.set_position(
    #                 kp_scale=.5,
    #                 kd_scale=.03,
    #                 position=math.nan,
    #                 velocity=2,
    #                 stop_position=15,
    #                 watchdog_timeout=math.nan,
    #                 query=True)
    #             print(state)
    #             await asyncio.sleep(0.02)
    #     return state

    # for i,controller in enumerate(self.controllerArray):
    #         await controller.set_stop()
    #         while not self.settingAngle:
    #             state = await controller.set_position(
    #                 kp_scale=.5,
    #                 kd_scale=.03,
    #                 position=math.nan,
    #                 velocity=self.velocity[i],
    #                 stop_position=self.moteusAngles[i],
    #                 watchdog_timeout=math.nan,
    #                 query=True)
    #             print(state)
    #             await asyncio.sleep(0.02)
    #     return state


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
    KNEE_GEAR_RATIO = 1.5

    def __init__(
        self, name, abductor_id, hip_id, knee_id, z_dir_forward, y_dir_to_right, defaultAngle
    ):
        print("Setting Up Leg: ", name)
        self.name = name
        self.abductor = Servo(abductor_id)
        self.hip = Servo(hip_id)
        self.knee = Servo(knee_id)
        self.offset_forward = z_dir_forward
        self.offset_right = y_dir_to_right
        self.defaultPos = self.fk(defaultAngle)

    def ik(self, position, retRad=False):
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
        if self.name == "FR" or self.name == "BL":
            t1 = np.arctan(b / r2) - np.arctan(x / y)
        else:
            t1 = np.arctan(b / r2) + np.arctan(x / y)

        t2 = np.arctan((z - r1) / b) - np.arccos(
            np.clip((r4 * r4 - c * c - r3 * r3) / (-2 * c * r3), -1, 1)
        )
        t3 = np.pi - np.arccos(
            np.clip((c * c - r3 * r3 - r4 * r4) / (-2 * r3 * r4), -1, 1)
        )

        if retRad:
            return np.array([t1, t2, t3])
        else:
            return np.array([180 / np.pi * t1, 180 / np.pi * t2, 180 / np.pi * t3])

    def fk(self, currAngles, inputRad=False):
        t1 = currAngles[0]
        t2 = currAngles[1]
        t3 = currAngles[2]
        if not inputRad:
            t1 = currAngles[0] * np.pi / 180
            t2 = currAngles[1] * np.pi / 180
            t3 = currAngles[2] * np.pi / 180
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
        if self.name == "FR" or self.name == "BL":
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

        return [x, y, z]

    def setServos(self, angles, use_rad=False):
        self.abductor.set_target_angle(angles[0], use_rad)
        self.hip.set_target_angle(angles[1], use_rad)
        self.knee.set_target_angle(angles[2] * self.KNEE_GEAR_RATIO, use_rad)

    def getServos(self, use_rad=False):
        t1 = self.abductor.get_angle(use_rad)  # abductor_rad
        t2 = self.hip.get_angle(use_rad)  # hip_rad
        t3 = self.knee.get_angle(use_rad)  # knee_rad
        return np.array([t1, t2, t3])
    def getPos(self):
        return np.array(self.fk(self.getServos(False),False))
    def is_grounded(self):
        # determines if a leg is grounded based on position and torques

        return


class Quad:
    def __init__(self):
        # create the leg objects
        self.defaultAngle = np.array([0, -15, 60])
        self.FR_leg = Leg("FR", 3, 4, 5, True, True,self.defaultAngle)
        self.FL_leg = Leg("FL", 0, 1, 2, True, False,self.defaultAngle)
        self.BR_leg = Leg("BR", 9, 10, 11, False, True,self.defaultAngle)
        self.BL_leg = Leg("BL", 6, 7, 8, False, False,self.defaultAngle)
        self.state = 100
        self.legToMove = 0
        self.legs = [self.FR_leg, self.FL_leg, self.BR_leg, self.BL_leg]
        # self.controllers = Moteus()
        #self.legs = [self.FL_leg, self.FR_leg]
        self.maxVel = 15
        self.maxAccel= 60
        
        return

    def setLeg(self, legID, arr, fk=False, use_rad=True):
        # Set all legs to same value, either set angles or position
        limb = self.legs[legID]
        if fk:
            limb.setServos(arr, use_rad)
            #self.controllers.setAngle(arr, i, False)
        else:
            [t1, t2, t3] = limb.ik(arr)
           # self.controllers.setAngle([t1, t2, t3], i, False)
            limb.setServos([t1, t2, t3], use_rad)
            print("%.4f, %.4f, %.4f" % (t1, t2, t3))
        return

    def getLegAngles(self, retRad=False):
        return np.array(self.controllers.inputAngles).reshape(2, 3)

    def pvt(self, legID, targetAngles, sampleRate, duration, rampTime, useRad=False):
        targetAngles = np.array(targetAngles)
        currAngles = np.array(
            self.controllers.inputAngles[:3]
            if legID == 1
            else self.controllers.inputAngles[-3:]
        )
        difference = np.subtract(targetAngles, currAngles)
        length = int(duration / sampleRate)
        position = np.zeros([length, 3])
        if np.sum(np.abs(difference)) < 0.001:
            position = np.tile(currAngles, (length, 1))
        else:
            time = np.reshape(np.arange(0, duration, sampleRate), (length, 1))
            ## y=self.maxVel/rampTime*(t)
            velRampUp = np.linspace(0, self.maxVel, rampTime / sampleRate, dtype=float)

            velConst = self.maxVel * np.ones(
                (1 / sampleRate * (duration - 2 * rampTime), 1), dtype=float
            )
            velRampDown = np.linspace(
                self.maxVel, 0, rampTime / sampleRate, dtype=float
            )
            velocity = np.reshape([velRampUp, velConst, velRampDown], (length, 1))
            for i in range(0, 3):
                for j in time:
                    if j == 0:
                        position[j, i] = velocity[j] * sampleRate
                    else:
                        position[j, i] = position[j - 1, i] + velocity[j] * sampleRate

        return position, velocity, time

    def pvt_Sim(self, legID, targetAngles, sampleRate, useRad=False,length = np.nan,currAngles = np.array([np.nan,np.nan,np.nan])):
        radFactor = 1 if useRad else np.pi/180
        targetAngles = radFactor*targetAngles
        #currAngles = np.array([0,0,0])
        if np.isnan(currAngles).all():
            currAngles = self.legs[legID].getServos(True)
        difference = np.subtract(targetAngles, currAngles)
        duration = np.nanmax(np.abs(difference/(self.maxVel))+self.maxVel/self.maxAccel)
        
        
        
        if np.isnan(length):
            length =  int(np.ceil(duration / sampleRate))+1
            #print("No Input Len:",length)
        else:
            #print("Input Len:",length)
            duration = (length-1)*sampleRate
        #print("Duration",duration)
        #print(maxReachVel)
        position = np.zeros((length, 3))
        velocity = np.zeros((length, 3))
        time = np.reshape(np.arange(0, duration+sampleRate, sampleRate), (length, 1))
        for i in range(0, 3):
            if np.abs(difference[i]) < 0.01 or np.isnan(difference[i]):
                velocity[:,i] = np.reshape(np.zeros((length,1)),(length,))
                position[:,i] = np.reshape(np.tile(currAngles[i], (length, 1)),(length,))
            else:
                maxReachVel = (duration*self.maxAccel-np.sqrt(duration*duration*self.maxAccel*self.maxAccel-4*difference[i]*self.maxAccel))/(2)
                rampLength = min(int(np.abs((maxReachVel/self.maxAccel)/sampleRate)),int(np.abs((self.maxVel/self.maxAccel)/sampleRate)))
                
                dirFactor = -1 if difference[i] < 0 else 1
                accel = dirFactor*np.concatenate((self.maxAccel*np.ones((1,rampLength+1),dtype=float)[0,:],np.zeros((1,length-2*rampLength-1),dtype=float)[0,:],-1*self.maxAccel*np.ones((1,rampLength),dtype=float)[0,:]))
                for index, j in enumerate(time):
                    if(index ==0 ):
                        position[index,:] = currAngles
                    else:
                        velocity[index,i] = velocity[index-1,i]+accel[index]*sampleRate
                        position[index, i] = ( position[index - 1, i] + velocity[index,i] * sampleRate)
                


        #print("Max Vel",np.amax(np.abs(velocity),axis=0))
        print("Final Angles",180/np.pi*position[-1])
        #print("Final Position",self.legs[legID].fk(position[-1],False))
        # plt.figure()
        # plt.plot(time,velocity[:,0])
        # plt.plot(time,velocity[:,1])
        # plt.plot(time,velocity[:,2])
        # plt.show()
            
        return np.reshape(position,(length,3)), np.reshape(velocity,(length,3)), np.reshape(time,(length,1))
    def creepStep(self, legID, fHeight, fDistance, hDist,sampleRate):
        legID=self.legToMove
        print("\n\nStep", self.state, " Leg",legID)
        print("Current Leg Angles",self.legs[legID].getServos(False))
        
        if self.state >= 100:    
            print("Set Up")
            pos1 =  self.legs[legID].ik(np.add(self.legs[legID].defaultPos, [0, 0, fDistance]),True)
            pos2=  self.legs[legID].ik(np.add(self.legs[legID].defaultPos, [0, 0, int(fDistance*2/3)]),True)
            pos3=  self.legs[legID].ik(np.add(self.legs[legID].defaultPos, [0, 0, int(fDistance*1/3)]),True)
            pos4=  self.legs[legID].ik(self.legs[legID].defaultPos,True)
            position, velocity, time = self.pvt_Sim(legID,pos1, sampleRate,True) 
            position2,velocity2,time2 = self.pvt_Sim((legID + 1)%4,pos2, sampleRate,True,len(time))
            position3,velocity3,time3 = self.pvt_Sim((legID + 2)%4,pos3, sampleRate,True,len(time))
            position4,velocity4,time4 = self.pvt_Sim((legID + 3)%4,pos4, sampleRate,True,len(time))
            for i in range(0, len(position)):
                mujoco.mj_step(model, data)
                
                self.legs[0].setServos(position[i,:],True)
                self.legs[1].setServos(position2[i,:],True)
                self.legs[2].setServos(position3[i,:],True)
                self.legs[3].setServos(position4[i,:],True)
                
                viewer.sync()
                tm.sleep(sampleRate)
            self.legToMove = (legID + 1) % 4 
            self.state = (self.state + 1) % 101        
        else: 
            pos1_1 = self.legs[legID].ik(np.add(self.legs[legID].defaultPos, [fHeight, 0, 0]),True)
            pos1_2 =  self.legs[legID].ik(np.add(self.legs[legID].defaultPos, [fHeight, 0, fDistance]),True)
            pos1_3 =  self.legs[legID].ik(np.add(self.legs[legID].defaultPos, [0, 0, fDistance]),True)
            pos2=  self.legs[legID].ik(np.add(self.legs[legID].defaultPos, [0, 0, int(fDistance*2/3)]),True)
            pos3=  self.legs[legID].ik(np.add(self.legs[legID].defaultPos, [0, 0, int(fDistance*1/3)]),True)
            pos4=  self.legs[legID].ik(self.legs[legID].defaultPos,True)
            print("P1")
            position, velocity, time = self.pvt_Sim(legID,pos1_1, sampleRate,True)
            ptemp, vtemp, ttemp =  self.pvt_Sim(legID,pos1_2, sampleRate,True,int(np.floor(len(time))),pos1_1)
            position = np.vstack((position,ptemp))
            velocity = np.vstack((velocity,vtemp))
            time = np.vstack((time,ttemp))
            print("P2")
            position2,velocity2,time2 = self.pvt_Sim((legID + 1)%4,pos2, sampleRate,True,len(time))
            
            position3,velocity3,time3 = self.pvt_Sim((legID + 2)%4,pos3, sampleRate,True,len(time))
            print("P3", position3)
            print("P4")
            position4,velocity4,time4 = self.pvt_Sim((legID + 3)%4,pos4, sampleRate,True,len(time))
            
            print("targetAngles",180/np.pi*pos1_2 )
            print("targetAngles2",180/np.pi*pos2)
            print("targetAngles3",180/np.pi*pos3)
            print("targetAngles4",180/np.pi*pos4)
            
            
            for i,pos in enumerate(position):
                mujoco.mj_step(model, data)
                self.legs[legID].setServos(position[i,:],True)
                self.legs[(legID + 1)%4].setServos(position2[i,:],True)
                self.legs[(legID + 2)%4].setServos(position3[i,:],True)
                self.legs[(legID + 3)%4].setServos(position4[i,:],True)
                mujoco.mj_step(model, data)
                tm.sleep(sampleRate)
                viewer.sync()
                
                
            self.legToMove = (legID + 1) % 4 
            for i in time:
                mujoco.mj_step(model, data)
                viewer.sync()
            
            #print(self.controllers.inputAngles)
    def step(self, legID, fHeight, fDistance, hDist,sampleRate):
        legID=self.legToMove 
        lastLegID = [0,1,2,3]
        adjacentLegID  = {0: 1, 1: 0, 2: 3, 3: 2}.get(legID)
        supportLegID = (legID + 2) % 4
        lastLegID.remove(legID)
        lastLegID.remove(adjacentLegID)
        lastLegID.remove(supportLegID)
        lastLegID = lastLegID[0]
        print("\n\nStep", self.state, " Leg",legID," Adjacent Leg",adjacentLegID," Support Leg",supportLegID," Last Leg",lastLegID)
        print("Current Leg Angles",self.legs[legID].getServos(False))
        targetAngles = np.nan
        targetAngles2 = np.nan
        targetAngles3 = np.nan
        rearLegFactor = -1 if legID > 1 else 1
        tiltAngle = 10
        match self.state:
            case 0:
                targetAngles = np.pi/180*np.add(self.legs[legID].getServos(False),[0,0,0])
                targetAngles2 = np.pi/180*np.add(self.defaultAngle ,[0,0,tiltAngle])
                targetAngles3 = np.pi/180*self.defaultAngle 
            case 1:
                targetAngles = self.legs[legID].ik(np.add(self.legs[legID].defaultPos, [fHeight, 0, 0]),True)
                targetAngles2 = np.pi/180*np.add(self.defaultAngle ,[0,0,tiltAngle])
                targetAngles3 = np.pi/180*self.defaultAngle 
            case 2:
                
                targetAngles = self.legs[legID].ik(np.add(self.legs[legID].defaultPos, [fHeight, 0, rearLegFactor*fDistance]),True)
                targetAngles2 = np.pi/180*np.add(self.defaultAngle ,[0,0,tiltAngle])
                targetAngles3 = np.pi/180*self.defaultAngle 
            
            case 4:
                
                targetAngles = self.legs[legID].ik(np.add(self.legs[legID].defaultPos, [0, 0, rearLegFactor*fDistance]),True)
                targetAngles2 = np.pi/180*np.add(self.defaultAngle ,[0,0,tiltAngle])
                targetAngles3 = np.pi/180*self.defaultAngle
            case 3:
                
                targetAngles = np.pi/180*self.defaultAngle
                targetAngles2 = np.pi/180*self.defaultAngle
                targetAngles3 = np.pi/180*self.defaultAngle
            case _:
                
                
                print("Error with step")
                # self.controllers.setAngle(self.defaultAngle, 1, 2)
                # self.controllers.setAngle(self.defaultAngle, 2, 2)

        if self.state >= 100:    
            print("Set Up")
            targetAngles = self.defaultAngle
            position, velocity, time = self.pvt_Sim(legID,targetAngles, sampleRate,False)
            position2, velocity2, time2 = self.pvt_Sim(legID,np.add(targetAngles,[0,0,15]), sampleRate,False)
            for i in range(0, len(position)):
                mujoco.mj_step(model, data)
                
                self.legs[0].setServos(position[i,:],True)
                self.legs[1].setServos(position[i,:],True)
                self.legs[2].setServos(position[i,:],True)
                self.legs[3].setServos(position[i,:],True)
                
                viewer.sync()
                tm.sleep(sampleRate/2)
            self.legToMove = 0    
            self.state = (self.state + 1) % 101        
        else: 
            print("\ntargetAngles",180/np.pi*targetAngles)
            position, velocity, time = self.pvt_Sim(legID,targetAngles, sampleRate,True)
            print("targetAngles2",180/np.pi*targetAngles2)
            position2,velocity2,time2 = self.pvt_Sim(adjacentLegID,targetAngles2, sampleRate,True,len(time))
            print("targetAngles3",180/np.pi*targetAngles3)
            position3,velocity3,time3 = self.pvt_Sim(supportLegID,targetAngles3, sampleRate,True,len(time))
            
            for i,pos in enumerate(position):
                mujoco.mj_step(model, data)
                self.legs[legID].setServos(position[i,:],True)
                self.legs[adjacentLegID].setServos(position2[i,:],True)
                self.legs[lastLegID].setServos(position2[i,:],True)
                self.legs[supportLegID].setServos(position3[i,:],True)
                mujoco.mj_step(model, data)
                tm.sleep(sampleRate/2)
                viewer.sync()
            states = 4
            if(self.state < states):
                for i in time:
                    mujoco.mj_step(model, data)
                    viewer.sync()
            
            if(self.state == states-1):
                self.legToMove = (self.legToMove + 1) % 4
            
            self.state = (self.state + 1) % states
            #print(self.controllers.inputAngles)


def on_press(key):
    fHeight = 50
    fDist = 75
    hHeight = 50
    hDist = 50
    sampleRate=0.01
    legMoving = quadruped.legToMove
    try:
        charInput = key.char
        print((key.char))
        match charInput:
            case "w":
                quadruped.step(legMoving,fHeight, fDist, sampleRate)
            case "s":
                quadruped.step(legMoving,fHeight, -1 * fDist, sampleRate)
            case "a":
                quadruped.step(legMoving,hHeight, hDist, sampleRate)
            case "d":
                quadruped.step(legMoving,hHeight, -1 * hDist, sampleRate)
    except AttributeError:
        print("special key {0} pressed".format(key))


# setup quad

quadruped = Quad()
runSim = True
printOutput = False
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)
if runSim:
    # setup scene
    file_path = "scene.xml"
    joint_data_offset = (
        7  # there are 19 data points, 12 are the joints, first 7 aren't joints
    )
    viewer = mujoco.viewer.launch_passive(model, data)
    alpha = -np.pi / 2
    factor = 1
    fHeight = 100
    fDist = 150
    hHeight = 1
    hDist = 20
    sampleRate=0.0005
    legMoving = quadruped.legToMove
    count = 0
    while viewer.is_running():
        #input()
        mujoco.mj_step(model, data)
        
        #     #quadruped.setLegs([-198.8935, 93.6625, 350.8295],False,False)
        #     #quadruped.setLegs([0,0,90],True,False)

        # [t1, t2, t3] = quadruped.legs[0].ik([-222.25 - 244 * np.cos(alpha), 93.6625, 107.95 + 244 * np.sin(alpha)], False)
        # quadruped.legs[0].setServos([t1, t2, t3], False)
    
        quadruped.step(legMoving,fHeight, fDist, hDist, sampleRate)
        count+=1
        

        if printOutput:
            for i, leg in enumerate(quadruped.legs):
                if i == 0:
                    [t1, t2, t3] = leg.getServos()
                    [x, y, z] = leg.fk([t1, t2, t3], False)

                    print(leg.name + " FK Pos: %.4f, %.4f, %.4f" % (x, y, z))
                    print(
                        leg.name
                        + " IK Pos: %.4f, %.4f, %.4f"
                        % (
                            -222.25 - 244 * np.cos(alpha),
                            93.6625,
                            107.95 + 244 * np.sin(alpha),
                        )
                    )

                    print(leg.name + " Curr Angles: %.4f, %.4f, %.4f" % (t1, t2, t3))
                    [t1, t2, t3] = quadruped.legs[0].ik([t1,t2,t3],True)
                    print(leg.name + " IK Angles: %.4f, %.4f, %.4f\n\n " % (t1, t2, t3))

        #alpha = alpha + factor * np.pi / 180
        if np.abs(alpha - np.pi / 2) < 0.1:
            factor = -1
        elif np.abs(alpha + np.pi / 2) < 0.1:
            factor = 1
        viewer.sync()

else:
    print("No Sim")
    np.set_printoptions(threshold=sys.maxsize)
    #quadruped.step(0,100, 200,0.01)
    print(quadruped.legs[1].fk([0, 0, -90],False))
    angles = np.array(quadruped.legs[1].ik([0, 0, 0],True))
    print(angles)
    position, velocity, time = quadruped.pvt_Sim(0,angles, 0.0005,True)
    position2,velocity2,time2 = quadruped.pvt_Sim(2,angles, 0.0005,True,len(time))
    # listener = keyboard.Listener(on_press=on_press)
    # listener.start()
    # while True:
    #     pass


""""



Test Cases 

[-222.25,93.6625,315.95]

[0,90,0]: [0.0,93.6625,574.2]
[-173.1017,442.94, 107.95]
[-392, -267.88, 107.95]
"""
