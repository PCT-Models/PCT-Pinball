# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 23:57:14 2021

@authors: Tauseef Gulrez and Warren Mansell
"""

#!/usr/bin/env python
import sys, time
sys.path.append('D:/Anaconda/envs/spyder/Lib/site-packages')
import gym, time, cv2, collections
import numpy as np
from pynput.keyboard import Key, Controller

keyboard = Controller()
key1 = "1"
key2 = "2"
key3 = "3"
key4 = "4"
key5 = "5"

# VideoPinball-v0
# VideoPinball-v4
# VideoPinballDeterministic-v0
# VideoPinballDeterministic-v4
# VideoPinballNoFrameskip-v0
# VideoPinballNoFrameskip-v4
# VideoPinball-ram-v0
# VideoPinball-ram-v4
# VideoPinball-ramDeterministic-v0
# VideoPinball-ramDeterministic-v4
# VideoPinball-ramNoFrameskip-v0
# VideoPinball-ramNoFrameskip-v4
env = gym.make('VideoPinballNoFrameskip-v0' if len(sys.argv)<2 else sys.argv[1])

if not hasattr(env.action_space, 'n'):
    raise Exception('Keyboard agent only supports discrete action spaces')
ACTIONS = env.action_space.n
SKIP_CONTROL = 0    # Use previous control decision SKIP_CONTROL times, that's how you
                    # can test what skip is still usable.


human_agent_action = 0
human_wants_restart = False
human_sets_pause = False

def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    if key==0xff0d: human_wants_restart = True
    if key==32: human_sets_pause = not human_sets_pause
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    human_agent_action = a

def key_release(key, mod):
    global human_agent_action
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    if human_agent_action == a:
        human_agent_action = 0

env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release

# Game Dynamics List of Ball and Paddle Positions
x_balli = collections.deque([0] * 2, maxlen=2)
y_balli = collections.deque([0] * 2, maxlen=2)
x_plate1i = collections.deque([0] * 2, maxlen=2)
y_plate1i = collections.deque([0] * 2, maxlen=2)

x_plate2i = collections.deque([0] * 2, maxlen=2)
y_plate2i = collections.deque([0] * 2, maxlen=2)

rewardsi = collections.deque([0] * 100, maxlen=100)


def rollout(env):
    global human_agent_action, human_wants_restart, human_sets_pause
    human_wants_restart = False
    obser = env.reset()
    skip = 0
    total_reward = 0
    total_timesteps = 0
    nn = 0
    pp = 0
    itr = 0
    while 1:
        if not skip:
            #print("taking action {}".format(human_agent_action))
            a = human_agent_action
            total_timesteps += 1
            skip = SKIP_CONTROL
        else:
            skip -= 1

        obser, r, done, info = env.step(a)
        # if r != 0:
            # print("reward %0.3f" % r)
        total_reward += r
        window_still_open = env.render()
        
        
        
        
        ##%% Image Processing
        img = env.render(mode="rgb_array")
        
        # Continuous Throwing of Ball
        if (nn == 0) or (r == 0):
            # time.sleep(5)
            for iii in range(0,2,1):
                human_agent_action = 5
        nn = nn + 1
        
        if (nn == 5):
            # time.sleep(5)
            for iii in range(0,2,1):
                human_agent_action = 1
                nn = 0
               
        
        
        # Image Processing
        # For Ball to detect crop Image Just under the Bricks
        # yc_ball, xc_ball, hc_ball, wc_ball = 175, 60, 40, 40
        yc_ball, xc_ball, hc_ball, wc_ball = 170, 60, 50, 40
        # Crop Image
        img_ball = img[yc_ball:yc_ball+hc_ball, xc_ball:xc_ball+wc_ball]
         # Image Processing PCT - Convert it to Grayscale
        grayscale_ball = cv2.cvtColor(img_ball, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(grayscale_ball, 100, 255,cv2.THRESH_OTSU)
        bin_ball = binary
        ## Find Contours and Start Algo
        contours_ball = cv2.findContours(bin_ball, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        # print(len(contours_ball))
        
        
        for i in range(0,len(contours_ball)):
            x0, y0, w0, h0 = cv2.boundingRect(contours_ball[i])
            cv2.rectangle(img_ball,(x0,y0), (x0+w0,y0+h0), (0,255,0), 1)
            a, b = round(x0+w0/2), round(y0)
            # cv2.circle(img_ball, (a,b), radius=1, color=(0, 0, 255), thickness=-1)
            # cv2.circle(img_ball, (0,5), radius=1, color=(0, 0, 255), thickness=-1)
            # cv2.circle(img_ball, (20,5), radius=1, color=(0, 255, 0), thickness=-1)
            # cv2.circle(img_ball, (38,5), radius=1, color=(255, 0, 0), thickness=-1)
            
        if len(contours_ball) == 3:
            x1, y1, w1, h1 = cv2.boundingRect(contours_ball[2])
            a1, b1 = round(x1+w1/2), round(y1+h1)
            cv2.circle(img_ball, (a1,b1), radius=1, color=(0, 255, 0), thickness=-1)
            cv2.rectangle(img_ball,(x1,y1), (x1+w1,y1+h1), (0,255,0), 1)
            
        # To Reset the Game with One Contour Only
        if len(contours_ball) == 0:
            print('Nothing is Detecte Something Wrong')
            continue
        elif len(contours_ball) == 3:
            cnt_plate_1 = contours_ball[0]
            cnt_plate_2 = contours_ball[1]
            cnt_ball = contours_ball[2]
            
            ## Get the Plate's bounding rect
            x_plate1,y_plate1,w_plate1,h_plate1 = cv2.boundingRect(cnt_plate_1)
            x_plate2,y_plate2,w_plate2,h_plate2 = cv2.boundingRect(cnt_plate_2)
            
            cv2.rectangle(img_ball,(x_plate1,y_plate1),(x_plate1+w_plate1,y_plate1+h_plate1),(0,255,0),2)
            cv2.rectangle(img_ball,(x_plate2,y_plate2),(x_plate2+w_plate2,y_plate2+h_plate2),(0,0,255),2)
            
            # Plate Coordinates
            x_cent_plate1 = float( x_plate1  + (w_plate1/2))
            x_cent_plate2 = float( x_plate2  + (w_plate2/2))
            
            cv2.circle(img_ball, (0,5), radius=1, color=(0, 0, 255), thickness=-1)
            cv2.circle(img_ball, (25,5), radius=1, color=(0, 255, 0), thickness=-1)
            cv2.circle(img_ball, (50,5), radius=1, color=(255, 0, 0), thickness=-1)
            
            
            
            y_cent_plate1 = float(y_plate1 + h_plate1/2)
            y_cent_plate2 = float(y_plate2 + h_plate2/2)
            
            # cv2.circle(img_ball, (round(x_cent_plate1),round(y_cent_plate1)), radius=2, color=(0, 0, 255), thickness=2)
            # cv2.circle(img_ball, (round(x_cent_plate2),round(y_cent_plate2)), radius=2, color=(255, 0, 0), thickness=2)
            
            cv2.circle(img_ball, (round((x_cent_plate1 + x_cent_plate2)/2),round(y_cent_plate2)), radius=2, color=(255, 0, 0), thickness=2)
            
            # print('plate1',x_cent_plate1,y_plate1)
            # print('plate2',x_cent_plate2,y_plate2)
            
            
            # human_agent_action = 1
        
            # ## Get the Ball's bounding rect
            bbox_ball = cv2.boundingRect(cnt_ball)
            x_ball,y_ball,w_ball,h_ball = bbox_ball
              # Ball Coordinates
            x_ball = float(x_ball+(w_ball/2))
            y_ball = float(y_ball)
            x_balli.append(x_ball)
            y_balli.append(y_ball)
            # print(y_ball)
            
            
            base = x_ball
            # Verifying the Center
            # cv2.rectangle(img,(65,100), (95,90), (0,255,0), 1)
            cntrx = ((x_cent_plate1 + x_cent_plate2)/2)
            
            if total_reward > -1:
                # print(y_ball)
                
                
                # If Ball going Down
                if ( ((94-y_ball) - (94-y_balli[0]) < 0) ):
        
                            # Hiearchical Loop
                            # First Perceptual Reference
                            R1 = 0
                            # All Gains
                            k1 = -1
                            k2 = 1
                            k3 = 1
                            k4 = 1
                            # Distance Control
                            D = abs(base - cntrx)
                            e1 = R1 - D
                            R2 = e1 * k1
                            # Movement Control
                            MD = np.sign(base - cntrx)
                            # print('MD = ',MD, 'D = ', D, 'Base = ', base, 'Mid',round((x_cent_plate1 + x_cent_plate2)/2))
                            e2 = R2 - MD
                            
                            # x_cent_plate1 Right
                            if MD < 0:
                                human_agent_action = 4
                                R3 = e2*k2
                                # Position Control
                                e3 = R3 - x_cent_plate1
                                R4 = e3*k3
                                e4 = R4 + base
                                BP = e4*k4
                                itr = itr + 1
                                if itr > 3:
                                      human_agent_action = 0
                                      itr = 0
                            
                            # x_cent_plate2 Left
                            if MD > 0:
                                human_agent_action = 3
                                R3 = e2*k2
                                # Position Control
                                e3 = R3 + x_cent_plate2
                                R4 = e3*k3
                                e4 = R4 - base
                                BP = e4*k4
                                itr = itr + 1
                                if itr > 3:
                                      human_agent_action = 0
                                      itr = 0
        
                
        # imS = cv2.resize(img, (400, 600)) 
        # cv2.imshow('pctAgent', img)
        
        
        if window_still_open==False: return False
        if done: break
        if human_wants_restart: break
        while human_sets_pause:
            env.render()
            time.sleep(0.1)
        time.sleep(0.01)
        
        
    print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))
    with open('C:/Users/gtaus/Breakout-master/results_Pinball_scores.txt', 'a') as f:
        f.write(str(total_reward))
        f.write('\n')
        f.close()
print("ACTIONS={}".format(ACTIONS))
print("Press keys 1 2 3 ... to take actions 1 2 3 ...")
print("No keys pressed is taking action 0")

while 1:
    window_still_open = rollout(env)
    if window_still_open==False: break
