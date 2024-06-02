 

# In[2]:


import torch
device = torch.device('cuda')


# In[3]:


import torch
from torch2trt import TRTModule

model_trt = TRTModule()
model_trt.load_state_dict(torch.load('best_steering_model_xy_trt.pth'))  

model_trt_collision = TRTModule()
model_trt_collision.load_state_dict(torch.load('best_model_trt.pth')) 


# In[4]:


import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import PIL.Image
import numpy as np

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()

def preprocess(image):
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device).half()
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


# In[5]:


from IPython.display import display
import ipywidgets
import traitlets
from jetbot import Camera, bgr8_to_jpeg

camera = Camera.instance(width=224, height=224, fps=10)


# In[6]:


image_widget = ipywidgets.Image()

traitlets.dlink((camera, 'value'), (image_widget, 'value'), transform=bgr8_to_jpeg)


# In[7]:


import os
password = 'jetbot'
command = "sudo -S sh -c 'echo 0 > /sys/devices/pwm-fan/target_pwm'"

print(command)

os.system('echo %s | %s'%(password,command))


# In[8]:


from jetbot import Robot

robot = Robot()


# In[9]:



speed_control_slider = ipywidgets.FloatSlider(min=0.0, max=1.0, step=0.01, description='speed control')
steering_gain_slider = ipywidgets.FloatSlider(min=0.0, max=1.0, step=0.01, value=0.04, description='steering gain')
steering_dgain_slider = ipywidgets.FloatSlider(min=0.0, max=0.5, step=0.001, value=0.0, description='steering kd')
steering_bias_slider = ipywidgets.FloatSlider(min=-0.3, max=0.3, step=0.01, value=0.0, description='steering bias')

display(speed_control_slider, steering_gain_slider, steering_dgain_slider, steering_bias_slider)


blocked_slider = ipywidgets.FloatSlider(min=0.0, max=1.0, orientation='horizontal', description='blocked')
stopduration_slider= ipywidgets.IntSlider(min=1, max=1000, step=1, value=10, description='time for stop') 
blocked_threshold= ipywidgets.FloatSlider(min=0, max=1.0, step=0.01, value=0.8, description='blocked threshold')

display(image_widget)

display(ipywidgets.HBox([blocked_slider, blocked_threshold, stopduration_slider]))


# In[10]:


import os
password = 'jetbot'
command = "sudo -S sh -c 'echo 0 > /sys/devices/pwm-fan/target_pwm'"

print(command)

os.system('echo %s | %s'%(password,command))


# In[11]:


import math

angle = 0.0
angle_last = 0.0
count_stops = 0
go_on = 1
stop_time = 5  
x = 0.0
y = 0.0
speed_value = speed_control_slider.value

def execute(change):
    global angle, angle_last, blocked_slider, robot, count_stops, stop_time, go_on, x, y, blocked_threshold
    global speed_value, steer_gain, steer_dgain, steer_bias
                
    steer_gain = steering_gain_slider.value
    steer_dgain = steering_dgain_slider.value
    steer_bias = steering_bias_slider.value
       
    image_preproc = preprocess(change['new']).to(device)
     
     
    
    prob_blocked = float(F.softmax(model_trt_collision(image_preproc), dim=1).flatten()[0])
    
    blocked_slider.value = prob_blocked    
    stop_time=stopduration_slider.value
    
    if go_on == 1:    
        if prob_blocked > blocked_threshold.value: 
            count_stops += 1
            go_on = 2
        else:
             
            go_on = 1
            count_stops = 0
            xy = model_trt(image_preproc).detach().float().cpu().numpy().flatten()        
            x = xy[0]            
            y = (0.5 - xy[1]) / 2.0
            speed_value = speed_control_slider.value
    else:
        count_stops += 1
        if count_stops < stop_time:
            x = 0.0  
            y = 0.0  
            speed_value = 0 
        else:
            go_on = 1
            count_stops = 0
            

    angle = math.atan2(x, y)        
    pid = angle * steer_gain + (angle - angle_last) * steer_dgain
    steer_val = pid + steer_bias 
    angle_last = angle
    robot.left_motor.value = max(min(speed_value + steer_val, 0.6), 0.0)
    robot.right_motor.value = max(min(speed_value - steer_val, 0.6), 0.0) 

execute({'new': camera.value})


# In[12]:


camera.observe(execute, names='value')


# In[13]:


import time

camera.unobserve(execute, names='value')

time.sleep(0.1)   
robot.stop()


# In[ ]:




