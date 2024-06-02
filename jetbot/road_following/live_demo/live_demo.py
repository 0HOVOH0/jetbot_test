 

# In[1]:


model_file = 'best_steering_model_xy.pth'


 

# In[2]:


 
from jetbot import Robot, Camera, bgr8_to_jpeg

robot=Robot()
 


 
# ## 設定功率到10W

# In[3]:


import os
password = 'jetbot'
command_set_power = "sudo -S nvpmodel -m10"

print(command_set_power)

os.system('echo %s | %s'%(password,command_set_power))


 

# In[ ]:


import os
password = 'jetbot'
command = "sudo -S sh -c 'echo 0 > /sys/devices/pwm-fan/target_pwm'"

print(command)

os.system('echo %s | %s'%(password,command))


 
# In[5]:


import torchvision
import torch

model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 2)



# In[6]:


model.load_state_dict(torch.load('best_steering_model_xy.pth'))


 

# In[7]:


device = torch.device('cuda')
model = model.to(device)
model = model.eval().half()


 
# In[8]:


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


 

# In[9]:


from IPython.display import display
import ipywidgets
import traitlets

 
camera = Camera.instance(width=224, height=224, fps=5)
image_widget = ipywidgets.Image()

camera_link = traitlets.dlink((camera, 'value'), (image_widget, 'value'), transform=bgr8_to_jpeg)

display(image_widget)


 

# In[10]:


speed_gain_slider = ipywidgets.FloatSlider(min=0.0, max=1.0, step=0.01, description='speed gain')
steering_gain_slider = ipywidgets.FloatSlider(min=0.0, max=1.0, step=0.01, value=0.2, description='steering gain')
steering_dgain_slider = ipywidgets.FloatSlider(min=0.0, max=0.5, step=0.001, value=0.0, description='steering kd')
steering_bias_slider = ipywidgets.FloatSlider(min=-0.3, max=0.3, step=0.01, value=0.0, description='steering bias')

slider_box = ipywidgets.VBox([speed_gain_slider, steering_gain_slider, steering_dgain_slider, steering_bias_slider])

x_slider = ipywidgets.FloatSlider(min=-1.0, max=1.0, description='x')
y_slider = ipywidgets.FloatSlider(min=0, max=1.0, orientation='vertical', description='y')
steering_slider = ipywidgets.FloatSlider(min=-1.0, max=1.0, description='steering')
speed_slider = ipywidgets.FloatSlider(min=0, max=1.0, orientation='vertical', description='speed')
y_box = ipywidgets.HBox([y_slider, speed_slider])
xy_box = ipywidgets.VBox([y_box,x_slider, steering_slider])
final_box = ipywidgets.HBox([xy_box,slider_box])

display(final_box)


 

# In[11]:


camera_link.unlink()   

# In[12]:


angle = 0.0
angle_last = 0.0

def execute(change):
    global angle, angle_last
    image = change['new']
    xy = model(preprocess(image)).detach().float().cpu().numpy().flatten()
    x = xy[0]
    y = (0.5 - xy[1]) / 2.0
    
    x_slider.value = x
    y_slider.value = y
    
    speed_slider.value = speed_gain_slider.value
    
    angle = np.arctan2(x, y)
    pid = angle * steering_gain_slider.value + (angle - angle_last) * steering_dgain_slider.value
    angle_last = angle
    
    steering_slider.value = pid + steering_bias_slider.value
    
    robot.left_motor.value = max(min(speed_slider.value + steering_slider.value, 1.0), 0.0)
    robot.right_motor.value = max(min(speed_slider.value - steering_slider.value, 1.0), 0.0)
    
execute({'new': camera.value})


 
# In[13]:


camera.observe(execute, names='value')


 
# In[1]:


camera.unobserve(execute, names='value')
robot.stop()


 
# In[16]:


import os
password ='jetbot'
command = "sudo -S sh -c 'echo 0 > /sys/devices/pwm-fan/target_pwm'"

print(command)

os.system('echo %s | %s'%(password,command))


 