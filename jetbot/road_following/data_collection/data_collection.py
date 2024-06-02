#設定路徑
dataset_path = 'best_steering_model_xy_green' 



from jetbot import Robot, Camera, bgr8_to_jpeg

robot=Robot()
 
# In[3]:


import os
password = 'jetbot'
command_set_power = "sudo -S nvpmodel -m0"

print(command_set_power)

os.system('echo %s | %s'%(password,command_set_power))


# ### 開啟散熱風扇

# In[4]:


import os
password = 'jetbot'
command = "sudo -S sh -c 'echo 150 > /sys/devices/pwm-fan/target_pwm'"

print(command)

os.system('echo %s | %s'%(password,command))



# In[5]:


 
import traitlets
import ipywidgets.widgets as widgets
from IPython.display import display

 
from uuid import uuid1
import os
import json
import glob
import datetime
import numpy as np
import cv2
import time


 
# In[6]:


 

DATASET_DIR = dataset_path

try:
    os.makedirs(DATASET_DIR)
except FileExistsError:
    print('Directory already exist')

 
def xy_uuid(x, y):
    return 'xy_%03d_%03d_%s' % (x * 50 + 50, y * 50 + 50, uuid1())

 

motor_speed_ratio = 0.4
time_interval = 0.5

##### Button Function ######

def stop(change):
    robot.stop()
    
def step_forward(change):
    robot.forward(motor_speed_ratio)
    time.sleep(time_interval)
    robot.stop()

def step_backward(change):
    robot.backward(motor_speed_ratio)
    time.sleep(time_interval)
    robot.stop()

def step_left(change):
    robot.left(motor_speed_ratio)
    time.sleep(time_interval)
    robot.stop()

def step_right(change):
    robot.right(motor_speed_ratio)
    time.sleep(time_interval)
    robot.stop()

def save_snapshot(change):
    uuid = xy_uuid(x_slider.value, y_slider.value)
    image_path = os.path.join(DATASET_DIR, uuid + '.jpg')
    with open(image_path, 'wb') as f:
        f.write(image_widget.value)
    count_widget.value = len(glob.glob(os.path.join(DATASET_DIR, '*.jpg')))

##########################

##### Camera settings ########

camera = Camera.instance(width=224, height=224, fps=5)
image_widget = widgets.Image(format='jpeg', width=224, height=224)
target_widget = widgets.Image(format='jpeg', width=224, height=224)

#######################################

##### Self-defined layout ######

button_layout = widgets.Layout(width='100px', height='80px', align_self='center')

stop_button = widgets.Button(description='stop', button_style='danger', layout=button_layout)
forward_button = widgets.Button(description='forward', layout=button_layout)
backward_button = widgets.Button(description='backward', layout=button_layout)
left_button = widgets.Button(description='left', layout=button_layout)
right_button = widgets.Button(description='right', layout=button_layout)

middle_box = widgets.HBox([left_button, stop_button, right_button], layout=widgets.Layout(align_self='center'))
controls_box = widgets.VBox([forward_button, middle_box, backward_button])

save_button = widgets.Button(description='save', layout=button_layout)##
count_widget = widgets.IntText(description='count', value=len(glob.glob(os.path.join(DATASET_DIR, '*.jpg'))))


x_slider = widgets.FloatSlider(min=-1.0, max=1.0, step=0.001, description='x')
y_slider = widgets.FloatSlider(min=-1.0, max=1.0, step=0.001, description='y')
slider_box = widgets.VBox([x_slider, y_slider,count_widget,save_button])

def display_xy(camera_image):
    image = np.copy(camera_image)
    x = x_slider.value
    y = y_slider.value
    x = int(x * 224 / 2 + 112)
    y = int(y * 224 / 2 + 112)
    image = cv2.circle(image, (x, y), 8, (0, 255, 0), 3)
    image = cv2.circle(image, (112, 224), 8, (0, 0,255), 3)
    image = cv2.line(image, (x,y), (112,224), (255,0,0), 3)
    jpeg_image = bgr8_to_jpeg(image)
    return jpeg_image

time.sleep(1)
camera_link_1 = traitlets.dlink((camera, 'value'), (image_widget, 'value'), transform=bgr8_to_jpeg)
camera_link_2 = traitlets.dlink((camera, 'value'), (target_widget, 'value'), transform=display_xy)

##### link buttons to actions #####
save_button.on_click(save_snapshot)
stop_button.on_click(stop)
forward_button.on_click(step_forward)
backward_button.on_click(step_backward)
left_button.on_click(step_left)
right_button.on_click(step_right)

###################################

display(widgets.HBox([target_widget,slider_box,controls_box]))


 
# In[7]:


camera_link_1.unlink() # don't stream to browser (will still run camera)
camera_link_2.unlink() # don't stream to browser (will still run camera)


 
# In[ ]:


print(len(glob.glob(os.path.join(DATASET_DIR, '*.jpg'))))


# In[ ]:


def timestr():
    return str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

get_ipython().system('zip -r -q road_following_{DATASET_DIR}_{timestr()}.zip {DATASET_DIR}')


 

# In[7]:


import os
password = 'jetbot'
command = "sudo -S sh -c 'echo 0 > /sys/devices/pwm-fan/target_pwm'"

print(command)

os.system('echo %s | %s'%(password,command))


# In[ ]:




