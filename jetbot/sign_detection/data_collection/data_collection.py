 
# In[1]:


import traitlets
import ipywidgets.widgets as widgets
from IPython.display import display
from jetbot import Camera, bgr8_to_jpeg

camera = Camera.instance(width=224, height=224)

image = widgets.Image(format='jpeg', width=224, height=224)   

camera_link = traitlets.dlink((camera, 'value'), (image, 'value'), transform=bgr8_to_jpeg)

display(image)


 

# In[4]:


import os

left_dir = 'dataset/turn_left'
red_sign_dir = 'dataset/red_sign'
right_dir = 'dataset/turn_right'
free_dir = 'dataset/free'


 
try:
    os.makedirs(free_dir)
    os.makedirs(blocked_dir)
except FileExistsError:
    print('Directories not created because they already exist')


 
# In[6]:


button_layout = widgets.Layout(width='128px', height='64px')

free_button = widgets.Button(description='add free', button_style='success', layout=button_layout)
red_sign_button = widgets.Button(description='add red', button_style='danger', layout=button_layout)
left_button = widgets.Button(description='add left', button_style='danger', layout=button_layout)
right_button = widgets.Button(description='add right', button_style='danger', layout=button_layout)


free_count = widgets.IntText(layout=button_layout, value=len(os.listdir(free_dir)))
red_sign_count = widgets.IntText(layout=button_layout, value=len(os.listdir(red_sign_dir)))
left_count = widgets.IntText(layout=button_layout, value=len(os.listdir(left_dir)))
right_count = widgets.IntText(layout=button_layout, value=len(os.listdir(right_dir)))


display(widgets.HBox([free_count, free_button]))
display(widgets.HBox([red_sign_count, red_sign_button]))
display(widgets.HBox([left_count, left_button]))
display(widgets.HBox([right_count, right_button]))


 
# In[7]:


from uuid import uuid1

def save_snapshot(directory):
    image_path = os.path.join(directory, str(uuid1()) + '.jpg')
    with open(image_path, 'wb') as f:
        f.write(image.value)

def save_free():
    global free_dir, free_count
    save_snapshot(free_dir)
    free_count.value = len(os.listdir(free_dir))
    
def save_red_sign():
    global red_sign_dir, red_sign_count
    save_snapshot(red_sign_dir)
    red_sign_count.value = len(os.listdir(red_sign_dir))
    
def save_left():
    global left_dir, left_count
    save_snapshot(left_dir)
    left_count.value = len(os.listdir(left_dir))
    
def save_right():
    global right_dir, right_count
    save_snapshot(right_dir)
    right_count.value = len(os.listdir(right_dir))
    

    
 
free_button.on_click(lambda x: save_free())
red_sign_button.on_click(lambda x: save_red_sign())
left_button.on_click(lambda x: save_left())
right_button.on_click(lambda x: save_right())


 

# In[8]:


display(image)
display(widgets.HBox([free_count, free_button]))
display(widgets.HBox([red_sign_count, red_sign_button]))
display(widgets.HBox([left_count, left_button]))
display(widgets.HBox([right_count, right_button]))


 

# In[27]:


camera.stop()


 

# In[7]:


get_ipython().system('zip -r -q dataset.zip dataset')


 
