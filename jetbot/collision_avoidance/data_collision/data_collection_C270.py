 

# In[18]:


import traitlets
import ipywidgets.widgets as widgets
from IPython.display import display
from jetbot import Camera, bgr8_to_jpeg

image = widgets.Image(format='jpeg', width=224, height=224)   


 
# In[19]:


import os

block_dir = 'dataset/block'
free_dir = 'dataset/free'

try:
    os.makedirs(free_dir)
    os.makedirs(blocked_dir)
except FileExistsError:
    print('Directories not created becasue they already exist')


 
# In[20]:


button_layout = widgets.Layout(width='128px', height='64px')
free_button = widgets.Button(description='add free', button_style='success', layout=button_layout)
blocked_button = widgets.Button(description='add blocked', button_style='danger', layout=button_layout)
 
free_count = widgets.IntText(layout=button_layout, value=len(os.listdir(free_dir)))
blocked_count = widgets.IntText(layout=button_layout, value=len(os.listdir(blocked_dir)))
  

display(widgets.HBox([free_count, free_button]))
display(widgets.HBox([blocked_count, blocked_button]))
 


 

# In[21]:


from uuid import uuid1

def save_snapshot(directory):
    image_path = os.path.join(directory, str(uuid1()) + '.jpg')
    with open(image_path, 'wb') as f:
        f.write(image.value)

def save_free():
    global free_dir, free_count
    save_snapshot(free_dir)
    free_count.value = len(os.listdir(free_dir))
    
def save_blocked():
    global blocked_dir, blocked_count
    save_snapshot(blocked_dir)
    blocked_count.value = len(os.listdir(blocked_dir))
    
 
free_button.on_click(lambda x: save_free())
blocked_button.on_click(lambda x: save_blocked())


 

# In[22]:


display(image)
display(widgets.HBox([free_count, free_button]))
display(widgets.HBox([blocked_count, blocked_button]))


# In[23]:


import cv2
import threading
import time

def C270_on():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    try:
        while True:
            
            ret,frame = cap.read()
            image.value = cv2.imencode('.jpg',frame)[1].tobytes()
             
    finally:
        cap.release()


thread =threading.Thread(target=C270_on)
thread.start()


 

# In[ ]:


get_ipython().system('zip -r -q dataset.zip dataset')


 
