# In[ ]:


# In[2]:


dataset_path = 'best_steering_model_xy_green'


# In[3]:


model_file = 'best_steering_model_xy.pth'


# In[4]:


set_epochs = 50


# In[5]:


import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import glob
import PIL.Image
import os
import numpy as np


 
# In[6]:


import os
password = 'jetbot'
command_set_power = "sudo -S nvpmodel -m1"
print(command_set_power)
os.system('echo %s | %s'%(password,command_set_power))


 

# In[7]:


import os
password = 'jetbot'
command = "sudo -S sh -c 'echo 255 > /sys/devices/pwm-fan/target_pwm'"
print(command)
os.system('echo %s | %s'%(password,command))


 
# In[8]:


def get_x(path):
    """Gets the x value from the image filename"""
    return (float(int(path[3:6])) - 50.0) / 50.0

def get_y(path):
    """Gets the y value from the image filename"""
    return (float(int(path[7:10])) - 50.0) / 50.0

class XYDataset(torch.utils.data.Dataset):
    
    def __init__(self, directory, random_hflips=False):
        self.directory = directory
        self.random_hflips = random_hflips
        self.image_paths = glob.glob(os.path.join(self.directory, '*.jpg'))
        self.color_jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        image = PIL.Image.open(image_path)
        x = float(get_x(os.path.basename(image_path)))
        y = float(get_y(os.path.basename(image_path)))
        
        if float(np.random.rand(1)) > 0.5:
            image = transforms.functional.hflip(image)
            x = -x
        
        image = self.color_jitter(image)
        image = transforms.functional.resize(image, (224, 224))
        image = transforms.functional.to_tensor(image)
        image = image.numpy()[::-1].copy()
        image = torch.from_numpy(image)
        image = transforms.functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        return image, torch.tensor([x, y]).float()
    
 
dataset = XYDataset(dataset_path, random_hflips=False)


 

# In[9]:


test_percent = 0.1
num_test = int(test_percent * len(dataset))
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - num_test, num_test])


 

# In[10]:


train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4
)


 

# In[11]:


model = models.resnet18(pretrained=True)


 
# In[12]:


model.fc = torch.nn.Linear(512, 2)
device = torch.device('cuda')
model = model.to(device)


 

# In[13]:


NUM_EPOCHS = 50 #訓練次數
BEST_MODEL_PATH = 'best_steering_model_xy_green' #導入要訓練的模型
best_loss = 1e9

optimizer = optim.Adam(model.parameters())

for epoch in range(NUM_EPOCHS):
    
    model.train()  #訓練模式
    train_loss = 0.0
    for images, labels in iter(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = F.mse_loss(outputs, labels)
        train_loss += loss
        loss.backward()
        optimizer.step()
    train_loss /= len(train_loader)
    
    model.eval()
    test_loss = 0.0
    for images, labels in iter(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = F.mse_loss(outputs, labels)
        test_loss += loss
    test_loss /= len(test_loader)
    
    print('round %d: %f, %f' % (epoch, train_loss, test_loss))
    if test_loss < best_loss:
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        best_loss = test_loss
print('Done!')


 
# In[ ]:


import os
password = 'jetbot'
command = "sudo -S sh -c 'echo 0 > /sys/devices/pwm-fan/target_pwm'"

print(command)

os.system('echo %s | %s'%(password,command))


# In[ ]:




