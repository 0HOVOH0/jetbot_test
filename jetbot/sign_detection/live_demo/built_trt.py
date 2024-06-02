 
# In[1]:


import torch
import torchvision

model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 3)
model = model.cuda().eval().half()


# In[2]:


model.load_state_dict(torch.load('best_model_resnet18.pth'))


# In[3]:


device = torch.device('cuda')


# In[4]:


from torch2trt import torch2trt

data = torch.zeros((1, 3, 224, 224)).cuda().half()

model_trt = torch2trt(model, [data], fp16_mode=True)


# In[5]:


torch.save(model_trt.state_dict(), 'best_model_trt.pth')


# In[ ]:





 




