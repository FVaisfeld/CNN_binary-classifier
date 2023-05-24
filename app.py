import streamlit as st
from torchvision import datasets, transforms, models
import torch
from PIL import Image
from torch import nn, optim
import logging
import numpy as np
import datetime
import matplotlib.pyplot as plt



logging.basicConfig(filename='logfile.log',
                            filemode='a', level=logging.INFO)


#load pre trained model from torchvision and adjust the last layer to fine tuned state
def load_create_model():
    #load pre trained model from torchvision 
    resnet = models.resnet18(pretrained=True)
    #adjust the last layer to fit our problem
    resnet.fc = nn.Linear(resnet.fc.in_features, 2)
    resnet.load_state_dict(torch.load('model.pth'))
    return resnet

#use a device a cpu or gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
st.title('apple or banana?!')

#upload the file as jpg
file_up = st.file_uploader("Upload an image of a banana or an apple (as jpg)", type="jpg")

if file_up is not None:
    
    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image.', width=500)



    transform = transforms.Compose([
        transforms.Resize(100),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        ])


    resnet = load_create_model()

    #do inference with the uploaded image 
    with torch.no_grad():
        #in case of an alpha channel remove it 
        image = transform(image)[:3]
        image = torch.unsqueeze(image, 0)
        image = image.to(device)        
        resnet.eval()
        out = resnet(image).argmax(dim=1)
        st.write("It seems to be... ")
        if out[0]==0:
            st.write("an apple!")
        else:
            st.write("a banana!")
    
    #log the result
    #use logging to store outputs of out
    logging.info(f"Image classified as {out}")
    #save logger

    keep_phrases = ["Image classified as tensor([0])","Image classified as tensor([1])"]
    times = []
    values = []

    with open('logfile.log') as f:
        f = f.readlines()
        for line in f:
            for phrase in keep_phrases:
                if phrase in line:
                    time = (line[:line.find("Image classified as")])
                    value = (line[ line.find("tensor([") : line.find("]") ])[-1]
                    times.append(time)
                    values.append(int(value))

    n_bananas = sum(values)
    n_apples = sum(abs(np.array(values)-1))
  



    fig, ax = plt.subplots()
    ax.set_title("apples vs bananas")
    ax.hist(values, bins=2)
    ax.set_ylabel('amount of classifications')
    ax.set_xticks(np.arange(2), labels=["apple","banana"])
    st.pyplot(fig)


    #calculate time difference 
    time_diff = []
    for i in range(len(times)-1):
        time_diff.append((datetime.datetime.strptime(times[i+1], "%Y-%m-%d %H:%M:%S.%f ") - datetime.datetime.strptime(times[i], "%Y-%m-%d %H:%M:%S.%f ")).total_seconds())

    mean_diff = (np.mean(time_diff))
    st.write(f"Average time difference between classifications: {mean_diff} seconds")

    #total time difference
    total_time = (datetime.datetime.strptime(times[-1], "%Y-%m-%d %H:%M:%S.%f ") - datetime.datetime.strptime(times[0], "%Y-%m-%d %H:%M:%S.%f ")).total_seconds()
    st.write(f"Total time from first till last classification: {total_time} seconds")











        
