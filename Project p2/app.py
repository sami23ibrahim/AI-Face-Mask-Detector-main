import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_score
from datetime import datetime

from sklearn.metrics import classification_report
from skorch import NeuralNetClassifier
from torch.utils.data import random_split
from skorch.helper import SliceDataset
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
from sklearn.metrics import f1_score
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import torchvision.datasets as datasets
path = 'cnnmodel.pth'
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
mode = input("Press 1 to train the model, display metrics and save it. Press 2 to and predict an image using a pretrained model. Press 3 to predict the test dataset and display metrics \n")

transform = transforms.Compose([
    transforms.Resize((150,150)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
])


all_dataset =datasets.ImageFolder(
    root='input/masks/train',transform = transform)

m = len(all_dataset)

def display_img(img,label):
    print(f"Label : {all_dataset.classes[label]}")
    plt.imshow(img.permute(1,2,0))

#split dataset

testSize = m - int(m - m * 0.2)

train_data,test_data = random_split(all_dataset,[int(m - m * 0.2), testSize])

torch.manual_seed(1)
y_train = np.array([y for x, y in iter(train_data)])

#define model
class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv_layer = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.LeakyReLU(inplace=True),
      nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.LeakyReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
      nn.BatchNorm2d(64),
      nn.LeakyReLU(inplace=True),
      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
      nn.BatchNorm2d(64),
      nn.LeakyReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
             )
    self.fc_layer = nn.Sequential(
        nn.Dropout(p=0.1),
        nn.Linear(87616, 1000),
        nn.ReLU(inplace=True),
        nn.Linear(1000, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.1),
        nn.Linear(512, 4)
    )
  def forward(self, x):
        # conv layers
        x = self.conv_layer(x)
         # flatten
        x = x.view(x.size(0), -1)
         # fc layer
        x = self.fc_layer(x)
        return x
    
if int(mode) == 2 or int(mode) == 3 :
    #loading the model
    loaded_net = NeuralNetClassifier(
    CNN,
    max_epochs=10,
    iterator_train__num_workers=0,
    iterator_valid__num_workers=0,
    lr= 1e-3,
    batch_size=64,
    optimizer=optim.Adam,
    criterion=nn.CrossEntropyLoss)
    loaded_net.initialize()  # This is important!
    print("Please select a trained model (.pth file) \n")
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
     
    loaded_net.load_params(f_params=filename)
    print("Model selected. Thank you")    
    
    
    
if int(mode) == 1:
    print(f"Length of Train Data : {len(train_data)}")
    print(f"Length of testing Data : {len(test_data)}")
    print("Training a new model on the training data...")
    
    net = NeuralNetClassifier(
    CNN,
    max_epochs=5,
    iterator_train__num_workers=0,
    iterator_valid__num_workers=0,
    lr= 1e-3,
    batch_size=64,
    optimizer=optim.Adam,
    criterion=nn.CrossEntropyLoss
    )
    with torch.no_grad():
      net.fit(train_data, y=y_train)    
    y_pred = net.predict(test_data)
    y_test = np.array([y for x, y in iter(test_data)])
    print("displaying confusion matrix. Please close this matrix to save the model and see classification report")
    plot_confusion_matrix(net, test_data, y_test.reshape(-1, 1))
    plt.show()
    plt.close()
    
    print(classification_report(y_test, y_pred, labels=[0,1,2,3], zero_division=0))
    #saving the model
    time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

    
    net.save_params(f_params='cnnTrained {}.pth'.format(time))
    print("saving the trained model as cnnTrained {}.pth".format(time))

    
    
elif int(mode) == 2:
    print("predict individual image mode selected")
    while True:
        print("Please select an image to predict")
        convert_tensor = transforms.ToTensor()
        Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
        filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
        print(filename)
        img = Image.open(filename)
        transformedImg = transform(img)
        transformedImg = transformedImg.unsqueeze(0)
    
        prediction = (loaded_net.predict(transformedImg))[0]
        again = input("Your prediction was label: {}. Press 1 to predict another image. Press 0 to exit \n".format(all_dataset.classes[prediction]))
        if int(again)==1:
            continue
        else:
            print("Looks like you did not select to predict another image. See you next time!")
            break
elif int(mode) ==3:
    print("predicting test dataset mode")
    y_pred = loaded_net.predict(test_data)
    y_test = np.array([y for x, y in iter(test_data)])
    print("accuracy score")
    accuracy_score(y_test, y_pred)
    print("confusion matrix")
    plot_confusion_matrix(loaded_net, test_data, y_test.reshape(-1, 1))
    plt.show()
    print(classification_report(y_test, y_pred, labels=[0,1,2,3]))
    
else:
    print("invalid command")
#{'(cloth) face mask': 0, 'FFP2N95KN95-type mask': 1, 'Person without a face mask': 2, 'surgical (procedural) mask': 3}
