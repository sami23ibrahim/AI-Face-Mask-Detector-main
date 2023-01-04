
from google.colab import drive
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import random_split
from torchvision.utils import make_grid

from skorch import NeuralNetClassifier
from skorch.helper import SliceDataset
from PIL import Image

drive.flush_and_unmount()
drive.mount('/content/drive')
path = '/content/drive/MyDrive/cnnmodel.pth'
device = torch.device("cuda")


# Loading the original biased data
transform = transforms.Compose([
    transforms.Resize((150,150)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


all_dataset = datasets.ImageFolder(
    root='/content/drive/MyDrive/Dataset/Cropped Pics',transform = transform)

m = len(all_dataset)

print(all_dataset.class_to_idx)
print(f"Lenght of Data: {len(all_dataset)}")

def display_img(img,label):
    print(f"Label : {all_dataset.classes[label]}")
    plt.imshow(img.permute(1,2,0))

#split dataset
train_data,test_data = random_split(all_dataset,[m-400, 400])
print(f"Length of Train Data : {len(train_data)}")
print(f"Length of testing Data : {len(test_data)}")


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
convnet = CNN()
#move stuff to gpu

net = NeuralNetClassifier(
CNN,
max_epochs=10,
iterator_train__num_workers=0,
iterator_valid__num_workers=0,
lr= 1e-3,
batch_size=64,
optimizer=optim.Adam,
criterion=nn.CrossEntropyLoss,
device = device
)


# Training the model

y_train = np.array([y for x, y in iter(train_data)])

with torch.no_grad():
  net.fit(train_data, y=y_train)


y_pred = net.predict(test_data)
y_test = np.array([y for x, y in iter(test_data)])

print("accuracy score")
accuracy_score(y_test, y_pred)
print("confusion matrix")
plot_confusion_matrix(net, test_data, y_test.reshape(-1, 1))
plt.show()

#saving the model
net.save_params(f_params='/content/drive/MyDrive/skorchCNN.pth')

#{'(cloth) face mask': 0, 'FFP2N95KN95-type mask': 1, 'Person without a face mask': 2, 'surgical (procedural) mask': 3}

net.fit(train_data, y=y_train)
train_sliceable = SliceDataset(train_data)
scores = cross_val_score(net, train_sliceable, y_train, cv=10,
scoring="accuracy")
# cross validation for original part 1 data
print(scores)
print(classification_report(y_test, y_pred, labels=[0,1,2,3]))


# Load only female data
transform = transforms.Compose([
    transforms.Resize((150,150)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


female_only = datasets.ImageFolder(
    root='/content/drive/MyDrive/Dataset/Gender Bias Cropped/Female test data cropped',transform = transform)

m = len(female_only)

print(female_only.class_to_idx)
print(f"Lenght of Data: {len(female_only)}")



# test only female data

y_pred = net.predict(female_only)
y_test = np.array([y for x, y in iter(female_only)])

print("accuracy score")
accuracy_score(y_test, y_pred)
print("confusion matrix")
plot_confusion_matrix(net, female_only, y_test.reshape(-1, 1))
plt.show()

print(classification_report(y_test, y_pred, labels=[0,1,2,3]))
#{'(cloth) face mask': 0, 'FFP2N95KN95-type mask': 1, 'Person without a face mask': 2, 'surgical (procedural) mask': 3}

# Cross validate only female
net.fit(female_only, y=y_test)
train_sliceable = SliceDataset(female_only)
scores = cross_val_score(net, train_sliceable, y_test, cv=10,
scoring="accuracy")

print(scores)

# Load only male data
transform = transforms.Compose([
    transforms.Resize((150,150)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


male_only = datasets.ImageFolder(
    root='/content/drive/MyDrive/Dataset/Gender Bias Cropped/Male test data cropped',transform = transform)

m = len(male_only)

print(male_only.class_to_idx)
print(f"Lenght of Data: {len(male_only)}")




# test only male data

y_pred = net.predict(male_only)
y_test = np.array([y for x, y in iter(male_only)])

print("accuracy score")
accuracy_score(y_test, y_pred)
print("confusion matrix")
plot_confusion_matrix(net, male_only, y_test.reshape(-1, 1))
plt.show()

print(classification_report(y_test, y_pred, labels=[0,1,2,3]))
#{'(cloth) face mask': 0, 'FFP2N95KN95-type mask': 1, 'Person without a face mask': 2, 'surgical (procedural) mask': 3}


# Cross validate only male
net.fit(male_only, y=y_test)
train_sliceable = SliceDataset(male_only)
scores = cross_val_score(net, train_sliceable, y_test, cv=10,
scoring="accuracy")

print(scores)


# Load Dark skin data
transform = transforms.Compose([
    transforms.Resize((150,150)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


dark_skin_only = datasets.ImageFolder(
    root='/content/drive/MyDrive/Dataset/Race Bias Cropped/Dark skinned cropped',transform = transform)

m = len(dark_skin_only)

print(dark_skin_only.class_to_idx)
print(f"Lenght of Data: {len(dark_skin_only)}")



# test only dark skin data

y_pred = net.predict(dark_skin_only)
y_test = np.array([y for x, y in iter(dark_skin_only)])

print("accuracy score")
accuracy_score(y_test, y_pred)
print("confusion matrix")
plot_confusion_matrix(net, dark_skin_only, y_test.reshape(-1, 1))
plt.show()

print(classification_report(y_test, y_pred, labels=[0,1,2,3]))
#{'(cloth) face mask': 0, 'FFP2N95KN95-type mask': 1, 'Person without a face mask': 2, 'surgical (procedural) mask': 3}

# Cross validate only dark skin data
net.fit(dark_skin_only, y=y_test)
train_sliceable = SliceDataset(dark_skin_only)
scores = cross_val_score(net, train_sliceable, y_test, cv=10,
scoring="accuracy")

print(scores)

# Load light skin data
transform = transforms.Compose([
    transforms.Resize((150,150)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


light_skin_only = datasets.ImageFolder(
    root='/content/drive/MyDrive/Dataset/Race Bias Cropped/Light skinned cropped',transform = transform)

m = len(light_skin_only)

print(light_skin_only.class_to_idx)
print(f"Lenght of Data: {len(light_skin_only)}")


# test only light skin data

y_pred = net.predict(light_skin_only)
y_test = np.array([y for x, y in iter(light_skin_only)])

print("accuracy score")
accuracy_score(y_test, y_pred)
print("confusion matrix")
plot_confusion_matrix(net, light_skin_only, y_test.reshape(-1, 1))
plt.show()

print(classification_report(y_test, y_pred, labels=[0,1,2,3]))

#{'(cloth) face mask': 0, 'FFP2N95KN95-type mask': 1, 'Person without a face mask': 2, 'surgical (procedural) mask': 3}

# Cross validate only dark skin data
net.fit(light_skin_only, y=y_test)
train_sliceable = SliceDataset(light_skin_only)
scores = cross_val_score(net, train_sliceable, y_test, cv=10,
scoring="accuracy")

print(scores)

# Load the unbiased data
transform = transforms.Compose([
    transforms.Resize((150,150)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


all_dataset = datasets.ImageFolder(
    root='/content/drive/MyDrive/Dataset/Unbiased data',transform = transform)

m = len(all_dataset)

print(all_dataset.class_to_idx)
print(f"Lenght of Data: {len(all_dataset)}")



#split dataset
train_data,test_data = random_split(all_dataset,[m-400, 400])
print(f"Length of Train Data : {len(train_data)}")
print(f"Length of testing Data : {len(test_data)}")

# Training the model using the old biased data

y_train = np.array([y for x, y in iter(train_data)])

with torch.no_grad():
  net.fit(train_data, y=y_train)


y_pred = net.predict(test_data)
y_test = np.array([y for x, y in iter(test_data)])

print("accuracy score")
accuracy_score(y_test, y_pred)
print("confusion matrix")
plot_confusion_matrix(net, test_data, y_test.reshape(-1, 1))
plt.show()

#saving the new unbiased model
net.save_params(f_params='/content/drive/MyDrive/UnBiasedskorchCNN.pth')

#{'(cloth) face mask': 0, 'FFP2N95KN95-type mask': 1, 'Person without a face mask': 2, 'surgical (procedural) mask': 3}


#cross validating the new model
net.fit(train_data, y=y_train)
train_sliceable = SliceDataset(train_data)
scores = cross_val_score(net, train_sliceable, y_train, cv=10,
scoring="accuracy")
# cross validation for original part 1 data
print(scores)
print(classification_report(y_test, y_pred, labels=[0,1,2,3]))


# Load only female data
transform = transforms.Compose([
    transforms.Resize((150,150)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


female_only = datasets.ImageFolder(
    root='/content/drive/MyDrive/Dataset/Gender Bias Cropped/Female test data cropped',transform = transform)

m = len(female_only)

print(female_only.class_to_idx)
print(f"Lenght of Data: {len(female_only)}")



# test only female data

y_pred = net.predict(female_only)
y_test = np.array([y for x, y in iter(female_only)])

print("accuracy score")
accuracy_score(y_test, y_pred)
print("confusion matrix")
plot_confusion_matrix(net, female_only, y_test.reshape(-1, 1))
plt.show()

print(classification_report(y_test, y_pred, labels=[0,1,2,3]))
#{'(cloth) face mask': 0, 'FFP2N95KN95-type mask': 1, 'Person without a face mask': 2, 'surgical (procedural) mask': 3}

# Cross validate only female
net.fit(female_only, y=y_test)
train_sliceable = SliceDataset(female_only)
scores = cross_val_score(net, train_sliceable, y_test, cv=10,
scoring="accuracy")

print(scores)


# Load only male data
transform = transforms.Compose([
    transforms.Resize((150,150)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


male_only = datasets.ImageFolder(
    root='/content/drive/MyDrive/Dataset/Gender Bias Cropped/Male test data cropped',transform = transform)

m = len(male_only)

print(male_only.class_to_idx)
print(f"Lenght of Data: {len(male_only)}")



# test only male data

y_pred = net.predict(male_only)
y_test = np.array([y for x, y in iter(male_only)])

print("accuracy score")
accuracy_score(y_test, y_pred)
print("confusion matrix")
plot_confusion_matrix(net, male_only, y_test.reshape(-1, 1))
plt.show()

print(classification_report(y_test, y_pred, labels=[0,1,2,3]))
#{'(cloth) face mask': 0, 'FFP2N95KN95-type mask': 1, 'Person without a face mask': 2, 'surgical (procedural) mask': 3}


# Cross validate only male
net.fit(male_only, y=y_test)
train_sliceable = SliceDataset(male_only)
scores = cross_val_score(net, train_sliceable, y_test, cv=10,
scoring="accuracy")

print(scores)

# Load Dark skin data
transform = transforms.Compose([
    transforms.Resize((150,150)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


dark_skin_only = datasets.ImageFolder(
    root='/content/drive/MyDrive/Dataset/Race Bias Cropped/Dark skinned cropped',transform = transform)

m = len(dark_skin_only)

print(dark_skin_only.class_to_idx)
print(f"Lenght of Data: {len(dark_skin_only)}")


# test only dark skin data

y_pred = net.predict(dark_skin_only)
y_test = np.array([y for x, y in iter(dark_skin_only)])

print("accuracy score")
accuracy_score(y_test, y_pred)
print("confusion matrix")
plot_confusion_matrix(net, dark_skin_only, y_test.reshape(-1, 1))
plt.show()

print(classification_report(y_test, y_pred, labels=[0,1,2,3]))
#{'(cloth) face mask': 0, 'FFP2N95KN95-type mask': 1, 'Person without a face mask': 2, 'surgical (procedural) mask': 3}


# Cross validate only dark skin data
net.fit(dark_skin_only, y=y_test)
train_sliceable = SliceDataset(dark_skin_only)
scores = cross_val_score(net, train_sliceable, y_test, cv=10,
scoring="accuracy")

print(scores)

# Load light skin data
transform = transforms.Compose([
    transforms.Resize((150,150)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


light_skin_only = datasets.ImageFolder(
    root='/content/drive/MyDrive/Dataset/Race Bias Cropped/Light skinned cropped',transform = transform)

m = len(light_skin_only)

print(light_skin_only.class_to_idx)
print(f"Lenght of Data: {len(light_skin_only)}")



# test only light skin data

y_pred = net.predict(light_skin_only)
y_test = np.array([y for x, y in iter(light_skin_only)])

print("accuracy score")
accuracy_score(y_test, y_pred)
print("confusion matrix")
plot_confusion_matrix(net, light_skin_only, y_test.reshape(-1, 1))
plt.show()

print(classification_report(y_test, y_pred, labels=[0,1,2,3]))

#{'(cloth) face mask': 0, 'FFP2N95KN95-type mask': 1, 'Person without a face mask': 2, 'surgical (procedural) mask': 3}


# Cross validate only dark skin data
net.fit(light_skin_only, y=y_test)
train_sliceable = SliceDataset(light_skin_only)
scores = cross_val_score(net, train_sliceable, y_test, cv=10,
scoring="accuracy")

print(scores)