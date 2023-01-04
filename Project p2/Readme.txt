COMP 472 - Project, Part 1: Readme


This document contains the specifics of how to run the source code provided in training as well as application mode, including generating the evaluation results.


Link to trained model to download:
https://drive.google.com/file/d/1vCRP5Jmq32IY1qbuymwowXDNAgtVTWsk/view?usp=sharing
Link to the full dataset (1600 images)
https://drive.google.com/drive/folders/1wO8CYo5M_Yiv7bHRvah_yps2S2GgbYGp?usp=sharing


All files:
*  CNN.py - Source code
* input/train - Folder for training data (not the full data, only 400 images)
* README - The .txt readme file
* Part 1 Report.pdf - The report


How to run the project: 
Ensure you have installed on your environment:
* matplot lib
* tkinder (pip install tk)
* PIL (pip install pillow)
Now that the above requirements are met, navigate to the project folder and run the command: python CNN.py


There are 3 modes, which can be accessed by typing 1, 2, and 3 respectively: 
1. to train the model, display metrics and save it. 
2. to and predict an image using a pretrained model. 
3. to predict the test dataset and display metrics 


Mode 1:
This mode will train a new model on the 400 images we provided in this submission.
It will then try to predict the test data that was split automatically and output a confusion matrix.
It will save the trained model as “cnnTrained +(the current time).pth”  in the same directory as the python file.


Mode 2:
In this mode, you first will be prompted to select a saved model using the file explorer. Select a pretrained model (.pth file) and click open. Then you will now be prompted to select an image to predict. We recommend you select an image from our Sample Images folder. Note: make sure to select the model before you select the image, else you will have an error.


Mode 3:
In this mode, you will also be prompted to select a saved model. To use the trained model, download the .pth file using the link above and drag it into the same directory as the python file. Then, It will then try to predict the testing data and output the accuracy metrics.