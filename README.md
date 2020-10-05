# Gui-Sudoku-Solver-AI</br>


![4haagu](https://user-images.githubusercontent.com/58811384/95017594-ebb03280-0677-11eb-952d-5e55e4a080d6.gif)

![4hackn (1)](https://user-images.githubusercontent.com/58811384/95018152-6c246280-067b-11eb-8dac-7bc27b276877.gif)

## Overview</br>
This is a simple fun play python GUI app to solve the Soduku using the input in the form of image , Video or the Webcam ( both from the system webcam or by using the ip webcam ) . It is build on the top of Keras API . The trained model ( ocr/model.hdf5 ) takes the picture of each digit and predict it accordingly . </br>
</br>
Opencv is highly used in it to find the block Inside the picture or the video </br>
## Motivation </br>
What could have been a perfect way to utilise the lockdown period ? Like most of my time in painting , Netflix . I thought of to start with the Deep Learning . After completing the Artificial Neural Network . I move on to CNN .So this is under that I made .

## Technical Aspects</br>
The Project is divided into three parts:
  1-> Preprocessing the data, Finding the outer box of sudoku , Using waped transform ( like that in scanner ) , Forming a hough Lines .</br>
  2 -> Forming a number image . i. e . it containg only the numbers , Then forming the 2 - Dimension array by predicting each digits inside . </br>
  3 -> Solving the sudoku and , Then finally  using insceptive wraping to print the answer on the original frame </br>
## Installation </br>
The code is written in python 3.7 . It you don't have installed you can find it on google . If you have a lower version of Python you can upgrade using the pip package , ensuring you have the latest version of pip . To install the required Packages and libraries , run teh command in the project directory after cloning the repository. </br>

### pip install -r requirements.txt
</br>
  
 ## Running the code </br>
 After following the above steps of installation . Open the terminal( cmd, powershell ) in the project directory and use the command </br> 
 ### python Gui_Final_Sudoku_solver.py
 </br>


## Main Libraries required-
Numpy ( for n-dimension array )</br>
PIL ( for image manipulation )</br>
Keras ( To train the model )</br>
Tensorflow-gpu ( Google API for deep learning )</br>
Tensorboard ( Visual analysis os model )</br>
Opencv-python ( Scientific library for Image Related stuffs )</br>
Tkinter ( GUI for the program )</br>
