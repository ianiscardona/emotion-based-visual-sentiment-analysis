#----------------------------------------------------------------------------------------------------
#                 IAN CARDONA
#                 linkedin: https://www.linkedin.com/in/ian-cardona-863a3b1a6/
#                 github: https://github.com/ianiscardona
# 
#                 2022
#----------------------------------------------------------------------------------------------------

#import the important libraries
from cgitb import text
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import tkinter
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2 as cv
import glob
from deepface import DeepFace

#set parameters for figures that will be used (10x10in = 800x800px)
plt.rcParams["figure.figsize"] = (10,10)

#lists as data structure & data storage
showimages_list = []
images_list = []
participants_list = []
emotions_list = []
sentiments_list = []

#column names for excel file
column1 = "Participants"
column2 = "Emotions"
column3 = "Sentiment"

# Main sentiment analysis classification process
def sentiment_analysis(face_info):
    analyze_output = {
        "happy": "satisfied",
        "neutral": "neutral",
        "sad": "not satisfied",
        "fear": "not satisfied",
        "angry": "not satisfied",
        "surprise": "not satisfied",
        "disgust": "not satisfied",
    }

    return analyze_output.get(face_info, "nothing")

# Directory finder
def select_folder():
    global path
    global current_directory_label
    images_folder = filedialog.askdirectory()

    path = images_folder
    path = path + "/*.*"

    current_directory_label = tkinter.Label(
    root, 
    text="Current folder directory:\n " + path
    ).grid(row=3, column=0, padx= 25)
    return path

# Main image processing method
def process_images():
    print(path)
    global process_done_label
    tkinter.messagebox.showinfo(title=None, message="Please wait. Do not exit the program.")
    for file in glob.glob(path): #iterate through each file
        filename = os.path.basename(file) #set the name
        print(filename)

        participants_list.append(filename) #add to participants

        image_read = cv.imread(file) #read each file
        image_read_brg2rgb = cv.cvtColor(image_read, cv.COLOR_BGR2RGB)

        images_list.append(image_read_brg2rgb) #add the images to the list

        #using deepface library, analyze the image
        prediction = DeepFace.analyze(image_read_brg2rgb)

        #only the dominant emotion is needed
        face_info = prediction['dominant_emotion']

        #execute the analyzation of infos as standalone
        if __name__ == "__main__":
            emotions_list.append(face_info) #add the emotion result
            sentiments_list.append(sentiment_analysis(face_info)) #add the sentiment result
            print(sentiment_analysis(face_info))

    #the lists to a pandas dataframe
    results_list = pd.DataFrame({column1:participants_list,column2:emotions_list,column3:sentiments_list}) 
    results_list.to_excel('results_batch1.xlsx', sheet_name= "event_name", index= False) #save to excel
    tkinter.messagebox.showinfo(title=None, message="The images have been processed and saved into an excel file.")
    # dislay done in the GUI
    process_done_label = tkinter.Label(root, text="The images have been processed.").grid(row=7, column=0, padx= 25)

    return print("Done")

def selected_images_folder():
    global label_var
    global text_var
    global back_button
    global forward_button
    global folder_length1
    
    # if len(showimages_list) == 0:
    #     return print("invalid")
    for file_image in glob.glob(path):
        process_read_image = Image.open(file_image)
        
        current_image= ImageTk.PhotoImage(process_read_image.resize((300,300)))
        showimages_list.append(current_image)
        print("file")
    folder_length1 = len(showimages_list)

    label_var= tkinter.Label(
        image= showimages_list[0],
    )
    label_var.grid(row= 1, column= 2, columnspan= 3, rowspan= 7, padx= 5)
    text_var= tkinter.Label(
        text= participants_list[0] + "\n Emotion: " + emotions_list[0] + "\n Sentiment: " + sentiments_list[0]
    )
    text_var.grid(row= 8, column= 2, columnspan= 3, rowspan= 3)
    forward_button = tkinter.Button(
        root,
        text=">>",
        command= lambda : forward(1)
    )
    forward_button.grid(row = 1, column = 5, pady = (0, 50), padx = 25)
    back_button = tkinter.Button(
        root,
        text="<<",
        command= lambda: backward,
        state=DISABLED
    )
    back_button.grid(row = 1, column = 1, pady = (0, 50), padx = 25)


def forward(image_number):
    global label_var
    global text_var
    global back_button
    global forward_button

    label_var.grid_forget()
    text_var.grid_forget()
    label_var= tkinter.Label(
        image= showimages_list[image_number]
    )
    label_var.grid(row= 1, column= 2, columnspan= 3, rowspan= 7, padx= 5)
    text_var= tkinter.Label(
        text= participants_list[image_number] + "\n Emotion: " + emotions_list[image_number] + "\n Sentiment: " + sentiments_list[image_number]
    )
    text_var.grid(row= 8, column= 2, columnspan= 3, rowspan= 3)
    if image_number == folder_length1-1:
        forward_button = tkinter.Button(root,text=">>",state=DISABLED)
    else:
        forward_button = tkinter.Button(
            root,
            text=">>",
            command= lambda : forward(image_number+1)
        )
    forward_button.grid(row = 1, column = 5, pady = (0, 50), padx = 25)
    if image_number == 0:
        back_button=Button(root,text="<<",state=DISABLED)
    else:
        back_button = tkinter.Button(
            root,
            text="<<",
            command = lambda: backward(image_number-1),
        )
    back_button.grid(row = 1, column = 1, pady = (0, 50), padx = 25)


    
def backward(image_number):
    global label_var
    global text_var
    global back_button
    global forward_button

    label_var.grid_forget()
    text_var.grid_forget()
    label_var = tkinter.Label(
        image= showimages_list[image_number]
    )
    label_var.grid(row= 1, column= 2, columnspan= 3, rowspan= 7, padx= 5)
    text_var= tkinter.Label(
        text= participants_list[image_number] + "\n Emotion: " + emotions_list[image_number] + "\n Sentiment: " + sentiments_list[image_number]
    )
    text_var.grid(row= 8, column= 2, columnspan= 3, rowspan= 3)
    if image_number == folder_length1-1:
        forward_button=Button(root,text=">>",state=DISABLED)
    else:
        forward_button = tkinter.Button(
            root,
            text=">>",
            command= lambda : forward(image_number+1)
        )
    forward_button.grid(row = 1, column = 5, pady = (0, 50), padx = 25)
    if image_number == 0:
        back_button=Button(root,text="<<",state=DISABLED)
    else:
        back_button = tkinter.Button(
            root,
            text="<<",
            command = lambda: backward(image_number-1),
        )
    back_button.grid(row = 1, column = 1, pady = (0, 50), padx = 25)

    

# create the root window
root = tkinter.Tk()
root.title('VSA for Online Meeting')

## FRAME 1
#Texts and padys
title_label = tkinter.Label(
    root, 
    text="Wanna check the vibe of your last online meeting?"
).grid(row=0, column=0, columnspan=6, pady= 50, padx= 25)

step1_label = tkinter.Label(
    root, 
    text="Step 1:"
).grid(row=1, column=0, padx= 25)
pady1_label = tkinter.Label(
    root,
).grid(row=4, column=0, pady=(0,50), padx= 25)

step2_label = tkinter.Label(
    root, 
    text="Step 2:"
).grid(row=5, column=0, padx= 25)
pady2_label = tkinter.Label(
    root,
).grid(row=8, column=0, pady=(0,50), padx= 25)

step3_label = tkinter.Label(
    root, 
    text="Step 3:"
).grid(row=9, column=0, padx= 25)

#clickables
open_button = tkinter.Button(
    root,
    text='Select Folder Location',
    width=20,
    command=select_folder
).grid(row= 2, column=0, padx= 25)
process_button = tkinter.Button(
    root,
    text='Process Folder',
    width=20,
    command= process_images
).grid(row = 6, padx= 25)
view_images = tkinter.Button(
    root, 
    text='Show Images',
    width=20,
    command = selected_images_folder
).grid(row=10, pady=(0,50), padx=25)
exit_button = tkinter.Button(
    root,
    text="Exit",
    command=root.destroy
).grid(row = 11, pady = (0,50), padx= 25)

# run the application
root.mainloop()
