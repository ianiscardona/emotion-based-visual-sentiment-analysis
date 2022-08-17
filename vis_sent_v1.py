#----------------------------------------------------------------------------------------------------
#                 IAN CARDONA
#                 linkedin: https://www.linkedin.com/in/ian-cardona-863a3b1a6/
#                 github: https://github.com/ianiscardona
# 
#                 2022
#----------------------------------------------------------------------------------------------------

#import the important libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2 as cv
import glob
from deepface import DeepFace

#set parameters for figures that will be used (10x10in = 800x800px)
plt.rcParams["figure.figsize"] = (10,10)

images_list = []
participants_list = []
emotions_list = []
sentiments_list = []

column1 = "Participants"
column2 = "Emotions"
column3 = "Sentiment"

#select the path
path = "Visual Sentiment Analysis/images/*.*"


#start an iterator for large number 
img_number = 1 #this number can be later added to output image file names
event_name = input('Event name: ')

def sentiment_analysis(face_info):
    analyze_output = {
        "happy": "satisfied",
        "neutral": "neutral",
        "sad": "not satisfied",
        "fear": "not satisfied",
        "angry": "not satisfied",
        "surprised": "not satisfied",
        "disgust": "not satisfied",
    }

    return analyze_output.get(face_info, "nothing")

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

    #show current image
    plt.imshow(image_read_brg2rgb)
    plt.show()

#the lists to a pandas dataframe
results_list = pd.DataFrame({column1:participants_list,column2:emotions_list,column3:sentiments_list}) 
results_list.to_excel('results_batch1.xlsx', sheet_name= event_name, index= False) #save to excel
