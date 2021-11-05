# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 01:24:15 2021

@author: gs935
"""


# Import the Gtts module for text  
# to speech conversion 
from gtts import gTTS 
  
# import Os module to start the audio file
import os 


def ttos(mytext):
    
      
    # Language we want to use 
    language = 'en'
      
    
    myobj = gTTS(text=mytext, lang=language, slow=False) 
      
    
    myobj.save("output.mp3") 
      
    # Play the converted file 
    os.system("start output.mp3") 