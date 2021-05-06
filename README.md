# Live_Sentiment_Analysis

#### Live Sentiment Analysis using Concolutional Neural Network

 ## Convolutional Neural Network - CNN
 
 Concolutional Layer - 3

 Flattening Layer - True

    Saved model for the later use = ***fer.json*** file 

    Saved weights for the later use = ***fer.h5*** file.

    Dataset on which model is trained = ***fer2013.csv***
 
 #### fer2013.cvs - Description 
 
 ***fer2013.csv*** contains approximately 30,000 RGB facial images of different expressions with size restricted to 48x48 pixels.
 
 It has following 7 different types of label for 7 different types of facial expression: 

    **0 - Angry**

    **1 - Disgust**

    **2 - Fear**

    **3 - Happy**

    **4 - Sad**

    **5 - Surprise**

    **6 - Neutral**
 
 CNN model is trained to identitfy the above mentioned sentiments live. Model is now trained with 10 epochs due to time limitation. Its accuracy can be increased by applying very few changes in the CNN model, i.e. changing the number of *epochs*, changing the *batch size* etc etc.
