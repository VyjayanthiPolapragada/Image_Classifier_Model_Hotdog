Training a model to classify images using transfer learning with python

Model is trained to classify given pictures into two categories: Hotdog and not Hotdog

Dataset used is provided as a zip file (need not extract, as we can do that in python file itself)

Used different python files to prepare the data, train the model and test to succesfully classify the given images

1. display_image.py : contains customized function to display the images that we try to load/loaded succesfully

2. dataset_hotdog.py : used to tranform and preprocess data to load into the model (zip file is extracted here)

3. image_classifier_model.py: a framework is developed to classify the given images, this framework is used to train,validate and test the model

4. main_code.py : main python script to incorporate the above mentioned functions/classes and perform training, testing of the model

Libraries used: pytorch,numpy,pandas,requests,typing,zipfile,os,matplotlib,tensor,copy (except pytorch, other libraries are included in Jupyter notebook)

Outcomes:

This trained image classifier model, has an accuracy of 84.44% to classify an image it has never seen before into a hotdog or not hotdog!







