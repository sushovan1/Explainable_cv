# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 22:28:53 2023

@author: CSU5KOR
"""



import os
import json
import numpy as np
import streamlit as st 
from keras.applications.resnet import ResNet50
from keras.applications.resnet import preprocess_input,decode_predictions
from keras.preprocessing import image
import scipy as sp
import matplotlib.pyplot as plt
from keras.models import Model
from io import StringIO
from PIL import Image
# Function to Read and Manupilate Images

data_dir=r'C:\Users\CSU5KOR\OneDrive - Bosch Group\CV_Training_Udemy\Feature_maps_experiments'
weights_dir=r'C:\Users\CSU5KOR\OneDrive - Bosch Group\CV_Training_Udemy'
model=ResNet50(input_shape=(224,224,3),weights=os.path.join(weights_dir,'resnet50_weights_tf_dim_ordering_tf_kernels.h5'),include_top=True)
conv_names=[]
for layer in model.layers:
    name=layer.name.split('_')[0]
    if name.find('conv')!=-1 and layer.output_shape[-1]==2048:
        conv_names.append(layer.name)

file_name=os.path.join(data_dir,'imagenet_class_index.json')
with open(file_name,'rb') as f:
    class_data=json.load(f)

def load_image(image_file):
    image = Image.open(image_file)
    image=image.resize((224,224))
    img_array = np.array(image)
    return img_array

# Uploading the File to the Page
uploadFile = st.sidebar.file_uploader(label="Upload image", type=['jpg', 'png'])

# Checking the Format of the page
if uploadFile is not None:
    # Perform your Manupilations (In my Case applying Filters)
    img = load_image(uploadFile)
    #st.image(img)
    img=np.expand_dims(img, axis=0)
    model_image=preprocess_input(img)
    pred=model.predict(model_image)
    #pred_prob=pred[0][np.argmax(pred)]
    highest_index_prob=np.argmax(pred)
    classes=class_data[str(highest_index_prob)][1]
    st.write('predicted class :'+ classes)
    layer_names=st.sidebar.selectbox(label='select conv layer for calculation', 
                                     options=np.array(conv_names),
                                     index=(len(conv_names)-1))
    #get the output from the last layer before flatten-Check summary for the names
    output_layer=model.get_layer(conv_names[-1]) #'activation_49'
    feature_analysis_model=Model(inputs=model.input,outputs=output_layer.output)

    final_layer=model.get_layer('predictions')
    #get the weights
    weights=final_layer.get_weights()
    coeffs=weights[0]
    bias=weights[1]
    #trying to gauge the features
    features=feature_analysis_model.predict(model_image)
    weight_vec=coeffs[:,highest_index_prob]
    bias_num=bias[highest_index_prob]
    #estimate the features
    feature_sub=features[0]
    #trial=weight_vec[0]*feature_sub[:,:,0]

    test_sum=features[0].dot(weight_vec)

    zoom_factor=224//test_sum.shape[0]	

    final_feature_img=sp.ndimage.zoom(test_sum,(zoom_factor,zoom_factor),order=1)
    fig=plt.figure()
    fig.add_subplot(1,2,1)
    plt.imshow(img[0]/255, alpha=0.8)
    plt.imshow(final_feature_img, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.title('feature maps')
    fig.add_subplot(1,2,2)
    plt.imshow(img[0]/255)
    plt.axis('off')
    plt.title('original images')
    #fig.show()
    st.pyplot(fig)
    
    #st.write(classes)
else:
    st.write("Make sure you image is in JPG/PNG Format.")
