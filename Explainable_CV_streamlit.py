# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 22:28:53 2023

@author: Sushovan
"""



import os
import numpy as np
import streamlit as st 
from keras.applications.resnet import ResNet50
from keras.applications.resnet import preprocess_input,decode_predictions
from keras.preprocessing import image
import scipy as sp
import matplotlib.pyplot as plt
from keras.models import Model
from PIL import Image


model=ResNet50(input_shape=(224,224,3),weights="imagenet",include_top=True)
conv_names=[]
for layer in model.layers:
    name=layer.name.split('_')[0]
    if name.find('conv')!=-1 and layer.output_shape[-1]==2048:
        conv_names.append(layer.name)



def load_image(image_file):
    image = Image.open(image_file)
    image=image.resize((224,224))
    img_array = np.array(image)
    return img_array

# Uploading the File to the Page
uploadFile = st.sidebar.file_uploader(label="Upload image", type=['jpg', 'png'])

# Checking the Format of the page
if uploadFile is not None:
    img = load_image(uploadFile)
    #st.image(img)
    img=np.expand_dims(img, axis=0)
    model_image=preprocess_input(img)
    pred=model.predict(model_image)
    pred_op=decode_predictions(pred)[0]
    classes=pred_op[0][1]
    st.write('predicted class :'+ classes)
    layer_names=st.sidebar.selectbox(label='select conv layer for calculation', 
                                     options=np.array(conv_names),
                                     index=(len(conv_names)-1))

    output_layer=model.get_layer(layer_names) 
    feature_analysis_model=Model(inputs=model.input,outputs=output_layer.output)

    final_layer=model.get_layer('predictions')
    weights=final_layer.get_weights()
    coeffs=weights[0]
    bias=weights[1]
    
    features=feature_analysis_model.predict(model_image)
    weight_vec=coeffs[:,highest_index_prob]
    bias_num=bias[highest_index_prob]
    
    feature_sub=features[0]
    

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
