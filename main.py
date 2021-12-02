#!/usr/bin/env python
# encoding: utf-8

# @author: Zhipeng Ye
# @contact: Zhipeng.ye19@xjtlu.edu.cn
# @file: main.py
# @time: 2021-11-29 09:44
# @desc:

import os

import cv2
import streamlit as st
from sklearn.externals import joblib
from streamlit_drawable_canvas import st_canvas
import processing as ps

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
if not os.path.isdir(MODEL_DIR):
    os.system('runipy train.ipynb')


@st.cache(allow_output_mutation=True)
def load_model():
    model = joblib.load('mlp.pkl')
    return model


# st.markdown('<style>body{color: White; background-color: DarkSlateGrey}</style>', unsafe_allow_html=True)
model = load_model()
st.title('My Digit Recognizer')
st.markdown('''
Try to write a digit!
''')

SIZE = 192
RESIZE = 28
mode = st.checkbox("Draw (or Delete)?", True)
canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=20,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=SIZE,
    height=SIZE,
    drawing_mode="freedraw" if mode else "transform",
    key='canvas')

if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (RESIZE, RESIZE), interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # st.write(img.shape)
    # st.write(img)
    st.write(img.shape)
    img = ps.centering(img)
    # st.image(img, clamp = True)
    #
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
    rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    st.write('Model Input')
    st.image(rescaled, clamp = True)

if st.button('Predict'):

    test_x = img.reshape(1, RESIZE * RESIZE)
    st.write(test_x.shape)
    st.write(test_x)
    val = model.predict(test_x)
    # val = model.predict(test_x.reshape(1, 28, 28))
    st.write(f'result: {val}')
