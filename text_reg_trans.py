import numpy as np
import streamlit as st
from PIL import Image
import pytesseract
import pdf2image
from googletrans import Translator 
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import cv2
import skimage.transform
from keras.models import load_model
from test import suggest_word

# load model
model = load_model('CNN_model.h5')


with st.sidebar:
    st.markdown("<div><img src='https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExNjI0MDVkZjhmZDgzMTRkNzlhYTMyYjllMGY1OGM3ZmZjYjMzNjc0MyZlcD12MV9pbnRlcm5hbF9naWZzX2dpZklkJmN0PXM/UvPvsX9oMlMWs/giphy.gif' width = 200></h1></div>", unsafe_allow_html=True)
    st.title("English Machine")
    choice = st.radio("Choose Function", ["Translator", "Spelling"])
    st.info("This app can accelerate the learning process English learners")

if choice == "Translator":
    st.title("Page Translator")
    pdf_file = st.file_uploader("Choose a pdf file", type=["pdf"])

    if pdf_file is not None:
        poppler_path = r"C:\poppler-0.68.0\bin"
        images = pdf2image.convert_from_bytes(pdf_file.read(), poppler_path=poppler_path)
        
        translator = Translator() 
        
        for i, image in enumerate(images):
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption=f"Page {i+1}")

            with col2:
                myconfig = r"--psm 6 --oem 3"
                text = pytesseract.image_to_string(image, config=myconfig, lang='deu')
                translation = translator.translate(text, src='en', dest='vi') 
                st.write(translation.text) 
            st.write('---')
            

    
if choice == "Spelling":
    st.title("Enhance your spelling")
    
    co1, co2, co3 = st.columns(3)
    if "score" not in st.session_state:
        st.session_state["score"] = 10
    if "count" not in st.session_state:
        st.session_state["count"] = 0
    
    synonym_list, selected_word = suggest_word()
    if st.button('Random word now'):
        st.write(f'The length of the word is {len(selected_word)}')
        st.write('relevant word:', synonym_list[st.session_state["count"]])
    
    
    if st.button('Reveal more relevant words'):
        st.session_state["count"] = st.session_state["count"] + 1
        st.write(synonym_list[st.session_state["count"]])
    
    if st.button('Reveal the word now'):
        st.write(selected_word)
        st.session_state["score"] = 0
        st.session_state["count"] = 0
    
    score = int(st.session_state["score"])
    count = int(st.session_state["count"])
    st.info(f"Your current score: {score - count} points")
    
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 10)
    realtime_update = st.sidebar.checkbox("Update in realtime", True)

    result_str = ""
    num_cols = st.number_input ('Enter the number of the word length', min_value=7, max_value=12, value=8, step=1)
    cols = st.columns(num_cols)
    
    for i, col in enumerate (cols):
        # Create a canvas component
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color="#ffffff",
            background_color= "#000000",
            update_streamlit=realtime_update,
            drawing_mode="freedraw",
            height=150,
            width=150,
            key="canvas" + str(i),
        )
        if canvas_result.image_data is not None:
            gray = cv2.cvtColor(canvas_result.image_data, cv2.COLOR_BGR2GRAY)
            resized = skimage.transform.resize(gray, output_shape=(28, 28))
            input = resized.reshape((1,28,28,1))
            prediction = model.predict(input)
            label = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'] 
            st.write(label[np.argmax(prediction)])
            result_str += label[np.argmax(prediction)]
            
  
    st.info(result_str)
    translator = Translator() 
    translation = translator.translate(result_str, src='en', dest='vi') 
    st.info(translation.text) 
    if selected_word == result_str.lower():
        st.success("Correct")
    else:
        st.error("Incorrect")
    