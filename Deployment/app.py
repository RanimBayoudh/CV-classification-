# IMPORT LIBRARIES
import re
import PyPDF2
import docx2txt
import pdfplumber
import pandas as pd
import streamlit as st

import en_core_web_sm
nlp = en_core_web_sm.load()
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

#----------------------------------------------------------------------------------------------------

import pickle as pk

# Charger le modèle
model = pk.load(open('model_GradientBoost.pkl', 'rb'))

# Charger le vectorizer
Vectorizer = pk.load(open('vector.pkl', 'rb'))

categories = ['télécommunications', 'Data science', 'Génie logiciel', 'IOT', 'Electronique et systéme embarqués',
              'mecanique', 'technologies de linformation']
select = st.selectbox('Fields', categories)

upload_file = st.file_uploader('Upload Your Resumes', type=['jpg'], accept_multiple_files=True)

for img_file in upload_file:
    if img_file is not None:
        filename.append(img_file.name)
        # Prétraiter le fichier image (utilisez une bibliothèque comme OpenCV ou Tesseract pour extraire le texte)
        cleaned_text = preprocess(process_image(img_file))
        prediction = model.predict(Vectorizer.transform([cleaned_text]))[0]
        predicted.append(prediction)
        skills.append(extract_skills(cleaned_text))

file_type['Uploaded File'] = filename
file_type['Skills'] = skills
file_type['Predicted Profile'] = predicted

st.table(file_type[file_type['Predicted Profile'] == select])
