"""**EDA**"""

# Commented out IPython magic to ensure Python compatibility.
import os
import spacy
import docx2txt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from textblob import TextBlob
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer

import warnings
warnings.filterwarnings('ignore')

file_path   = r'/content/drive/MyDrive/cvs/'
jpg_file    = []
folder_name = []

for folder in os.listdir(file_path):
    folder_path = file_path+folder
    for file in os.listdir(folder_path):
        if file.endswith('.jpg'):
            jpg_file.append(file)
            folder_name.append(folder)
        else:
            pdf_file.append(file)
            folder_name.append(folder)

print('Number of .jpg Files  = {}'.format(len(jpg_file)))

file_path = r'/content/drive/MyDrive/cvs/'
file_name = []
profile   = []

for folder in os.listdir(file_path):
    folder_path = file_path+folder
    for file in os.listdir(folder_path):
        if file.endswith('.jpg'):
            profile.append(folder)
            file_name.append(file)

        else:
            profile.append(folder)
            file_name.append(file)

resume_data = pd.DataFrame()
resume_data['Profile'] = profile
resume_data['Resumes'] = file_name
resume_data

resume_data.Profile.value_counts().index

resume_data.Profile.value_counts()

from matplotlib import rcParams
fig = plt.figure(figsize=(8,8))

sizes = resume_data.Profile.value_counts()
labels = resume_data.Profile.value_counts().index
colors = ['#03F6E4', '#0380F6', '#C603F6', '#F65B03'] #, '#4dc0b5', '#03F6E4', '#0380F6', '#C603F6', '#E8C110'
explode = (0.01, 0.01, 0.01, 0.01,0.01,0.01,0.01)

plt.pie(sizes, colors= colors, labels= labels, autopct= lambda x:'{:.0f}'.format(x*sizes.sum()/100),
        pctdistance= 0.85, explode= explode, startangle=0, textprops= {'size':'large', 'fontweight':'bold'})

centre_circle = plt.Circle((0,0), 0.60, fc='white')
fig.gca().add_artist(centre_circle)
plt.title('Number of Profiles in Resumes', fontsize= 18, fontweight= 'bold')
plt.legend(labels, loc="center")

pylab.rcParams.update(rcParams)
fig.tight_layout()
plt.show()

from PIL import Image
import pytesseract

def extract_text_from_image(image_path):
    try:
        # Ouvrir l'image avec la bibliothèque PIL
        img = Image.open(image_path)

        # Utiliser pytesseract pour extraire le texte de l'image
        text = pytesseract.image_to_string(img)

        return text
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")
        return None

# Exemple d'utilisation
image_text = extract_text_from_image('/content/drive/MyDrive/cvs/Data science/0a8ba46d-c2e3-4152-a7b0-240be9ed9a5d.jpg')
print(image_text)

resume_data = pd.read_csv('Cleaned_Resumes.csv')
resume_data

TextBlob(resume_data['Resume_Details'][1]).ngrams(1)[:20]

TextBlob(resume_data['Resume_Details'][1]).ngrams(2)[:20]

TextBlob(resume_data['Resume_Details'][1]).ngrams(3)[:20]

resume_data['Resume_Details']

countvec = CountVectorizer(stop_words=stopwords.words('english'), ngram_range=(1,2))
ngrams = countvec.fit_transform(resume_data['Resume_Details']) # matrix of ngrams
count_values = ngrams.toarray().sum(axis=0) # count frequency of ngrams

vocab = countvec.vocabulary_ # list of ngrams
df_ngram = pd.DataFrame(sorted([(count_values[i],k) for k, i in vocab.items()],
                               reverse=True)).rename(columns={0: 'Frequency', 1:'Unigram_Bigram'})
df_ngram.head(20)

fig, axe = plt.subplots(1,1, figsize=(12,6), dpi=200)
ax = sns.barplot(x=df_ngram['Unigram_Bigram'].head(25), y=df_ngram.Frequency.head(25), data=resume_data, ax = axe,
            label='Total Pofile Category : {}'.format(len(resume_data.Category.unique())))

axe.set_xlabel('Words', size=16,fontweight= 'bold')
axe.set_ylabel('Frequency', size=16, fontweight= 'bold')
plt.xticks(rotation = 90)
plt.legend(loc='best', fontsize= 'x-large')
plt.title('Top 25 Most used Words in Resumes', fontsize= 18, fontweight= 'bold')

for i in ax.containers:
    ax.bar_label(i,color = 'black', fontweight = 'bold', fontsize= 12)

pylab.rcParams.update(rcParams)
fig.tight_layout()
plt.show()