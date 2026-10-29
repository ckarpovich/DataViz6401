# Chris Karpovich - Week 8:
# Run with:  streamlit run Chris_Karpovich_HW8.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from bertopic import BERTopic
import umap
import plotly.express as px
from sklearn.datasets import fetch_olivetti_faces
import streamlit.components.v1 as components
import numpy as np
from sklearn.linear_model import LogisticRegression

faces = fetch_olivetti_faces()

@st.cache_resource
def train_model():
    model = LogisticRegression(max_iter=1000)
    model.fit(faces.data, faces.target)
    return model


pic = faces.images[[10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42, 50, 51, 52, 60, 61, 62]]
label = faces.target[[10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42, 50, 51, 52, 60, 61, 62]]

#Montage
st.header("Face Image Montage")

fig, axes = plt.subplots(3, 6, figsize=(8, 4))
for ax, img, person in zip(axes.ravel(), pic, label):
    ax.imshow(img, cmap="gray"); ax.set_title("Person #" + str(person)); ax.axis("off")

st.pyplot(fig)
plt.close(fig)

st.caption("This montage shows three images from three different people photographed in the dataset.  It demonstrates that facial expressions change across images, as well as angle.  Some also have accessories, such as glasses.  The photos all do appear relatively similar, with similiar framing and brightness, especially when the photos are all in grayscale.")

#Intensity

st.header("Pixel Intensity Distribution")

img = pic[0]

fig, axes = plt.subplots()
plt.hist(img.ravel(), bins=50, alpha=0.5, label="Grayscale")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.legend()

st.pyplot(fig)
plt.close(fig)

st.caption("The histogram shows distribution of pixel intensities in a selected face image.  It reveals whether they contain mostly dark or light pixels.  Since the histogram is centered around 0.6, it shows that the majority of the pixels have a high brightness.  It is also common for a few dark pixels to be present, especially in the background of the faces, which can be seen in the montage.")

#Saliency

def occlude(model, img, cls, patch=3):
    base = model.predict_proba([img.ravel()])[0][cls]
    sal, r = np.zeros(img.shape), patch // 2
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            t = img.copy()
            t[max(0, i - r):i + r + 1, max(0, j - r):j + r + 1] = 0
            sal[i, j] = base - model.predict_proba([t.ravel()])[0][cls]
    return sal

model = train_model()

img = pic[0]
cls = int(model.predict([img.ravel()])[0])
sal = occlude(model, img, cls)

og, sali = st.columns(2)

with og:
    fig, ax = plt.subplots()
    ax.imshow(img, cmap = "gray")
    ax.set_title("Original Picture")

    st.pyplot(fig)
    plt.close(fig)

with sali:
    fig, ax = plt.subplots()
    ax.imshow(sal, cmap = "inferno")
    ax.set_title("Occlusion Saliency Map")

    st.pyplot(fig)
    plt.close(fig)

    st.caption("The visualization shows us which parts of a face image influence a logistic regression model's predictions.  The darker regions show where the model's confidence increases, and vice versa for lighter areas.  Looking at the image, those regions that increase the model's confidence include nose, mouth, and accessories, such as the glasses that are present in the picture.")