#!pip install pyreadstat
import warnings 
warnings.filterwarnings("ignore")
import streamlit as st
from project import clean_data, vectorizer, model_lr
from PIL import Image

# set up the page

st.set_page_config(
        page_title = "Hate Speech Detection Application",
        page_icon = "tbs_icon.png",
    )

# define the style of UI with css file in html

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html = True)

local_css("style.css")

image = Image.open('tbs.png')
st.image(image)

# set the appilcation's title

st.title('Hate Speech Detection Application')

# set the subheader

st.write("An application to detect whether the text input is hate speech, offensive language, or neither.")

# define user input of strings

user_input = st.text_input("Your text input:")

# perform data cleaning and transformation for detection

input_clean = clean_data(user_input)
user_tfidf = vectorizer.transform(input_clean)
user_pred = model_lr.predict(user_tfidf)

# display the type of speech
       
if (st.button("Start Detection")):
    st.text("Your result: ")
    
    try:
        
        # detect whether there is any unrecognizable word
        
        if user_tfidf.data.any() == False:
            st.info("Neither")
        
        # detect the type of speech
        
        else:
            if user_pred == 0:
                st.error("Hate Speech")
            elif user_pred == 1:
                st.warning("Offensive Language")
            else:
                st.info("Neither")
    except:
        st.text("Invalid input. Please type again and make sure it's in English.")
