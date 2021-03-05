# Hate Speech Detection Application

According to the United Nation, the term hate speech is **understood as any kind of communication in speech, writing or behaviour, that attacks or uses pejorative or discriminatory language with reference to a person or a group on the basis of who they are, in other words, based on their religion, ethnicity, nationality, race, colour, descent, gender or other identity factor.** It is believed that hate speech is strongly associated with actual hate crime. To avoid such crime, we have to detect them before it happens.

Social media is a platform where some people publish their own opinions, some of which may be hate speech or use offensive language. Twitter is one of the most common social media; hence in this project we use tweets to analyze and predict whether they are hate speech, offensive language, or neither of them.

<p align = "center">
  <img src = "http://www.coe.int/documents/2323735/7720949/HateSpeech-webinar+cybercrime.jpg/8c9efedc-8c26-6996-cf55-915ac800282c"
       </p>
  
I use a number of machine learning algorithms to predict the results and compare their accuracy. They include Logistic Regression, Random Forest, SVM, Decision Tree, K-nearest Neighbors, and Naive Bayes. The dataset contains the below:

- count: number of experts who coded each tweet
- hate_speech: number of experts who define the tweet to be hate speech
- offensive_language: number of experts who judged the tweet to be offensive
- neither: number of experts who judged the tweet to be neither offensive nor non-offensive
- class: class label for majority of experts: 
  - 0 - hate speech
  - 1 - offensive language
  - 2 - neither
- tweet

The algorithm and application is written in python. The application uses the open-source library stremalit. The css file is the code to define some of the appearance of the application.
