# Hate Speech Detection Project
    
# !pip install streamlit
import warnings 
warnings.filterwarnings("ignore")
import pandas as pd
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer 
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from textstat.textstat import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix

#%%

# read dataset1 csv file

data_tweet = pd.read_csv("dataset/Dataset1.csv", sep = ",")
data_tweet = data_tweet[['class', 'tweet']]

#data_tweet.head()

# summary of the dataset

#data_tweet.info()

#%%

"""
# plot the graph of number of tweets of each class

count_class = pd.DataFrame({'class_0': [np.sum(data_tweet['class'] == 0)], 'class_1': [np.sum(data_tweet['class'] == 1)],\
     'class_2': [np.sum(data_tweet['class'] == 2)]})

x, y = count_class.columns, count_class.iloc[0]
plt.bar(x, y, width = 0.3, color = 'green')
plt.title('Number of Tweets of Each Class')
plt.ylabel('Number of Tweets')
plt.show()

# number of words

df1 = pd.DataFrame()

df1['number_of_words'] = data_tweet['tweet'].apply(lambda x: len(str(x).split(" ")))
df2 = pd.concat([data_tweet['tweet'], data_tweet['class'], df1], axis = 1)

# number of characters

df2['number_of_characters'] = data_tweet['tweet'].str.len()
df2

# average word length

def avg_word_len(tweet):
    words = tweet.split()
    return (sum(len(word) for word in words)/len(words))

df2['average_word_length'] = data_tweet['tweet'].apply(lambda x: avg_word_len(x))
df2

# visualize the number of words

graph1 = sns.FacetGrid(df2, col = 'class')
graph1.map(plt.hist, 'number_of_words', bins = 30)

# visualize the number of characters

graph1 = sns.FacetGrid(df2, col = 'class')
graph1.map(plt.hist, 'number_of_characters', bins = 50)

# visualize the average word length

graph2 = sns.FacetGrid(df2, col = 'class')
graph2.map(plt.hist, 'average_word_length', bins = 40)
"""

#%%

# create a optimal stopword list

stop = set(stopwords.words('english'))
negation = set(['not', 'don', "don't", 'ain', 'aren', "aren't", "couldn't", 'didn', "didn't", 'doesn', "doesn't", 
            'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",'mightn', "mightn't", 'mustn',
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
            "weren't", 'won', "won't", 'wouldn', "wouldn't", 'couldn'])
stop = list(stop - negation)

# self-define a stopword list 

more_stopwords = ["english", "rt", 'ya', 'yea', 'another', 'yo', 'us', 'cant', 'im', 'yeah', 'tweets', 'tweet', 
                  'lmao', 'Lmao', 'LMAO', 'u', 'lol', 'amp', 'thats', 'youre', 'would']

# combine the two stopword lists together

for i in range(len(more_stopwords)):
    stop.append(more_stopwords[i])
    
#%%

# # define a function to clean data and make it suitable for modelling

def clean_data(dataset):
    
    dataset = pd.Series(dataset)

    # make all words to lowercase, and remove redundant space

    data_tweet_str = dataset.apply(lambda x: ' '.join(x.lower().strip() for x in x.split()))

    # remove user names

    data_tweet_wo_user = data_tweet_str.apply(lambda x: " ".join(x for x in x.split() if x[0] != '@'))

    # remove all punctuations

    data_tweet_str_wo_punc = data_tweet_wo_user.str.replace('[^\w\s]','')

    # remove stopwords

    data_tweet_str_wo_punc_wo_stop = data_tweet_str_wo_punc.apply(lambda x: " ".join(x for x in x.split() if x not in stop))

    # remove web address

    data_tweet_str_wo_punc_wo_stop_wo_http = data_tweet_str_wo_punc_wo_stop.apply(lambda x: " ".join(x for x in x.split() \
                                                                                                 if x[0:4] != 'http'))
    # remove numbers and words starting with numbers

    number = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

    data_tweet_str_wo_punc_wo_stop_wo_http_wo_no = data_tweet_str_wo_punc_wo_stop_wo_http\
                                                    .apply(lambda x: " ".join(x for x in x.split() \
                                                            if x[0] not in number))

    # lemmatization

    lemmatizer = WordNetLemmatizer() 

    dataset_clean = data_tweet_str_wo_punc_wo_stop_wo_http_wo_no \
                    .apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
    
    return (dataset_clean)

# combine clean tweets with original dataset

data_tweet['clean_tweets'] = clean_data(data_tweet['tweet'])
#data_tweet[['tweet','clean_tweets']]

#%%

"""
# most frequent words

freq = pd.Series(' '.join(data_tweet['clean_tweets']).split()).value_counts()[:10]
plt.bar(freq.index, freq, color = 'green')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Most Frequent Words')
plt.show()

# visualize most frequent words in all tweets

mask = np.array(Image.open('heart.png'))

all_words = ' '.join([text for text in data_tweet['clean_tweets'] ])
wordcloud = WordCloud(width = 800, height = 800, background_color = "white", mask = mask).generate(all_words)
plt.figure(figsize = (10, 10))
plt.imshow(wordcloud, interpolation = "bilinear")
plt.axis('off')
plt.show()

# overall tweet sentiment

overall_sentiment = TextBlob(str(data_tweet['clean_tweets'])).sentiment[0]
print("The overall sentiment of all tweets is: " + str(overall_sentiment), "1: positive", "-1: negative", sep = "\n")

"""

from data_preparation import hate_speech_words, offensive_words, fox_comments, microposts_final, linkedin_post

# combine other datasets with the original dataset of tweets

hate_speech_words['clean_tweets'] = clean_data(hate_speech_words['tweet'])
offensive_words['clean_tweets'] = clean_data(offensive_words['tweet'])
fox_comments['clean_tweets'] = clean_data(fox_comments['tweet'])
microposts_final['clean_tweets'] = clean_data(microposts_final['tweet'])
linkedin_post['clean_tweets'] = clean_data(linkedin_post['tweet'])

data_tweet = pd.concat([data_tweet, hate_speech_words, offensive_words, fox_comments, microposts_final, linkedin_post]).reset_index(drop = True)
data_tweet.columns = ['class', 'texts', 'clean_texts']

# convert the tweets into statistical numbers

vectorizer = TfidfVectorizer(ngram_range = (1, 4), max_features = 10000, use_idf = True)
tfidf = vectorizer.fit_transform(data_tweet['clean_texts'])

#%%

# data split

X = tfidf
y = data_tweet['class'].astype(int)
x_train_tfidf, x_test_tfidf, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# Logistic Regression 

model_lr = LogisticRegression(C = 2, max_iter = 100).fit(x_train_tfidf, y_train)
y_pred_lr = model_lr.predict(x_test_tfidf)
report_lr = classification_report(y_test, y_pred_lr)

"""
print(report_lr)
acc_lr = accuracy_score(y_test, y_pred_lr)
label_lr = "Logistic Regression"
print(label_lr)
print('Accuracy: ', acc_lr)

# Decision Tree

model_dt = tree.DecisionTreeClassifier().fit(x_train_tfidf, y_train)
y_pred_dt = model_dt.predict(x_test_tfidf)
report_dt = classification_report(y_test, y_pred_dt)

print(report_dt)
acc_dt = accuracy_score(y_test, y_pred_dt)
label_dt = 'Decision Tree'
print(label_dt)
print('Accuracy: ', acc_dt)

# kNN

model_knn = KNeighborsClassifier(n_neighbors = 3, weights = 'distance', p = 1).fit(x_train_tfidf, y_train)
y_pred_knn = model_knn.predict(x_test_tfidf)
report_knn = classification_report(y_test, y_pred_knn)

print(report_knn)
acc_knn = accuracy_score(y_test, y_pred_knn)
label_knn = 'K-nearest Neighbors'
print(label_knn)
print('Accuracy: ', acc_knn)

# Random Forest

model_rf = RandomForestClassifier(n_estimators = 200).fit(x_train_tfidf, y_train)
y_pred_rf = model_rf.predict(x_test_tfidf)
report_rf = classification_report(y_test, y_pred_rf)

print(report_rf)
acc_rf = accuracy_score(y_test, y_pred_rf)
label_rf = 'Random Forest'
print(label_rf)
print('Accuracy: ', acc_rf)

# SVM

model_svm = LinearSVC().fit(x_train_tfidf, y_train)
y_pred_svm = model_svm.predict(x_test_tfidf)
report_svm = classification_report(y_test, y_pred_svm)

print(report_svm)
acc_svm = accuracy_score(y_test, y_pred_svm)
label_svm = 'SVM'
print(label_svm)
print('Accuracy: ', acc_svm)

# Naive Bayes

x_train_tfidf, x_test_tfidf, y_train, y_test = train_test_split(X.toarray(), y, random_state = 42, test_size = 0.3)
model_nb = GaussianNB().fit(x_train_tfidf, y_train)
y_pred_nb = model_nb.predict(x_test_tfidf)
report_nb = classification_report(y_test, y_pred_nb)

print(report_nb)
acc_nb = accuracy_score(y_test, y_pred_nb)
label_nb = 'Naive Bayes'
print(label_nb)
print('Accuracy: ', acc_nb)

#%%

# summary of modelling results

d0 = pd.DataFrame([label_lr, label_dt, label_knn, label_rf, label_svm, label_nb])
d1 = pd.DataFrame([acc_lr, acc_dt, acc_knn, acc_rf, acc_svm, acc_nb])
d2 = pd.concat([d0, d1], axis = 1)
d2.columns = ['Model', 'Accuracy']
d2 = d2.sort_values(by = ['Accuracy'], ascending = False)

display(d2)
print('The best model is', d2.iloc[0, 0])

# plot the graph of accuracy of algorithms

plt.figure(figsize = (10, 6))
plt.bar(d2['Model'], d2['Accuracy'], width = 0.5, color = 'green')
plt.title('Accuracy of Algorithms')
plt.ylabel('Accuracy')

lower_bound = d2.iloc[-1, -1] - 0.05
upper_bound = d2.iloc[0, 1] + 0.05
plt.ylim(lower_bound, upper_bound)
plt.show()

# visualize the confusion matrix of the most accurate algorithm, 

confusion_matrix_final = confusion_matrix(y_test, y_pred_lr)
matrix_proportions = np.zeros((3, 3))
for i in range(0,3):
    matrix_proportions[i, :] = confusion_matrix_final[i, :] / float(confusion_matrix_final[i, :].sum())
names = ['Hate', 'Offensive', 'Neither']
confusion_df = pd.DataFrame(matrix_proportions, index = names, columns = names)
plt.figure(figsize = (6, 6))
sns.heatmap(confusion_df, annot = True, annot_kws = {"size": 20}, cmap = 'Greens', cbar = False, square = True,fmt = '.2f')
plt.title('Confusion Matrix of Logistic Regression', fontsize = 16)
plt.ylabel(r'True Value', fontsize = 14)
plt.xlabel(r'Predicted Value', fontsize = 14)
plt.tick_params(labelsize = 12)

"""


