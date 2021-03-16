import pandas as pd
import numpy as np

#%%

# Dataset2

# read file

with open("dataset/Dataset2.txt", 'r+') as f:
          dataset2 = f.readlines()
f.close()

dataset2 = dataset2[24:]

# remove unnecessary elements

try:
    for i in dataset2:
        dataset2.remove('\n')
except:
    print("finished")
dataset2.remove('************************************\n')

# split the elements

df1 = pd.Series(dataset2).apply(lambda x: x.split('\t'))
df2 = df1.apply(lambda x: pd.Series(x).str.replace('\n', ''))

# split into offensive word list (first list of the text file)

offensive_words = df2.iloc[0:404, 0]
offensive_words = pd.DataFrame(offensive_words)

# create the class label 1 for each word, which is identified as offensive language

df_ones = pd.DataFrame(np.ones(len(offensive_words)), dtype = int)
offensive_words = pd.concat([df_ones, offensive_words], axis = 1)
offensive_words.columns = ['class', 'tweet']

# split into hate speech word list (second list of the text file)

hate_speech_words = df2.iloc[405:793, 0]
hate_speech_words = pd.DataFrame(hate_speech_words).reset_index(drop = True)

# create the class label 0 for each word, which is identified as offensive language

df_zeros = pd.DataFrame(np.zeros(len(hate_speech_words)), dtype = int)
hate_speech_words = pd.concat([df_zeros, hate_speech_words], axis = 1)
hate_speech_words.columns = ['class', 'tweet']

#%%

# Dataset3

# read the file

with open("dataset/Dataset3.txt") as f:
    dataset3 = f.readlines()
f.close()
dataset3 = dataset3[8:]

# create the class column for the comments

df_zeros = pd.DataFrame(np.zeros(len(dataset3)), dtype = int)
fox_comments = pd.DataFrame(dataset3)
fox_comments = pd.concat([df_zeros, fox_comments], axis = 1)
fox_comments.columns = ['class', 'tweet']

# change to class 2 if it is not hate speech nor offensive language to fit the definition of dataset1

for i in range(len(fox_comments)):
    if fox_comments['tweet'][i][0] == '0':
        fox_comments.at[i, 'class'] = 2

#%%

# Dataset4

# read the file

with open("dataset/Dataset4.txt") as f:
    microposts = f.readlines()
f.close()

# data cleaning

try:
    for i in microposts:
        microposts.remove('#*#*#*#*#*#*#\n')
except:
    print("finished")
microposts.remove('#*#*#*#*#*#*#')

# separate into 2 lists

df3, df4, df5 = [], [], [] 

# number of iterations

num = int(len(microposts)/2)

# create a list of labels

for i in range(num):
    df3.append(microposts[i * 2])

# create a list of comments

for i in range(num):
    df4.append(microposts[i * 2 + 1])

# assign the class according to the label 
    
for i in range(num):
    if "abusive" in df3[i]:
        df5.append(0)
    else:
        df5.append(2)

# combine the list of class and comments

microposts_final = pd.concat([pd.DataFrame(df5), pd.DataFrame(df4)], axis = 1)
microposts_final.columns = ['class', 'tweet']

#%%

# Dataset5
# https://www.kaggle.com/usharengaraju/dynamically-generated-hate-speech-dataset?select=2020-12-31-DynamicallyGeneratedHateDataset-entries-v0.1.csv

# read the file

dataset5 = pd.read_csv("dataset/Dataset5.csv", sep = ",")
dataset5 = dataset5[['text', 'label']]

# create a column of class for the dataset

df6 = pd.DataFrame(np.zeros(len(dataset5), dtype = int))
linkedin_post = pd.concat([df6, dataset5], axis = 1)
linkedin_post.columns = ['class', 'tweet', 'label']

# define the non hate speech comments as class 2

for i in range(len(linkedin_post)):
    if "nothate" in linkedin_post['label'][i]:
        linkedin_post.at[i, 'class'] = 2
linkedin_post = linkedin_post.drop('label', axis = 1)


