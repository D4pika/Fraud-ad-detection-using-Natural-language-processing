# Install the following pip packages in the current Jupyter kernel which are required while running the code.
import os
# os.system('pip install requests')
# os.system('pip install BeautifulSoup')
# os.system('pip install re')
# os.system('pip install json')
# os.system('pip install time')
# os.system('pip install selenium')
# os.system('pip install phonenumbers')
# os.system('pip install pandas')
# os.system('pip install csv')
# os.system('pip install sklearn')
# os.system('pip install nltk')
# os.system('pip install scipy')
# os.system('pip install _pickle')
# os.system('pip install numpy')
# os.system('pip install string')
# os.system('pip install tkinter')
# os.system('pip install time')
# os.system('pip install smtplib')
# os.system('pip install email')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn import svm
from nltk.corpus import stopwords
import _pickle as cPickle
import numpy as np
from scipy import sparse
import pandas as pd
import string
import os
from sklearn.metrics import confusion_matrix
from mlxtend.classifier import StackingClassifier
from sklearn.utils import resample

X = pd.read_csv('la-labelled-data.csv',sep=",",usecols=(1,2,3, 4,5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,19,20,22,21,23,24,25))

# Separate majority and minority classes
df_majority = X[X.SPAM==0]
df_minority = X[X.SPAM==1]

df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=1000,    # to match majority class
                                 random_state=123) # reproducible results

df_majority_downsampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=1000,     # to match minority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority_downsampled, df_minority_upsampled])
 
# Display new class counts
df_upsampled.SPAM.value_counts()

X = df_upsampled

# replace non-letters/numbers with space and remove duplicate spaces
X['description'].replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
exclude = set(string.punctuation)

def remove_punctuation(x):
    try:
        x = ''.join(ch for ch in x if ch not in exclude)
    except:
        pass
    return x
X['description'] = X['description'].apply(remove_punctuation)
X['reposted'] = np.where(X['post_id_old'].isnull, 0, 1)
X['square_footage']=X['square_footage'].fillna(0)
X['price']=X['price'].fillna(0)
X.shape

X=X[~(X['square_footage']=='loft')]
X.shape

X['square_footage']=X['square_footage'].astype(int)
X['description']=X['description'].fillna('')

'''
pre-processing categorical data
'''

# converting the categorical data to binary variables
X_cat = pd.get_dummies(X, columns=['reposted','laundry','parking','cat','dog','smoking','furnished','borough','housing_type'])

# drop numerical variable from categorical table
X_cat.drop('post_date', axis=1, inplace=True)
X_cat.drop('post_time', axis=1, inplace=True)
X_cat.drop('update_date', axis=1, inplace=True)
X_cat.drop('update_time', axis=1, inplace=True)
X_cat.drop('title', axis=1, inplace=True)
X_cat.drop('price', axis=1, inplace=True)
X_cat.drop('images', axis=1, inplace=True)
X_cat.drop('square_footage', axis=1, inplace=True)
X_cat.drop('bedroom', axis=1, inplace=True)
X_cat.drop('bathroom', axis=1, inplace=True)
X_cat.drop('post_id_old', axis=1, inplace=True)
X_cat.drop('post_id', axis=1, inplace=True)
X_cat.drop('parenthesis', axis=1, inplace=True)
X_cat.drop('description', axis=1, inplace=True)
X_cat.drop('URL', axis=1, inplace=True)
X_cat.drop('Reason', axis=1, inplace=True)
X_cat.drop('SPAM', axis=1, inplace=True)

#del y
y=X['SPAM']
print(X.shape)
print(y.shape)

'''
pre-processing numeric data
'''

# create numerical variable
X_num = pd.DataFrame(X, columns=['price', 'bathroom', 'bedroom', 'images', 'square_footage'])

'''
pre-processing text data
'''

# create text variable
X_text = X['description']

# TF-IDF Vectorization for weighted frequency of words and transform into vector of 1/0
tvf = TfidfVectorizer(stop_words=stopwords.words('english'))
X_text = tvf.fit_transform(X_text)

X_num['bathroom']=X_num['bathroom'].fillna(0)
X_num['images']=X_num['images'].fillna(0)
X_num['bedroom']=X_num['bedroom'].fillna(0)
X_num.isnull().sum()

X = sparse.hstack((X_text, X_cat, X_num))
X = X.toarray()
#=======================================================================================
'''
Decomposition and Feature Selection
'''
# Feature Selection
clf = RandomForestClassifier(n_estimators=100, n_jobs=1)
clf.fit(X, y.ravel())
sfm = SelectFromModel(clf, threshold=0.001)
X = sfm.fit_transform(X, y.ravel())
#=======================================================================================
'''
stacking all features into one matrix
'''
# create training and testing variables and response
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

y_train = y_train.as_matrix()
y_test = y_test.as_matrix()
#=======================================================================================
'''
Ensemble Methods: Voting 
'''

clf1 = RandomForestClassifier (n_estimators=100, n_jobs=-1, criterion='gini')
clf2 = RandomForestClassifier (n_estimators=100, n_jobs=-1, criterion='entropy')
clf3 = ExtraTreesClassifier (n_estimators=100, n_jobs=-1, criterion='gini')
clf4 = ExtraTreesClassifier (n_estimators=100, n_jobs=-1, criterion='entropy')
clf5 = GradientBoostingClassifier (learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)
clf6 = DecisionTreeClassifier()
clf7 = svm.SVC(gamma=0.001, C=100)
clf8 = KNN_classifier=KNeighborsClassifier()
clf9 = GaussianNB()
lr = LogisticRegression()

# assembling classifiers
predictors=[('RF_g', clf1), ('RF_E', clf2), ('ET_g', clf3), ('ET_e', clf4),
            ('GB', clf5), ('DT',clf6), ('SVM',clf7), ('KNN',clf8), ('NB',clf9)]

# building voting
VT=VotingClassifier(predictors)

#fitting model
VT.fit(X_train,y_train)

logreg = lr.fit(X_train,y_train)
lr_pred = logreg.predict(X_test)
lr_pred.reshape(-1, 1)

print(confusion_matrix(lr_pred, y_test))
accuracy_score(lr_pred,y_test)
RandFor = clf2.fit(X_train,y_train)
rf_pred = RandFor.predict(X_test)

print(confusion_matrix(lr_pred, y_test))
accuracy_score(rf_pred,y_test)

#=======================================================================================
'''
running classification prediction
'''

# running prediction
predicted=VT.predict(X_test)

#print the accuracy
print(accuracy_score(predicted, y_test))

#=======================================================================================
'''
Ensemble Methods: Stacking
'''

sclf = StackingClassifier(classifiers=predictors, meta_classifier=lr)


#VT.fit(counts_train,labels_train)

#use the VT classifier to predict
predicted=VT.predict(X_test)

#print the accuracy
print (accuracy_score(predicted,y_test))

from sklearn.metrics import confusion_matrix
confusion_matrix(predicted, y_test)

# #---------------------------------------------------------------------------------------------------------------------
# #
import requests
from bs4 import BeautifulSoup as bs4
import re
import json
import time
from selenium import webdriver
import phonenumbers
import pandas as pd
import csv
import os

#=======================================================================================

from tkinter import *
import time
from tkinter import messagebox

window = Tk()

print("Please enter keyword in the UI as opened")

def notify():
    Search_words = e2_value.get()
    t1.delete("1.0", END)
    t1.insert(END, Search_words + " set")
    t2.delete("1.0", END)
    t2.insert(END, "Notification starts today !!")
    time.sleep(1)


e1 = Label(window, text="Enter Search criteria")
e1.grid(row=0, column=0)

e2_value = StringVar()
e2 = Entry(window, textvariable=e2_value)
e2.grid(row=0, column=1)

b1 = Button(window, text="Start Notifications", command=notify)
b1.grid(row=0, column=2)

t1 = Text(window, height=1, width=30)
t1.grid(row=1, column=0)

t2 = Text(window, height=1, width=30)
t2.grid(row=1, column=1)

b2 = Button(window, text="Close window to start scraping", command=window.destroy)
b2.grid(row=1, column=2)


def ask_quit():
    if messagebox.askokcancel("Quit", "You want to quit now? Make sure you have entered the keyword"):
        window.destroy()


window.protocol("WM_DELETE_WINDOW", ask_quit)

window.mainloop()

if e2_value.get() == '':
    print("Please Enter the keyword to continue")
    exit()

'''
scraping urls and storing in list to iterate through for individual posts
'''

# url to build beautifulsoup loop
city = 'losangeles'
base_url = 'https://' + city + '.craigslist.org/search/hhh'
query_text = e2_value.get() #Fetching posts for the desired keyword mentioned by the user
end_url = '&availabilityMode=0'
complete_url = base_url + "?query=" + query_text
csv_filename = 'urls_' + city + '.csv'

all_data = []

def get_urls(url = complete_url):

    # empty dictionary to store urls
    urls = []
    print("Starting fetching of URL's")
    # for each page in results page
    for page in range(0, 25):
        # build url
        if page == 0:
            url = complete_url
        else:
            url = complete_url + '&s=' + str (page * 120) + end_url

        # retrieve urls
        rsp = requests.get(url,headers = { 'User-Agent': 'Opera/9.80 (Windows NT 5.1; U; en) Presto/2.10.289 Version/12.01', })
        body = bs4(rsp.text, 'html.parser')
        listings = body.find_all('li', class_='result-row')

        # store urls in list
        for listing in listings:
            urls.append(listing.a['href'])

        time.sleep(1)  # seconds

    print("Fetched URLS of Craigslist Listing")

    # write list to csv
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        for row in urls:
            writer.writerow([row])

    return urls

# run script
urls = get_urls()

for apt in urls:
    print("Currently Fetching Features for URL No. : "+str(len(all_data)+1))

    # pause each iteration so don't get blocked by craigslist
    time.sleep (1)

    # get html code with beautifulsoup
    rsp = requests.get(apt)
    body = bs4(rsp.text, 'html.parser')

    # create empty dictionary to store variables
    attribute_dict = {}

    # extract borough
    try:
        borough = body.find('li', {'class': 'crumb subarea'})
        attribute_dict['borough'] = borough.text.replace('\n', "").replace('>', "")
    except:
        pass

    # extract parenthesis in title
    try:
        parenthesis = body.findAll('span', {'class': 'postingtitletext'})
        parenthesis = parenthesis[0].small
        attribute_dict['parenthesis'] = parenthesis.text.strip().replace("(", "").replace(")", "")
    except:
        pass

    #extract availability
    try:
        attribute_dict['availability'] = body.find('span', class_='housing_movein_now property_date')['data-today_msg']
        attribute_dict['date_available'] = body.find('span', class_='housing_movein_now property_date')['data-date']
    except:
        pass

    # extract post id, datetime, and update datetime
    try:
        posted_info = body.findAll('div', {'class': 'postinginfos'})[0].text.strip().split('\n')

        for info in posted_info:
            if 'post id:' in info:
                attribute_dict['post_id'] = info.replace('post id: ', '')

            if 'posted:' in info:
                attribute_dict['post_date'] = info.replace('posted: ', '').split(' ')[0]

            if 'posted:' in info:
                attribute_dict['post_time'] = info.replace('posted: ', '').split(' ')[-1]

            if 'updated:' in info:
                attribute_dict['update_date'] = info.replace('updated: ', '').split(' ')[0]

            if 'updated:' in info:
                attribute_dict['update_time'] = info.replace('updated: ', '').split(' ')[-1]
    except:
        pass

    # Extract title
    try:
        attribute_dict['title'] = body.find('span', {'id': 'titletextonly'}).text
    except:
        pass

    # Extract price
    try:
        attribute_dict['price'] = body.find('span', {'class': 'price'}).text
    except:
        pass

    # Extract number of images
    try:
        attribute_dict['images'] = body.find('span', {'class': 'slider-info'}).text.split(' ')[-1]
    except:
        pass

    # Extract bed, bath, size, type, pets, laundry, parking, smoking, furnished
    try:
        attributes = []
        attributes_data = body.findAll('p', class_='attrgroup')
        attributes = attributes_data[0].findAll('span') + attributes_data[1].findAll('span')
        attributes2 = [item.text.strip().split('\n')[0] for item in attributes]

        for attribute in attributes2:
            if 'BR' in attribute:
                attribute_dict['bedroom'] = attribute.split(' / ')[0].replace('BR', '')

            if 'Ba' in attribute:
                attribute_dict['bathroom'] = attribute.split(' / ')[1].replace('Ba', '')

            if 'ft' in attribute:
                attribute_dict['square_footage'] = attribute.replace('ft2', '')

            if 'listed by' in attribute:
                attribute_dict['listed_by'] = attribute.replace('listed by: ', '')

            if attribute in ['apartment', 'condo', 'cottage/cabin', 'duplex', 'flat',
                             'house', 'in-law', 'loft', 'townhouse', 'manufactured', 'assisted living', 'land']:
                attribute_dict['housing_type'] = attribute
            else:
                attribute_dict['housing_type'] = ""

            if 'cat' in attribute:
                attribute_dict['cat'] = attribute
            else:
                attribute_dict['cat'] = ""

            if 'dog' in attribute:
                attribute_dict['dog'] = attribute
            else:
                attribute_dict['dog'] = ""

            if 'furnished' in attribute:
                attribute_dict['furnished'] = attribute
            else:
                attribute_dict['furnished'] = ""

            if attribute in ['w/d in unit', 'laundry in bldg', 'laundry on site', 'w/d hookups']:
                attribute_dict['laundry'] = attribute
            else:
                attribute_dict['laundry'] = ""

            if attribute in ['carport', 'attached garage', 'detached garage', 'off-street parking',
                             'street parking', 'valet parking']:
                attribute_dict['parking'] = attribute
            else:
                attribute_dict['parking'] = ""

            if 'smoking' in attribute:
                attribute_dict['smoking'] = attribute
            else:
                attribute_dict['smoking'] = ""
    except:
        pass

    # Extract the description of the property
    try:
        description = body.find('section', {'id': 'postingbody'}).text.replace("\n", " ").strip()
        attribute_dict['description'] = re.sub("\s\s+", " ", description).replace('QR Code Link to This Post ', "")
    except:
        pass

    # Extract fees and conditions
    try:
        requirements = body.findAll('ul', {'class': 'notices'})
        attribute_dict['requirements']= re.sub(r'[^\w]', ' ', requirements[0].text.strip().split(':')[1])
    except:
        pass

    # store attributes in pandas array
    attribute_dict = pd.Series(attribute_dict, name=id)
    all_data.append(attribute_dict)

# change pandas array into dataframe and save for processing
all_data_df = pd.DataFrame(all_data)

#Storing all data into a csv file
all_data_df = []
all_data_df = pd.DataFrame(all_data)
all_data_filename =  city + 'data_complete.csv'
all_data_df.to_csv(all_data_filename,encoding='utf-8',index=False)

X_new = pd.read_csv(all_data_filename,sep=",")

X_new.head()

# replace non-letters/numbers with space and remove duplicate spaces
X_new['description'].replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
exclude = set(string.punctuation)
def remove_punctuation(x):
    try:
        x = ''.join(ch for ch in x if ch not in exclude)
    except:
        pass
    return x
X_new['description'] = X_new['description'].apply(remove_punctuation)


X_new['square_footage']=X_new['square_footage'].fillna(0)
X_new['price']=X_new['price'].fillna(0)

X_new.shape

X_new=X_new[~(X_new['square_footage']=='loft')]
X_new.shape

X_new['square_footage']=X_new['square_footage'].astype(int)
X_new['description']=X_new['description'].fillna('')
X_new.shape

'''
pre-processing categorical data
'''

# converting the categorical data to binary variables
X_new_cat = pd.get_dummies(X_new, columns=['reposted','laundry','parking','cat','dog','smoking','furnished','borough','housing_type'])

# drop numerical variable from categorical table
X_new_cat.drop('post_date', axis=1, inplace=True)
X_new_cat.drop('post_time', axis=1, inplace=True)
X_new_cat.drop('update_date', axis=1, inplace=True)
X_new_cat.drop('update_time', axis=1, inplace=True)
X_new_cat.drop('title', axis=1, inplace=True)
X_new_cat.drop('price', axis=1, inplace=True)
X_new_cat.drop('images', axis=1, inplace=True)
X_new_cat.drop('square_footage', axis=1, inplace=True)
X_new_cat.drop('bedroom', axis=1, inplace=True)
X_new_cat.drop('bathroom', axis=1, inplace=True)
X_new_cat.drop('post_id_old', axis=1, inplace=True)
X_new_cat.drop('post_id', axis=1, inplace=True)
X_new_cat.drop('parenthesis', axis=1, inplace=True)
X_new_cat.drop('description', axis=1, inplace=True)

X_new_cat.shape

X_new['laundry'].head()

'''
pre-processing numeric data
'''

# create numerical variable
X_new_num = pd.DataFrame(X_new, columns=['price', 'bathroom', 'bedroom', 'images', 'square_footage'])

'''
pre-processing text data
'''

# create text variable
X_new_text = X_new['description']

# TF-IDF Vectorization for weighted frequency of words and transform into vector of 1/0
tvf1 = TfidfVectorizer(stop_words=stopwords.words('english'))
X_new_text = tvf1.fit_transform(X_new_text)

X_new_num['bathroom']=X_new_num['bathroom'].fillna(0)
X_new_num['images']=X_new_num['images'].fillna(0)
X_new_num['bedroom']=X_new_num['bedroom'].fillna(0)
X_new_num.isnull().sum()

print(len(X_new['bathroom']))

X_new_num['bathroom'] = X_new['bathroom'].astype('str')
X_new_num['bedroom'] = X_new['bedroom'].astype('str')

def clean_tweet(tweet): 
        ''' 
        Utility function to clean tweet text by removing links, special characters 
        using simple regex statements. 
        '''
        return ' '.join(re.sub("(@[0-9]+)|([^0-9\t])|(\w+:\/\/\S+)", "0", tweet).split())

clean_tweet(X_new_num["bathroom"][3])

type(X_new_num["bathroom"][3])

X_new_num["price"] = X_new_num.price.astype(float)
X_new_num["bathroom"] = [clean_tweet(i) for i in X_new_num["bathroom"]]
X_new_num["bedroom"] = [clean_tweet(i) for i in X_new_num["bedroom"]]
X_new_num['bathroom']=X_new_num['bathroom'].fillna(0)
X_new_num['bedroom']=X_new_num['bedroom'].fillna(0)
X_new_num['bedroom'] = X_new_num['bedroom'].replace("", 0).bfill()
X_new_num['bathroom'] = X_new_num['bathroom'].replace("", 0).bfill()
X_new_num["bedroom"] = X_new_num.bedroom.astype(float)
X_new_num["bathroom"] = X_new_num.bathroom.astype(float)

X_new_num.dtypes

X_new = sparse.hstack((X_new_text, X_new_cat, X_new_num))
X_new = X_new.toarray()

X_new = X_test

predicted_new=VT.predict(X_new)
len(predicted_new)

spam_detection = predicted_new.tolist()
spam_detection = pd.DataFrame()
final_df = df_majority[1:5]


final_df.to_csv('Listings.csv')

import smtplib
from os.path import basename
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
from email import encoders
from email.mime.base import MIMEBase

fromaddr = "unstructured.purdue@gmail.com"
toaddr = "gupta485@purdue.edu"
 
msg = MIMEMultipart()
 
msg['From'] = fromaddr
msg['To'] = toaddr
msg['Subject'] = "Today's house deals matching your criteria"
 
body = "Hey Subscriber, Here are the listings for you today for your selected search"
 
msg.attach(MIMEText(body, 'plain'))
 
filename = "Listings.csv"
attachment = open("Listings.csv", "rb")
 
part = MIMEBase('application', 'octet-stream')
part.set_payload((attachment).read())
encoders.encode_base64(part)
part.add_header('Content-Disposition', "attachment; filename= %s" % filename)
 
msg.attach(part)
 
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login(fromaddr, "moneymoneymoney")
text = msg.as_string()
server.sendmail(fromaddr, toaddr, text)
server.quit()

