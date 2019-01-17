###########################################################################################
#S  				T					A 						R						 T#
###########################################################################################

import json, os
import pandas as pd
from glob import glob
import geocoder

import time
import datetime

from collections import Counter
import numpy as np
import unicodedata

from nltk.corpus import stopwords
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#for Anaconda python2.7:
#source activate py27
#data has to be in same dir as the file

#path to the current dir
#all_files to the dir of txt files
# Tweets are stored in in file "fname". In the file used for this script, 
# each tweet was stored on one line
#fname = 'data2/energy20180322T083452.txt'
path = ''
all_files = glob(os.path.join(path, "*.txt"))

#loop through the all_files dir
for fname in all_files:
    with open(fname, 'r') as f:
        #http://www.mikaelbrunila.fi/2017/03/27/scraping-extracting-mapping-geodata-twitter/
        #https://opensas.wordpress.com/2013/06/30/using-openrefine-to-geocode-your-data-using-google-and-openstreetmap-api/
        #Create dictionary to later be stored as JSON. All data will be included
        # in the list 'data'
        users_with_geodata = {
            "data": []
        }
        all_users = []
        total_tweets = 0
        geo_tweets  = 0
        for line in f:
            tweet = json.loads(line)
            if tweet['user']['id']:
                total_tweets += 1 
                user_id = tweet['user']['id']
                if user_id not in all_users:
                    all_users.append(user_id)
                
                    #Give users some data to find them by. User_id listed separately 
                    # to make iterating this data later easier
                    user_data = {
                        "user_id" : tweet['user']['id'],
                        "features" : {
                            "name" : tweet['user']['name'],
                            "id": tweet['user']['id'],
                            "screen_name": tweet['user']['screen_name'],
                            "tweets" : 1,
                            "location": tweet['user']['location'],
                            "text": tweet['text'],
                            "created_at": tweet['created_at'],
                        }
                    }
                    #Iterate through different types of geodata to get the variable primary_geo
                    if tweet['coordinates']:
                        user_data["features"]["primary_geo"] = str(tweet['coordinates'][tweet['coordinates'].keys()[1]][1]) + ", " + str(tweet['coordinates'][tweet['coordinates'].keys()[1]][0])
                        user_data["features"]["geo_type"] = "Tweet coordinates"
                    elif tweet['place']:
                        user_data["features"]["primary_geo"] = tweet['place']['full_name'] + ", " + tweet['place']['country']
                        user_data["features"]["geo_type"] = "Tweet place"
                    else:
                        user_data["features"]["primary_geo"] = tweet['user']['location']
                        user_data["features"]["geo_type"] = "User location"
                    #Add only tweets with some geo data to .json. Comment this if you want to include all tweets.
                    if user_data["features"]["primary_geo"]:
                        users_with_geodata['data'].append(user_data)
                        geo_tweets += 1
            
                #If user already listed, increase their tweet count
                elif user_id in all_users:
                    for user in users_with_geodata["data"]:
                        if user_id == user["user_id"]:
                            user["features"]["tweets"] += 1
    
        #Count the total amount of tweets for those users that had geodata            
        for user in users_with_geodata["data"]:
            geo_tweets = geo_tweets + user["features"]["tweets"]
        #Get some aggregated numbers on the data
        print fname + " included " + str(len(all_users)) + " unique users who tweeted with or without geo data"
        print fname + " included " + str(len(users_with_geodata['data'])) + " unique users who tweeted with geo data, including 'location'"
        print fname + " users with geo data tweeted " + str(geo_tweets) + " out of the total " + str(total_tweets) + " of tweets."
    # Save data to JSON file
    with open('user_loc_' + fname + '.json', 'w') as fout:
        fout.write(json.dumps(users_with_geodata, indent=4))
#create a glob list of the json files in our dir
path = ''
all_files = glob(os.path.join(path, "*.json"))
#loop through the glob list of json files
for data in all_files:
    df = pd.read_json(data)
    tweets = pd.read_json((df['data']).to_json(), orient='index')
    tweets1 = pd.read_json((tweets['features']).to_json(), orient='index')
    tweets1['coord'] = 'coord'

    #create a list to append the geo data to, geocode based on location column
    for index, row in tweets1.iterrows():
        try:
        	print(row['location'])
        	time.sleep(1.01)
        	g = geocoder.osm(row['location'])
        	geo = g.latlng
        	print(geo)
        	tweets1.at[index, 'coord'] = geo
        except:
        	pass

    #split the coord column in y and x columns
    #tweets1['coord'] = pd.Series(coord)
    tweets1['coord'] = tweets1['coord'].astype(str)
    tweets1['coord'] = tweets1['coord'].str.strip('[]')
    tweets1['y'], tweets1['x'] = tweets1['coord'].str.split(',', 1).str
    
    #save to a json file
    #tweets1.to_json(data + '_geo.json')
    print "Geocoded " + data
    
    #sidestep an error reading a string
    tweets1 = tweets1[tweets1['text'].notnull()]
    
    #remove stopwords
    stop = stopwords.words('english')
    tweets1['tweet_without_stopwords'] = tweets1['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    stop =  ['The','RT','&amp;', '-', 'A', 'https:', '.', '2']
    tweets1['tweet_without_stopwords'] = tweets1['tweet_without_stopwords'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    #remove periods
    tweets1['tweet_without_stopwords'] = tweets1['tweet_without_stopwords'].str.replace('[\.]','')
    #remove commas
    tweets1['tweet_without_stopwords'] = tweets1['tweet_without_stopwords'].str.replace('[\,]','')
    #remove -
    tweets1['tweet_without_stopwords'] = tweets1['tweet_without_stopwords'].str.replace('[-]','')
    #remove @
    tweets1['tweet_without_stopwords'] = tweets1['tweet_without_stopwords'].str.replace('[@]','')
    
    #sentiment analysis using VADER
    tweets1["compound"] = ''
    tweets1["neg"] = ''
    tweets1["neu"] = ''
    tweets1["pos"] = ''
    sid = SentimentIntensityAnalyzer()
    
    for user, row in tweets1.T.iteritems():
        try:
            sentence = unicodedata.normalize('NFKD', tweets1.loc[user, 'tweet_without_stopwords'])
            ss = sid.polarity_scores(sentence)
            tweets1.set_value(user, 'compound', ss['compound'])
            tweets1.set_value(user, 'neg', ss['neg'])
            tweets1.set_value(user, 'neu', ss['neu'])
            tweets1.set_value(user, 'pos', ss['pos'])
        except TypeError:
            print(tweets1.loc[user, 'tweet_without_stopwords'])
    
    #print a positive message and save the file
    print "Sentiment analyzed " + data
    tweets1.to_json(data + '_geo_sent.json')

###########################################################################################
#F  									I 												 N#
###########################################################################################

