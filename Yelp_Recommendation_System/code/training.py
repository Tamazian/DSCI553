# training XGBoost/CatBoost models and pickling it to import in competition.py
# this file is just for reference, and is not run during submission (but rather before)
    # sys is not set up to be run ; though inputting the folder directory as the first/only argument should work
    # not tested on Vocareum since it was not part of the requirements; but it works locally
    
############################################################################################################################

from pyspark import SparkContext
import os
import sys
import csv
import json
import time
import statistics
import itertools
import numpy as np
from collections import Counter
import pickle
import xgboost as xgb
import catboost as cat


#os.environ["SPARK_HOME"] = '/Users/alaintamazian/spark-3.1.2-bin-hadoop3.2'
#os.environ["PYTHONPATH"] = '/Users/alaintamazian/spark-3.1.2-bin-hadoop3.2/python'
#os.environ['PYSPARK_PYTHON'] = '/Applications/anaconda3/envs/dsci553/bin/python'
#os.environ['PYSPARK_DRIVER_PYTHON'] = '/Applications/anaconda3/envs/dsci553/bin/python'


start_time = time.time()


# sc = SparkContext('local[*]', 'task1')
sc = SparkContext.getOrCreate()
# what's the difference; what exactly does each do ??? ;

sc.setLogLevel("OFF")
# does this work ???



# Param: folder_path: the path of dataset folder, which contains exactly the same file as the google drive.
# folder_path = "/Users/alaintamazian/DocumentsAT/DSCI553/Homework/HW3/data"
folder_path = sys.argv[1]

yelp_train_path = folder_path + "/" + "yelp_train.csv"
yelp_val_path = folder_path + "/" + "yelp_val.csv"
business_path = folder_path + "/" + "business.json"
user_path = folder_path + "/" + "user.json"
review_path = folder_path + "/" + "review_train.json"
checkin_path = folder_path + "/" + "checkin.json"
# photo_path = folder_path + "/" + "photo.json"
# tip_path = folder_path + "/" + "tip.json"



############################################################################################################################


# Regression Model-Based CF (using XGBRegressor)


businessRDD = sc.textFile(business_path)
userRDD = sc.textFile(user_path)
reviewRDD = sc.textFile(review_path)
checkinRDD = sc.textFile(checkin_path)
# tipRDD = sc.textFile(tip_path)
# photoRDD = sc.textFile(photo_path)


with open(yelp_val_path, 'r') as f:
    csv_reader = csv.reader(f)
    # first line is the column names: [user_id,business_id,stars]
    val_dataL = list(csv_reader)[1:]
valRDD = sc.parallelize(val_dataL)


#####


friends_avg_star_ratings_userRDD1 = userRDD.map(lambda row: json.loads(row)).filter(lambda rowD: ("friends" in rowD) and (rowD["friends"]!="None") ).map(lambda rowD: ( rowD["user_id"], rowD["friends"].split(", ") )).zipWithIndex().filter(lambda rowL: rowL[-1] < 300000).map(lambda rowL: rowL[0])
friends_avg_star_ratings_userD1 = friends_avg_star_ratings_userRDD1.collectAsMap()

friends_avg_star_ratings_userRDD2 = userRDD.map(lambda row: json.loads(row)).filter(lambda rowD: ("friends" in rowD) and (rowD["friends"]!="None") ).map(lambda rowD: ( rowD["user_id"], rowD["friends"].split(", ") )).zipWithIndex().filter(lambda rowL: (rowL[-1] >= 300000) and (rowL[-1] < 400000) ).map(lambda rowL: rowL[0])
friends_avg_star_ratings_userD2 = friends_avg_star_ratings_userRDD2.collectAsMap()

friends_avg_star_ratings_userRDD3 = userRDD.map(lambda row: json.loads(row)).filter(lambda rowD: ("friends" in rowD) and (rowD["friends"]!="None") ).map(lambda rowD: ( rowD["user_id"], rowD["friends"].split(", ") )).zipWithIndex().filter(lambda rowL: (rowL[-1] >= 400000) and (rowL[-1] < 500000) ).map(lambda rowL: rowL[0])
friends_avg_star_ratings_userD3 = friends_avg_star_ratings_userRDD3.collectAsMap()

friends_avg_star_ratings_userRDD4 = userRDD.map(lambda row: json.loads(row)).filter(lambda rowD: ("friends" in rowD) and (rowD["friends"]!="None") ).map(lambda rowD: ( rowD["user_id"], rowD["friends"].split(", ") )).zipWithIndex().filter(lambda rowL: (rowL[-1] >= 500000) and (rowL[-1] < 600000) ).map(lambda rowL: rowL[0])
friends_avg_star_ratings_userD4 = friends_avg_star_ratings_userRDD4.collectAsMap()

friends_avg_star_ratings_userRDD5 = userRDD.map(lambda row: json.loads(row)).filter(lambda rowD: ("friends" in rowD) and (rowD["friends"]!="None") ).map(lambda rowD: ( rowD["user_id"], rowD["friends"].split(", ") )).zipWithIndex().filter(lambda rowL: (rowL[-1] >= 600000) and (rowL[-1] < 700000) ).map(lambda rowL: rowL[0])
friends_avg_star_ratings_userD5 = friends_avg_star_ratings_userRDD5.collectAsMap()

friends_avg_star_ratings_userRDD6 = userRDD.map(lambda row: json.loads(row)).filter(lambda rowD: ("friends" in rowD) and (rowD["friends"]!="None") ).map(lambda rowD: ( rowD["user_id"], rowD["friends"].split(", ") )).zipWithIndex().filter(lambda rowL: rowL[-1] >= 700000).map(lambda rowL: rowL[0])
friends_avg_star_ratings_userD6 = friends_avg_star_ratings_userRDD6.collectAsMap()

friends_avg_star_ratings_userD = {**friends_avg_star_ratings_userD1, **friends_avg_star_ratings_userD2, **friends_avg_star_ratings_userD3, **friends_avg_star_ratings_userD4, **friends_avg_star_ratings_userD5, **friends_avg_star_ratings_userD6}
del friends_avg_star_ratings_userD1
del friends_avg_star_ratings_userD2
del friends_avg_star_ratings_userD3
del friends_avg_star_ratings_userD4
del friends_avg_star_ratings_userD5
del friends_avg_star_ratings_userD6

# avg_star_ratings_businessD = businessRDD.map(lambda row: json.loads(row)).map(lambda rowD: (rowD["business_id"], rowD["stars"])).collectAsMap()
user_business_ratingD = reviewRDD.map(lambda row: json.loads(row)).map(lambda rowD:( (rowD["user_id"], rowD["business_id"]), rowD["stars"]) ).collectAsMap()
avg_star_ratings_userD = userRDD.map(lambda row: json.loads(row)).map(lambda rowD: (rowD["user_id"], rowD["average_stars"])).collectAsMap()

# def multimode_friends(some_list):
#     try:
#         return statistics.mode(some_list)
#     except:
#         modesL = []
#         num_countsL = Counter(some_list)
#         max_mode_count = num_countsL.most_common(1)[0][1]
#
#         for num in some_list:
#             if (num not in modesL) and (some_list.count(num) == max_mode_count):
#                 modesL.append(num)
#
#         if len(modesL) % 2:
#             return statistics.median(modesL)
#         else:
#             return max(modesL)

def friends_avg_star_ratings_userF(user_id, business_id):
    if (user_id not in friends_avg_star_ratings_userD) or (friends_avg_star_ratings_userD[user_id] == ['None']):
        return None
    else:
        friend_ratings = []

        for friend in friends_avg_star_ratings_userD[user_id]:
            if (friend, business_id) in user_business_ratingD:
                friend_ratings.append(user_business_ratingD[(friend, business_id)])

        if len(friend_ratings) == 0:
            for friend in friends_avg_star_ratings_userD[user_id]:
                if friend in avg_star_ratings_userD:
                    # friend_ratings.append(avg_star_ratings_businessD[business_id])
                    friend_ratings.append(avg_star_ratings_userD[friend])

        if len(friend_ratings) == 0:
            # return avg_star_ratings_userD[user_id]
            return None
        else:
            rating = sum(friend_ratings)/len(friend_ratings)
            # rating = multimode_friends(friend_ratings)
            return rating

train_X_frendsRatingL = []
for i, (user, bus) in enumerate( reviewRDD.map(lambda row: json.loads(row)).map(lambda rowD: (rowD["user_id"], rowD["business_id"]) ).collect() ):
    stars = friends_avg_star_ratings_userF(user, bus)
    # print(i, user, bus, stars)
    train_X_frendsRatingL.append([stars])

val_X_frendsRatingL = []
for i, (user, bus) in enumerate( valRDD.map(lambda rowL: (rowL[0], rowL[1])).collect() ):
    stars = friends_avg_star_ratings_userF(user, bus)
    # print(i, user, bus, stars)
    val_X_frendsRatingL.append([stars])


days_openD = businessRDD.map(lambda row: json.loads(row)).filter(lambda rowD: ("hours" in rowD) and (rowD["hours"])).map(lambda rowD: (rowD["business_id"], len(rowD["hours"]))).collectAsMap()
days_openD_avg = sum([count for count in days_openD.values()])/len(days_openD)

def days_openF(business_id):
    if (business_id in days_openD) and (days_openD[business_id]):
        return days_openD[business_id]
    else:
        return days_openD_avg
        # return None


# based on opening time average
opening_timeD = businessRDD.map(lambda row: json.loads(row)).map(lambda rowD: ( rowD["business_id"], [ float(".".join(hours.split("-")[0].split(":"))) for hours in rowD["hours"].values() ] ) if ("hours" in rowD) and (rowD["hours"]) else (rowD["business_id"], None)).map(lambda rowL: ( rowL[0], sum(rowL[1])/len(rowL[1]) ) if rowL[1] else rowL).collectAsMap()
opening_timeD_avg = sum([num for num in opening_timeD.values() if num])/len(opening_timeD)

def opening_timeF(business_id):
    if (business_id in opening_timeD) and (opening_timeD[business_id]):
        return opening_timeD[business_id]
    else:
        return opening_timeD_avg
        # return None


def multimode(some_list):
    try:
        return statistics.mode(some_list)
    except:
        modesL = []
        num_countsL = Counter(some_list)
        max_mode_count = num_countsL.most_common(1)[0][1]

        for num in some_list:
            if (num not in modesL) and (some_list.count(num) == max_mode_count):
                modesL.append(num)

        if min(modesL) <= 6:
            return min(modesL)
        else:
            return max(modesL)

# based on closing time mode
closing_timeD = businessRDD.map(lambda row: json.loads(row)).map(lambda rowD: ( rowD["business_id"], multimode([ float(".".join(hours.split("-")[1].split(":"))) for hours in rowD["hours"].values() ]) ) if ("hours" in rowD) and (rowD["hours"]) else (rowD["business_id"], None)).collectAsMap()
closing_timeD_mode = multimode([num for num in closing_timeD.values() if num])

def closing_timeF(business_id):
    if (business_id in closing_timeD) and (closing_timeD[business_id]):
        return closing_timeD[business_id]
    else:
        return closing_timeD_mode
        # return None



# Review count per business
    # is this "review_count" feature from "business.json" more accurate than the one we calculate from review_train.json
review_count_businessD = businessRDD.map(lambda row: json.loads(row)).map(lambda rowD: (rowD["business_id"], rowD["review_count"])).collectAsMap()

# overall average used for imputation
review_count_businessD_avg = sum([count for count in review_count_businessD.values()])/len(review_count_businessD)
def review_count_businessF(business_id):
    if (business_id in review_count_businessD) and (review_count_businessD[business_id] or review_count_businessD[business_id] == 0):
        return review_count_businessD[business_id]
    else:
        # the average
        return review_count_businessD_avg



avg_star_ratings_businessD = businessRDD.map(lambda row: json.loads(row)).map(lambda rowD: (rowD["business_id"], rowD["stars"])).collectAsMap()

# overall average used for imputation
avg_star_ratings_businessD_avg = sum([stars for stars in avg_star_ratings_businessD.values()])/len(avg_star_ratings_businessD)
# print(avg_star_ratings_businessD_avg)
def avg_star_ratings_businessF(business_id):
    if (business_id in avg_star_ratings_businessD) and (avg_star_ratings_businessD[business_id]):
        return avg_star_ratings_businessD[business_id]
    else:
        # the average
        return avg_star_ratings_businessD_avg



# is_open binary
is_open_businessD = businessRDD.map(lambda row: json.loads(row)).map(lambda rowD: (rowD["business_id"], rowD["is_open"])).collectAsMap()

def is_open_businessF(business_id):
    if (business_id in is_open_businessD) and (is_open_businessD[business_id] or is_open_businessD[business_id] == 0):
        return is_open_businessD[business_id]
    else:
        # Since there are more businesses in the validation set that are included in our business.json than not,
            # and usually the dummy will be 0
        # when we have no category info on business i, we mark it as a 0 ???
        return 0



# despite being a number, latitude is actually a categorical variable; so technically, inlcuding it like this is wrong
# it is unintuitive but it seems to help
latD = businessRDD.map(lambda row: json.loads(row)).map(lambda rowD: (rowD["business_id"], rowD["latitude"]) ).collectAsMap()

def latF(business_id):
    if (business_id in latD) and (latD[business_id]):
        return latD[business_id]
    else:
        # just an arbitrary number
        return 0


longD = businessRDD.map(lambda row: json.loads(row)).map(lambda rowD: (rowD["business_id"], rowD["longitude"]) ).collectAsMap()

def longF(business_id):
    if (business_id in longD) and (longD[business_id]):
        return longD[business_id]
    else:
        # just an arbitrary number
        return 0



# key is user_id ; value is the number of every business that the key user rated
business_ratings_per_userD = reviewRDD.map(lambda row: json.loads(row)).map(lambda rowD: (rowD["user_id"], rowD["business_id"])).groupByKey().map(lambda kv: (kv[0], len(kv[1]))).collectAsMap()
business_ratings_per_userD_avg = sum([num for num in business_ratings_per_userD.values()])/len(business_ratings_per_userD)

def business_ratings_per_userF(user_id):
    if (user_id in business_ratings_per_userD) and (business_ratings_per_userD[user_id] or business_ratings_per_userD[user_id] == 0):
        return business_ratings_per_userD[user_id]
    else:
        # return 0
        return business_ratings_per_userD_avg


# key is business_id ; value is the number of every user that rated the key business rated
user_raters_per_businessD = reviewRDD.map(lambda row: json.loads(row)).map(lambda rowD: (rowD["business_id"], rowD["user_id"])).groupByKey().map(lambda kv: (kv[0], len(kv[1]))).collectAsMap()
user_raters_per_businessD_avg = sum([num for num in user_raters_per_businessD.values()])/len(user_raters_per_businessD)

def user_raters_per_businessF(business_id):
    if (business_id in user_raters_per_businessD) and (user_raters_per_businessD[business_id] or user_raters_per_businessD[business_id] == 0):
        return user_raters_per_businessD[business_id]
    else:
        # return 0
        ### *** what are others setting for their defaults
        return user_raters_per_businessD_avg



checkins_businessD = checkinRDD.map(lambda row: json.loads(row)).map(lambda rowD: (rowD["business_id"], list(rowD["time"].values()) ) ).collectAsMap()

def checkin_days_businessF(business_id):
    if business_id in checkins_businessD:
        return len(checkins_businessD[business_id])
    else:
        return 0



def binary_encode(binary):
    if (binary == "True") or (binary is True):
        good = 1
    else:
        good = 0
    return good

good_for_kidsD = businessRDD.map(lambda row: json.loads(row)).map(lambda rowD: ( rowD["business_id"], binary_encode(rowD["attributes"]["GoodForKids"]) ) if \
    (("attributes" in rowD) and (rowD["attributes"]) and ("GoodForKids" in rowD["attributes"])) else (rowD["business_id"], 0) ).collectAsMap()
# change default to 1 or something else ???

def good_for_kidsF(business_id):
    if business_id in good_for_kidsD:
        return good_for_kidsD[business_id]
    else:
        return 0
# change default value for when business isn't in our data or has NA for the feature to None ???



good_for_groupsD = businessRDD.map(lambda row: json.loads(row)).map(lambda rowD: ( rowD["business_id"], binary_encode(rowD["attributes"]["RestaurantsGoodForGroups"]) ) if \
    (("attributes" in rowD) and (rowD["attributes"]) and ("RestaurantsGoodForGroups" in rowD["attributes"])) else (rowD["business_id"], 0) ).collectAsMap()
# change default to 1 or something else ???

def good_for_groupsF(business_id):
    if (business_id in good_for_groupsD) and (good_for_groupsD[business_id] or good_for_groupsD[business_id] == 0):
        return good_for_groupsD[business_id]
    else:
        return 0



wifiD = businessRDD.map(lambda row: json.loads(row)).map(lambda rowD: ( rowD["business_id"], rowD["attributes"]["WiFi"] ) if \
    (("attributes" in rowD) and (rowD["attributes"]) and ("WiFi" in rowD["attributes"])) else (rowD["business_id"], "n/a") ).collectAsMap()
# change default to 1 or something else ???

def dummy_wifiF(business_id, answer):
    # could be "free", "paid", "no", (or "n/a") ; change "n/a" to "no" ???
        # "n/a" is like have a seperate dummy asking whether this is a business where having wifi makes sense
    if (business_id in wifiD) and (wifiD[business_id] == answer):
        return 1
    else:
        return 0



rest_price_rangeD = businessRDD.map(lambda row: json.loads(row)).map(lambda rowD: ( rowD["business_id"], int(rowD["attributes"]["RestaurantsPriceRange2"]) ) if \
    (("attributes" in rowD) and (rowD["attributes"]) and ("RestaurantsPriceRange2" in rowD["attributes"])) else (rowD["business_id"], 0) ).collectAsMap()

def rest_price_rangeF(business_id):
    if business_id in rest_price_rangeD:
        return rest_price_rangeD[business_id]
    else:
        # just an arbitrary number 0 or 2
        return 0



# variance of user review dates
var_review_date_userD = reviewRDD.map(lambda row: json.loads(row)).map(lambda rowD: (rowD["user_id"], int("".join(rowD["date"].split('-'))) )).groupByKey().map(lambda row: ( row[0], statistics.variance(list(row[1])) ) if len(list(row[1]))>=2 else ( row[0], 0 ) ).collectAsMap()

# overall average used for imputation
var_review_date_userD_avg = sum([date for date in var_review_date_userD.values()])/len(var_review_date_userD)
def var_review_date_userF(user_id):
    if (user_id in var_review_date_userD) and (var_review_date_userD[user_id] or var_review_date_userD[user_id] == 0):
        return var_review_date_userD[user_id]
    else:
        # the average
        return var_review_date_userD_avg



avg_star_ratings_userD = userRDD.map(lambda row: json.loads(row)).map(lambda rowD: (rowD["user_id"], rowD["average_stars"])).collectAsMap()

# overall average used for imputation
# avg_star_ratings_userD_avg = sum([stars for stars in avg_star_ratings_userD.values()])/len(avg_star_ratings_userD)
def avg_star_ratings_userF(user_id):
    if (user_id in avg_star_ratings_userD) and (avg_star_ratings_userD[user_id]):
        return avg_star_ratings_userD[user_id]
    else:
        # the average
        # return avg_star_ratings_userD_avg
        return avg_star_ratings_businessD_avg



# Review count per user
review_count_userD = userRDD.map(lambda row: json.loads(row)).map(lambda rowD: (rowD["user_id"], rowD["review_count"])).collectAsMap()

# overall average used for imputation
review_count_userD_avg = sum([count for count in review_count_userD.values()])/len(review_count_userD)
def review_count_userF(user_id):
    if (user_id in review_count_userD) and (review_count_userD[user_id] or review_count_userD[user_id] == 0):
        return review_count_userD[user_id]
    else:
        # the average
        return review_count_userD_avg



yelping_since_userD = userRDD.map(lambda row: json.loads(row)).map(lambda rowD: ( rowD["user_id"], int("".join(rowD["yelping_since"].split('-'))) )).collectAsMap()

yelping_since_userD_max = max([date for date in yelping_since_userD.values()])
def yelping_since_userF(user_id):
    if (user_id in yelping_since_userD) and (yelping_since_userD[user_id]):
        return yelping_since_userD[user_id]
    else:
        # ??? ???
        return yelping_since_userD_max + 1



fans_userD = userRDD.map(lambda row: json.loads(row)).map(lambda rowD: ( rowD["user_id"], rowD["fans"]) ).collectAsMap()

def fans_userF(user_id):
    if (user_id in fans_userD) and (fans_userD[user_id] or fans_userD[user_id] == 0):
        return fans_userD[user_id]
    else:
        # the average
        return 0



funny_userD = userRDD.map(lambda row: json.loads(row)).map(lambda rowD: (rowD["user_id"], rowD["funny"])).collectAsMap()

# funny_userD_avg = sum([count for count in funny_userD.values()])/len(funny_userD)
def funny_userF(user_id):
    if (user_id in funny_userD) and (funny_userD[user_id] or funny_userD[user_id] == 0):
        return funny_userD[user_id]
    else:
        return 0


useful_userD = userRDD.map(lambda row: json.loads(row)).map(lambda rowD: (rowD["user_id"], rowD["useful"])).collectAsMap()

useful_userD_avg = sum([count for count in useful_userD.values()])/len(useful_userD)
def useful_userF(user_id):
    if (user_id in useful_userD) and (useful_userD[user_id] or useful_userD[user_id] == 0):
        return useful_userD[user_id]
    else:
        return useful_userD_avg


cool_userD = userRDD.map(lambda row: json.loads(row)).map(lambda rowD: (rowD["user_id"], rowD["cool"])).collectAsMap()

cool_userD_avg = sum([count for count in cool_userD.values()])/len(cool_userD)
def cool_userF(user_id):
    if (user_id in cool_userD) and (cool_userD[user_id] or cool_userD[user_id] == 0):
        return cool_userD[user_id]
    else:
        return cool_userD_avg



comp_hot_userD = userRDD.map(lambda row: json.loads(row)).map(lambda rowD: (rowD["user_id"], rowD["compliment_hot"])).collectAsMap()

# comp_hot_userD_avg = sum([count for count in comp_hot_userD.values()])/len(comp_hot_userD)
def comp_hot_userF(user_id):
    if (user_id in comp_hot_userD) and (comp_hot_userD[user_id] or comp_hot_userD[user_id] == 0):
        return comp_hot_userD[user_id]
    else:
        return 0


comp_note_userD = userRDD.map(lambda row: json.loads(row)).map(lambda rowD: (rowD["user_id"], rowD["compliment_note"])).collectAsMap()

# average
def comp_note_userF(user_id):
    if (user_id in comp_note_userD) and (comp_note_userD[user_id] or comp_note_userD[user_id] == 0):
        return comp_note_userD[user_id]
    else:
        return 0


#comp_photos_userD = userRDD.map(lambda row: json.loads(row)).map(lambda rowD: (rowD["user_id"], rowD["compliment_photos"])).collectAsMap()

# average
#def comp_photos_userF(user_id):
#    if (user_id in comp_photos_userD) and (comp_photos_userD[user_id] or comp_photos_userD[user_id] == 0):
#        return comp_photos_userD[user_id]
#    else:
#        return 0


# other compliments didn't seem to help



friends_userD = userRDD.map(lambda row: json.loads(row)).map(lambda rowD: (rowD["user_id"], rowD["friends"].split(", ")) if rowD["friends"] else (rowD["user_id"], ["None"]) ).map(lambda rowL: (rowL[0], 0) if rowL[1]==['None'] else (rowL[0], len(rowL[1])) ).collectAsMap()

friends_userD_avg = sum([count for count in friends_userD.values()])/len(friends_userD)

def friends_userF(user_id):
    if (user_id in friends_userD):
        return friends_userD[user_id]
    else:
        return friends_userD_avg
        # return 0



# A json file I created using online common name sources like "List of common male names." and "List of common female names." from CMU; as well as other guess and check work
# this json/dictionary can approximately map most names (common) names to their associated gender
# Could have simply hard coded this dictionary but this seemed neater
name_gender_path = "name_gender.json"
with open(name_gender_path, 'r') as openfile:
    name_genderD = json.load(openfile)

names_femaleRDD = userRDD.map(lambda row: json.loads(row)).map(lambda rowD: (rowD["user_id"], rowD["name"], name_genderD[rowD["name"].lower()]) if rowD["name"].lower() in name_genderD else (rowD["user_id"], rowD["name"], None) ).\
    map(lambda rowL: (rowL[0], rowL[1], "1") if "F" == rowL[2] else (rowL[0], rowL[1], rowL[2])).\
    map(lambda rowL: (rowL[0], rowL[1], "0") if "M" == rowL[2] else (rowL[0], rowL[1], rowL[2])).\
    map(lambda rowL: (rowL[0], int(rowL[2])) if type(rowL[2]) == "str" else (rowL[0], rowL[2]))

names_femaleD = names_femaleRDD.collectAsMap()

def names_female_dummyF(user_id):
    if user_id in names_femaleD:
        return names_femaleD[user_id]
    else:
        return None



#############################################################################################################################

# Regression Model-Based CF


train_X1 = reviewRDD.map(lambda row: json.loads(row)).map(lambda rowD: (rowD["user_id"], rowD["business_id"]) ).map(lambda rowL: [ avg_star_ratings_userF(rowL[0]), avg_star_ratings_businessF(rowL[1]), var_review_date_userF(rowL[0]), review_count_userF(rowL[0]), review_count_businessF(rowL[1]), fans_userF(rowL[0]), user_raters_per_businessF(rowL[1]), business_ratings_per_userF(rowL[0]), avg_star_ratings_userF(rowL[0])*avg_star_ratings_businessF(rowL[1]), latF(rowL[1]), longF(rowL[1]) ] ).collect()

train_X2 = reviewRDD.map(lambda row: json.loads(row)).map(lambda rowD: (rowD["user_id"], rowD["business_id"]) ).map(lambda rowL: [ good_for_groupsF(rowL[1]), yelping_since_userF(rowL[0]), dummy_wifiF(rowL[1], "no"), dummy_wifiF(rowL[1], "n/a"), rest_price_rangeF(rowL[1]), good_for_kidsF(rowL[1]), checkin_days_businessF(rowL[1]) ] ).collect()

train_X3 = reviewRDD.map(lambda row: json.loads(row)).map(lambda rowD: (rowD["user_id"], rowD["business_id"]) ).map(lambda rowL: [ funny_userF(rowL[0]), useful_userF(rowL[0]), cool_userF(rowL[0]), funny_userF(rowL[0])+useful_userF(rowL[0])+cool_userF(rowL[0]), comp_hot_userF(rowL[0]), comp_note_userF(rowL[0]) ] ).collect()

train_X4 = reviewRDD.map(lambda row: json.loads(row)).map(lambda rowD: (rowD["user_id"], rowD["business_id"]) ).map(lambda rowL: [ friends_userF(rowL[0]), names_female_dummyF(rowL[0]), days_openF(rowL[1]), opening_timeF(rowL[1]), closing_timeF(rowL[1]) ] ).collect()

train_X = [l1+l2+l3+l4+l5 for l1, l2, l3, l4, l5 in zip(train_X1, train_X2, train_X3, train_X4, train_X_frendsRatingL)]
# had to break it down into chunks; or else Vocaruem can't handle it for some reason (ie 406); makes no difference on PyCharm
# hopefully if instructors choose to rerun on Vocareum it should work; though this wasn't noted as a requirement
    # worst case, please run in locally if you need to verify something



val_X1 = valRDD.map(lambda rowL: [ avg_star_ratings_userF(rowL[0]), avg_star_ratings_businessF(rowL[1]), var_review_date_userF(rowL[0]), review_count_userF(rowL[0]), review_count_businessF(rowL[1]), fans_userF(rowL[0]), user_raters_per_businessF(rowL[1]), business_ratings_per_userF(rowL[0]), avg_star_ratings_userF(rowL[0])*avg_star_ratings_businessF(rowL[1]), latF(rowL[1]), longF(rowL[1]) ] ).collect()

val_X2 = valRDD.map(lambda rowL: [ good_for_groupsF(rowL[1]), yelping_since_userF(rowL[0]), dummy_wifiF(rowL[1], "no"), dummy_wifiF(rowL[1], "n/a"), rest_price_rangeF(rowL[1]), good_for_kidsF(rowL[1]), checkin_days_businessF(rowL[1]) ] ).collect()

val_X3 = valRDD.map(lambda rowL: [ funny_userF(rowL[0]), useful_userF(rowL[0]), cool_userF(rowL[0]), funny_userF(rowL[0])+useful_userF(rowL[0])+cool_userF(rowL[0]), comp_hot_userF(rowL[0]), comp_note_userF(rowL[0]) ] ).collect()

val_X4 = valRDD.map(lambda rowL: [ friends_userF(rowL[0]), names_female_dummyF(rowL[0]), days_openF(rowL[1]), opening_timeF(rowL[1]), closing_timeF(rowL[1]) ] ).collect()

val_X = [l1+l2+l3+l4+l5 for l1, l2, l3, l4, l5 in zip(val_X1, val_X2, val_X3, val_X4, val_X_frendsRatingL)]


all_train_X = train_X.copy() + val_X.copy()
# As a general rule of thumb in data science, the more (good) data a model is trained on, the better it performs
# So, although also using the validation set to train the regressor will cause the leaderboard RMSE to not be represenative due to data leakage, there should still be an overall improvement compared to having not used it



train_y = reviewRDD.map(lambda row: json.loads(row)).map(lambda rowD: rowD["stars"]).collect()

val_y = valRDD.map(lambda rowL: rowL[-1]).collect()

all_train_y = train_y.copy()+val_y.copy()




regressor = xgb.XGBRegressor(random_state=1, learning_rate=0.05, reg_alpha=0.2, n_estimators=1000, max_depth=6, objective="reg:linear", base_score=1, tree_method="auto", grow_policy="depthwise", max_bin=256, gamma=0, reg_lambda=1, colsample_bytree=0.7)

# regressor.fit(np.array(train_X), train_y)
regressor.fit(np.array(all_train_X), all_train_y)



with open ("competition_xgb_pickle", "wb") as f:
    pickle.dump(regressor, f)


    
regressor_cat = cat.CatBoostRegressor(random_state=1, learning_rate=0.05, n_estimators=1000, max_depth=10, max_bin=256, verbose=0, colsample_bylevel=0.8)

# regressor.fit(np.array(train_X), train_y)
regressor_cat.fit(np.array(all_train_X), all_train_y)

with open ("competition_cat_pickle", "wb") as f:
    pickle.dump(regressor_cat, f)



end_time = time.time()
print(end_time-start_time)

