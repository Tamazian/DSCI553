# Alain Tamazian


"""
Method Description:
For the competition, I implemented a Weighted Hybrid Recommendation System, using Item-Based and Model-Based collaborative filtering. The Item-Based CF considers the business’s 20 closest neighbors and takes advantage of the case amplification extension. For negative weights, I interpret their corresponding rating as a negative and then interpolate it within a range of 1-5 (also changing its PCC to a positive).
The predictor models, XGBoost and CatBoost Regressors – which use gradient boosted decision trees – are combined with equal-weight ensemble learning. Hyperparameters were tuned through a combination of GridSearchCV and manual testing; the key parameters were learning_rate, n_estimators, max_depth, and colsample_by*. However, most of the improvements to the regression models were the results of increasing the predictor features; I used 30. Features were extracted from “business.json”, “user.json”, “review_train.json”, and “checkin.json” – though mostly from the former two.
The features from “business.json” are mostly comprised of binary dummy variables. Two useful and interesting features that were extracted from “user.json” were based on its “name” and “friends” properties. For the former, I used logic, guess-and-check work, and common gender-specific names to surmise the users' gender from their first name. For the latter, I used the average known ratings (on the target business) of a user’s friends as a predictor; similar to the intuition for user-based CF – where friends act as neighbors. Some other useful features were yelping_since, review count, and rating averages.
Overall, as mentioned, the two recommendation systems were combined into a weighted hybrid. However, the method weighting uses an element of switching. On the occasion that the user-business pair is well-suited to be predicted through Item-Based CF (e.g. if the business has many known neighbors), a weight of 0.2 is given to the Item-Based CF and 0.8 to the regressor-ensemble; otherwise the weighting is 0.1 and 0.9, respectively.

Error Distribution:
>=0 and <1: 110749
>=1 and <2: 26854
>=2 and <3: 4153
>=3 and <4: 288
>=4: 0

RMSE:
0.86362

Execution Time: 641 seconds
"""


#############################################################################################################################


# Extra Notes about Error Distributions and RMSE

"""
The Error Distributions and RMSE seen above can be misleading due to the data leakage resulting from including the validation set in the hybrid model's training.
Neither of these will be representative of the model's actual performance; the item-based CF seems to be especially susceptible to data leakage.
Still, using the validation set (as was allowed in Piazza @1174) should improve the model's overall performance; just not that much.
Below are the error distribution and RMSE of this weighted hybrid when trained on only the training set -- making predictions for the validation set.

Error Distribution:
>=0 and <1: 102634
>=1 and <2: 32556
>=2 and <3: 6066
>=3 and <4: 788
>=4: 0

RMSE:
0.97278
"""


#############################################################################################################################


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



#outFile_csv = "competition_voc.csv"
# # <business_filepath>
outFile_csv = sys.argv[3]


#inFile_test = "/Users/alaintamazian/DocumentsAT/DSCI553/Homework/HW3/Datasets/yelp_val_in.csv"
inFile_test = sys.argv[2]



# Param: folder_path: the path of dataset folder, which contains exactly the same file as the google drive.
#folder_path = "/Users/alaintamazian/DocumentsAT/DSCI553/Homework/HW3/data"
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


# Item-based CF Recommendation System (using Pearson Correlation Coefficients)


with open(yelp_train_path, 'r') as f:
    csv_reader = csv.reader(f)
    # first line is the column names [user_id, business_id, stars]
    dataL_train = list(csv_reader)[1:]

with open(yelp_val_path, 'r') as f:
    csv_reader = csv.reader(f)
    # first line is the column names [user_id, business_id, stars]
    dataL_val = list(csv_reader)[1:]

dataL = dataL_train.copy() + dataL_val.copy()
dataRDD = sc.parallelize(dataL)



# dictionary where the key is (user_id, business_id) and value is rating
# will be used to get all of the r[u,n], r[u,i], r[u,j] values for the PCC and Prediction formulas
ratingsD = dataRDD.map(lambda row: ((row[0], row[1]), row[2])).collectAsMap()
training_rating_avg = sum([float(rating) for rating in ratingsD.values()])/len(ratingsD)


# key is user_id ; value is every business that the key user rated ; value is a list of businesses
businesses_rated_by_userD = dataRDD.map(lambda row: (row[0], row[1])).groupByKey().map(lambda kv: (kv[0], tuple(kv[1]))).collectAsMap()

# key is user_id ; value is the rating for every business that the key user rated ; value is a list of ratings
businesses_ratings_userD = dataRDD.map(lambda row: (row[0], float(row[2]))).groupByKey().map(lambda kv: (kv[0], tuple(kv[1]))).collectAsMap()


# key is business_id ; value is every user rating that the key business rated ; value is a list of users
users_rated_businessesD = dataRDD.map(lambda row: (row[1], row[0])).groupByKey().map(lambda kv: (kv[0], tuple(kv[1]))).collectAsMap()

# key is business_id ; value is every user rating of that the key business ; value is a list of ratings
users_ratings_businessesD = dataRDD.map(lambda row: (row[1], float(row[2]))).groupByKey().map(lambda kv: (kv[0], tuple(kv[1]))).collectAsMap()



with open(inFile_test, 'r') as f:
    csv_reader = csv.reader(f)
    # first line is the column names [user_id, business_id]
    data_to_predictL = list(csv_reader)[1:]



# linear interpolation formula: to map a number into a different specific range
# x_cord is a list of the min and max of our number x's old range
# y_cord is a list of the min and max of the new range (we want to map to)
def linear_interp(x, x_cord, y_cord):
    x1 = x_cord[0]
    x2 = x_cord[1]
    y1 = y_cord[0]
    y2 = y_cord[1]

    # y1+(x-x1)*(y2-y1)/(x2-x1)
    y = y1+(x-x1)*(y2-y1)/(x2-x1)
    return y


def custom_sort(lst):
    lst.insert(0, (0,0,0))
    lst = sorted(lst, key=lambda tup: -(tup)[-1])
    index0 = lst.index((0,0,0))
    lst1 = lst[:index0]
    lst2 = sorted(lst[index0+1:], key=lambda tup: tup[-1])
    return lst1+lst2


# number of item i's top neighbors which will be used for the predictions
n_neighbors = 20
    # try other numbers
# significant difference between n_neighbors 15 and 17
# even if we find optimal N here, it'll likely be different on the test set !!! ???



def find_coraters(x):
    good_for_ib_cfL = True
    user_u = x[0]
    business_i = x[1]

    if user_u in businesses_rated_by_userD:
        n_businessesL = businesses_rated_by_userD[user_u]
    else:
        n_businessesL = []
        good_for_ib_cfL = False

    return (user_u, business_i, n_businessesL, good_for_ib_cfL)


# PCC
def pcc(x):
    user_u = x[0]
    business_i = x[1]
    n_businessesL = x[2]
    good_for_ib_cfL = x[-1]

    if (business_i not in users_ratings_businessesD) or ( user_u not in businesses_rated_by_userD):
        good_for_ib_cfL = False

        return (user_u, business_i, sorted([], key=lambda tup: -tup[-1])[:n_neighbors], good_for_ib_cfL)

    pcc_weightsL = []
    # pcc
    for business_j in n_businessesL:
        # find number of users that rated both i and j
        users_that_rated_i = set(users_rated_businessesD[business_i])
        users_that_rated_j = set(users_rated_businessesD[business_j])
        users_that_rated_ij = users_that_rated_i.intersection(users_that_rated_j)


        ### what if no coraters
        if not users_that_rated_ij:
            # choose a default w ??? !!! arbitrary
            # ***
            pcc_weightsL.append((business_j, 0.65))
            continue

        ### what if only 1 coraters ; correlation can only be calculated when the array for i and j is at least 2
        elif len(users_that_rated_ij) == 1:
            corater = next(iter(users_that_rated_ij))

            # choose a default w ??? !!! arbitrary ; try different hardcoded numbers
            if (float(ratingsD[(corater, business_i)]) > training_rating_avg) and (float(ratingsD[(corater, business_j)]) > training_rating_avg):
                # intuitive, rough correlation
                pcc_weightsL.append((business_j, 0.75))
                # make more dynamic ???
                continue
            elif (float(ratingsD[(corater, business_i)]) < training_rating_avg) and (float(ratingsD[(corater, business_j)]) < training_rating_avg):
                pcc_weightsL.append((business_j, 0.80))
                continue
            else:
                pcc_weightsL.append((business_j, 0.35))
                continue
            # something more custom related to the one co-rater/user !!! ???

        # pcc is technically calculable but not enough data to be accurate
        # elif (len(users_that_rated_ij) == 2) or (len(users_that_rated_ij) == 3) or (len(users_that_rated_ij) == 4):
        elif (len(users_that_rated_ij) == 2) or (len(users_that_rated_ij) == 3):
            pcc_weightsL.append((business_j, 0.65))
            continue


        # mean based on coraters ; increases accuracy
        i_avg = 0
        j_avg = 0
        for user in users_that_rated_ij:
            i_avg += float(ratingsD[user, business_i])
            j_avg += float(ratingsD[user, business_j])
        business_i_avg_rating = i_avg / len(users_that_rated_ij)
        business_j_avg_rating = j_avg / len(users_that_rated_ij)


        # pearson correlation coefficient formula
        w_num = 0
        w_den_i = 0
        w_den_j = 0
        smooth = 0
        for x, pcc_user in enumerate(users_that_rated_ij):
            # smooth += x/1000000
            # w_num += (float(ratingsD[(pcc_user, business_i)])+smooth - business_i_avg_rating) * (float(ratingsD[(pcc_user, business_j)])+smooth - business_j_avg_rating)
            # w_den_i += ((float(ratingsD[(pcc_user, business_i)]) - business_i_avg_rating)+smooth) ** 2
            # w_den_j += ((float(ratingsD[(pcc_user, business_j)]) - business_j_avg_rating)+smooth) ** 2

            w_num += (float(ratingsD[(pcc_user, business_i)]) - business_i_avg_rating) * (float(ratingsD[(pcc_user, business_j)]) - business_j_avg_rating)
            w_den_i += (float(ratingsD[(pcc_user, business_i)]) - business_i_avg_rating) ** 2
            w_den_j += (float(ratingsD[(pcc_user, business_j)]) - business_j_avg_rating) ** 2


        ### what are we supposed to do when the PCC weight denominator is 0
        if (not w_den_i) or (not w_den_j):
            w = 0.7
        elif (not w_num):
            w = 0.95
        else:
            w = (w_num) / ( w_den_i**(1/2) * w_den_j**(1/2) )


        # during float division, sometimes it may go over (like 1.00000002)
        if w > 1:
            w = 1
        if w < -1:
            w = -1

        # if w > 0:
        pcc_weightsL.append((business_j, w))
        # compare with actual pcc value


    # Im assuming actual PCC weights aren't such neatly rounded decimals; so its unlikely for there to be unintended overlap ???
    hard_coded_weights = (0.65, 0.75, 0.8, 0.35, 0.65, 0.7, 0.95)
    true_weightsL = [w[1] for w in pcc_weightsL if w[1] not in hard_coded_weights]
    if len(true_weightsL) < 2:
        # try increasing this number ??????
        good_for_ib_cfL = False



    if len(pcc_weightsL) <= n_neighbors:
        return (user_u, business_i, pcc_weightsL, good_for_ib_cfL)
    else:
        ### How to select "best" neighbors -- while considering negatives
        return ( user_u, business_i, sorted(pcc_weightsL, key=lambda tup: -tup[-1])[:n_neighbors], good_for_ib_cfL )
            ### why is this version better than the below ???
        # return ( user_u, business_i, sorted(pcc_weightsL, key=lambda tup: -abs(tup[-1]))[:n_neighbors] )
        # return ( user_u, business_i, custom_sort(pcc_weightsL)[:n_neighbors] )


def cf_prediction(x):
    user_u = x[0]
    business_i = x[1]
    pcc_weightsL = x[2]
    good_for_ib_cfL = x[-1]

    ### what if no user id and business id
    if (business_i not in users_ratings_businessesD) and (user_u not in businesses_rated_by_userD):
        good_for_ib_cfL = False
        return (user_u, business_i, training_rating_avg, good_for_ib_cfL)
        # return (user_u, business_i, 3)
    ### what if no user id in training set -- ie what if no corated businesses
    elif user_u not in businesses_rated_by_userD:
        good_for_ib_cfL = False
        i_avg = sum(users_ratings_businessesD[business_i]) / len(users_ratings_businessesD[business_i])
        # use training_rating_avg instead ???
        return (user_u, business_i, i_avg, good_for_ib_cfL)
    ### what if no business id
    ### what if no corater users ???
    elif (business_i not in users_ratings_businessesD) or (not pcc_weightsL):
        good_for_ib_cfL = False
        # the "or" is redundant based on "######" ???
        u_avg = sum(businesses_ratings_userD[user_u]) / len(businesses_ratings_userD[user_u])
        ### use training_rating_avg instead ???
        return (user_u, business_i, u_avg, good_for_ib_cfL)

    prediction_num = 0
    prediction_den = 0

    # normalized item-based CF formula
        # need the average rating of business j (from all its users)
    i_avg = sum(users_ratings_businessesD[business_i])/len(users_ratings_businessesD[business_i])

    for tup in pcc_weightsL:
        business_j = tup[0]
        w = tup[1]

        j_avg = sum(users_ratings_businessesD[business_j]) / len(users_ratings_businessesD[business_j])

        # handling negatives
        if w < 0:
            # if business i and j have a PCC (correlation) of -1 (i.e.) inverse, if business j has rating of 5, it is the opposite for i
            # but the opposite of 5 isn't -5; need to map (ie interpolate correctly) so that a 5 score when weight is negative becomes 1
            # so, we interpolate the rating value (to account for the negative correlation) and change the weight to positive
            rating_j = linear_interp(-float(ratingsD[(user_u, business_j)]), [-5, -1], [1, 5])
            w = abs(w)
            # this is more intuition than math !!! ???
        else:
            rating_j = float(ratingsD[(user_u, business_j)])


        # case amplification extension
        prediction_num += (rating_j-j_avg) * w**2.5
        # prediction_num += float(ratingsD[(user_u, business_j)]) * w
        prediction_den += abs(w)**2.5



    ### what are we supposed to do when the prediction denominator is 0 ; when main formula basically can't be used
    if not prediction_den:
        good_for_ib_cfL = False
        if business_i in users_ratings_businessesD:
            bus_ratings_len = len(users_ratings_businessesD[business_i])
            business_avg_rating = sum(users_ratings_businessesD[business_i]) / len(users_ratings_businessesD[business_i])
        else:
            business_avg_rating = None
            bus_ratings_len = 0

        if user_u in businesses_ratings_userD:
            user_ratings_len = len(businesses_ratings_userD[user_u])
            user_avg_rating = sum(businesses_ratings_userD[user_u]) / len(businesses_ratings_userD[user_u])
        else:
            user_avg_rating = None
            user_ratings_len = 0


        # weigh the joint "average" based on how much data was available for each type (i.e. len)
        if (bus_ratings_len) and (user_ratings_len):
            prediction = ((business_avg_rating*bus_ratings_len) / (bus_ratings_len + user_ratings_len)) + \
                         ((user_avg_rating*user_ratings_len) / (bus_ratings_len + user_ratings_len))

        elif not bus_ratings_len:
            prediction = user_avg_rating
        elif not user_ratings_len:
            prediction = business_avg_rating
        else:
            # prediction = 3
            # instead use the mean ??? !!!
            prediction = training_rating_avg

    else:
        prediction = prediction_num / prediction_den + i_avg


    if prediction > 5:
        prediction = 5.0
    elif prediction < 1:
        prediction = 1.0

    return (user_u, business_i, prediction, good_for_ib_cfL)
    # return (user_u, business_i, abs(prediction))


ib_predictionsRDD = sc.parallelize(data_to_predictL).map(find_coraters).map(pcc).map(cf_prediction)




#############################################################################################################################


# Regression Model-Based CF (using XGBRegressor)


businessRDD = sc.textFile(business_path)
userRDD = sc.textFile(user_path)
reviewRDD = sc.textFile(review_path)
checkinRDD = sc.textFile(checkin_path)
# tipRDD = sc.textFile(tip_path)
# photoRDD = sc.textFile(photo_path)

with open(inFile_test, 'r') as f:
    csv_reader = csv.reader(f)
    # first line is the column names: [user_id,business_id,stars]
    test_dataL = list(csv_reader)[1:]
testRDD = sc.parallelize(test_dataL)


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

test_X_frendsRatingL = []
for i, (user, bus) in enumerate( testRDD.map(lambda rowL: (rowL[0], rowL[1])).collect() ):
    stars = friends_avg_star_ratings_userF(user, bus)
    # print(i, user, bus, stars)
    test_X_frendsRatingL.append([stars])




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



# A json file I created based on common names I know and found from sources like "List of common male names." and "List of common female names." from CMU; as well as other guess and check work
# unisex names were ignored; For some, I also went through the names that appeared most often in user.json (eg in the 1000s) and manually assigned them their corresponding gender
# this resulting json/dictionary (I made) can approximately map most (common) names to their associated gender
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


test_X1 = testRDD.map(lambda rowL: [ avg_star_ratings_userF(rowL[0]), avg_star_ratings_businessF(rowL[1]), var_review_date_userF(rowL[0]), review_count_userF(rowL[0]), review_count_businessF(rowL[1]) ] ).collect()

test_X2 = testRDD.map(lambda rowL: [ fans_userF(rowL[0]), user_raters_per_businessF(rowL[1]), business_ratings_per_userF(rowL[0]), avg_star_ratings_userF(rowL[0])*avg_star_ratings_businessF(rowL[1]), latF(rowL[1]), longF(rowL[1]) ] ).collect()

test_X3 = testRDD.map(lambda rowL: [ good_for_groupsF(rowL[1]), yelping_since_userF(rowL[0]) ] ).collect()

test_X4 = testRDD.map(lambda rowL: [ dummy_wifiF(rowL[1], "no"), dummy_wifiF(rowL[1], "n/a"), rest_price_rangeF(rowL[1]), good_for_kidsF(rowL[1]), checkin_days_businessF(rowL[1]) ] ).collect()

test_X5 = testRDD.map(lambda rowL: [ funny_userF(rowL[0]), useful_userF(rowL[0]), cool_userF(rowL[0]), funny_userF(rowL[0])+useful_userF(rowL[0])+cool_userF(rowL[0]) ] ).collect()

test_X6 = testRDD.map(lambda rowL: [ comp_hot_userF(rowL[0]), comp_note_userF(rowL[0]) ] ).collect()

test_X7 = testRDD.map(lambda rowL: [ friends_userF(rowL[0]), names_female_dummyF(rowL[0]) ] ).collect()

test_X8 = testRDD.map(lambda rowL: [ days_openF(rowL[1]), opening_timeF(rowL[1]), closing_timeF(rowL[1]) ] ).collect()

test_X = [l1+l2+l3+l4+l5+l6+l7+l8+l9 for l1, l2, l3, l4, l5, l6, l7, l8, l9 in zip(test_X1, test_X2, test_X3, test_X4, test_X5, test_X6, test_X7, test_X8, test_X_frendsRatingL)]
# had to break it down into chunks; or else Vocaruem can't handle it for some reason (ie 406); makes no difference on PyCharm
# Vocareum was inconsistent regarding this



picklesL = ["competition_xgb_pickle", "competition_cat_pickle"]
modelsL = ["xgb", "cat"]

ensemble_predsD = {}

for pickle_path, model in zip(picklesL, modelsL):
    with open (pickle_path, "rb") as f:
        regressor = pickle.load(f)

    y_predictions_model = regressor.predict(np.array(test_X))
    ensemble_predsD[model] = y_predictions_model.copy()

    del regressor

y_predictionsL = [0.5*p1+0.5*p2 for p1, p2 in zip(ensemble_predsD["xgb"], ensemble_predsD["cat"])]


############################################################################################################################


# weighted hybrid

# float multiplication in Spark is worse than in Python !!! ???
hybrid_predRDD = ib_predictionsRDD.zipWithIndex().map(lambda rowL: ( rowL[0][0], rowL[0][1], rowL[0][2]*0.2+y_predictionsL[rowL[1]]*0.8 ) \
    if rowL[0][3] else ( rowL[0][0], rowL[0][1], rowL[0][2]*0.1+y_predictionsL[rowL[1]]*0.9 ) ).map(lambda rowL: (rowL[0], rowL[1], 5.0) \
    if rowL[2]>5 else rowL ).map(lambda rowL: (rowL[0], rowL[1], 1.0) if rowL[2]<1 else rowL )
predictionsL = hybrid_predRDD.collect()

# weighted_predL = []
# ib_predictionsL = ib_predictionsRDD.map(lambda row: row[-1]).collect()
# for ib_rating, reg_rating in zip(ib_predictionsL, y_predictionsL):
#     weighted_rating = ib_rating*0.2 + reg_rating*0.8
#     weighted_predL.append(weighted_rating)
# predictionsL = ib_predictionsRDD.zipWithIndex().map(lambda rowL: ( rowL[0][0], rowL[0][1], weighted_predL[rowL[1]] ) ).collect()



with open(outFile_csv, "w") as f:
    f.write("user_id, business_id, prediction\n")
    writer = csv.writer(f)
    writer.writerows(predictionsL)


end_time = time.time()
print(end_time-start_time)


