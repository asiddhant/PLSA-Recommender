#!/usr/bin/python
# -*- coding: latin-1 -*- 
import numpy as np
import csv
from os import sys
import random 
import time
from operator import itemgetter
import math
import pickle   
import numpy as np
import matplotlib.pyplot as plt

# These are dictionaries storing mean and variance for each movie,cluster ->  mean[movieName][clusterId]
# meanRating = None
# varRating = None

# # Access by [userId][movieId]
# ratingData = None

# # Dictionary for number of reviews for a movie
# movieList = None

# # Dictionary to store probabilities P(z|user) , accessed by [userId][clusterId]
# latent_given_user = None

# # Dictionary for storing P(z | user,movie,rating) , access by [userId][movie][clusterId] , as of now ratings are not included because they are stored separately 
# latent_given_all = None

# Returns a dictionary of ratings as  ratingData[userId][movieId]

__name__="__plsa_package__.py"


def readData(path,fileName):   
    uMap = {}
    mMap = {}
    tstart = time.time()
    reader = open(path+fileName,'r');
    print(path+fileName)
    # f = open(fileName,'r');
    # reader = csv.reader(f,delimiter="::");
    maxUser = 0;
    maxMovie = 0;
    ctr = 0;
    userList = {}
    movieList = {}
    time2000 = 60*60*24*365*30;
    for line in reader:
        ctr += 1;
        if ctr == 1:
            continue;
        lineVector = line.split(',');
        user = int(lineVector[0])
        movie = int(lineVector[1])
        ratingTime = int(lineVector[3])
        if ratingTime > time2000:
            if user in userList:
                userList[user] += 1
            else:
                userList[user] = 1

            if movie in movieList:
                movieList[movie] += 1
            else:
                movieList[movie] = 1

    reader.close();
    reader = open(path+fileName,'r');

    ctr = 0;
    print("it should be : ",len(userList.keys()))
    print("it should be : ",len(movieList.keys()))
    for user in list(userList.keys()):
        # if userList[user] > 250:
        ctr += 1;           
        uMap[user] = ctr;

    maxUser = ctr;
    ctr = 0;
    for movie in list(movieList.keys()):
        # if movieList[movie] > 50:
        ctr += 1;           
        mMap[movie] = ctr;

    # temp = [movieList[movie] for movie in movieList.keys() if movieList[movie] < 30]
    # print temp
    
    maxMovie =  ctr
    # ratingData is maxUser+1 because users are given IDs starting from 1 and I don't want to use -1 with userId everywhere I try to access them

    print(maxUser,maxMovie)
    # ratingData = np.zeros((maxUser+1,maxMovie+1))
    # print 'RatingData Size',ratingData.shape
    ctr = 0;
    ratingData = {}
    for line in reader:
        ctr += 1;
        if ctr == 1:
            continue;

        lineVector = line.split(',');
        user = int(lineVector[0])
        movie = int(lineVector[1])
        rating = float(lineVector[2]);
        if user in uMap and movie in mMap:
            if uMap[user] in ratingData:
                ratingData[uMap[user]][mMap[movie]] = rating;
            else:
                ratingData[uMap[user]] = {}
                ratingData[uMap[user]][mMap[movie]] = rating;

    # for key in uMap.keys():
    #   print key,uMap[key]
    # print '------------'
    # for key in mMap.keys():
    #   print key,mMap[key]
    dumpUserMovieMapping(path,uMap,mMap);
    return ratingData,maxUser,maxMovie

# Writes testData to a file, testData is dictionary with key being user and item being tuple of movie,rating
def writeTestData(path,fileName,testData):
    f = open(path+fileName,'w');
    writer = csv.writer(f);
    for user in list(testData.keys()):
        writer.writerow([user,testData[user][0],testData[user][1]]);
    
    f.close;

# Reads and returns testData as a dictionary
def readTestData(path,fileName):
    f = open(path+fileName,'r');
    reader      = csv.reader(f);
    testData    = {}
    for line in reader:
        user    = int(line[0]);
        movie   = int(line[1]);
        rating  = float(line[2]);
        testData[user] = (movie,rating);
    
    f.close();
    return testData;

# Writes the parameters to a file
def dumpData(path,meanRating,varRating,latent_given_user,userMeanRating,userVarRating):
    
    f = open(path+'modelDump/'+'meanRating','wb');
    pickle.dump(meanRating, f)
    f.close();

    f = open(path+'modelDump/'+'varRating','wb');
    pickle.dump(varRating, f)
    f.close();

    f = open(path+'modelDump/'+'latent_given_user','wb');
    pickle.dump(latent_given_user, f)
    f.close();

    f = open(path+'modelDump/'+'userMeanRating','wb');
    pickle.dump(userMeanRating, f)
    f.close();

    f = open(path+'modelDump/'+'userVarRating','wb');
    pickle.dump(userVarRating, f)
    f.close();

# Reads parameters from the file
def readFromDump(path):
    
    f = open(path+'modelDump/'+'meanRating','rb');
    meanRating = pickle.load(f, encoding="latin-1");
    f.close();

    f = open(path+'modelDump/'+'varRating','rb');
    varRating = pickle.load(f, encoding="latin-1");
    f.close();

    f = open(path+'modelDump/'+'latent_given_user','rb');
    latent_given_user = pickle.load(f, encoding="latin-1");
    f.close();

    f = open(path+'modelDump/'+'userMeanRating','rb');
    userMeanRating = pickle.load(f, encoding="latin-1");
    f.close();

    f = open(path+'modelDump/'+'userVarRating','rb');
    userVarRating = pickle.load(f, encoding="latin-1");
    f.close();


    return meanRating,varRating,latent_given_user,userMeanRating,userVarRating

def readClusterInfo(path):
    f = open(path+'modelDump/'+'movieRankList','rb');
    movieRankList = pickle.load(f, encoding="latin-1");
    f.close();

    f = open(path+'modelDump/'+'genreScore','rb');
    genreScore = pickle.load(f, encoding="latin-1");
    f.close();
    return movieRankList,genreScore

# saves uMap and mMap which . uMap maps real users to userIds in range 1-maxUser . mMap also does the same
def dumpUserMovieMapping(path,uMap,mMap):

    f = open(path+'modelDump/'+'userMapping','wb');
    pickle.dump(uMap, f)
    f.close();

    f = open(path+'modelDump/'+'movieMapping','wb');
    pickle.dump(mMap, f)
    f.close();

# returns uMap and mMap which . uMap maps real users to userIds in range 1-maxUser . mMap also does the same
def readUserMovieMapping(path):

    f = open(path+'modelDump/'+'userMapping','rb');
    uMap = pickle.load(f, encoding="latin-1");
    f.close();

    f = open(path+'modelDump/'+'movieMapping','rb');
    mMap = pickle.load(f, encoding="latin-1");
    f.close();

    return uMap,mMap

# Returns meanRating,varRating,latent_given_user,latent_given_all
def initParameters(ratingData,numLatent,maxUser,maxMovie):
    
    meanRating  =  5*np.random.rand(maxMovie+1,numLatent);
    varRating   =  1.5*np.random.rand(maxMovie+1,numLatent);

    print('Initialising latent variable data')
    latent_given_user = np.random.rand(maxUser+1,numLatent);

    latent_given_all  = {}
    for user in list(ratingData.keys()):
        tempSum = np.sum(latent_given_user[user]);
        latent_given_user[user] = latent_given_user[user]/tempSum ;

        # Initialise for only movies which user has rated
        for movie in list(ratingData[user].keys()):
            tempSum = 0;
            if user not in latent_given_all:
                latent_given_all[user] = {}

            if movie not in latent_given_all[user]:
                latent_given_all[user][movie] = np.zeros((numLatent,1))

            for latentVar in range(0,numLatent):
                latent_given_all[user][movie][latentVar] = random.random();
                tempSum += latent_given_all[user][movie][latentVar];

            for latentVar in range(0,numLatent):
                latent_given_all[user][movie][latentVar] = latent_given_all[user][movie][latentVar]/tempSum;

    return meanRating,varRating,latent_given_user,latent_given_all

# Returns density as point in gaussian distribution with given mean and variance 
def calcGaussian(point,mean,variance):
    exponent = -1*(point-mean)*(point-mean)/(2*variance);
    denominator = math.sqrt(2*3.14*variance);
    temp = math.exp(exponent);
    return temp/denominator

# Returns testData and updated ratingData with testData entries removed from it.
def partitionData(ratingData):
    
    print('--------------------------')
    testData = {}
    for user in list(ratingData.keys()):
        tempMovieList = []
        for movie in list(ratingData[user].keys()):
            tempMovieList += [movie]
        if len(tempMovieList) > 5:
            movieChosen = random.choice(tempMovieList);
            testData[user] = (movieChosen,ratingData[user][movieChosen])
            del ratingData[user][movieChosen]
        else:
            print('User has very few ratings',user,len(tempMovieList))
        
    return testData,ratingData;

# Returns normalised ratingData , userMeanRating and userVarRating; Normalisation for each user
def normaliseRatings(ratingData,maxUser):

    smoothingConstant = 3;

    # Calculating overall mean of ratings 
    numNonZero  = 0;
    ratingSum   = 0;
    for user in list(ratingData.keys()):
        for movie in list(ratingData[user].keys()):
            ratingSum +=  ratingData[user][movie]
            numNonZero += 1
        
    overallMeanRating = ratingSum/numNonZero;

    overallVarSum = 0;
    # Calculating overall variance of ratings 
    for user in list(ratingData.keys()):
        for movie in list(ratingData[user].keys()):
            temp = (ratingData[user][movie] - overallMeanRating);
            overallVarSum += temp*temp

    overallVarRating = overallVarSum/numNonZero;
    print('Over all rating mean',overallMeanRating)
    print('Over all rating variance',overallVarRating)
    userMeanRating = np.zeros((maxUser+1,1))
    userVarRating  = np.zeros((maxUser+1,1))

    # Calculating mean for each user and smoothening it.
    for user in list(ratingData.keys()):
        tempRatingSum   = 0.0;
        tempNZSize      = 0.0
        for movie in list(ratingData[user].keys()):
            tempRatingSum +=  ratingData[user][movie]
            tempNZSize += 1
        userMeanRating[user] = (tempRatingSum + smoothingConstant*overallMeanRating)/(tempNZSize + smoothingConstant);

    # Calculating variance for each user and smoothening it.
    for user in list(ratingData.keys()):
        tempNZSize = 0.0;
        tempVarSum = 0.0;
        for movie in list(ratingData[user].keys()):
            temp = (ratingData[user][movie] - userMeanRating[user]);
            tempVarSum += temp*temp
            tempNZSize += 1;

        userVarRating[user] = (tempVarSum + smoothingConstant*overallVarRating)/(tempNZSize + smoothingConstant);

    # Normalising rating for each user
    for user in list(ratingData.keys()):
        for movie in list(ratingData[user].keys()):
            ratingData[user][movie] -= userMeanRating[user];
            temp = math.sqrt(userVarRating[user]);
            ratingData[user][movie] /= temp;

    return ratingData,userMeanRating,userVarRating

# Predicts rating for trainingData itself.
def predictRating(ratingData,meanRating,varRating,latent_given_user,userMeanRating,userVarRating):
    
    s,numLatent     = meanRating.shape;
    ratingList  = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5];
    error = 0;
    ctr   = 0;
    rms   = 0;
    for user in list(ratingData.keys()):
        for movie in list(ratingData[user].keys()):
            bestRating = 3;
            maxRatingProb = -1;
            for ratingVal in ratingList:
                tempRatingVal = (ratingVal - userMeanRating[user])/(math.sqrt(userVarRating[user]));
                tempRating = 0;
                tempRatingProb = 0;
                for latentVar in range(0,numLatent):
                    tempMean    = meanRating[movie][latentVar]
                    tempVar     = varRating[movie][latentVar]
                    if tempVar != 0:
                        tempProb    = calcGaussian(tempRatingVal,tempMean,tempVar) 
                        if math.isnan(tempProb) == False:
                            tempRatingProb += latent_given_user[user][latentVar]*tempProb
                # print ratingVal,'\t--->',tempRatingProb
                if tempRatingProb > maxRatingProb :
                    maxRatingProb = tempRatingProb
                    bestRating = ratingVal
                    # print 'BestRating',bestRating,maxRatingProb,tempRatingProb

            tempMovieRating = ratingData[user][movie]*(math.sqrt(userVarRating[user])) + userMeanRating[user];
            # print 'Actual Rating',tempMovieRating
            # print 'Predicted Rating',bestRating
            error += abs(tempMovieRating - bestRating);
            rms   += (tempMovieRating - bestRating)*(tempMovieRating - bestRating);
            ctr += 1;

    rms = rms/ctr;
    rms = math.sqrt(rms);
    print('Error',error,ctr)
    print('Avg',error/ctr)
    print('RMS',rms)
    return rms,error/ctr

# Predicts rating on testData and calculates mean error
def predictTestData(testData,meanRating,varRating,latent_given_user,userMeanRating,userVarRating):
    s,numLatent     = meanRating.shape;
    # ratingList  = [1,2,3,4,5];
    ratingList  = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
    error = 0;
    ctr   = 0;
    rms = 0
    for user in list(testData.keys()):
        movie,realRating = testData[user]
        bestRating = 3;
        maxRatingProb = -1;
        for ratingVal in ratingList:
            tempRatingVal = (ratingVal - userMeanRating[user])/(math.sqrt(userVarRating[user]));
            tempRating = 0;
            tempRatingProb = 0;
            for latentVar in range(0,numLatent):
                tempMean    = meanRating[movie][latentVar]
                tempVar     = varRating[movie][latentVar]

                if tempVar != 0:
                    tempProb    = calcGaussian(tempRatingVal,tempMean,tempVar) 
                    if math.isnan(tempProb) == False:
                        tempRatingProb += latent_given_user[user][latentVar]*tempProb
            
            if tempRatingProb > maxRatingProb :
                maxRatingProb = tempRatingProb
                bestRating = ratingVal

        # print 'Actual Rating', realRating
        # print 'Predicted Rating',bestRating
        diff    = realRating - bestRating;
        error   += abs(diff);
        rms     += diff*diff;
        ctr     += 1;

    print('Error',error,ctr)
    rms = rms/ctr
    rms = math.sqrt(rms);
    print('RMS',rms)
    print('Avg',error/ctr)
    return rms,error/ctr

# Returns latent_given_all[user][movie][latentVar]
def expectationStep(ratingData,meanRating,varRating,latent_given_user,latent_given_all):
    
    s,numLatent     = meanRating.shape;
    print("In expectation step")
    ctr = 0
    # Updating P(z|user,movie,rating)
    dist = {}

    for user in list(ratingData.keys()):
        for movie in list(ratingData[user].keys()):
            tempSum = 0;
            for latentVar in range(0,numLatent):

                tempMean    = meanRating[movie][latentVar]
                tempVar     = varRating[movie][latentVar]
                
                if (tempVar != 0):
                    ratingProb  = calcGaussian(ratingData[user][movie],tempMean,tempVar);
                    if math.isnan(ratingProb) == False:
                        latent_given_all[user][movie][latentVar] = latent_given_user[user][latentVar]*ratingProb;
                        
                else:
                    pass
                    # print movie,latentVar,tempMean,tempVar

                tempSum     += latent_given_all[user][movie][latentVar];

            if tempSum != 0:
                latent_given_all[user][movie] = latent_given_all[user][movie]/tempSum;
            else:
                print('TempSUm is zero')
                pass

    return latent_given_all

# Returns latent_given_user,meanRating,varRating
def maximisationStep(ratingData,meanRating,varRating,latent_given_user,latent_given_all):
    
    maxMovie,numLatent  = meanRating.shape;
    maxMovie    = maxMovie - 1;

    # Updating mean and variance for each movie,latentVariable pair
    print("In maximisation step")
    for movie in range(1,maxMovie+1):
        for latentVar in range(0,numLatent):
            newMean = 0.0;
            newVariance = 0.0;
            denominator = 0.0;
            numMovieCtr = 0.0;
            for user in list(ratingData.keys()):
                if movie in ratingData[user]:
                    numMovieCtr += 1;
                    tempRating = ratingData[user][movie];
                    newMean = newMean + tempRating*latent_given_all[user][movie][latentVar];
                    denominator = denominator + latent_given_all[user][movie][latentVar];

            if newMean != 0.0:
                newMean = newMean/denominator;

                numMovieCtr = 0;
                for user in list(ratingData.keys()):
                    if movie in ratingData[user]:
                        numMovieCtr += 1
                        tempRating = ratingData[user][movie];
                        gg1 = pow((tempRating - newMean),2)
                        gg2 = latent_given_all[user][movie][latentVar];
                        newVariance = newVariance + gg1*gg2;

                newVariance = newVariance/denominator;
            else:
                newMean = 0.1
                newVariance = 0.1
                # print 'MAX::movieCtr',numMovieCtr,movie,latentVar

            meanRating[movie][latentVar]    = newMean;
            varRating[movie][latentVar]     = newVariance;

    # Updating P(z|user) 
    print('Updating P(z|user)')
    for user in list(ratingData.keys()):
        probSum = 0;
        for latentVar in range(0,numLatent):
            newProb     = 0;
            for movie in list(ratingData[user].keys()):
                newProb += latent_given_all[user][movie][latentVar];

            latent_given_user[user][latentVar] = newProb;
            if newProb == 0:
                print('Zero  in maximisation step')
            probSum = probSum + newProb;

        latent_given_user[user] = latent_given_user[user]/probSum;

    return latent_given_user,meanRating,varRating

# Returns log0likelihood of the training data as per the given parameters
def calculateLikelihood(ratingData,meanRating,varRating,latent_given_user):
    
    s,numLatent         = meanRating.shape;
    LL = 0
    print('In calculate log likelihood')
    exceptionCtr = 0
    ratingProb = 0
    for user in list(ratingData.keys()):
        for movie in list(ratingData[user].keys()):
            for latentVar in range(0,numLatent):
                tempMean    = meanRating[movie][latentVar]
                tempVar     = varRating[movie][latentVar]
                if (tempVar != 0):
                    ratingProb = 1
                    try:
                        ratingProb  = calcGaussian(ratingData[user][movie],tempMean,tempVar);
                    except:
                        exceptionCtr += 1
                        ratingProb = 1

                    try:
                        temp = math.log(latent_given_user[user][latentVar]);
                        if math.isnan(ratingProb) == False:
                            temp += math.log(ratingProb);
                        
                        LL += temp
                    except:
                        pass

    print('-----------------------------------------------------',exceptionCtr,LL)
    return -1*LL;

# Performs EM step and calculates log-likelihood to measure convergence
def performEM(ratingData,meanRating,varRating,latent_given_user,latent_given_all):
    
    s,numLatent         = meanRating.shape;
    converged   = False;
    oldLL       = 0;
    epsilon     = 10000;
    ctr         = 0;

    tstart = time.time();
    while( not converged):
        ctr += 1;
        t1 = time.time();
        latent_given_all = expectationStep(ratingData,meanRating,varRating,latent_given_user,latent_given_all);
        t2 = time.time();
        print('Time taken by expectation step',t2-t1)
        
        t1 = time.time();
        latent_given_user,meanRating,varRating = maximisationStep(ratingData,meanRating,varRating,latent_given_user,latent_given_all);
        t2 = time.time();
        print('Time taken by maximisation step',t2-t1)

        newLL = calculateLikelihood(ratingData,meanRating,varRating,latent_given_user);
        diff = abs(newLL - oldLL);
        print('Iteration::',ctr,'Difference::',diff,'LogLikelihood',newLL)

        if diff < epsilon:
            converged = True;

        if ctr > 10:
            converged = True;

        oldLL = newLL;

    tfinish = time.time();
    print('Total time taken is ',tfinish - tstart);
    return meanRating,varRating,latent_given_user,latent_given_all

# Returns score of a movie as per parameters passed
def calcScore(bestRating,maxRatingProb,numRating):
    N_0 = 390
    P_0 = 0.4
    numRatingFactor = 1- (math.exp(-1*numRating/float(N_0)))
    ratingProbFactor = 1- (math.exp(-1*maxRatingProb/float(P_0)))
    score =  bestRating*ratingProbFactor*numRatingFactor;
    if bestRating*score < 0:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n\n\n\n\n\n")
        print(bestRating,maxRatingProb,numRating)
        print(ratingProbFactor,numRatingFactor)
        print(score)
    return score

# Returns a dictionary of dictionaries. movieList[movieId] gives a dictionary which stores name ,genre, and other data about movie
def getMovieData(path,fileName):

    uMap,mMap = readUserMovieMapping(path);
    reader      = open(path+fileName,'r',encoding='latin-1');
    reader = csv.reader(reader);
    movieList   = {};
    ctr = 0
    for line in reader:
        ctr += 1
        if ctr ==1:
            continue;
        lineVector  = line
        movieId     = int(lineVector[0])
        movieName   = lineVector[1]
        numRating   = int(lineVector[3])
        movieYear   = int(lineVector[4])
        avgRating   = float(lineVector[5])
        temp        = lineVector[6].split(' ');
        genreList=["adventure","animation","children","comedy","fantasy","romance","drama","action","crime", "thriller","horror","mystery","sci.fi","imax","documentary","war","musical","western","film.noir"]
        movieGenre  = []
        ctr = 0;
        for genre in temp:
            if genre.strip() == '1':
                movieGenre += [genreList[ctr]]
            ctr += 1

        temp = lineVector[7].split(' ');
        starList = []
        for star in temp:
            try:
                starList += [int(star)];
            except:
                pass

        if movieId in mMap:
            movieId = mMap[movieId];
            movieList[movieId] = {'name':movieName,'numRating':numRating,'avgRating':avgRating,'year':movieYear,'star':starList,'genre':movieGenre}
    
    return movieList;

def getStarData(path,fileName):

    reader      = open(path+fileName,'r',encoding='latin-1');
    reader      = csv.reader(reader);
    starList    = {};
    ctrQ = 0
    for line in reader:
        #print(ctrQ)
        ctrQ += 1
        if ctrQ ==1:
            continue;

        # "star_id","stars","mstars","noofmovies","movies","ratings","pop","year","genre"
        starId      = int(line[0])
        starName    = line[1]
        numMovies   = int(line[3])
        tempMovieList = line[4].split();
        avgRating   = float(line[5])
        popularity  = int(line[6]);
        year        = int(line[7]);
        tempGenreList = line[8].split();

        temp        = line[6].split(' ');
        genreList=["adventure","animation","children","comedy","fantasy","romance","drama","action","crime", "thriller","horror","mystery","sci.fi","imax","documentary","war","musical","western","film.noir"]
        starGenreList  = []
        ctr = 0;
        for genre in tempGenreList:
            if genre.strip() == '1':
                starGenreList += [genreList[ctr]]
            ctr += 1

        starMovieList = []
        for movie in tempMovieList:
            starMovieList += [int(movie)];

        starList[starId] = {'name':starName,'numMovies':numMovies,'avgRating':avgRating,'popularity':popularity,'year':year,'genreList':starGenreList,'movieList':starMovieList}
    
    return starList;

# Removes user,movie tuple present in testData from ratingData and returns updated ratingData
def removeFromTrainData(path,testFileName,ratingData):

    testData = readTestData(path,testFileName);
    for user in list(testData.keys()):
        movie = testData[user][0];
        del ratingData[user][movie]

    return ratingData;

# Just does prediction of ratings on testData
def readAndTest(path,testFileName):
    meanRating,varRating,latent_given_user,userMeanRating,userVarRating = readFromDump(path,);

    # predictRating(ratingData,meanRating,varRating,latent_given_user);
    testData = readTestData(path,testFileName);
    predictTestData(testData,meanRating,varRating,latent_given_user,userMeanRating,userVarRating);

    # tempList = getRecommendations_collaborative(1,meanRating,varRating,latent_given_user,userMeanRating[1],userVarRating[1]);
    # ctr = 0;
    # for item in tempList:
    #   ctr += 1
    #   print ctr,item


    # n = input('Press enter to continue')
    # print 'Bye'

# Updates P(z|u) , mean and variance for y,z where <u,y,v> is the new rating by an old user for an old movie
def updateMaximisation(userParam,movieParam,ratingData,meanRating,varRating,latent_given_user,latent_given_all):
    
    s,numLatent         = meanRating.shape;
    
    # Updating mean and variance for each movie,latentVariable pair
    print("In maximisation step")

    for latentVar in range(0,numLatent):
        newMean     = 0.0;
        newVariance = 0.0;
        denominator = 0.0;
        numMovieCtr = 0.0;
        for user in list(ratingData.keys()):
            if movieParam in ratingData[user]:
                numMovieCtr += 1;
                tempRating  = ratingData[user][movieParam];
                newMean     = newMean       + tempRating*latent_given_all[user][movieParam][latentVar];
                denominator = denominator   + latent_given_all[user][movieParam][latentVar];

        if newMean != 0.0:
            newMean = newMean/denominator;
            numMovieCtr = 0;
            for user in list(ratingData.keys()):
                if movieParam in ratingData[user]:
                    numMovieCtr += 1
                    tempRating = ratingData[user][movieParam];
                    gg1 = pow((tempRating - newMean),2)
                    gg2 = latent_given_all[user][movieParam][latentVar];
                    newVariance = newVariance + gg1*gg2;

            newVariance = newVariance/denominator;
        else:
            newMean = 0.1
            newVariance = 0.1
            # print 'MAX::movieCtr',numMovieCtr,movieParam,latentVar

        meanRating[movieParam][latentVar]   = newMean;
        varRating[movieParam][latentVar]    =  newVariance;

    # Updating P(z|user) 
    print('Updating P(z|user)')
    probSum = 0;
    for latentVar in range(0,numLatent):
        newProb     = 0;
        for movie in list(ratingData[userParam].keys()):
            newProb += latent_given_all[userParam][movie][latentVar];

        latent_given_user[userParam][latentVar] = newProb;
        probSum = probSum + newProb;

    latent_given_user[userParam] = latent_given_user[userParam]/probSum;

    return latent_given_user,meanRating,varRating

# Updates P(z|u,y,v) for the new rating given by an old user for an old movie
def updateExpectation(user,movie,rating,meanRating,varRating,latent_given_user,latent_given_all):
    s,numLatent     = meanRating.shape;
    print("In Update expectation step")
    # Updating P(z|user,movie,rating)
    tempSum = 0;
    for latentVar in range(0,numLatent):
        tempMean    = meanRating[movie][latentVar]
        tempVar     = varRating[movie][latentVar]
        
        if (tempVar != 0):
            ratingProb  = calcGaussian(rating,tempMean,tempVar);
            if math.isnan(ratingProb) == False:
                latent_given_all[user][movie][latentVar] = latent_given_user[user][latentVar]*ratingProb;                           
        else:
            pass
            # print movie,latentVar,tempMean,tempVar

        tempSum     += latent_given_all[user][movie][latentVar];

    latent_given_all[user][movie] = latent_given_all[user][movie]/tempSum;
    return latent_given_all

# Performs EM  to update parameters accordingly if a new rating is added for old movie by old user
def performIncrementalEM(updateTuple,ratingData,parameters):

    (user,movie,rating) = updateTuple;
    (meanRating,varRating,latent_given_user,latent_given_all) = parameters;

    ratingData[user][movie] = rating;
    print('Inside rating',ratingData[user][movie])
    converged   = False;
    oldLL       = 0;
    epsilon     = 100;
    ctr         = 0;

    tstart = time.time();
    while( not converged):
        ctr += 1;
        print('~~~~~1~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(meanRating[movie])
        print('~~~~~1~~~~~~~~~~~~~~~~~~~~~~~~~~~')

        t1 = time.time();
        latent_given_all = updateExpectation(user,movie,rating,meanRating,varRating,latent_given_user,latent_given_all);
        t2 = time.time();
        print('Time taken by expectation step',t2-t1)
        
        print('..............................................')
        print(latent_given_all[user][movie])
        print('..............................................')

        t1 = time.time();
        latent_given_user,meanRating,varRating = updateMaximisation(user,movie,ratingData,meanRating,varRating,latent_given_user,latent_given_all);
        t2 = time.time();
        print('Time taken by maximisation step',t2-t1)

        # newLL = calculateLikelihood(ratingData,meanRating,varRating,latent_given_user);
        # diff = abs(newLL - oldLL);
        # print 'Iteration::',ctr,'Difference::',diff,'LogLikelihood',newLL

        # if diff < epsilon:
        #   converged = True;

        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(meanRating[movie])
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

        if ctr > 3:
            converged = True;

        newLL = 9
        oldLL = newLL;

    tfinish = time.time();
    print('Total time taken for update is ',tfinish - tstart);
    return ratingData,meanRating,varRating,latent_given_user,latent_given_all

def testIncrement(path,ratingData,meanRating,varRating,latent_given_user,latent_given_all,userMeanRating,userVarRating):
    
    s,numLatent     = meanRating.shape;

    testData = readTestData(path,'testFile_1.csv');
    tempUser    = None
    tempMovie   = None
    tempRating  = None
    temp = list(testData.keys());
    tempUser = temp[0];
    tempMovie,tempRating = testData[tempUser];

    normRating = (tempRating - userMeanRating[tempUser])/(math.sqrt(userVarRating[tempUser]));

    print('User',tempUser)
    print('Movie',tempMovie)
    print('Rating',tempRating)
    print('Rating2',normRating)
    print('Rating3',ratingData[tempUser][tempMovie])
    ctr = 0
    for user in list(ratingData.keys()):
        if tempMovie in ratingData[user]:
            ctr+=1

    print('Number of ratings for this movie',ctr)
    print('Number of movies rated by this user',len(list(ratingData[tempUser].keys())))
    print('latent_given_all[tempUser][tempMovie]',latent_given_all[tempUser][tempMovie])
    print('latent_given_user[tempUser]',latent_given_user[tempUser])
    print('meanRating[tempMovie]',meanRating[tempMovie])
    print('varRating[tempMovie]',varRating[tempMovie])
    print('=======================================================================================================')


    update = (tempUser,tempMovie,normRating);
    parameters = (meanRating,varRating,latent_given_user,latent_given_all);
    ratingData,meanRating,varRating,latent_given_user,latent_given_all = performIncrementalEM(update,ratingData,parameters);

    print('=======================================================================================================')
    ctr = 0;
    for user in list(ratingData.keys()):
        if tempMovie in ratingData[user]:
            ctr+=1

    print('Number of ratings for this movie',ctr)
    ctr = len(list(ratingData[tempUser].keys()))
    
    print('Number of movies rated by this user',ctr)
    print('latent_given_all[tempUser][tempMovie]',latent_given_all[tempUser][tempMovie])
    print('latent_given_user[tempUser]',latent_given_user[tempUser])
    print('meanRating[tempMovie]',meanRating[tempMovie])
    print('varRating[tempMovie]',varRating[tempMovie])

# Returns ratingData,meanRating,varRating,latent_given_user,latent_given_all . testFileName is optional
def run(path,numLatent,ratingFileName,testFileName=None):

    logFile = open('logFile.txt','a');
    logFile.write('-----------------------------------------------------------------\n');
    if testFileName != None:
        logFile.write('Running PLSI with numLatent:' + str(numLatent)+' and file:'+ testFileName+'\n');
    else:
        logFile.write('Running PLSI with numLatent:' + str(numLatent)+'\n');
    tstart = time.time();

    print('-----------------------------------------------------------------')
    print('----------------Running PLSI with numLatent',numLatent,'--------------')
    ratingData,maxUser,maxMovie = readData(path,ratingFileName);
    print('Read Data')

    # testData,ratingData = partitionData(ratingData);
    # writeTestData(testFileName,testData);
    if testFileName != None :
        ratingData = removeFromTrainData(path,testFileName,ratingData);

    tstart = time.time();
    meanRating,varRating,latent_given_user,latent_given_all = initParameters(ratingData,numLatent,maxUser,maxMovie);
    tfinish = time.time();
    print('Init took',tfinish - tstart)
    ctr = 0;
    for user in list(ratingData.keys()):
        for movie in list(ratingData[user].keys()):
            ctr+=1

    print('Number of ratings remaining',ctr)
    logFile.write('Number of ratings remaining:'+str(ctr)+'\n');

    ratingData,userMeanRating,userVarRating = normaliseRatings(ratingData,maxUser);
    print('Normalised ratings as well')

    t1 = time.time()
    meanRating,varRating,latent_given_user,latent_given_all = performEM(ratingData,meanRating,varRating,latent_given_user,latent_given_all);
    t2 = time.time()
    print('Done with EM'  ,t2-t1)
    logFile.write('Done with EM:'+str(t2-t1)+'\n');

    dumpData(meanRating,varRating,latent_given_user,userMeanRating,userVarRating);
    print("Done dumping");

    logFile.close();
    logFile = open('logFile.txt','a');

    t1 = time.time()
    rmsError,absError = predictRating(ratingData,meanRating,varRating,latent_given_user,userMeanRating,userVarRating);
    logFile.write('Error of TrainData:\n\tRMS:'+str(rmsError)+'\n\tABS:'+str(absError)+'\n');
    t2 = time.time()
    
    print('Predicting on test data,training one took ',t2-t1)

    if testFileName != None:
        testData = readTestData(path,testFileName);
        rmsError,absError = predictTestData(testData,meanRating,varRating,latent_given_user,userMeanRating,userVarRating);
        logFile.write('Error of TestData:\n\tRMS:'+str(rmsError)+'\n\tABS:'+str(absError)+'\n');

    print('----------------Done with PLSI having numLatent',numLatent,'--------------')
    print('-----------------------------------------------------------------')

    tfinish = time.time();
    logFile.write('Running PLSI with numLatent:' + str(numLatent)+'   Took time:'+str(tfinish - tstart)+'\n');
    logFile.write('-----------------------------------------------------------------\n');
    logFile.close();

    # testIncrement(ratingData,meanRating,varRating,latent_given_user,latent_given_all,userMeanRating,userVarRating);

    return ratingData,meanRating,varRating,latent_given_user,latent_given_all

# To see which how genres are ranked in different clusters
def mineClusterType(path,ratingFileName,movieFileName):
    
    meanRating,varRating,latent_given_user,userMeanRating,userVarRating = readFromDump(path);
    s,numLatent     = meanRating.shape;
    ratingData,maxUser,maxMovie =  readData(path,ratingFileName);

    
    movieRankList   = calcMovieGivenLatent(path,meanRating,varRating,ratingFileName);
    movieData       = getMovieData(path,movieFileName);
    print('Starting')
    # Dictionary of dictionaries, first key: latentVar , second key genre , third key, score and count
    genreScore ={}

    # for i in range(0,maxMovie):
    #   for latentVar in range(0,numLatent):
    #       print i,movieRankList[latentVar][i]
    #   print ""

    # for movieId in movieData.keys():
    #   print movieId,movieData[movieId]

    for latentVar in range(0,numLatent):
        genreScore[latentVar] = {}

    # maxMovie = len(movieRankList[0])
    # print maxMovie

    # for latentVar in range(0,numLatent):
    #   print '///',len(movieRankList[latentVar])
    # g,genreWeights = buildGenreCoOccurence(movieFileName);

    numMovieToConsider = maxMovie;
    if maxMovie > 500:
        numMovieToConsider = 500

    for i in range(0,numMovieToConsider):
        for latentVar in range(0,numLatent):
            movieId         = movieRankList[latentVar][i][0]
            movieRating     = movieRankList[latentVar][i][1]
            movieRatingProb = movieRankList[latentVar][i][2]
            movieScore      = movieRankList[latentVar][i][4]


            movieName = movieData[movieId]['name'];
            movieGenre = movieData[movieId]['genre'];
        
            for genre in movieGenre:
                if genre in genreScore[latentVar]:
                    genreScore[latentVar][genre]['score']   += movieScore#*(math.exp(-1*i/float(numMovieToConsider)));
                    genreScore[latentVar][genre]['rating']  += movieRating;
                    genreScore[latentVar][genre]['count']   += 1
                else:
                    genreScore[latentVar][genre] = {}
                    genreScore[latentVar][genre]['score']   = movieScore#*(math.exp(-1*i/float(numMovieToConsider)));
                    genreScore[latentVar][genre]['rating']  = movieRating;
                    genreScore[latentVar][genre]['count']   = 1

                    # print genre,"{0:.3f}".format(tempweights[genre]),"{0:.3f}".format(movieRating),"{0:.3f}".format(tempweights[genre]*movieRating)
                
            # print i,movieRankList[latentVar][i]
    
    #  Need to some how calcuate score so as to effectively calculate cluster which are representative of mostly orthogonal genre choices
    for latentVar in range(0,numLatent):
        for genre in list(genreScore[latentVar].keys()):
            genreScore[latentVar][genre]['rating'] /= genreScore[latentVar][genre]['count']
            genreScore[latentVar][genre]['score'] /= genreScore[latentVar][genre]['count']

    for latentVar in range(0,numLatent):
        tempDict = genreScore[latentVar]
        genreScore[latentVar] = sorted(iter(tempDict.items()), key=lambda k_v: k_v[1]['score'] , reverse=True)

        ctr = 0
        for (genre,data) in genreScore[latentVar]:
            ctr += 1
            if len(genre) < 4:
                print(ctr,'\t',genre+'\t\t\t\t'+"{0:.3f}".format(data['rating'])+'\t' + "{0:.3f}".format(data['score']));
            elif len(genre) < 8:
                print(ctr,'\t',genre+'\t\t\t'+"{0:.3f}".format(data['rating'])+'\t' + "{0:.3f}".format(data['score']));
            elif len(genre) < 12:
                print(ctr,'\t',genre+'\t\t'+"{0:.3f}".format(data['rating'])+'\t' + "{0:.3f}".format(data['score']));
            else:
                print(ctr,'\t',genre+'\t'+"{0:.3f}".format(data['rating'])+'\t' + "{0:.3f}".format(data['score']));


        # print genreScore[latentVar]
        print('-------------------------------------------')


    f = open(path+'modelDump/genreScore','wb')
    pickle.dump(genreScore, f)
    f.close();

    f = open(path+'modelDump/movieRankList','wb')
    pickle.dump(movieRankList, f)
    f.close();

    return genreScore

# To see which movies are in top movies in each cluster
def mineClusterMovies(path,ratingFileName,movieFileName):
    
    meanRating,varRating,latent_given_user,userMeanRating,userVarRating = readFromDump();
    s,numLatent     = meanRating.shape;
    movieRankList   = calcMovieGivenLatent(path,meanRating,varRating,ratingFileName);
    movieData       = getMovieData(path,movieFileName);
    print('Starting')

    for latentVar in range(0,numLatent):
        for i in range(0,200):
            movieId         = movieRankList[latentVar][i][0]
            movieRating     = movieRankList[latentVar][i][1]
            movieRatingProb = movieRankList[latentVar][i][2]
            movieScore      = movieRankList[latentVar][i][4]
            movieName = 'NAN'
            if movieId in movieData:
                movieName = movieData[movieId]['name']

            printStr = str(i)+'\t'+"{0:.3f}".format(movieRating) + '\t' + "{0:.3f}".format(movieScore) + '\t' + movieName;
            print(printStr)
        print('----------------------------------------')
    

# Returns a dictionary with abg rating for each genre along with number of times that genre has been rated
def calcAvgGenreRating(path,ratingFileName,movieFileName):

    ratingData,maxUser,maxMovie     = readData(path,ratingFileName);
    movieData   = getMovieData(path,movieFileName);

    print('Starting')
    
    movieRating = {}
    gtr = 0
    for movie in range(1,maxMovie+1):
        numNonZero  = 0;
        ratingSum   = 0;
        for user in list(ratingData.keys()):
            if movie in ratingData[user]:
                ratingSum   += ratingData[user][movie];
                numNonZero  += 1;
        if numNonZero != 0:
            movieRating[movie] = ratingSum/numNonZero;

        if numNonZero < 5:
            gtr += 1

    print('Going to genre',gtr)
    genreRating = {}
    for movieId in list(movieRating.keys()):
        try:
            genreList = movieData[movieId]['genre']
            for genre in genreList:
                if genre in genreRating:
                    genreRating[genre]['rating'] += movieRating[movieId]
                    genreRating[genre]['count'] += 1
                else:
                    genreRating[genre] = {};
                    genreRating[genre]['rating'] = movieRating[movieId]
                    genreRating[genre]['count'] = 1

        except:
            # print movieId,'not present'
            pass


    for genre in list(genreRating.keys()):
        genreRating[genre]['rating'] /= genreRating[genre]['count'];
        # print genre,genreRating[genre]['rating']

    temp = sorted(iter(genreRating.items()), key=lambda k_v2: k_v2[1]['rating'] , reverse=True)

    ctr = 0
    for item in temp:
        ctr += 1
        genreRating[item[0]] = item[1]
        if len(item[0]) < 4:
            print(ctr,'\t',item[0]+'\t\t\t\t'+str(item[1]))
        elif len(item[0]) < 8:
            print(ctr,'\t',item[0]+'\t\t\t'+str(item[1]))
        elif len(item[0]) < 12:
            print(ctr,'\t',item[0]+'\t\t'+str(item[1]))
        else:
            print(ctr,'\t',item[0]+'\t'+str(item[1]))

        # print item[0] , item[1]

    # for genre in genreRating.keys():
    #   print genre,genreRating[genre]['rating']

    return genreRating,movieRating

# Returns a dictionary with clusterId as key , item being ranked list of (movie,rating,ratingProb) in each cluster.
def calcMovieGivenLatent(path,meanRating,varRating,ratingFileName):

    ratingData,maxUser,maxMovie     = readData(path,ratingFileName)
    s,numLatent         = meanRating.shape;

    # ratingList        = [-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3]
    ratingList = np.arange(-3, 3, 0.2)
    movieRankList   = {}

    print("Starting calcMovieGivenLatent")
    
    for latentVar in range(0,numLatent):
        movieRankList[latentVar] = []
        for movie in range(1,maxMovie+1):
            bestRating = meanRating[movie][latentVar];
            maxRatingProb = 0;
            if varRating[movie][latentVar] != 0:
                bestRating      = random.gauss(meanRating[movie][latentVar], math.sqrt(varRating[movie][latentVar]));
                maxRatingProb   = calcGaussian(bestRating,meanRating[movie][latentVar],varRating[movie][latentVar]);

            numRating = 0
            for user in list(ratingData.keys()):
                if movie in ratingData[user]:
                    numRating += 1

            score       = calcScore(bestRating,maxRatingProb,numRating);
            movieRankList[latentVar] += [(movie,bestRating,maxRatingProb,numRating,score)];

        movieRankList[latentVar] = sorted(movieRankList[latentVar], key=lambda dataTuple: dataTuple[4] , reverse=True)   # sort by rating

    # for i in range(0,maxMovie):
    #   for latentVar in range(0,numLatent):
    #       print i,movieRankList[latentVar][i]
    #   print ""

    return movieRankList

#. Returns a dictionary with movieId as  key and item being a dictionary having name,genreList,avgRating for that movie
# Strict = False : give movies with genre (can be occuring with other genres as well())
# Strict = True : give movies with genre (Has to occur alone)
def getMovieByGenre(path,genre,ratingFileName,movieFileName,strict=False):

    movieData   = getMovieData(path,movieFileName);
    ratingData,maxUser,maxMovie     = readData(path,ratingFileName);

    movieList = {}
    for movieId in list(movieData.keys()):
        movieName = movieData[movieId]['name'];
        movieGenre = movieData[movieId]['genre'];
        movieRating = 0;
        ratingCount = 0;
        for user in list(ratingData.keys()):
            if movieId in ratingData[user]:
                movieRating += ratingData[user][movieId];
                ratingCount += 1

        tempRating = 0.0;
        if ratingCount > 0:
            tempRating = movieRating/float(ratingCount);

        if genre in movieGenre:
            if (strict and (len(movieGenre) == 1)):
                movieList[movieId] = {'name':movieName, 'genre':movieGenre, 'rating':tempRating ,'numRating':ratingCount}
            elif (strict == False):
                movieList[movieId] = {'name':movieName, 'genre':movieGenre, 'rating':tempRating ,'numRating':ratingCount}

    
    temp = sorted(iter(movieList.items()), key=lambda k_v3: k_v3[1]['rating'] , reverse=True)

    for item in temp:
        movieList[item[0]] = item[1]
        # if movieList[item[0]]['numRating'] > 50:
        #     print(item[0],"{0:.3f}".format(round(movieList[item[0]]['rating'])),movieList[item[0]]['numRating'],movieList[item[0]]['name'],movieList[item[0]]['genre'])

    # for movieId in movieList.keys():
    #   print movieId,"{0:.3f}".format(movieList[movieId]['rating']),movieList[movieId]['name']

    return movieList;

# Returns a dictionary with keys as all possible combination of genres in data
def getGenreCombinations(path,fileName):
    reader      = open(path+fileName,'r');
    genreList   = {};
    for line in reader:
        lineVector  = line.split('::');
        temp        = lineVector[2].strip('\n');
        try:
            genreList[temp] += 1
        except:
            genreList[temp] = 1

    reader.close();
    for key in sorted(genreList.keys()):
        print(key,'.',genreList[key])
    return genreList;

def testRecommendations(userid):
    path='C:/Python34/Lib/collabrec/'
    uMap,mMap = readUserMovieMapping(path);
    meanRating,varRating,latent_given_user,userMeanRating,userVarRating = readFromDump(path);
    recommendations = getRecommendations_collaborative(path,uMap[userid],meanRating,varRating,latent_given_user,userMeanRating[uMap[userid]],userVarRating[uMap[userid]],'data/ratings.csv',[]);#,numRecommendation = None):
    movieData   = getMovieData(path,'data/movienames.csv');
    tempProb = []
    tempNum = []
    for item in recommendations:
        movieId     = item[0];
        rating      = item[1];
        ratingProb  = item[2];
        numRating   = item[3];
        score       = item[4];
        tempProb += [ratingProb]
        tempNum  += [numRating]
        # if movieId in movieData:
        #     print(str(movieId).zfill(4)+'\t'+"{0:.3f}".format(score)+'\t'+str(numRating).zfill(4) +'\t'+"{0:.3f}".format(rating)+ '\t'+"{0:.3f}".format(ratingProb),movieData[movieId]['name'],movieData[movieId]['genre'])
        #     # print str(movieId)+'\t'+str(rating)+'\t'+str(numRating) +'\t'+"{0:.3f}".format(score)+ str(movieData[movieId]['name']),movieData[movieId]['genre']
        # else:
        #     print(movieId,rating)
        #     pass
        

    # print 'MeanProb::',np.mean(np.array(tempProb))
    # print 'DvnProb::',math.sqrt(np.var(np.array(tempProb)))
    # print 'MeanNum::',np.mean(np.array(tempNum))
    # print 'DvnNum::',math.sqrt(np.var(np.array(tempNum)))

# Returns coOccurence score for each genre , also score for each a weight list having weight for each genre ,
# this weight will be used for mining cluster type
def buildGenreCoOccurence(path,movieFileName):

    movieData   = getMovieData(path,movieFileName);
    genreCoScore = {}
    for movieId in list(movieData.keys()):
        movieName = movieData[movieId]['name']
        movieGenre = movieData[movieId]['genre']
        for genre in movieGenre:
            if genre in genreCoScore:
                pass
            else:
                genreCoScore[genre] = {}

        for genre in movieGenre:
            for innerGenre in movieGenre:
                if innerGenre in genreCoScore[genre]:
                    if genre != innerGenre:
                        genreCoScore[genre][innerGenre] += 1
                    elif len(movieGenre) == 1:
                        genreCoScore[genre][innerGenre] += 1
                else:
                    if genre != innerGenre:
                        genreCoScore[genre][innerGenre] = 1
                    elif len(movieGenre) == 1:
                        genreCoScore[genre][innerGenre] = 1
                    else:
                        genreCoScore[genre][innerGenre] = 0

    # for genreX in sorted(genreCoScore.iterkeys()):
    #   if len(genreX) < 4:
    #       printStr = genreX + '\t\t\t'
    #   elif len(genreX) < 8:
    #       printStr = genreX + '\t\t'
    #   else:
    #       printStr = genreX + '\t'


    #   for genreY in sorted(genreCoScore.iterkeys()):
    #       try:
    #           printStr += str(genreCoScore[genreX][genreY]) + '\t'
    #       except:
    #           printStr += str(0) + '\t'
    #   print printStr

    genreWeights = {}
    for genreX in sorted(genreCoScore.keys()):
        tempList = []
        for genreY in sorted(genreCoScore.keys()):
            try:
                if genreX != genreY:
                    tempList += [genreCoScore[genreX][genreY]]
            except:
                pass
                # tempList += [0]

        tempArray = np.array(tempList)
        mean =  np.mean(tempArray)
        variance = np.var(tempArray)
        genreWeights[genreX] = {'mean':mean,'variance':variance}
        if len(genreX) < 4:
            printStr = genreX + '\t\t\t'
        elif len(genreX) < 8:
            printStr = genreX + '\t\t'
        else:
            printStr = genreX + '\t'

        printStr += "{0:.3f}".format(mean) + '\t' + "{0:.3f}".format(math.sqrt(variance))
        # print printStr

    return genreCoScore,genreWeights

# Gives back a list of movies with their "predicted" ratings and confidence ,sorted by the ratings.
# Expects userMeanRating and userVarRating to be specific to the userId and not arrays 
def getRecommendations_collaborative(path,userId,meanRating,varRating,latent_given_user,userMeanRating,userVarRating,ratingFileName,numMovieRating,exclusionList = [],numRecommendation = None):
    
    ratingData,maxUser,maxMovie = readData(path,ratingFileName);
    s,numLatent     = meanRating.shape;
    ratingList  = np.arange(0.5, 5, 0.1)
    movieRating = []
    printLog = False;

    probData = {}
    print(meanRating.shape)
    for movie in range(1,maxMovie+1):
        probData[movie] = {}
        bestRating      = 3;
        maxRatingProb   = -1;
        for ratingVal in ratingList:
            tempRatingVal   = (ratingVal - userMeanRating)/(math.sqrt(userVarRating));
            tempRatingProb  = 0;
            for latentVar in range(0,numLatent):
                tempMean    = meanRating[movie][latentVar]
                tempVar     = varRating[movie][latentVar]
                if tempVar != 0:
                    if probData[movie].has_key(latentVar):
                        tempProb = probData[movie][latentVar]
                    else:
                        tempProb    = calcGaussian(tempRatingVal,tempMean,tempVar)
                        probData[movie][latentVar] = tempProb;
                        
                    if math.isnan(tempProb) == False:
                        tempRatingProb += latent_given_user[userId][latentVar]*tempProb
                        
            
            if tempRatingProb > maxRatingProb :
                maxRatingProb   = tempRatingProb
                bestRating      = ratingVal
                
        #numRating = 0
        #for user in list(ratingData.keys()):
        #    if movie in ratingData[user]:
        #        numRating += 1
        numRating = numMovieRating[movie];

        score       = calcScore(bestRating,maxRatingProb,numRating);
        movieRating += [(movie,bestRating,maxRatingProb,numRating,score)]

    filteredList = []
    for dataTuple in movieRating:
        if str(dataTuple[0]) not in exclusionList:
            filteredList += [dataTuple];

    sortedList = sorted(filteredList, key=lambda dataTuple:dataTuple[4] , reverse=True)   # sort by score

    if numRecommendation == None:
        print('Giving back ',len(sortedList) ,' recommendations')
        return sortedList
    else:
        if numRecommendation > len(sortedList):
            print('Giving back ',len(sortedList) ,' recommendations')
            return sortedList
        else:
            print('Giving back ',numRecommendation ,' recommendations')
            return sortedList[0:numRecommendation];

# Returns a dictionary key: userId ,genrePreference[userId][genre] = {'rating': ,'numRating':}
# Based on user's ratings
def getUserGenrePreference(path,ratingFileName,movieFileName):

    ratingData,maxUser,maxMovie = readData(ratingFileName)

    movieData = getMovieData(movieFileName);
    genreList = {}
    genrePreference = {}

    for user in list(ratingData.keys()):
        genrePreference[user] = {}
        for movie in list(ratingData[user].keys()):
            movieGenre = movieData[movie]['genre']
            for genre in movieGenre:
                if genre in genrePreference[user]:
                    genrePreference[user][genre]['rating']      += ratingData[user][movie]
                    genrePreference[user][genre]['numRating']   += 1
                else:
                    genrePreference[user][genre] = {}
                    genrePreference[user][genre]['rating']      = ratingData[user][movie]
                    genrePreference[user][genre]['numRating']   = 1

                genreList[genre] = 1

        for genre in list(genrePreference[user].keys()):
            genrePreference[user][genre]['rating']  /=  genrePreference[user][genre]['numRating']


    printStr = ''
    for genre in sorted(genreList.keys()):
        try:
            printStr += genre[0:4] + '\t\t'
        except:
            printStr += genre + '\t\t'


    print(printStr) 
    # for user in range(1,maxUser+1):
    #   printStr = str(user).zfill(4) + '\t'
    #   for genre in sorted(genreList.iterkeys()):
    #       if genre in genrePreference[user].keys():
    #           printStr += "{0:.2f}".format(genrePreference[user][genre]['rating'])+","+str(int(genrePreference[user][genre]['numRating'])).zfill(3) + '\t';
    #       else:
    #           printStr += "{0:.2f}".format(0)+',000\t';

    #   print printStr
    
    for user in list(ratingData.keys()):
        printStr = str(user).zfill(4) + '\t'
        for genre,item in sorted(iter(genrePreference[user].items()), key=lambda k_v1: k_v1[1]['rating'] , reverse=True):
            if genre in list(genrePreference[user].keys()):
                printStr += genre[0:3] + "{0:.2f}".format(genrePreference[user][genre]['rating'])+ '\t';
                # printStr += genre[0:3] + "{0:.2f}".format(genrePreference[user][genre]['rating'])+","+str(int(genrePreference[user][genre]['numRating'])).zfill(3) + '\t';
            else:
                printStr += "{0:.2f}".format(0)+',000\t';

        print(printStr)

    return genrePreference

# Returns a dictionary for user preference , with score for features like Genre, director. Need to extend this possibly
'''def getContentBasedPreference(path,ratingFileName,movieData):

    ratingData,maxUser,maxMovie = readData(path,ratingFileName)
    genreList =["adventure","animation","children","comedy","fantasy","romance","drama","action","crime", "thriller","horror","mystery","sci.fi","imax","documentary","war","musical","western","film.noir"];

    userPreference = {}
    maxUserStar = 20

    for user in list(ratingData.keys()):
        for movie in list(ratingData[user].keys()):
            movieGenreList = movieData[movie]['genre'];
            moviestarList  = movieData[movie]['star'];

            if user in userPreference:
                for star in moviestarList:
                    if star in userPreference[user]['star']:
                        userPreference[user]['star'][star]['ratings'] += [ratingData[user][movie]]
                    else:
                        userPreference[user]['star'][star] = {}
                        userPreference[user]['star'][star]['ratings'] = [ratingData[user][movie]]

                for movieGenre in movieGenreList:
                    userPreference[user]['genre'][movieGenre]['ratings'] += [ratingData[user][movie]]

            else:
                userPreference[user] = {}
                userPreference[user]['genre']   = {}
                userPreference[user]['star']    = {}

                for star in moviestarList:
                    userPreference[user]['star'][star] = {}
                    userPreference[user]['star'][star]['ratings'] = [ratingData[user][movie]]

                for genre in genreList:
                    userPreference[user]['genre'][genre] = {} 
                    userPreference[user]['genre'][genre]['ratings'] = [] # List of ratings

                for movieGenre in movieGenreList:
                    userPreference[user]['genre'][movieGenre]['ratings'] += [ratingData[user][movie]]


        # Calculating star preferences of user
        ctr = 0.0
        meanNumStarRating = 0.0
        for star in list(userPreference[user]['star'].keys()):
            if len(userPreference[user]['star'][star]['ratings']) > 0:
                meanNumStarRating += len(np.array(userPreference[user]['star'][star]['ratings']))
                ctr += 1

        meanNumStarRating /= ctr;
        for star in list(userPreference[user]['star'].keys()):
            if len(userPreference[user]['star'][star]['ratings']) > 0:
                tempMeanRating  = np.mean(np.array(userPreference[user]['star'][star]['ratings']))
                tempNumRating   = len(userPreference[user]['star'][star]['ratings'])
                userPreference[user]['star'][star]['score']         = tempMeanRating*( 1 - math.exp(-1*tempNumRating/float(meanNumStarRating)));
                # print user,star,'star',userPreference[user]['star'][star]['score'],tempMeanRating,math.exp(-1*tempNumRating/float(meanNumStarRating)),tempNumRating,meanNumStarRating
            else:
                userPreference[user]['star'][star]['score']         = 0

            del userPreference[user]['star'][star]['ratings']

        sortedList = sorted(iter(userPreference[user]['star'].items()), key = lambda key_val:key_val[1]['score'] , reverse = True)
        ctr = 0
        # print user , len(sortedList)
        starTotalScore = 0.0
        for (key,val) in sortedList:
            ctr += 1
            if ctr > maxUserStar:
                del userPreference[user]['star'][key];
            else:
                # print key,val
                starTotalScore += userPreference[user]['star'][key]['score']
                pass

        for star in list(userPreference[user]['star'].keys()):
            userPreference[user]['star'][star]['score'] /= starTotalScore;
            # print user,'-->',userPreference[user]['star'][star]['score']

        # sortedList = sorted(userPreference[user]['star'].iteritems(), key = lambda (key,val):val['score'] , reverse = True)
        # print '-->',user , len(sortedList)

        # Calculating movie preference for user
        meanNumGenreRating = 0.0;
        ctr = 0;
        for movieGenre in genreList:
            if len(userPreference[user]['genre'][movieGenre]['ratings']) > 0:
                meanNumGenreRating  += len(userPreference[user]['genre'][movieGenre]['ratings']);
                ctr += 1;

        meanNumGenreRating /= ctr;

        genreTotalScore = 0.0
        for movieGenre in genreList:
            if len(userPreference[user]['genre'][movieGenre]['ratings']) > 0:
                tempMeanRating  = np.mean(np.array(userPreference[user]['genre'][movieGenre]['ratings']))
                tempNumRating   = len(userPreference[user]['genre'][movieGenre]['ratings']);
                userPreference[user]['genre'][movieGenre]['score']      = tempMeanRating*( 1 -  math.exp(-1*tempNumRating/float(meanNumGenreRating)));
                # print user,movieGenre,'genre',userPreference[user]['genre'][movieGenre]['score']
            else:
                userPreference[user]['genre'][movieGenre]['score']      = 0;

            genreTotalScore += userPreference[user]['genre'][movieGenre]['score'];
            del userPreference[user]['genre'][movieGenre]['ratings']

        for movieGenre in genreList:
            userPreference[user]['genre'][movieGenre]['score'] /= genreTotalScore;
            # print user,'-->',userPreference[user]['genre'][movieGenre]['score']

    f = open('modelDump/userContentPreference','wb')
    pickle.dump(userPreference, f)
    f.close();
    return userPreference'''
def new_contentBased(path,ratingData,maxUser,maxMovie,movieData):
    uMap,mMap = readUserMovieMapping(path); 
    surveyweight = 0.6;

    # ratingData,maxUser,maxMovie = readData(ratingFileName)
    genreList =["adventure","animation","children","comedy","fantasy","romance","drama","action","crime", "thriller","horror","mystery","sci.fi","imax","documentary","war","musical","western","film.noir"];
    f = open(path+'data/userlists.csv');
    reader = csv.reader(f);

    userAltPreference = {}
    sCtr = 0
    for line in reader:
        sCtr += 1;
        if sCtr == 1:
            continue;
        userId = int(line[0]); 
        userId = uMap[userId];      
    
        userAltPreference[userId] = {}
        userAltPreference[userId]['weights'] = {}
        userAltPreference[userId]['genre'] = {}

        weightNormaliser = 0.0;
        userAltPreference[userId]['weights']['genre']   = int(line[7]);
        weightNormaliser += int(line[7])
        userAltPreference[userId]['weights']['star']    = int(line[8]);
        weightNormaliser += int(line[8])
        userAltPreference[userId]['weights']['ratings'] = int(line[9])
        weightNormaliser += int(line[9])
        for key in list(userAltPreference[userId]['weights'].keys()):
            userAltPreference[userId]['weights'][key] = userAltPreference[userId]['weights'][key]/weightNormaliser;

        genreNormaliser = 0.0 
        # ACF
        userAltPreference[userId]['genre']["animation"] = int(line[11])
        genreNormaliser += int(line[11])
        userAltPreference[userId]['genre']["children"]  = int(line[11])
        genreNormaliser += int(line[11])
        userAltPreference[userId]['genre']["fantasy"]   = int(line[11])
        genreNormaliser += int(line[11])
        # CTM
        userAltPreference[userId]['genre']["crime"]     = int(line[12])
        genreNormaliser += int(line[12])
        userAltPreference[userId]['genre']["thriller"]  = int(line[12])
        genreNormaliser += int(line[12])
        userAltPreference[userId]['genre']["mystery"]   = int(line[12])
        genreNormaliser += int(line[12])
        # AAS
        userAltPreference[userId]['genre']["adventure"] = int(line[13])
        genreNormaliser += int(line[13])
        userAltPreference[userId]['genre']["action"]    = int(line[13])
        genreNormaliser += int(line[13])
        userAltPreference[userId]['genre']["sci.fi"]    = int(line[13])
        genreNormaliser += int(line[13])
        # H
        userAltPreference[userId]['genre']["horror"]    = int(line[14])
        genreNormaliser += int(line[14])
        # RCD
        userAltPreference[userId]['genre']["romance"]   = int(line[15])
        genreNormaliser += int(line[15])
        userAltPreference[userId]['genre']["comedy"]    = int(line[15])
        genreNormaliser += int(line[15])
        userAltPreference[userId]['genre']["drama"]     = int(line[15])
        genreNormaliser += int(line[15])
        # WWMF
        userAltPreference[userId]['genre']["war"]       = int(line[16])
        genreNormaliser += int(line[16])
        userAltPreference[userId]['genre']["musical"]   = int(line[16])
        genreNormaliser += int(line[16])
        userAltPreference[userId]['genre']["western"]   = int(line[16])
        genreNormaliser += int(line[16])
        userAltPreference[userId]['genre']["film.noir"] = int(line[16])
        genreNormaliser += int(line[16])

        userAltPreference[userId]['genre']["imax"]      = 3;
        genreNormaliser += 3
        userAltPreference[userId]['genre']["documentary"]   = 2;
        genreNormaliser += 2

        for key in list(userAltPreference[userId]['genre'].keys()):
            userAltPreference[userId]['genre'][key] = userAltPreference[userId]['genre'][key] /float(genreNormaliser);

        #for key in list(userAltPreference[userId]['genre'].keys()):
            #print(userId,key,"{0:.3f}".format(userAltPreference[userId]['genre'][key]))
#           try:
#               print userId,key,"{0:.3f}".format(userPreference[userId]['genre'][key]['score'])
#           except:
#               print "-->",userId,key

        # print userId,"{0.:3f}".format(userAltPreference[userId]['genre']);
        # print userId,"{0.:3f}".format(userPreference[userId]['genre']);

#   f = open('modelDump/userContentPreference','wb')
#   pickle.dump(userPreference, f)
#   f.close();

    userPreference = {}
    maxUserStar = 20
         
    for user in list(userAltPreference.keys()):
        for movie in list(ratingData[user].keys()):
            movieGenreList = movieData[movie]['genre'];
            moviestarList  = movieData[movie]['star'];

            if user in userPreference:
                for star in moviestarList:
                    if star in userPreference[user]['star']:
                        userPreference[user]['star'][star]['ratings'] += [ratingData[user][movie]]
                    else:
                        userPreference[user]['star'][star] = {}
                        userPreference[user]['star'][star]['ratings'] = [ratingData[user][movie]]

                for movieGenre in movieGenreList:
                    userPreference[user]['genre'][movieGenre]['ratings'] += [ratingData[user][movie]]

            else:
                userPreference[user] = {}
                userPreference[user]['genre']   = {}
                userPreference[user]['star']    = {}

                for star in moviestarList:
                    userPreference[user]['star'][star] = {}
                    userPreference[user]['star'][star]['ratings'] = [ratingData[user][movie]]

                for genre in genreList:
                    userPreference[user]['genre'][genre] = {} 
                    userPreference[user]['genre'][genre]['ratings'] = [] # List of ratings

                for movieGenre in movieGenreList:
                    userPreference[user]['genre'][movieGenre]['ratings'] += [ratingData[user][movie]]


        # Calculating star preferences of user
        ctr = 0.0
        meanNumStarRating = 0.0
        for star in list(userPreference[user]['star'].keys()):
            if len(userPreference[user]['star'][star]['ratings']) > 0:
                meanNumStarRating += len(np.array(userPreference[user]['star'][star]['ratings']))
                ctr += 1

        meanNumStarRating /= ctr;
        for star in list(userPreference[user]['star'].keys()):
            if len(userPreference[user]['star'][star]['ratings']) > 0:
                tempMeanRating  = np.mean(np.array(userPreference[user]['star'][star]['ratings']))
                tempNumRating   = len(userPreference[user]['star'][star]['ratings'])
                userPreference[user]['star'][star]['score']         = tempMeanRating*( 1 - math.exp(-1*tempNumRating/float(meanNumStarRating)));
                # print user,star,'star',userPreference[user]['star'][star]['score'],tempMeanRating,math.exp(-1*tempNumRating/float(meanNumStarRating)),tempNumRating,meanNumStarRating
            else:
                userPreference[user]['star'][star]['score']         = 0

            del userPreference[user]['star'][star]['ratings']

        sortedList = sorted(iter(userPreference[user]['star'].items()), key = lambda key_val:key_val[1]['score'] , reverse = True)
        ctr = 0
        # print user , len(sortedList)
        starTotalScore = 0.0
        for (key,val) in sortedList:
            ctr += 1
            if ctr > maxUserStar:
                del userPreference[user]['star'][key];
            else:
                # print key,val
                starTotalScore += userPreference[user]['star'][key]['score']
                pass

        for star in list(userPreference[user]['star'].keys()):
            userPreference[user]['star'][star]['score'] /= starTotalScore;
            # print user,'-->',userPreference[user]['star'][star]['score']

        # sortedList = sorted(userPreference[user]['star'].iteritems(), key = lambda (key,val):val['score'] , reverse = True)
        # print '-->',user , len(sortedList)

        # Calculating movie preference for user
        meanNumGenreRating = 0.0;
        ctr = 0;
        for movieGenre in genreList:
            if len(userPreference[user]['genre'][movieGenre]['ratings']) > 0:
                meanNumGenreRating  += len(userPreference[user]['genre'][movieGenre]['ratings']);
                ctr += 1;

        meanNumGenreRating /= ctr;

        genreTotalScore = 0.0
        for movieGenre in genreList:
            if len(userPreference[user]['genre'][movieGenre]['ratings']) > 0:
                tempMeanRating  = np.mean(np.array(userPreference[user]['genre'][movieGenre]['ratings']))
                tempNumRating   = len(userPreference[user]['genre'][movieGenre]['ratings']);
                userPreference[user]['genre'][movieGenre]['score']      = tempMeanRating*( 1 -  math.exp(-1*tempNumRating/float(meanNumGenreRating)));
                # print user,movieGenre,'genre',userPreference[user]['genre'][movieGenre]['score']
            else:
                userPreference[user]['genre'][movieGenre]['score']      = 0;

            genreTotalScore += userPreference[user]['genre'][movieGenre]['score'];
            del userPreference[user]['genre'][movieGenre]['ratings']

        for movieGenre in genreList:
            userPreference[user]['genre'][movieGenre]['score'] /= genreTotalScore;
            # print user,'-->',userPreference[user]['genre'][movieGenre]['score']

        for user in list(userPreference.keys()):
            for genre in genreList:
                if userPreference[user]['genre'][genre]['score']==0:
                    userPreference[user]['genre'][genre]['score']=userAltPreference[user]['genre'][genre]
                else:
                    userPreference[user]['genre'][genre]['score']=surveyweight*userAltPreference[user]['genre'][genre]+(1-surveyweight)*userPreference[user]['genre'][genre]['score']
            userPreference[user]['weights']=userAltPreference[user]['weights']
    
    return userPreference


# Returns recommnedations based on user content preference from past rated movies as sorted list of (movie,score) tuple
# userPreference sent is not complete dictionary for all users but only for the user sent as argument
def getRecommendation_contentBased(user,userPreference,movieData,genreWeight,starWeight):
    
    print('Getting content based recommendations')
    movieScore = {}

    for movieId in list(movieData.keys()):
        movieGenreList  = movieData[movieId]['genre']
        movieStarList   = movieData[movieId]['star']
        movieScore[movieId] = 0;
        ctr = 0
        for movieGenre in movieGenreList:
            movieScore[movieId] += genreWeight*userPreference['genre'][movieGenre]['score'];
            ctr += 1;

        movieScore[movieId] = movieScore[movieId]/float(ctr);

        for star in movieStarList:
            if star in list(userPreference['star'].keys()):
                movieScore[movieId] += starWeight*userPreference['star'][star]['score'];

        # print movieScore[movieId]
    movieScoreList = sorted(iter(movieScore.items()), key = lambda key_val4:key_val4[1] , reverse=True);

    #for movieId,score in movieScoreList:
        #print("{0:.3f}".format(score),movieData[movieId]['name'],movieData[movieId]['genre'],movieData[movieId]['star'])

    return movieScoreList

# To visualise P(z|user)
def printLatent_given_user():

    meanRating,varRating,latent_given_user,userMeanRating,userVarRating = readFromDump(path);

    tempMaxUser,numLatent = latent_given_user.shape
    print(latent_given_user.shape);
    tempMaxUser -= 1

    for user in range(1,tempMaxUser+1):
        printStr = str(user).zfill(4) + '\t'
        X = []
        Y = []
        for latentVar in range(0,numLatent):
            X += [latentVar]
            Y += [latent_given_user[user][latentVar]]
            printStr += "{0:.3f}".format(latent_given_user[user][latentVar]) +'\t'

        plt.plot(X,Y);

        # f = input("Hi")
        print(printStr)
    plt.show()

# Returns user-user similarity based of P(z|u)
def calcUserSimilarity():
    meanRating,varRating,latent_given_user,userMeanRating,userVarRating = readFromDump(path);
    tempMaxUser,numLatent = latent_given_user.shape
    print(latent_given_user.shape);
    tempMaxUser =  tempMaxUser - 1;

    userSimilarity = np.empty([tempMaxUser+1,tempMaxUser+1]);
    for user1 in range(1,tempMaxUser+1):
        for user2 in range(1,tempMaxUser+1):
            diff = np.sum(abs(latent_given_user[user1] - latent_given_user[user2]));
            userSimilarity[user1][user2] = diff;

    return userSimilarity

# Returns combinedList of recommendations
def combineContent_Collaborative(user,contentWeight,collaborativeWeight,ratingData,maxUser,maxMovie,movieData):

    movieData                   = getMovieData(path,movieFileName)
    userPreference              = getContentBasedPreference(path,ratingData,maxUser,maxMovie,movieData);
    contentBasedRecommendation  = getRecommendation_contentBased(user,userPreference[user],movieData,0.5,0.1);
    print("Got Content Based Recommendations")
    meanRating,varRating,latent_given_user,userMeanRating,userVarRating = readFromDump(path);
    collaborativeRecommendation = getRecommendations_collaborative(path,user,meanRating,varRating,latent_given_user,userMeanRating[user],userVarRating[user],ratingData,maxUser,maxMovie,[]);#,numRecommendation = None):
    print("Got Recommendations based on collaborative filtering")
    
    contentBasedDict = {}
    collaborativeDict = {}
    combinedDict = {}
    for (movie,score) in contentBasedRecommendation:
        contentBasedDict[movie] = score

    for (movie,r,p,n,score) in collaborativeRecommendation:
        collaborativeDict[movie] = score

    print("Combining Recommendations")
    for movie in list(collaborativeDict.keys()):
        if movie in contentBasedDict:
            combinedDict[movie] = contentWeight*contentBasedDict[movie] + collaborativeWeight*collaborativeDict[movie]


    combinedList = sorted(iter(combinedDict.items()) ,key= lambda movie_score : movie_score[1] , reverse=True);

    ctr = 0;
    for (key,val) in combinedList:
        ctr += 1
        print(str(ctr).zfill(4)+ '\t' +"{0:.3f}".format(val)+ '\t'+ str(movieData[key]['name']), movieData[key]['genre'],movieData[key]['star'])

    return combinedList

def getMovieRatingProb(meanRating,varRating,userMeanRating,userVarRating,maxMovie):    
    s,numLatent     = meanRating.shape;
    ratingList  = np.arange(0.5, 5, 0.5)
    print(meanRating.shape)
    probData = {}
    for movie in range(1,maxMovie+1):
        probData[movie] = {}
        for ratingVal in ratingList:
            probData[movie][ratingVal] = {}
            tempRatingVal   = (ratingVal - userMeanRating)/(math.sqrt(userVarRating));
            for latentVar in range(0,numLatent):
                tempMean    = meanRating[movie][latentVar]
                tempVar     = varRating[movie][latentVar]
                if tempVar != 0:
                    probData[movie][ratingVal][latentVar] = calcGaussian(tempRatingVal,tempMean,tempVar);
                else:
                    probData[movie][ratingVal][latentVar] = 0;

    return probData


# def new_collaborative(path,userId,meanRating,varRating,latent_given_user,userMeanRating,userVarRating,ratingData,maxUser,maxMovie,numMovieRating,exclusionList = [],numRecommendation = None):    
def new_collaborative(userId,meanRating,latent_given_user,maxMovie,numMovieRating,probData,exclusionList = [],numRecommendation = None):    
    s,numLatent     = meanRating.shape;
    ratingList  = np.arange(0.5, 5, 0.5)
    movieRating = []
    printLog = False;

    print(meanRating.shape)
    for movie in range(1,maxMovie+1):
        bestRating      = 3;
        maxRatingProb   = -1;
        for ratingVal in ratingList:
            # tempRatingVal   = (ratingVal - userMeanRating)/(math.sqrt(userVarRating));
            tempRatingProb  = 0;
            for latentVar in range(0,numLatent):
                # tempMean    = meanRating[movie][latentVar]
                # tempVar     = varRating[movie][latentVar]
                try:
                    tempProb = probData[movie][ratingVal][latentVar];
                except:
                    print("Key error in probData")
                    tempProb = 0
                    
                tempRatingProb += latent_given_user[userId][latentVar]*tempProb 
                if math.isnan(tempProb):
                    print("Got a NAN!!\n\n")
                         
            if tempRatingProb > maxRatingProb :
                maxRatingProb   = tempRatingProb
                bestRating      = ratingVal
                
        numRating   = numMovieRating[movie];
        score       = calcScore(bestRating,maxRatingProb,numRating);
        movieRating += [(movie,bestRating,maxRatingProb,numRating,score)]

    filteredList = []
    for dataTuple in movieRating:
        if str(dataTuple[0]) not in exclusionList:
            filteredList += [dataTuple];

    sortedList = sorted(filteredList, key=lambda dataTuple:dataTuple[4] , reverse=True)   # sort by score

    if numRecommendation == None:
        print('Giving back ',len(sortedList) ,' recommendations')
        return sortedList
    else:
        if numRecommendation > len(sortedList):
            print('Giving back ',len(sortedList) ,' recommendations')
            return sortedList
        else:
            print('Giving back ',numRecommendation ,' recommendations')
            return sortedList[0:numRecommendation];

# Returns a dictionary key: userId ,genrePreference[userId][genre] = {'rating': ,'numRating':}
# Based on user's ratings

'''def new_contentBased(path,ratingData,maxUser,maxMovie,movieData):

    genreList =["adventure","animation","children","comedy","fantasy","romance","drama","action","crime", "thriller","horror","mystery","sci.fi","imax","documentary","war","musical","western","film.noir"];

    userPreference = {}
    maxUserStar = 20

    for user in list(ratingData.keys()):
        for movie in list(ratingData[user].keys()):
            try:
                movieGenreList = movieData[movie]['genre'];
                moviestarList  = movieData[movie]['star'];
            except:
                print("An error occured here")
                print(user)
                print(ratingData[user])
                print(movie)
                print(ratingData[user][movie])

            if user in userPreference:
                for star in moviestarList:
                    if star in userPreference[user]['star']:
                        userPreference[user]['star'][star]['ratings'] += [ratingData[user][movie]]
                    else:
                        userPreference[user]['star'][star] = {}
                        userPreference[user]['star'][star]['ratings'] = [ratingData[user][movie]]

                for movieGenre in movieGenreList:
                    userPreference[user]['genre'][movieGenre]['ratings'] += [ratingData[user][movie]]

            else:
                userPreference[user] = {}
                userPreference[user]['genre']   = {}
                userPreference[user]['star']    = {}

                for star in moviestarList:
                    userPreference[user]['star'][star] = {}
                    userPreference[user]['star'][star]['ratings'] = [ratingData[user][movie]]

                for genre in genreList:
                    userPreference[user]['genre'][genre] = {} 
                    userPreference[user]['genre'][genre]['ratings'] = [] # List of ratings

                for movieGenre in movieGenreList:
                    userPreference[user]['genre'][movieGenre]['ratings'] += [ratingData[user][movie]]


        # Calculating star preferences of user
        ctr = 0.0
        meanNumStarRating = 0.0
        for star in list(userPreference[user]['star'].keys()):
            if len(userPreference[user]['star'][star]['ratings']) > 0:
                meanNumStarRating += len(np.array(userPreference[user]['star'][star]['ratings']))
                ctr += 1

        meanNumStarRating /= ctr;
        for star in list(userPreference[user]['star'].keys()):
            if len(userPreference[user]['star'][star]['ratings']) > 0:
                tempMeanRating  = np.mean(np.array(userPreference[user]['star'][star]['ratings']))
                tempNumRating   = len(userPreference[user]['star'][star]['ratings'])
                userPreference[user]['star'][star]['score']         = tempMeanRating*( 1 - math.exp(-1*tempNumRating/float(meanNumStarRating)));
                # print user,star,'star',userPreference[user]['star'][star]['score'],tempMeanRating,math.exp(-1*tempNumRating/float(meanNumStarRating)),tempNumRating,meanNumStarRating
            else:
                userPreference[user]['star'][star]['score']         = 0

            del userPreference[user]['star'][star]['ratings']

        sortedList = sorted(iter(userPreference[user]['star'].items()), key = lambda key_val:key_val[1]['score'] , reverse = True)
        ctr = 0
        # print user , len(sortedList)
        starTotalScore = 0.0
        for (key,val) in sortedList:
            ctr += 1
            if ctr > maxUserStar:
                del userPreference[user]['star'][key];
            else:
                # print key,val
                starTotalScore += userPreference[user]['star'][key]['score']
                pass

        for star in list(userPreference[user]['star'].keys()):
            userPreference[user]['star'][star]['score'] /= starTotalScore;
            # print user,'-->',userPreference[user]['star'][star]['score']

        # sortedList = sorted(userPreference[user]['star'].iteritems(), key = lambda (key,val):val['score'] , reverse = True)
        # print '-->',user , len(sortedList)

        # Calculating movie preference for user
        meanNumGenreRating = 0.0;
        ctr = 0;
        for movieGenre in genreList:
            if len(userPreference[user]['genre'][movieGenre]['ratings']) > 0:
                meanNumGenreRating  += len(userPreference[user]['genre'][movieGenre]['ratings']);
                ctr += 1;

        meanNumGenreRating /= ctr;

        genreTotalScore = 0.0
        for movieGenre in genreList:
            if len(userPreference[user]['genre'][movieGenre]['ratings']) > 0:
                tempMeanRating  = np.mean(np.array(userPreference[user]['genre'][movieGenre]['ratings']))
                tempNumRating   = len(userPreference[user]['genre'][movieGenre]['ratings']);
                userPreference[user]['genre'][movieGenre]['score']      = tempMeanRating*( 1 -  math.exp(-1*tempNumRating/float(meanNumGenreRating)));
                # print user,movieGenre,'genre',userPreference[user]['genre'][movieGenre]['score']
            else:
                userPreference[user]['genre'][movieGenre]['score']      = 0;

            genreTotalScore += userPreference[user]['genre'][movieGenre]['score'];
            del userPreference[user]['genre'][movieGenre]['ratings']

        for movieGenre in genreList:
            userPreference[user]['genre'][movieGenre]['score'] /= genreTotalScore;
            # print user,'-->',userPreference[user]['genre'][movieGenre]['score']

    f = open('modelDump/userContentPreference','wb')
    pickle.dump(userPreference, f)
    f.close();
    return userPreference'''

def returnlist(user,contentWeight,collaborativeWeight,meanRating,varRating,latent_given_user,userMeanRating,userVarRating,userPreference,ratingData,maxUser,maxMovie,movieData,numMovieRating,probData):

    # path='C:/Python34/Lib/collabrec/'
    path = 'C:/Users/nisyadav/AppData/Local/Programs/Python/Python35-32/Lib/collabrec/'
    tstart = time.time();
    contentBasedRecommendation  = getRecommendation_contentBased(user,userPreference[user],movieData,0.5,0.5);
    tfinish = time.time();
    print('Time taken by content based::'+str(tfinish-tstart));
    tstart = time.time();
    collaborativeRecommendation = new_collaborative(user,meanRating,latent_given_user,maxMovie,numMovieRating,probData,[]);#,numRecommendation = None):
    tfinish = time.time();
    print('Time taken by collaborative based::'+str(tfinish-tstart));
    tstart = time.time();
    contentBasedDict = {}
    collaborativeDict = {}
    combinedDict = {}
    maxConScore = 0;
    for (movie,score) in contentBasedRecommendation:
        contentBasedDict[movie] = score
        if maxConScore < score:
            maxConScore = score

    maxColabScore = 0;
    for (movie,r,p,n,score) in collaborativeRecommendation:
        collaborativeDict[movie] = score
        if maxColabScore < score:
            maxColabScore = score;

    for movie in list(collaborativeDict.keys()):
        if movie in contentBasedDict:
            combinedDict[movie] = 5*(contentWeight*contentBasedDict[movie]/maxConScore + collaborativeWeight*collaborativeDict[movie]/maxColabScore)

    tfinish = time.time();
    print('Time taken by loop thingies::'+str(tfinish-tstart));
    tstart = time.time();
    combinedList = sorted(iter(combinedDict.items()) ,key= lambda movie_score : movie_score[1] , reverse=True);

    totalDict={}
    ctr = 0
    for (key,val) in combinedList:
        ctr += 1
        if ctr > 100:
            break;
        totalDict[ctr]={'name':str(movieData[key]['name']),'key':key, 'total_rating': val, 'content_based': 5*contentBasedDict[key]/maxConScore,'collaborative_based':5*collaborativeDict[key]/maxColabScore}
    tfinish = time.time();
    print('Time taken by sorting::'+str(tfinish-tstart));
    #for (key,val) in combinedList:
        #ctr += 1
        #print(str(ctr).zfill(4)+ '\t' +"{0:.3f}".format(val)+ '\t'+ str(movieData[key]['name']), movieData[key]['genre'],movieData[key]['star'])

    return combinedList,totalDict

# Returns updated P(z|user). Sentiment is assumed to be on a scale of 1-5. genreScore is rankedList of genres for each cluster
# alpha is measure of how strong the update has to be. Smaller value of alpha might be helpful when updating genres extracted from movies/stars.
# alpha must be from 0 to 1
def updateGenrePreference(user,genre,sentiment,alpha,latent_given_user,genreScore,userContentPreference):

    #print("did i get here? ",1)
    s,numLatent =  latent_given_user.shape;
    numGenre = 18;
    normaliseSum = 0.0
    X = []
    Y = []
    for latentVar in range(0,numLatent):
        X += [latentVar]
        Y += [latent_given_user[user][latentVar]]

    #print("did i get here? ",2)
    #p1, = plt.plot(X,Y);

    #print("User,Genre,Sentiment On Scale 1-5")
    #print(user,genre,sentiment)

    # Updating Collaborative filtering based model
    for latentVar in range(0,numLatent):
        genreRank = 0;
        for (iterGenre,data) in genreScore[latentVar]:
            if iterGenre == genre:
                break;
            genreRank += 1;
        factor = alpha*(sentiment - 2.5)*((numGenre - genreRank)/float(numGenre));
        #print(latentVar,genreRank,"{0:.3f}".format(factor))
        
        latent_given_user[user][latentVar] = latent_given_user[user][latentVar]*math.exp(factor);
        
        normaliseSum += latent_given_user[user][latentVar]

    #print("did i get here? ",2)
    latent_given_user[user] /= normaliseSum;

    #print("did i get here? ",3)
    
    X = []
    Y = []
    for latentVar in range(0,numLatent):
        X += [latentVar]
        Y += [latent_given_user[user][latentVar]]

    #p2, = plt.plot(X,Y);
    #plt.legend([p2, p1], ["Updated", "Original"])
    #plt.show()

    # Updating content based preference

    #print("did i get here? ",4)
    
    tempSum = 0.0;
    ctr = 0
    X = []
    Y = []
    tempList = [];
    for movieGenre in list(userContentPreference[user]['genre'].keys()):
        # print "--->",ctr,movieGenre,userContentPreference[user]['genre'][movieGenre]['score']
        X       += [ctr]
        Y       += [userContentPreference[user]['genre'][movieGenre]['score']]
        tempList += [(ctr,movieGenre)]
        ctr += 1

    #print("did i get here? ",5)
    #p1, = plt.plot(X,Y);
    factor = (alpha*(sentiment - 2.5))/float(1);
    #print('Before',userContentPreference[user]['genre'][genre]['score'])
    #print('Factor',factor)
    userContentPreference[user]['genre'][genre]['score'] *= math.exp(factor);
   
    #print('After',userContentPreference[user]['genre'][genre]['score'])

    #print("did i get here? ",6)

    tempSum = 0.0;
    ctr = 0;
    for movieGenre in list(userContentPreference[user]['genre'].keys()):
        tempSum += userContentPreference[user]['genre'][movieGenre]['score']
        ctr += 1

    #print("did i get here? ",7)
    
    X = []
    Y = []
    ctr = 0;
    for movieGenre in list(userContentPreference[user]['genre'].keys()):
        userContentPreference[user]['genre'][movieGenre]['score'] /= tempSum;
        X += [ctr]
        Y += [userContentPreference[user]['genre'][movieGenre]['score']]
        # print "~~",ctr,movieGenre,userContentPreference[user]['genre'][movieGenre]['score']
        ctr += 1

    # print 'After Normalisation',userContentPreference[user]['genre'][genre]['score']
    #p2, = plt.plot(X,Y);
    #plt.legend([p2, p1], ["Updated", "Original"])
    #print(tempList)

    #print("did i get here? ",8)
    
    tempSum = 0.0;
    for movieGenre in list(userContentPreference[user]['genre'].keys()):
        tempSum += userContentPreference[user]['genre'][movieGenre]['score']

    #print('Sum Finally',tempSum)
    #plt.show()

    #print("did i get here? ",9)

    return latent_given_user,userContentPreference

def testGenreUpdate():
    
    uMap,mMap = readUserMovieMapping(path);

    meanRating,varRating,latent_given_user,userMeanRating,userVarRating = readFromDump();
    f = open('modelDump/movieClusters','rb')
    genreScore = pickle.load(f);
    f.close();
    print('Read genreScore')
    f = open('modelDump/userContentPreference','rb')
    userContentPreference = pickle.load(f);
    f.close();
    # userContentPreference = getContentBasedPreference('data/ratings.csv','data/movies.csv')
    print('Read userContentPreference')

    updateGenrePreference(uMap[6041],'Romance',5,1,latent_given_user,genreScore,userContentPreference)

# movieRankList is a dictionary with clusterId being key and ranked list of movies being item
# genreScore is ranked list of genres for each cluster
def updateMoviePreference(user,movie,sentiment,latent_given_user,movieRankList,genreScore,movieFileName):
    movieData = getMovieData(movieFileName);
    print('User:',user)
    print('Movie:',movie,movieData[movie])

    movieGenreList = movieData[movie]['genre'];

    for genre in movieGenreList:
        latent_given_user = updateGenrePreference(user,genre,sentiment,0.5,latent_given_user,genreScore)

    for latentVar in list(movieRankList.keys()):
        movieList = [item[0] for item in movieRankList[latentVar]]
        rank = movieList.index(movie)
        total = len(movieList)
        factor = 1 + (sentiment - 3.0)*(total - rank)/(total*3.0);
        latent_given_user[user][latentVar] *= factor;
        tempSum += latent_given_user[user][latentVar]

    latent_given_user[user] /= tempSum;

    return latent_given_user;
    # user movieRankList to update genre Probabilities as well

def getStarGenre(star,starDataFileName):
    starData = getStarData(starDataFileName);
    if star in list(starData.keys()):
        return starData[star]['genre'];
    else:
        return []

# Updates genrePreference for genres related to the actor, also update Content preference for the user by reducing score of this actor 
# if the actor is present in the user's list
def updateStarPreference(user,star,sentiment,latent_given_user,genreScore,userContentPreference,starDataFileName):

    # Updating preference for genre that the star acts in
    starGenreList = getStarGenre(path,star,starDataFileName);
    for genre in starGenreList:
        latent_given_user = updateGenrePreference(user,genre,sentiment,0.1,latent_given_user,genreScore);

    # As of now , not using the movies that the star has acted in to update the model
    # for movie in starMovieList:
    #   latent_given_user = updateMoviePreference(user,movie,sentiment,latent_given_user,genreScore);

    # Updatinf Content Based Preference
    alpha = 1;
    factor = 1 + alpha*(sentiment - 3)/5.0;
    if star in userContentPreference[user]['star']:
        userContentPreference[user]['star'][star]['score'] *= factor;

        tempSum = 0.0;
        for star in list(userContentPreference[user]['star'].keys()):
            tempSum += userContentPreference[user]['star'][star]['score']
        
        for star in list(userContentPreference[user]['star'].keys()):
            userContentPreference[user]['star'][star]['score'] /= tempSum;

    return latent_given_user, userContentPreference;

def new_updateMoviePreference(user,movie,sentiment,latent_given_user,userPreference,movieRankList,genreScore,movieData):
    print('User:',user)
    print('Movie:',movie,movieData[movie])

    movieGenreList = movieData[movie]['genre'];

    #print("this place ",1)
    for genre in movieGenreList:
        latent_given_user,userPreference = updateGenrePreference(user,genre,sentiment,0.5,latent_given_user,genreScore,userPreference)
    #print("this place ",2)
    tempSum = 0.0;
    for latentVar in list(movieRankList.keys()):
        movieList = [item[0] for item in movieRankList[latentVar]]
        rank = movieList.index(movie)
        total = len(movieList)
        factor = (sentiment - 2.5)*(total - rank)/float(total);
        
        latent_given_user[user][latentVar] *= math.exp(factor);
        
        tempSum += latent_given_user[user][latentVar]

    latent_given_user[user] /= tempSum;
    #print("this place")
    return latent_given_user,userPreference
    # user movieRankList to update genre Probabilities as well

def new_getStarGenre(star,starData):
    if star in list(starData.keys()):
        return starData[star]['genreList'];
    else:
        return []

# Updates genrePreference for genres related to the actor, also update Content preference for the user by reducing score of this actor 
# if the actor is present in the user's list
def new_updateStarPreference(user,star,sentiment,latent_given_user,genreScore,userContentPreference,starData):

    # Updating preference for genre that the star acts in
    starGenreList = new_getStarGenre(star,starData);
    for genre in starGenreList:
        latent_given_user,userContentPreference = updateGenrePreference(user,genre,sentiment,0.1,latent_given_user,genreScore,userContentPreference);

    # As of now , not using the movies that the star has acted in to update the model
    # for movie in starMovieList:
    #   latent_given_user = updateMoviePreference(user,movie,sentiment,latent_given_user,genreScore);

    # Updatinf Content Based Preference
    alpha = 1;
    factor = alpha*(sentiment - 2.5)/1;
    if star in userContentPreference[user]['star']:
        userContentPreference[user]['star'][star]['score'] *= math.exp(factor);
       
        tempSum = 0.0;
        for star in list(userContentPreference[user]['star'].keys()):
            tempSum += userContentPreference[user]['star'][star]['score']
        
        for star in list(userContentPreference[user]['star'].keys()):
            userContentPreference[user]['star'][star]['score'] /= tempSum;

    return latent_given_user, userContentPreference;

def pddf2fb(keysent,latent_given_user,genreScore,userPreference,starData,movieRankList,movieData,alpha,uMap,mMap,skip):
    print("Length of mMap which should be 10085 is : ",len(mMap))
    keysent.index=range(keysent.shape[0])
    for i in range(skip,keysent.shape[0]):
        tag=str(keysent['tag'][i])
        uX=keysent['userid'][i]
        #print (uX)
        try:
            user=uMap[int(uX)]
        except:
            print("I made a booboo at this step : 'user=uMap[int(uX)]'. So I'm skipping it.")
            continue
        key=keysent['key'][i]
        sent=keysent['sent_score'][i]
        try:
            if tag=='m':
                try:
                    mX=mMap[int(key)]
                except:
                    print("Python is stupid")
                latent_given_user,userPreference=new_updateMoviePreference(user,mX,sent,latent_given_user,userPreference,movieRankList,genreScore,movieData);
            elif tag=='s':
                latent_given_user,userPreference=new_updateStarPreference(user,int(key),sent,latent_given_user,genreScore,userPreference,starData);
            elif tag=='g':
                latent_given_user,userPreference=updateGenrePreference(user,key,sent,alpha,latent_given_user,genreScore,userPreference)
        except Exception as e:
            print("I made a booboo at this step : 'mMap[int(key)]'. So I'm skipping it. Also, key is : ", key)
            print("error is: ", e)
            continue
    return latent_given_user,userPreference


#print('is this running?')
#path='C:/Python34/Lib/collabrec/'
#mineClusterType(path,'data/ratings.csv','data/movienames.csv')
#mineClusterMovies(path,'data/ratings.csv','data/movienames.csv')
#getStarData(path,'data/starcast.csv')
if __name__ == '__main__':
    
    # r1,maxUser,maxMovie = readData(path,'data/ratings.csv');
    
    # numLatentList = [16];
    # fileNameList = ['testFile20m.csv']
    # for param in numLatentList:
    #     for testFileName in fileNameList:
    #         ratingData,meanRating,varRating,latent_given_user,latent_given_all = run(path,param,'data/ratings.csv',testFileName);

    # meanRating,varRating,latent_given_user,userMeanRating,userVarRating = readFromDump();
    # uMap,mMap = readUserMovieMapping();
    # recommendations = getRecommendations_collaborative(uMap[6041],meanRating,varRating,latent_given_user,userMeanRating[uMap[6041]],userVarRating[uMap[6041]],'data/ratings.csv','data/movies.csv');

    # print "~~1"
    # calcMovieGivenLatent(meanRating,varRating,'data/ratings.csv');
    # print "~~2--Failed"
    mineClusterType('data/ratings.csv','data/movienames.csv')
    print ("~~3--Failed")

    # calcAvgGenreRating('data/ratings.csv','data/movies.csv')
    # print "~~3b--Failed"
    # getMovieByGenre("Children's",'data/ratings.csv','data/movies.csv')
    # print "~~4--Failed"
    # getGenreCombinations('data/movies.csv')
    # print "~~5--Failed"
    # testRecommendations();
    # print "~~6--Failed"
    # buildGenreCoOccurence('data/movies.csv')
    # print "~~7--Failed"
    # getUserGenrePreference('data/ratings.csv','data/movies.csv')
    # print "~~8--Failed"
    # getMovieByGenre("Children's",'data/ratings.csv','data/movies.csv')
    # print "~~9--Failed"
     #mineClusterMovies('data/ratings.csv','data/movies.csv')
     #print "~~10--Failed"
    # userPreference = getContentBasedPreference('data/ratings.csv','data/movies.csv',0.5,0.1)
    # print "~~11--Failed"
    # getRecommendation_contentBased(uMap[6042],userPreference[uMap[6042]],'data/movies.csv')
    # print "~~12--Failed"

    # combineContent_Collaborative(uMap[6042],1.5,0.6,'data/ratings.csv','data/movies.csv')
    # printLatent_given_user()
    # testGenreUpdate()
    
    print(" ~~13--Failed")
