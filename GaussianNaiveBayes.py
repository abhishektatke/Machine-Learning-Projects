import numpy as np
from numpy import genfromtxt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics

#Read, shuffle and pre-process data
print ("Reading spambase data set")
data_set = genfromtxt('spambase.data', delimiter=',') 
data_length = len(data_set)
tar_values = data_set[:, -1]
print ("Shuffling spambase data set")
np.random.shuffle(data_set)
attributes_values = preprocessing.scale(data_set[:, 0:-1])
tar_values = data_set[:, -1]
#split data into training and testing data_set(50%)
X_train, X_test, y_train, y_test = train_test_split(
    attributes_values, tar_values, test_size=0.5, random_state=17)

#total probabilities for the spam and non- spam should ne nearly 40% and 60 %
TotalSpam = 0
train_set_length = len(X_train)
for eachrow in range(train_set_length):
    if y_train[eachrow] == 1:
        TotalSpam += 1		
prob_spammails = float(TotalSpam) / train_set_length
prob_nonspammails = 1 - prob_spammails
print("Spam mail probability is: \t",prob_spammails)
print("Non-spam mail probability is: \t",prob_nonspammails)

#mean and Standard Deviation for all the attributes_values(features)
mean_spammails,standev_spammails,mean_nonspammails,standev_nonspammails  = [], [],[],[]
for attributes_values in range(0,57):
    spam_values,nonspam_values = [],[]
    for eachrow in range(0, train_set_length):
        if (y_train[eachrow] == 1):
            spam_values.append(X_train[eachrow][attributes_values])
        else :
           nonspam_values.append(X_train[eachrow][attributes_values])
    mean_spammails.append(np.mean(spam_values))
    mean_nonspammails.append(np.mean(nonspam_values))
    standev_spammails.append(np.std(spam_values))
    standev_nonspammails.append(np.std(nonspam_values))
#replacing 0 standard deviation with .0001
for feature in range(0,57):
    if(standev_spammails[feature]==0):
        standev_spammails[feature] = .0001
    if(standev_nonspammails[feature]==0):
        standev_nonspammails[feature]=.0001
		
#precision, Recall and accuracy calculation
def cal_accuracy_precision_recall(tar_values, predicted_values, threshold_value):
    true_pos,false_pos,true_neg,false_neg = 0,0,0,0
    for eachrow in range(len(predicted_values)):
        if (predicted_values[eachrow] > threshold_value and tar_values[eachrow] == 1)  :
            true_pos += 1
        elif (predicted_values[eachrow] > threshold_value and tar_values[eachrow] == 0 )  :
            false_pos += 1
        elif (predicted_values[eachrow] <= threshold_value and tar_values[eachrow] == 1 )  :
            false_neg += 1
        elif (predicted_values[eachrow] <= threshold_value and tar_values[eachrow] == 0 )  :
            true_neg += 1
    accuracy = float(true_pos + true_neg) / len(predicted_values)
    recall = float(true_pos) / (true_pos + false_neg)
    precision = float(true_pos) / (true_pos + false_pos)
    return  accuracy, recall, precision
	
#probability calculation and predicting classes using the Gaussian Naive Bayes algorithm.
probability_spam,probability_non_spam = 0,0
pred = []
for eachrows in range(0,len(X_test)):
    NB_final_spam_prob,NB_final_nonspam_prob = [],[]
    NB_part_cal_1,NB_part_cal_2,NB_part_cal_3,NB_part_cal_4 = 0,0,0,0
    for attributes_values in range(0,57):
        NB_part_cal_1 = float(1)/ (np.sqrt(2 * np.pi) * standev_spammails[attributes_values])
        NB_part_cal_2 = (np.e) ** - (((X_test[eachrows][attributes_values] - mean_spammails[attributes_values]) ** 2) / (2 * standev_spammails[attributes_values] ** 2))
        NB_final_spam_prob.append(NB_part_cal_1 * NB_part_cal_2)
        NB_part_cal_3 = float(1)/ (np.sqrt(2 * np.pi) * standev_nonspammails[attributes_values])
        NB_part_cal_4 = (np.e) ** - (((X_test[eachrows][attributes_values] - mean_nonspammails[attributes_values]) ** 2) / (2 * standev_nonspammails[attributes_values] ** 2))
        NB_final_nonspam_prob.append(NB_part_cal_3 * NB_part_cal_4)
		#in case of 0 prob replace it with minimal value
        #if(NB_final_spam_prob[attributes_values]==0):
            #NB_final_spam_prob[attributes_values]=10**-10
        #if(NB_final_nonspam_prob[attributes_values]==0):
            #NB_final_nonspam_prob[attributes_values]=10**-10
    probability_spam = np.log(prob_spammails) + np.sum(np.log(np.asarray(NB_final_spam_prob)))
    probability_non_spam = np.log(prob_nonspammails) + np.sum(np.log(np.asarray(NB_final_nonspam_prob)))
    output = np.argmax([probability_non_spam, probability_spam])
    pred.append(output)
acc,rec,pre = cal_accuracy_precision_recall(y_test, pred, 0)
print("Confusion matrix:\n",metrics.confusion_matrix(y_test, pred))
print ("Accuracy Value: \t",acc)
print ("Precision Value: \t", pre)
print ("Recall Value: \t",rec)