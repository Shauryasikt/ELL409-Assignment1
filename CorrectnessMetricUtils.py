import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# Calculate accuracy
def accuracy_calc(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual))

# Calculate precision
def precision_calc(obs_label, actual, predicted):
	true_positive = 0
	denominator = 0
	for i in range(len(actual)):
		if predicted[i] == obs_label:
			denominator += 1
			if actual[i] == obs_label:
				true_positive += 1
	if denominator!=0:
		return true_positive / float(denominator)
	else:
		return 0

# Calculate recall
def recall_calc(obs_label, actual, predicted):
	true_positive = 0
	denominator = 0
	for i in range(len(actual)):
		if actual[i] == obs_label:
			denominator += 1
			if predicted[i] == obs_label:
				true_positive += 1
	if denominator!=0:
		return true_positive / float(denominator)
	else:
		return 0

# Calculate F1
def f1_calc(obs_label, actual, predicted):
	precision = precision_calc(obs_label, actual, predicted)
	recall = recall_calc(obs_label, actual, predicted)
	if (precision+recall)!=0:
		return (2*precision*recall)/(precision+recall)
	else:
		return 0

# Calculate specificity
def specificity_calc(obs_label, actual, predicted):
	true_negative = 0
	denominator = 0
	for i in range(len(actual)):
		if actual[i] != obs_label:
			denominator += 1
			if predicted[i] == actual[i]:
				true_negative += 1
	if denominator!=0:
		return true_negative / float(denominator)
	else:
		return 0

# ROC curve - to be corrected
def roc_auc(fpr, tpr):
	plt.plot(fpr, tpr)
	plt.show()
	auc = np.trapz(tpr, fpr)