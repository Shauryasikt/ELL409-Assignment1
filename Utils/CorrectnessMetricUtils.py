import numpy as np
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
	return true_positive / float(denominator)

# Calculate recall
def recall_calc(obs_label, actual, predicted):
	true_positive = 0
	denominator = 0
	for i in range(len(actual)):
		if actual[i] == obs_label:
			denominator += 1
			if predicted[i] == obs_label:
				true_positive += 1
	return true_positive / float(denominator)

# Calculate F1
def f1_calc(obs_label, actual, predicted):
	precision = precision_calc(obs_label, actual, predicted)
	recall = recall_calc(obs_label, actual, predicted)
	return (2*precision*recall)/(precision+recall)

# Calculate specificity
def specificity_calc(obs_label, actual, predicted):
	true_negative = 0
	denominator = 0
	for i in range(len(actual)):
		if actual[i] != obs_label:
			denominator += 1
			if predicted[i] == actual[i]:
				true_negative += 1
	return true_negative / float(denominator)

# ROC curve - to be corrected
def roc_auc(obs_label, actual, predicted):
	tpr = recall_calc(obs_label, actual, predicted)
	fpr = 1 - specificity_calc(obs_label, actual, predicted)
	plt.plot(fpr, tpr)
	plt.show()
	auc = np.trapz(tpr, fpr)