from sklearn import metrics
import numpy as np


def get_accuracy(y, y_pre):
	sambles = len(y)
	count = 0.0
	for i in range(sambles):
		y_true = 0
		all_y = 0
		for j in range(len(y[i])):
			if y[i][j] > 0 and y_pre[i][j] > 0:
				y_true += 1
			if y[i][j] > 0 or y_pre[i][j] > 0:
				all_y += 1
		if all_y <= 0:
			all_y = 1
		count += float(y_true) / float(all_y)
	acc = float(count) / float(sambles)
	acc=round(acc,4)
	return acc

def exact_match_ratio(y, y_pred):
    """
    Calculate the Exact Match Ratio (Subset Accuracy) for multi-label classification.

    Parameters:
    y (list of list of int): The ground truth labels.
    y_pred (list of list of int): The predicted labels.

    Returns:
    float: The calculated exact match ratio.
    """
    samples = len(y)
    correct_samples = 0

    for i in range(samples):
        if np.array_equal(y[i], y_pred[i]):
            correct_samples += 1

    emr = float(correct_samples) / float(samples)
    return round(emr, 4)

def get_metrics(y, y_pre):
	y = y.cpu().detach().numpy()
	y_pre = y_pre.cpu().detach().numpy()
	y=np.array(y)
	y_pre=np.array(y_pre)
	acc = get_accuracy(y, y_pre)
	y = np.array(y)
	y_pre = np.array(y_pre)
	micro_f1 = metrics.f1_score(y, y_pre, average='micro')
	micro_precision = metrics.precision_score(y, y_pre, average='micro')
	micro_recall = metrics.recall_score(y, y_pre, average='micro')
	macro_f1 = metrics.f1_score(y,y_pre,average='macro')
	hamming_loss = metrics.hamming_loss(y,y_pre)
	emr = exact_match_ratio(y,y_pre)
	return micro_f1, micro_precision, micro_recall, acc,macro_f1,emr



