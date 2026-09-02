import numpy as np
import torch as tch


class Metrics:
    """Provide common classification metrics for actual and predicted labels.

    The metrics are returned as percentages, except where a zero denominator
    causes the corresponding metric to return zero.
    """
    def __init__(self,y , y_hat):
        """Initialize metrics with actual and predicted labels.

        Args:
            y: The actual labels.
            y_hat: The predicted labels.

        Returns:
            None.
        """
        self.y = y
        self.y_hat = y_hat
    def accuracy(self):
        """Calculate the percentage of correctly predicted labels.

        Returns:
            The accuracy as a percentage.
        """
        score = 0
        for i in range(self.y.shape[0]):
            
            if self.y[i] == self.y_hat[i]:
                score += 1
        
        return (score / self.y.shape[0])*100

    def precision(self, positive_class=1):
        """Calculate the percentage of positive predictions that are correct.

        Returns:
            The precision as a percentage, or zero when no positive predictions
            are available.
        """
        true_positives = 0
        false_positives = 0
        for i in range(self.y.shape[0]):
            
            if self.y_hat[i] == positive_class:
                if self.y[i] == positive_class:
                    true_positives += 1
                else:
                    false_positives += 1
        if true_positives + false_positives == 0:
            return 0
        
        return (true_positives / (true_positives + false_positives)) * 100
        
    
    def recall(self, positive_class=1):
        """Calculate the percentage of actual positives that were identified.

        Returns:
            The recall as a percentage, or zero when no actual positives are
            available.
        """
        true_positives = 0
        false_negatives = 0
        for i in range(self.y.shape[0]):
            
            if self.y[i] == positive_class:
                if self.y_hat[i] == positive_class:
                    true_positives += 1
                else:
                    false_negatives += 1
            
        if true_positives + false_negatives == 0:
            return 0
            
        return (true_positives / (true_positives + false_negatives)) * 100
    
    def f1_score(self, positive_class=1):
        """Calculate the harmonic mean of precision and recall.

        Returns:
            The F1 score as a percentage, or zero when both component metrics
            are zero.
        """

        precision = self.precision(positive_class=positive_class)
        recall = self.recall(positive_class=positive_class)
        
        if precision + recall == 0:
            return 0
        
        return (2 * precision * recall) / (precision + recall)

    def confusion_matrix(self, positive_class=1):
        
        tp = fp = fn = tn = 0
        
        for i in range(self.y.shape[0]):
            
            if self.y[i] == positive_class:
                if self.y_hat[i] == positive_class:
                    tp+=1
                else:
                    fn+=1
            
            elif self.y[i] != positive_class:
                if self.y_hat[i] == positive_class:
                    fp+=1
                else:
                    tn+=1
                    
        print(f'\n Confusion Matrix: \n {tp} | {fp} \n {fn} | {tn}')
        return tp, fp, fn, tn
    
    def specificity(self, positive_class=1):
        _, fp, _, tn = self.confusion_matrix(positive_class=positive_class)
        return 0 if tn + fp == 0 else (tn / (tn + fp)) * 100
        
        
    def macro_precision(self):
        """Calculate the average precision across all classes.

        Returns:
            The macro precision as a percentage.
        """
        classes = np.unique(self.y)
        total_precision = 0
        
        for cls in classes:
            total_precision += self.precision(positive_class=cls)
        
        return total_precision / len(classes)
    
    def macro_recall(self):
        """Calculate the average recall across all classes.

        Returns:
            The macro recall as a percentage.
        """
        classes = np.unique(self.y)
        total_recall = 0
        
        for cls in classes:
            total_recall += self.recall(positive_class=cls)
        
        return total_recall / len(classes)
    
    def macro_f1_score(self):
        """Calculate the average F1 score across all classes.

        Returns:
            The macro F1 score as a percentage.
        """
        classes = np.unique(self.y)
        total_f1 = 0
        
        for cls in classes:
            total_f1 += self.f1_score(positive_class=cls)
        
        return total_f1 / len(classes)

    def all_confusion_matrices(self):
        """Calculate confusion matrices for all classes.

        Returns:
            A dictionary of confusion matrices for each class.
        """
        classes = np.unique(self.y)
        result = {}
        
        for cls in classes:
            result[cls] = self.confusion_matrix(positive_class=cls)
        
        return result
    
    def balanced_accuracy(self):
        return self.macro_recall()
    
    