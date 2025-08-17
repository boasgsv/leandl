import ast
import numpy as np

class EvaluationEngine:
    def get_conformity_scores_di(self, raw_outputs_di):
        def evaluate_single_prediction_format(y_pred_raw):
            initial_char = y_pred_raw[0]
            if initial_char != '{':
                return 0
        
            closed_bracket_idx = y_pred_raw.find('}')
            if closed_bracket_idx == -1:
                return 0
            
            y_pred_entry = y_pred_raw[0 : closed_bracket_idx + 1]
            if ':' not in y_pred_entry:
                return 0

            y_pred_entry_di = ast.literal_eval(y_pred_entry)
            entry_keys = y_pred_entry_di.keys()
            if len(entry_keys) != 1 or 'flag' not in entry_keys:
                return 0
            
            value = y_pred_entry_di['flag']
            if value not in ['legit', 'fraud']:
                return 0

            return 1

        scores = {}
        for page, y_preds_raw in raw_outputs_di.items():
            scores[page] = []
            for y_pred_raw in y_preds_raw:
                score = evaluate_single_prediction_format(y_pred_raw)
                scores[page].append(score)
        return scores

    def get_pred_freq_di(self, y_preds_di):
        runs = list(y_preds_di.keys())
        num_preds = len(y_preds_di[runs[0]])
        pred_freq_di = {'fraud': [0] * num_preds, 'legit': [0] * num_preds}

        for run in runs:
            for i in range(num_preds):
                y_pred = y_preds_di[run][i]
                if y_pred in pred_freq_di:
                    pred_freq_di[y_pred][i] += 1 / len(runs)
                
        return pred_freq_di


    def evaluate_conformity(self, conformity_scores_di):
        mean_scores = []
        for _, score_vec in conformity_scores_di.items():
            mean_scores.append(np.mean(score_vec))
        return np.min(mean_scores)

    def get_pred_consistency_scores_di(self, pred_freq_di):
        scores = {}
        for label, frequencies in pred_freq_di.items():
            scores[label] = []
            for freq in frequencies:
                distance = min(abs(freq - 0), abs(freq - 1))
                score = 1 - distance / 0.5
                scores[label].append(score)
        return scores

    def get_consistency_scores_di(self, pred_consistency_scores_di):
        mean_scores = {}
        for label, distances in pred_consistency_scores_di.items():
            mean_scores[label] = np.mean(distances)
        return mean_scores
    
    def evaluate_consistency(self, consistency_scores_di):
        return min(list(consistency_scores_di.values()))


    def get_confusion_matrix_di(self, y_preds_di, y_expecteds, positive_label='fraud'):
        confusion_matrix_di = {}
        for run, y_preds in y_preds_di.items():
            num_preds = len(y_preds)
            confusion_matrix_di[run] = {'tp': 0, 'fn': 0, 'fp': 0, 'tn': 0}

            for i in range(num_preds):
                y_expected = y_expecteds[i]
                y_pred = y_preds[i]
                if y_expected == positive_label:
                    if y_pred == positive_label:
                        confusion_matrix_di[run]['tp'] += 1
                    else:
                        confusion_matrix_di[run]['fn'] += 1
                else:
                    if y_pred == positive_label:
                        confusion_matrix_di[run]['fp'] += 1
                    else:
                        confusion_matrix_di[run]['tn'] += 1
        return confusion_matrix_di

    def get_recall_di(self, confusion_matrix_di):
        scores = {}
        for run, confusion_matrix in confusion_matrix_di.items():
            all_pos = (confusion_matrix['tp'] + confusion_matrix['fn'])
            scores[run] = 0 if all_pos == 0 else confusion_matrix['tp'] / all_pos
        return scores

    def evaluate_recall(self, recall_di):
        return np.mean(list(recall_di.values()))


    def get_fpr_di(self, confusion_matrix_di):
        scores = {}
        for run, confusion_matrix in confusion_matrix_di.items():
            all_neg = (confusion_matrix['fp'] + confusion_matrix['tn'])
            scores[run] = 0 if all_neg == 0 else confusion_matrix['fp'] / all_neg
        return scores
    
    def evaluate_fpr(self, fpr_di):
        return np.mean(list(fpr_di.values()))

    def clean_predictions(self, raw_outputs_di, conformity_scores_di):
        y_preds_di = {}
        for page, y_preds_raw in raw_outputs_di.items():
            y_preds_di[page] = []
            num_preds = len(y_preds_raw)
            for i in range(num_preds):
                y_pred_raw = y_preds_raw[i]
                if conformity_scores_di[page][i] == 1:
                    closed_bracket_idx = y_pred_raw.find('}')
                    y_pred_entry = y_pred_raw[0 : closed_bracket_idx + 1]
                    y_pred = ast.literal_eval(y_pred_entry)['flag']
                else:
                    y_pred = 'error'
                y_preds_di[page].append(y_pred)
        return y_preds_di