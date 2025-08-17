#TODO: implement logging for each validation step

import json
from .inference import InferenceEngine, InferenceHelper
from .evaluation import EvaluationEngine

class ValidationEngine:
    def __init__(self, X_validation, y_validation, semantics):
        self.X_validation = X_validation
        self.y_validation = y_validation
        self.semantics = semantics

    def sample_balanced(self, sample_size=20, majority_class_ratio=0.5):
        y_validation_legit = self.y_validation[self.y_validation.flag == 'legit']
        y_validation_fraud = self.y_validation[self.y_validation.flag == 'fraud']

        legit_samples, fraud_samples = y_validation_legit.sample(int(sample_size * majority_class_ratio)).index, y_validation_fraud.sample(int(sample_size * (1 - majority_class_ratio))).index
        samples = legit_samples.union(fraud_samples)
        return self.X_validation.loc[samples], self.y_validation.loc[samples]
    
    def get_inference_entries(self, X):
        entries = []
        inference_helper = InferenceHelper()
        for _, X_i in X.iterrows():
            account_opening_application_data = json.loads(X_i.to_json())
            entry = inference_helper.get_application_entry(input=account_opening_application_data, input_description=self.semantics, output=f'')
            entries.append(entry)
        return entries
    
    # def is_valid_format(self, conformity, min_conformity):
    #     return 1 if 1 - format_eval < epsilon else 0
    
    # def is_consistent_across_runs(self, consistency, epsilon=1e-6):
    #     return 1 if 1 - consistency < epsilon else 0
    
    # def has_acceptable_fpr(self, fpr, epsilon=1e-6):
    #     return 1 if abs(0 - fpr) < epsilon else 0
    
    # def has_acceptable_recall(self, recall, epsilon=1e-6):
    #     return 1 if 1 - recall < epsilon else 0


    def get_performance_metrics(
        self, 
        model, 
        tokenizer, 
        min_conformity_score = 0.99,
        min_consistency_score = 0.2,
        runs=1,
        sample_size=20,
        majority_class_ratio=0.5,
    ):
        performance_metrics = {
            'conformity': -1,
            'consistency': -1,
            'fpr': -1,
            'recall': -1
        }

        inference_engine = InferenceEngine(model=model, tokenizer=tokenizer)
        evaluation_engine = EvaluationEngine()

        X_bal, y_bal = self.sample_balanced(sample_size=sample_size, majority_class_ratio=majority_class_ratio)
        entries = self.get_inference_entries(X=X_bal)
        raw_outputs_di = inference_engine.run_ntimes(n=runs, entries=entries)

        conformity_scores_di = evaluation_engine.get_conformity_scores_di(raw_outputs_di=raw_outputs_di)
        performance_metrics['conformity'] = evaluation_engine.evaluate_conformity(conformity_scores_di=conformity_scores_di)
        if performance_metrics['conformity'] < min_conformity_score:
            print('Validation failed due to a low `OUTPUT CONFORMITY` evaluation.')
            print(f'Conformity = {performance_metrics["conformity"]}')
            print(f'conformity_di is = {conformity_scores_di}')
            print(f'Raw Outpts DI is = {raw_outputs_di}')
            return performance_metrics, raw_outputs_di

        y_preds_di = evaluation_engine.clean_predictions(raw_outputs_di=raw_outputs_di, conformity_scores_di=conformity_scores_di)

        pred_freq_di = evaluation_engine.get_pred_freq_di(y_preds_di=y_preds_di)
        pred_consistency_scores_di = evaluation_engine.get_pred_consistency_scores_di(pred_freq_di=pred_freq_di)
        consistency_scores_di = evaluation_engine.get_consistency_scores_di(pred_consistency_scores_di=pred_consistency_scores_di)
        performance_metrics['consistency'] = evaluation_engine.evaluate_consistency(consistency_scores_di=consistency_scores_di)
        if performance_metrics['consistency'] < min_consistency_score:
            print('Validation failed due to a low `OUTPUT CONSISTENCY` evaluation.')
            print(f'Raw Outpts DI is = {raw_outputs_di}')
            print(f'Consistency = {performance_metrics["consistency"]}')
            print(f'Consistency Scores = {consistency_scores_di}')
            print(f'y_preds_di = {y_preds_di}')
            return performance_metrics, y_preds_di

        confusion_matrix_di = evaluation_engine.get_confusion_matrix_di(y_preds_di=y_preds_di, y_expecteds=y_bal.flag.to_list())
        fpr_di = evaluation_engine.get_fpr_di(confusion_matrix_di=confusion_matrix_di)
        performance_metrics['fpr'] = evaluation_engine.evaluate_fpr(fpr_di=fpr_di)
        recall_di = evaluation_engine.get_recall_di(confusion_matrix_di=confusion_matrix_di)
        performance_metrics['recall'] = evaluation_engine.evaluate_recall(recall_di=recall_di)
        return performance_metrics, y_preds_di
