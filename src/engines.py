import ast
import numpy as np

from .utils import Configs
import os

class ValidationEngine:
    def get_assessment(self, report):
        return 1

class PerformanceReport:
    def __init__(self, conformity, consistency, confusion_matrix):
        self.conformity = conformity
        self.consistency = consistency
        self.confusion_matrix = confusion_matrix
    def get_accuracy(self):
        pass
    def get_fpr(self):
        pass
    def get_recall(self):
        pass
    def get_precision(self):
        pass
    def get_f1_score(self):
        pass


class EvaluationEngine:
    def get_performance_report(self, predictions_df, y):
        return 1
    def calc_consistency(self, predictions_df):
        predictions_df.groupby('')    
    def calc_conformity(self, predictions_df):
        return np.mean(np.where(predictions_df.y_pred.str.contains('error'), 1, 0))

class InferenceEngine:
    def run(self, texts, model, tokenizer, device='cuda', max_new_tokens=10):
        raw_generated_texts = []
        i = 0
        for text in texts:
            print(f'Running Inference Engine for instance {i+1}...')
            i+=1
            in_tensors = tokenizer([text], return_tensors='pt').to(device)
            out_tensors = model.generate(**in_tensors, max_new_tokens=max_new_tokens, use_cache=True)
            raw_generated_text = tokenizer.decode(out_tensors[0], skip_special_tokens=True)
            raw_generated_texts.append(raw_generated_text)
            del in_tensors, out_tensors
        return raw_generated_texts

class ParsingEngine:
    def __init__(self):
        self.raw_prediction_error = '''error'''
        
    def parse_single_prediction(self, raw_prediction, label_colname='flag', allowed_label_values=['fraud', 'legit']):
        if raw_prediction[0] != '{':
            return self.raw_prediction_error
        raw_prediction_end_idx = raw_prediction.find('}')
        if raw_prediction_end_idx == -1:
            return self.raw_prediction_error
        prediction = raw_prediction[0 : raw_prediction_end_idx + 1]
        key_val_split = prediction.split(':')
        if len(key_val_split) != 2:
            return self.raw_prediction_error
        prediction_di = ast.literal_eval(prediction)
        if label_colname not in prediction_di:
            return self.raw_prediction_error
        if prediction_di[label_colname] not in allowed_label_values:
            return self.raw_prediction_error
        return prediction_di[label_colname]
    
    def parse_predictions(self, raw_generated_texts: list[str]):
        predictions = []
        for raw_generated_text in raw_generated_texts:
            try: 
                raw_prediction = raw_generated_text.split('Output:')[1].strip()
                prediction = self.parse_single_prediction(raw_prediction=raw_prediction)
            except:
                prediction = self.raw_prediction_error
            predictions.append(prediction)
        return predictions

class DumpingEngine:
    def dump_model(self, model, model_name: str, version: str, step: str):
        dir = os.path.join(Configs.get_project_root(), 'outputs', model_name, version, 'model')
        if not os.path.exists(dir):
            os.makedirs(dir)
        model.save_pretrained(os.path.join(dir, step))
    def dump_predictions(self, predictions_df, model_name: str, version: str, cycle_id: int):
        dir = os.path.join(Configs.get_project_root(), 'outputs', model_name, version, 'predictions')
        if not os.path.exists(dir):
            os.makedirs(dir)
        predictions_df.to_parquet(os.path.join(dir, f'cycle_{cycle_id}.parquet'))
    def dump_validation_input_texts(self, texts_df, model_name: str, version: str):
        dir = os.path.join(Configs.get_project_root(), 'outputs', model_name, version)
        if not os.path.exists(dir):
            os.makedirs(dir)
        texts_df.to_parquet(os.path.join(dir, f'validation_input_texts.parquet'))
    def dump_raw_generated_texts(self, raw_generated_texts_df, model_name: str, version: str, cycle_id: int):
        dir = os.path.join(Configs.get_project_root(), 'outputs', model_name, version, 'raw_generated_texts')
        if not os.path.exists(dir):
            os.makedirs(dir)
        raw_generated_texts_df.to_parquet(os.path.join(dir, f'cycle_{cycle_id}.parquet'))


class LoggingEngine:
    def notify_cycle_start(self, i: int):
        print(f'Starting cycle {i}...')
    
    def notify_cycle_end(self, i: int):
        print(f'Cycle {i} finished.')