from abc import ABC
from .utils import get_model_tokenizer, get_trainer, build_inference_input_texts
from .engines import ValidationEngine, EvaluationEngine, InferenceEngine, ParsingEngine, DumpingEngine, LoggingEngine

import numpy as np
import pandas as pd

class Pipeline(ABC):
    def run(self, **kwargs):
        pass

class PreprocessingBuilder:
    def start_building(self, data):
        self.data = data
        return self
    
    def convert_category_cols(self):
        category_cols = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']
        for col in category_cols:
            self.data[col] = self.data[col].astype('category')
        return self
    
    def convert_label_to_fraud_flag(self):
        self.data['flag'] = np.where(
            self.data['fraud_bool'] == 0, 'legit',
            np.where(
                self.data['fraud_bool'] == 1, 'fraud',
                f'unknown - fraud_bool must be either 0 or 1 but {self.data["fraud_bool"]} found'
            )
        )
        return self
    
    def select_features(
        self,
        features_to_select = [
            'payment_type', 'employment_status', 'housing_status', 'source', 'device_os',
            'income', 'name_email_similarity', 'prev_address_months_count',
            'current_address_months_count', 'customer_age', 'intended_balcon_amount',
            'zip_count_4w', 'velocity_6h', 'velocity_24h', 'velocity_4w',
            'bank_branch_count_8w', 'date_of_birth_distinct_emails_4w',
            'credit_risk_score', 'email_is_free', 'phone_home_valid', 'phone_mobile_valid',
            'bank_months_count', 'has_other_cards', 'proposed_credit_limit', 'foreign_request',
            'session_length_in_minutes', 'keep_alive_session', 'device_distinct_emails_8w'
        ]
    ):
        cols = ['month', 'flag']
        cols.extend(features_to_select)
        self.data = self.data[cols]
        return self
    

    def train_test_split_bal(
        self, 
        sample_size_cap: int, 
        maj_cls_ratio: float, 
        max_train_month: int = 6,
    ):
        train = self.data[self.data.month < max_train_month].drop(columns=['month'])
        test = self.data[self.data.month >= max_train_month].drop(columns=['month'])

        train_fraud, train_legit = \
            train[train.flag == 'fraud'],\
            train[train.flag == 'legit']
        
        sample_size = \
            sample_size_cap\
                if sample_size_cap and sample_size_cap < 2 * train_fraud.shape[0]\
                else 2 * train_fraud.shape[0]
        
        train_fraud_samples, train_legit_samples =\
            train_fraud.sample(int(sample_size * (1-maj_cls_ratio))).index,\
            train_legit.sample(int(sample_size * maj_cls_ratio)).index
        train_samples = train_fraud_samples.union(train_legit_samples)
        train = train.loc[train_samples]

        labels = ['flag']
        features = [col for col in train.columns if (col not in labels)]
        self.X_train, self.y_train = train[features], train[labels]
        self.X_test, self.y_test = test[features], test[labels]
        return self

    def test_vl_split(
        self,
        test_vl_ratio: float = 0.5,
    ):
        self.X_vl = self.X_test.sample(frac=test_vl_ratio)
        sample_index = self.X_vl.index
        self.X_test = self.X_test.drop(sample_index)
        self.y_vl = self.y_test.loc[sample_index]
        self.y_test = self.y_test.drop(sample_index)
        return self

    def vl_undersample(
        self,
        sample_size: int = 100, 
        maj_cls_ratio: float = 0.5, 
    ):
        y_vl_legit = self.y_vl[self.y_vl.flag == 'legit']
        y_vl_fraud = self.y_vl[self.y_vl.flag == 'fraud']   
        legit_samples, fraud_samples = \
            y_vl_legit.sample(int(sample_size * maj_cls_ratio)).index, \
            y_vl_fraud.sample(int(sample_size * (1 - maj_cls_ratio))).index
        samples = legit_samples.union(fraud_samples)
        self.X_vl, self.y_vl = self.X_vl.loc[samples], self.y_vl.loc[samples]
        return self
        
        

class PreprocessingPipeline(Pipeline):
    def __init__(self, builder):
        self.builder = builder

    def run(
        self, 
        data,
        train_sample_size_cap: int = None,
        train_maj_cls_ratio: float = 0.5,
        test_vl_ratio: float = 0.5,
        vl_sample_size: int = 100,
        vl_maj_cls_ratio: float = 0.5,
    ):
        builder = (
            self.builder.start_building(data=data)
            .convert_category_cols()
            .convert_label_to_fraud_flag()
            .select_features()
            .train_test_split_bal(
                sample_size_cap=train_sample_size_cap, 
                maj_cls_ratio=train_maj_cls_ratio
            )
            .test_vl_split(test_vl_ratio=test_vl_ratio)
            .vl_undersample(
                sample_size=vl_sample_size,
                maj_cls_ratio=vl_maj_cls_ratio
            )
        )
        return \
            builder.X_train, builder.y_train, \
            builder.X_test, builder.y_test, \
            builder.X_vl, builder.y_vl 

class TrainingPipeline(Pipeline):  
    def __init__(
        self,
        validator: ValidationEngine,
        evaluator: EvaluationEngine,
        inference: InferenceEngine,
        parser: ParsingEngine,
        dumper: DumpingEngine,
        logger: LoggingEngine
    ):
        self.validator = validator
        self.evaluator = evaluator
        self.inference = inference
        self.parser = parser
        self.dumper = dumper
        self.logger = logger

    def run(
        self,
        X_train, y_train, 
        X_vl, y_vl,
        base_model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        cycles: int = 30,
        num_inferences: int = 5
    ):     
        model, tokenizer = get_model_tokenizer(model_name=base_model_name)
        trainer = get_trainer(model=model, tokenizer=tokenizer, X=X_train, y=y_train)
        texts = build_inference_input_texts(X=X_vl)
        texts_df = pd.DataFrame({'appid': X_vl.index.to_list(), 'text': texts})
        self.dumper.dump_validation_input_texts(texts_df, model_name='bob', version='v1')
        
        should_continue_training = True
        all_predictions_dfs = []
        for i in range(cycles):
            self.logger.notify_cycle_start(i)
            
            if not should_continue_training:
                break

            trainer.train()
            predictions_di = {'appid': X_vl.index.to_list()}
            raw_generated_texts_di = {'appid': X_vl.index.to_list()}
            for j in range(num_inferences):
                raw_generated_texts_di[f'r{j}'] = self.inference.run(texts=texts, model=model, tokenizer=tokenizer)
                predictions_di[f'r{j}'] = self.parser.parse_predictions(raw_generated_texts=raw_generated_texts_di[f'r{j}'])
            predictions_df = (
                pd.DataFrame(predictions_di)
                .melt(id_vars='appid', var_name='run', value_name='y_pred')
                .merge(y_vl.reset_index().rename(columns={'index': 'appid'}), how='left', on='appid')
                .merge(X_vl.reset_index().rename(columns={'index': 'appid'}), how='left', on='appid')
            )
            raw_generated_texts_df = (
                pd.DataFrame(raw_generated_texts_di)
                .melt(id_vars='appid', var_name='run', value_name='text')
            )
            self.dumper.dump_raw_generated_texts(raw_generated_texts_df=raw_generated_texts_df, model_name='bob', version='v1', cycle_id=i)
            
            report = self.evaluator.get_performance_report(predictions_df=predictions_df, y=y_vl)
            should_continue_training = self.validator.get_assessment(report=report)
            
            self.dumper.dump_predictions(predictions_df=predictions_df, model_name='bob', version='v1', cycle_id=i)
            if i % 5 == 0:
                self.dumper.dump_model(model=model, model_name='bob', version='v1', step=f'm{i}')
            all_predictions_dfs.append(predictions_df)

            self.logger.notify_cycle_end(i)
        return all_predictions_dfs
        
            