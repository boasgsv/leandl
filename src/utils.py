import os 
import pandas as pd
import json
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

def get_model_tokenizer(model_name: str, max_seq_length: int = 2048, dtype=None, load_in_4bit: bool = True):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    return model, tokenizer

def get_trainer(model, tokenizer, X, y, max_seq_length: int = 2048):
    training_set = build_training_set(X=X, y=y)
    trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = Dataset.from_pandas(training_set),
            dataset_text_field = "text",
            max_seq_length = 2048,
            packing = False, # Can make training 5x faster for short sequences.
            args = SFTConfig(
                per_device_train_batch_size = 2,
                gradient_accumulation_steps = 4,
                warmup_steps = 5,
                # num_train_epochs = 1, # Set this for 1 full training run.
                max_steps = 10,
                learning_rate = 2e-4,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                seed = 3407,
                output_dir = "outputs",
                report_to = "none", # Use this for WandB etc
            ),
        )
    return trainer

def build_training_set(X, y):
    semantics_df = DataLoader.load_semantics()
    semantics_di = dict(zip(semantics_df.column, semantics_df.description))
    entries = []
    for i in X.index:
        input = json.loads(X.loc[i].to_json())
        output = json.loads(y.loc[i].to_json())
        prompt = InstructionPrompt(input=input, input_description=semantics_di, output=output)
        entries.append(prompt.get_dict())
    df = pd.DataFrame(entries)
    return df

def build_inference_input_texts(X):
    semantics_df = DataLoader.load_semantics()
    semantics_di = dict(zip(semantics_df.column, semantics_df.description))
    texts = []
    for i in X.index:
        input = json.loads(X.loc[i].to_json())
        prompt = InstructionPrompt(input=input, input_description=semantics_di, output='')
        texts.append(prompt.get_text())
    return texts

class DataLoader:
    @staticmethod
    def load_semantics():
        return pd.read_parquet(os.path.join(Configs.get_project_root(), 'data', 'semantics.parquet'))
    
    def load_baf():
        return pd.read_csv(os.path.join(Configs.get_project_root(), 'data', 'Base.csv'))


class Configs:
    @staticmethod
    def get_project_root():
        return os.path.join('/', 'home', 'gsvboas', 'leandl')

class InstructionPrompt:
    def __init__(
        self,
        input,
        instruction="Your task is to analyze the following bank account opening application and classify it as either `fraud` or `legit`",
        input_description=None,
        output="",
    ):        
        self.instruction = instruction
        self.input_description = input_description
        self.input = input
        self.output = output

    def get_text(self):    
        # json_str = json.dumps({
        #     'instruction': self.instruction,
        #     'input_description': self.input_description,
        #     'input': self.input,
        #     'output': self.output
        # })
        return f'''
        You are an expert in fraud analysis for a bank account application department. 
        In our department we value a low amount of false positive fraud accusations, because they hinder people from their rights to a bank account.
        True frauds are rare, and we need your help in identifying frauds from the application data.
        The description for the application data is provided below: 
        {self.input_description}

        ### Instruction:
        {self.instruction}

        ### Input:
        {self.input}

        ### Output:
        {self.output}
        '''

    def get_dict(self):
        return {
            'instruction': self.instruction,
            'input': self.input,
            'output': self.output,
            'text': self.get_text()
        }