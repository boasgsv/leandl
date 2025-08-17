class InferenceHelper:
    def get_application_entry(
        self, 
        input, 
        input_description, 
        output='', 
        instruction='''Your task is to analyze the following bank account opening application and classify it as either `fraud` or `legit`''',
    ):
        return {
        'instruction': instruction,
        'input': input,
        'output': output,
        'text': \
            f"""
            You are an expert in fraud analysis for a bank account application department. 
            In our department we value a low amount of false positive fraud accusations, because they hinder people from their rights to a bank account.
            True frauds are rare, and we need your help in identifying frauds from the application data.
            The description for the application data is provided below: 
            {input_description}
            
            ### Instruction
            {instruction}:
            
            ### Input
            {input}
            
            ### Output:
            {output}
            """,
    } 

class InferenceEngine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = 'cuda'
        self.post_processer = InferencePostProcesser(tokenizer=tokenizer)

    def run_once(
        self, 
        entries,
        input_key='text',
        max_new_tokens=10,
    ):
        raw_outputs = []
        i = 0
        for entry in entries:
            print(f'Running inference for entry of index `{i}`')
            i += 1
            input = self.tokenizer([entry[input_key]], return_tensors='pt').to(self.device)
            output_tensors = self.model.generate(**input, max_new_tokens=max_new_tokens, use_cache=True)
            generated_text = self.post_processer.decode_generated_text_from_tensors(output_tensors=output_tensors)
            raw_output = self.post_processer.parse_raw_output(generated_text=generated_text)
            raw_outputs.append(raw_output)
            del input
        return raw_outputs
    
    def run_ntimes(
        self,
        n,
        entries,
        input_key='text',
        max_new_tokens=10
    ):
        runs = [f'r{i}' for i in range(n)]
        raw_outputs_di = {}
        for run in runs:
            print(f'Starting inference run `{run}`...')
            raw_outputs_di[run] = self.run_once(entries=entries, input_key=input_key, max_new_tokens=max_new_tokens)
        return raw_outputs_di
    
class InferencePostProcesser:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def decode_generated_text_from_tensors(self, output_tensors):
        return self.tokenizer.decode(output_tensors[0], skip_special_tokens=True)
    
    def parse_raw_output(self, generated_text):
        return generated_text.split('Output:')[1].strip()

    
            

