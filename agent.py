from openai import OpenAI
from typing import List
import os
import requests
import json
from datetime import datetime


class OriginalAgent:
    def __init__(self, model: str, temperature: float = 0.0, seed=None,  config=None ):
        self.model = model
        self.temperature = temperature

        if model.startswith('gpt'):
            self.client = OpenAI()
        
        self.thoughts: List[str] = []
        self.interactions: List[str] = []
        self.history = ""
        self.resulting_model = None
        self.seed = seed
        self.config = config
        
    def log_response(self, response, output_dir):
        """Append the GPT response as a single line JSON into the log file."""
        log_path = os.path.join(output_dir, 'log.jsonl')
        
        if hasattr(response, 'to_dict'):
            response_data = response.to_dict()
        else:
            response_data = dict(response)
        
        with open(log_path, 'a', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "response": response_data
            }, f, ensure_ascii=False)
            f.write('\n')



    def act(self, prompt: str, stop: List[str], max_new_tokens=1024, do_sample=False, temperature=0.0, top_k=1, top_p=1.0):
        if self.model.startswith('gpt'):
            res = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                stop=stop,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )

            self.log_response(res, self.config.output_dir)
            return res.choices[0].message.content
        
        else:    
            url = "http://0.0.0.0:5000/inference"
            prompt1 = [{'role': 'user', 'content': prompt}]
            data = {"prompt": prompt1, "stop_strings": stop, "max_new_tokens": max_new_tokens, 'do_sample': do_sample, 'temperature': None, 'top_k': None, 'top_p': None }

            response = requests.post(url, json=data)
            json_obj = response.json()
            if json_obj['error'] == True:
                print(json_obj['exception'])
                exit(1)
                
            # parse generated text based on model return
            if self.model =='gemma' or self.model == 'qwen':
                output = json_obj["output"][len(prompt):]
                if 'model' in output and self.model =='gemma':
                    output = 'model'.join(output.split('model')[1:]).strip()
                if 'assistant' in output and self.model =='qwen':
                    output = output[len('system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.\nuser\n'):]
                    output = output.split('assistant')[1].strip()
            else:
                output = response.json()["output"][0]['generated_text'][1]['content']
            return output
    
    
    def reset_model(self):
        if self.model.startswith('gpt'):
            return
        url = "http://0.0.0.0:5000/reset"
        data = {"seed": 42}
        response = requests.post(url, json=data)
        return response.json()['reset']

    def update(self, agent_output: str):
        actions = agent_output.split("\n")
        for action in actions:
            action = action.strip()
            if action.startswith("Thought:"):
                self.thoughts.append(action.split("Thought:")[1].strip())
            elif action.startswith("Action:") or action.startswith("Observation:"):
                self.interactions.append(action)
        self.history += agent_output

    def get_observations(self):
        observations = []
        for interaction in self.interactions:
            
            if interaction.startswith("Observation:") and 'Error' not in interaction:
                observations.append(interaction.split("Observation:")[1].strip())
        return observations


