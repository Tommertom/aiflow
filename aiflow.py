# aiflow.py
import json
from openai import OpenAI
from IPython.display import HTML

# chrome helper that converts a query result to a string, so we can use it in the class
def chroma_query_result_to_text(obj):
    documents = obj.get('documents')
    if documents:
        concatenated_string = ''.join(['\n'.join(doc) for doc in documents])
        return concatenated_string
    else:
        return ''
    

# models - gpt-4, gpt-4o, gpt-3.5-turbo
class AIFlow:
    def __init__(self, api_key, model='gpt-4', temperature=0,max_tokens=150):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        self.chat_messages = []
        self.context_map = {}
        self.images_map = {}
        self.audio_map = {}

    def set_temperature(self, temperature=0):
        self.temperature=temperature
        return self
            
    def set_model(self, model='gpt-4'):
        self.model=model
        return self
    
    def set_max_tokens(self, max_tokens=150):
        self.max_tokens=max_tokens
        return self
    
    def show_model_config(self):
        print(f"Model: {self.model}")
        print(f"Max Tokens: {self.max_tokens}")
        print(f"Temperature: {self.temperature}")
        return self

    def show_self_data(self):
        print("Chat Messages:")
        print(json.dumps(self.chat_messages, indent=4))
        print("\nContext Map:")
        print(json.dumps(self.context_map, indent=4))
        print("\nImages Map:")
        print(json.dumps(self.images_map, indent=4))
        print("\nAudio Map:")
        print(json.dumps(self.audio_map, indent=4))
        return self
        
    #
    # Chat methods
    #
    def pretty_print_messages(self):
        for message in self.chat_messages:
            role = message["role"]
            content = message["content"]
            print(f"{role}:")
            print(content)
            print() 
        return self
            
    def pretty_print_messages_to_file(self, file_name="output.txt",html=True):
        with open(file_name, 'w') as file:
                    for message in self.chat_messages:
                        role = message["role"]
                        content = message["content"]
                        file.write(f"{role}:\n")
                        file.write(content + "\n\n")
        if (html):
            return HTML(f'<a href="{file_name}" download>Click here to download the pretty-printed messages</a>')
        return self

    def set_system_prompt(self, prompt=""):
        # Remove existing "system" role message if it exists
        self.chat_messages = [msg for msg in self.chat_messages if msg.get("role") != "system"]

        print(prompt)
        print()
        prompt = self.replace_tags_with_content(prompt)
        
        # Insert new system message at the beginning of the list
        self.chat_messages.insert(0, {"role": "system", "content": prompt})
        return self

    def add_user_chat(self, prompt, label="latest"):
        print(prompt)
        prompt = self.replace_tags_with_content(prompt)
        self.context_map["latest_prompt"]=prompt
        self.chat_messages.append({"role": "user", "content": prompt})
        response = self.call_openai_chat_api(messages=self.chat_messages)
        self.chat_messages.append({"role": "assistant", "content": response})
        self.context_map["latest"] = response
        self.context_map[label] = response
        print(response)
        print()
        return self

    def filter_messages(self, func):
        if func is not None:
            self.chat_messages = func(self.chat_messages) 
        return self

    def reduce_messages_to_text(self, func):
        if func is not None:
            self.context_map["latest"] = func(self.chat_messages) 
            print(self.context_map["latest"])
        return self
    
    #
    # Simple completion
    #
    def completion(self, prompt, label='latest'):
        print(prompt)
        print() 
        prompt = self.replace_tags_with_content(prompt)
        self.context_map["latest_prompt"]=prompt
        messages=[{"role": "user", "content": prompt}]
        response = self.call_openai_chat_api(messages=messages)
        self.context_map["latest"] = response
        self.context_map[label] = response
        print(response)
        print()
        return self

    #
    # OpenAI caller
    #
    def call_openai_chat_api(self, messages=[]):
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=messages
            )
            return completion.choices[0].message.content
        except Exception as e:
            return str(e)
        
    #
    # Completing context in prompts
    #   
    def replace_tags_with_content(self, input_string=""):
        for key, value in self.context_map.items():
            input_string = input_string.replace(f'[{key}]', value)
        return input_string

    #
    # Context functions
    #
    def latest_to_context(self, label='latest'):
        self.context_map[label] = self.context_map["latest"]
        return self
    
    def transform_context(self, label="latest", func=lambda x: x):
        if label != 'latest' and label in self.context_map and func is not None:
                self.context_map[label] = func(self.context_map[label])
        return self
    
    def set_context(self, content="", label='latest'):
        if label != 'latest':
            self.context_map[label] = content
        return self
    
    def delete_context(self, label='latest'):
        if label != 'latest' and label in self.context_map:
            del self.context_map[label]
        return self
    
    def show_context(self, label='latest'):
        print(self.context_map[label])
        return self
    
    def read_textfile_to_context(self, filename, label="latest_file"):
        try:
            with open(filename, 'r') as file:
                content = file.read()
                self.context_map[label] = content
        except FileNotFoundError:
            print(f"The file {filename} does not exist.")
        except Exception as e:
            print(f"An error occurred: {e}")
        return self
    
    #
    # Functions to support embedding in other classes. Only implemented by returning strings
    #
    def return_latest_to_text(self):
        return self.context_map['latest']
    
    def return_context_to_text(self, label='latest'):
        return self.context_map(label)
    
    def return_reduce_messages_to_text(self, func):
        if func is not None:
            return func(self.chat_messages) 
        return self
    