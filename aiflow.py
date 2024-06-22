# aiflow.py
import json
from openai import OpenAI  # pip install openai
from IPython.display import HTML
from docx import Document  # pip install python-docx
import urllib.request
from pathlib import Path


# chrome helper that converts a query result to a string, so we can use it in the class
def chroma_query_result_to_text(obj):
    documents = obj.get("documents")
    if documents:
        concatenated_string = "".join(["\n".join(doc) for doc in documents])
        return concatenated_string
    else:
        return ""


# models - gpt-4, gpt-4o, gpt-3.5-turbo
class AIFlow:
    def __init__(self, api_key, model="gpt-4", temperature=0, max_tokens=150):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.json_mode = False

        self.chat_messages = []
        self.context_map = {}
        self.images_map = {}
        self.audio_map = {}

        self.default_folder_for_output = ""

    # model configs
    def set_temperature(self, temperature=0):
        self.temperature = temperature
        return self

    def set_model(self, model="gpt-4"):
        self.model = model
        return self

    def set_max_tokens(self, max_tokens=150):
        self.max_tokens = max_tokens
        return self

    def set_json_output(self, json_mode=False):
        self.json_mode = json_mode
        return self

    def show_model_config(self):
        print(f"Model: {self.model}")
        print(f"Max Tokens: {self.max_tokens}")
        print(f"Temperature: {self.temperature}")
        return self

    #
    def set_output_folder(self, folder=""):
        self.default_folder_for_output = folder
        return self

    #
    # Some debugging tools
    #
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

    def clear_self_data(self):
        self.chat_messages = []
        self.context_map = {}
        self.images_map = {}
        self.audio_map = {}
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

    def pretty_print_messages_to_file(self, file_name="output.txt", html=True):
        with open(file_name, "w") as file:
            for message in self.chat_messages:
                role = message["role"]
                content = message["content"]
                file.write(f"{role}:\n")
                file.write(content + "\n\n")
        if html:
            return HTML(
                f'<a href="{file_name}" download>Click here to download the pretty-printed messages</a>'
            )
        return self

    def set_system_prompt(self, prompt=""):
        # Remove existing "system" role message if it exists
        self.chat_messages = [
            msg for msg in self.chat_messages if msg.get("role") != "system"
        ]

        print(prompt)
        print()
        prompt = self.replace_tags_with_content(prompt)

        # Insert new system message at the beginning of the list
        self.chat_messages.insert(0, {"role": "system", "content": prompt})
        return self

    def add_user_chat(self, prompt, label="latest"):
        print(prompt)
        prompt = self.replace_tags_with_content(prompt)
        self.context_map["latest_prompt"] = prompt
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
    def completion(self, prompt, label="latest"):
        print(prompt)
        print()
        prompt = self.replace_tags_with_content(prompt)
        self.context_map["latest_prompt"] = prompt
        messages = [{"role": "user", "content": prompt}]
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
                messages=messages,
            )
            return completion.choices[0].message.content
        except Exception as e:
            return str(e)

    #
    # Completing context in prompts
    #
    def replace_tags_with_content(self, input_string=""):
        for key, value in self.context_map.items():
            input_string = input_string.replace(f"[{key}]", value)
        return input_string

    #
    # Context functions
    #
    def copy_latest_to(self, label="latest"):
        self.context_map[label] = self.context_map["latest"]
        return self

    def transform_context(self, label="latest", func=lambda x: x):
        if label != "latest" and label in self.context_map and func is not None:
            self.context_map[label] = func(self.context_map[label])
        return self

    def set_context_of(self, content="", label="latest"):
        if label != "latest":
            self.context_map[label] = content
        return self

    def delete_context(self, label="latest"):
        if label != "latest" and label in self.context_map:
            del self.context_map[label]
        return self

    def show_context_of(self, label="latest"):
        print(self.context_map[label])
        return self

    def show_context_keys(self):
        for key in self.context_map.items():
            print(key)
        return self

    def read_textfile_to_context(self, filename, label="latest_file"):
        try:
            with open(filename, "r") as file:
                content = file.read()
                self.context_map[label] = content
        except FileNotFoundError:
            print(f"The file {filename} does not exist.")
        except Exception as e:
            print(f"An error occurred: {e}")
        return self

    def dump_context_to_files(self):
        for key, value in self.context_map.items():
            filename = f"context_{key}.txt"
            with open(filename, "w") as file:
                file.write(str(value))
        return self

    def dump_context_to_markdown(self, output_filename="content.md"):
        with open(output_filename, "w") as file:
            for chapter, content in self.context_map.items():
                file.write(f"# {chapter}\n\n")
                file.write(f"{content}\n\n")
        return self

    def generate_heading_for_context(
        self,
        label="latest",
        prompt="Generate a short 10 word summary of the following content:\n",
    ):
        content = self.context_map.get(label, "")
        if not content:
            return self
        full_prompt = f"{prompt}{content}"

        messages = [{"role": "user", "content": full_prompt}]
        response = self.call_openai_chat_api(messages=messages)

        self.set_context(label=label + "_heading", content=response)
        return self

    def dump_context_to_docx(self, output_filename):
        document = Document()
        for chapter, content in self.context_map.items():
            heading_key = chapter + "_heading"
            if heading_key in self.context_map:
                document.add_heading(self.context_map[heading_key], level=1)
            else:
                document.add_heading(chapter, level=1)

            document.add_paragraph(content)

        document.save(output_filename)
        return self

    #
    # Functions to support inclusion in other chat instances. Only implemented by returning strings
    #
    def return_latest_to_text(self):
        return self.context_map["latest"]

    def return_context_to_text(self, label="latest"):
        return self.context_map(label)

    def return_reduce_messages_to_text(self, func):
        if func is not None:
            return func(self.chat_messages)
        return self

    #
    # Saving state
    #
    def save_state(self, filename="state.json"):
        state_to_save = self.__dict__.copy()  # Create a copy of __dict__
        state_to_save.pop("client", None)  # Remove 'client' key if present
        with open(filename, "w") as f:
            json.dump(state_to_save, f, indent=4)
        return self

    def load_state(self, filename="state.json"):
        with open(filename, "r") as f:
            state = json.load(f)
        self.__dict__.update(state)
        return self

    #
    # Image generation
    # dall-e-3 dall-e-2
    # 1024x1024 512x512
    # https://platform.openai.com/docs/api-reference/images/create
    #
    def generate_image(
        self,
        model="dall-e-2",
        style="vivid",
        response_format="url",
        prompt="A white siamese cat",
        size="1024x1024",
        quality="standard",
        n=1,
        label="latest_image",
        html=False,
    ):
        response = self.client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            quality=quality,
            n=n,
            response_format=response_format,
            style=style,
        )

        if response_format == "url":
            self.images_map[label] = response.data[0].url
            self.images_map["latest_image"] = response.data[0].url

        if response_format == "b64_json":
            self.images_map[label] = response.data[0].b64_json
            self.images_map["latest_image"] = response.data[0].b64_json

        self.context_map["latest_image_prompt"] = prompt
        self.context_map["latest_revised_image_prompt"] = response.data[
            0
        ].revised_prompt

        if html:
            return HTML(f'<img src="{self.images_map[label]}" />')

        return self

    def dump_image_to_file(self, label="latest_image", filename=""):
        if filename == "":
            output_filename = "image_" + label + ".jpg"
        else:
            output_filename = filename
            # Ensure the filename ends with '.jpg'
            if not output_filename.lower().endswith(".jpg"):
                output_filename += ".jpg"

        print("Saving ", label, output_filename)

        # Check if the label exists in context_map before retrieving the image
        if label in self.images_map:
            urllib.request.urlretrieve(self.images_map[label], output_filename)

        return self

    #
    # Vision model
    # https://platform.openai.com/docs/guides/vision
    #
    def interpret_image(
        self,
        image="",
        prompt="What's in this image?",
        model="gpt-4o",
        label="latest",
        detail="low",
        max_tokens=300,
    ):
        if image == "":
            return self

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image, "detail": detail},
                        },
                    ],
                }
            ],
            max_tokens=max_tokens,
        )

        print(response.choices[0].message)
        self.context_map["latest"] = response.choices[0].message
        self.context_map[label] = response.choices[0].message

        return self

    #
    # MP3 generation
    # https://platform.openai.com/docs/api-reference/audio/createSpeech
    #
    def generate_speech(
        self,
        model="tts-1",
        voice="alloy",
        response_format="mp3",
        prompt="A white siamese cat",
        speed=1,
        filename="",
        label="latest_speech",
        html=False,
    ):
        if filename == "":
            filename = label + ".mp3"

        speech_file_path = Path(__file__).parent / filename

        response = self.client.audio.speech.create(
            model=model,
            input=prompt,
            voice=voice,
            speed=speed,
            response_format=response_format,
        )
        response.stream_to_file(speech_file_path)

        self.audio_map[label] = filename
        self.context_map["latest_speech_file"] = filename
        self.context_map["latest_speech_prompt"] = prompt

        if html:
            return HTML(
                f'<a href="{speech_file_path}" download>Click here to download the speech generated</a>'
            )

        return self

    # Sound transcription
    # https://platform.openai.com/docs/api-reference/audio/createTranscription
    def generate_transcription(
        self,
        filename="",
        model="whisper-1",
        language="en",
        prompt="",
        response_format="text",
        temperature=0,
        label="latest",
    ):
        if filename == "":
            return self

        audio_file = open(filename, "rb")
        transcript = self.client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
        )

        self.context_map["latest"] = transcript
        self.context_map[label] = transcript

        print(transcript)
        return self

    #
    # Moderation
    # https://platform.openai.com/docs/guides/moderation/overview
    #
    def generate_moderation(self, prompt="", label="latest_moderation"):
        if prompt == "":
            return self

        moderation = self.client.moderations.create(input=prompt)

        print(moderation.to_json())

        self.context_map["latest_moderation"] = moderation.to_json()
        self.context_map[label] = moderation.to_json()

        return self
