# aiflow.py
import json
from openai import OpenAI
from IPython.display import HTML
from docx import Document
import urllib.request
from pathlib import Path
import os
from IPython.display import Markdown, display
import markdown


# chroma helper that converts a query result to a string, so we can use it in the class
def chroma_query_result_to_text(obj):
    documents = obj.get("documents")
    if documents:
        concatenated_string = "".join(["\n".join(doc) for doc in documents])
        return concatenated_string
    else:
        return ""


# chroma helper that converts the query to a list
def chroma_query_to_list(result):
    ids = result["ids"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    # combine each n-th item from ids, metadatas and distances and put it in an object. Append to a result list and return
    if len(ids) != len(metadatas) or len(ids) != len(distances):
        raise ValueError("Lengths of ids, metadatas, and distances must be the same")

    result_list = []
    for i in range(len(ids)):
        # Create an object with id, metadata, and distance
        item = {"id": ids[i], "metadata": metadatas[i], "distance": distances[i]}
        # Append the object to the result list
        result_list.append(item)

    return result_list


# models - gpt-4, gpt-4o, gpt-3.5-turbo
class AIFlow:
    def __init__(self, api_key, model="gpt-4", temperature=0, max_tokens=150):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.json_mode = False
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.total_tokens = 0

        self.chat_messages = []
        self.context_map = {}
        self.images_map = {}
        self.audio_map = {}

        self.default_folder_for_output = ""
        self.verbose = True
        self.latest_state_filename = ""
        self.save_state_per_step = False

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

    def get_token_usage(self):
        return {
            "completion_tokens": self.completion_tokens,
            "prompt_tokens": self.prompt_tokens,
            "total_tokens": self.total_tokens,
        }

    # other config
    def set_output_folder(self, folder=""):
        self.default_folder_for_output = folder
        if folder != "":
            os.makedirs(self.default_folder_for_output, exist_ok=True)
        return self

    def set_verbose(self, level=True):
        self.verbose = level
        return self

    def set_step_save(self, step=False):
        self.save_state_per_step = step
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

    # function to run another function that may return something or nothing - this to support running code in the chain
    def run(self, func=lambda: "", label=""):
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

        if self.verbose:
            print(prompt)
            print()

        if self.save_state_per_step:
            self.save_state()

        prompt = self.replace_tags_with_content(prompt)

        # Insert new system message at the beginning of the list
        self.chat_messages.insert(0, {"role": "system", "content": prompt})
        return self

    def add_user_chat(self, prompt, label="latest"):
        if self.verbose:
            print(prompt)

        prompt = self.replace_tags_with_content(prompt)

        self.context_map["latest_prompt"] = prompt
        self.chat_messages.append({"role": "user", "content": prompt})

        response = self.call_openai_chat_api(messages=self.chat_messages)

        self.chat_messages.append({"role": "assistant", "content": response})
        self.context_map["latest"] = response
        self.context_map[label] = response

        if self.verbose:
            print(response)
            print()

        if self.save_state_per_step:
            self.save_state()

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
        if self.verbose:
            print(prompt)
            print()

        prompt = self.replace_tags_with_content(prompt)

        self.context_map["latest_prompt"] = prompt
        messages = [{"role": "user", "content": prompt}]
        response = self.call_openai_chat_api(messages=messages)
        self.context_map["latest"] = response
        self.context_map[label] = response

        if self.verbose:
            print(response)
            print()

        if self.save_state_per_step:
            self.save_state()

        return self

    #
    # OpenAI caller
    #
    def call_openai_chat_api(self, messages=[]):
        params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages,
        }

        # Conditionally add response_format
        if self.json_mode:
            params["response_format"] = {"type": "json_object"}

        try:
            completion = self.client.chat.completions.create(**params)

            # increase the tokens using completion.usage
            self.add_token_usage(completion.usage)

            return completion.choices[0].message.content
        except Exception as e:
            return str(e)

    #
    # Completing context in prompts
    #
    def replace_tags_with_content(self, input_string=""):
        for key, value in self.context_map.items():
            input_string = input_string.replace(f"[{key}]", str(value))
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
        keys_list = list(self.context_map.keys())
        keys_str = ", ".join(keys_list)
        print(keys_str)
        return self

    def return_context_keys(self):
        return self.context_map.keys()

    def load_to_context(self, filename, label="latest_file"):
        try:
            with open(filename, "r") as file:
                content = file.read()
                self.context_map[label] = content
        except FileNotFoundError:
            print(f"The file {filename} does not exist.")
        except Exception as e:
            print(f"An error occurred: {e}")
        return self

    def dump_context_to_file(self, label="latest", filename=""):
        if self.default_folder_for_output != "":
            filename_2 = os.path.join(
                self.default_folder_for_output, f"context_{label}.txt"
            )
        else:
            filename_2 = f"context_{label}.txt"

        if filename != "":
            filename_2 = filename

        with open(filename_2, "w") as file:
            file.write(str(self.context_map[label]))

        return self

    def dump_context_to_files(self):
        for key, value in self.context_map.items():
            self.dump_context_to_file(label=key)
        return self

    def dump_context_to_markdown(self, output_filename="content.md"):
        with open(output_filename, "w") as file:
            for chapter, content in self.context_map.items():
                file.write(f"# {chapter}\n\n")
                file.write(f"{content}\n\n")
        return self

    def generate_headings_for_context(
        self,
        labels=[],
        prompt="Generate a short 10 word summary of the following content:\n",
        replace=True,
    ):
        # iterate through all labels and call generate_heading_for_context for each label
        for label in labels:
            self.generate_heading_for_context(
                label=label, prompt=prompt, replace=replace
            )

        return self

    def generate_heading_for_context(
        self,
        label="latest",
        prompt="Generate a short 10 word summary of the following content:\n",
        replace=True,
    ):
        content = self.context_map.get(label, "")
        if not content:
            return self

        # Check if the heading already exists
        heading_label = label + "_heading"
        existing_heading = self.context_map.get(heading_label)

        # Conditionally call the API
        if replace or not existing_heading:
            # Generate the prompt
            full_prompt = f"{prompt}{content}"
            messages = [{"role": "user", "content": full_prompt}]
            response = self.call_openai_chat_api(messages=messages)

            self.set_context_of(label=heading_label, content=response)

            if self.save_state_per_step:
                self.save_state()

        return self

    def dump_context_to_docx(self, output_filename, chapters_to_include=[]):
        document = Document()
        for chapter, content in self.context_map.items():
            if chapter in chapters_to_include:
                heading_key = chapter + "_heading"
                if heading_key in self.context_map:
                    document.add_heading(self.context_map[heading_key], level=1)
                else:
                    document.add_heading(chapter, level=1)

                document.add_paragraph(str(content))

        document.save(output_filename)

        return self

    # def dump_context_to_html(self, output_filename, chapters_to_include=[]):
    #     html_content = "<html><body>"

    #     for chapter, content in self.context_map.items():
    #         if chapter in chapters_to_include or chapters_to_include == []:
    #             heading_key = chapter + "_heading"
    #             if heading_key in self.context_map:
    #                 heading = self.context_map[heading_key]
    #             else:
    #                 heading = chapter

    #             html_content += f"<h1>{heading}</h1>"
    #             html_content += markdown.markdown(str(content))

    #     html_content += "</body></html>"

    #     # Ensure the directory exists
    #     os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    #     with open(output_filename, "w", encoding="utf-8") as f:
    #         f.write(html_content)

    #     return self

    # def dump_context_to_html(self, output_filename, chapters_to_include=[]):
    #     html_content = "<html><body>"

    #     for chapter, content in self.context_map.items():
    #         if chapter in chapters_to_include or chapters_to_include == []:
    #             heading_key = chapter + "_heading"
    #             if heading_key in self.context_map:
    #                 heading = self.context_map[heading_key]
    #             else:
    #                 heading = chapter

    #             html_content += f"<h1>{heading}</h1>"
    #             html_content += markdown.markdown(str(content))

    #     html_content += "</body></html>"

    #     # Ensure the directory exists
    #     os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    #     with open(output_filename, "w", encoding="utf-8") as f:
    #         f.write(html_content)

    #     return self

    def dump_context_to_html(self, output_filename, chapters_to_include=[]):
        html_content = "<html><body>"

        if chapters_to_include:
            for chapter in chapters_to_include:
                if chapter in self.context_map:
                    heading_key = chapter + "_heading"
                    if heading_key in self.context_map:
                        heading = self.context_map[heading_key]
                    else:
                        heading = chapter

                    html_content += f"<h1>{heading}</h1>"
                    html_content += markdown.markdown(str(self.context_map[chapter]))
        else:
            for chapter, content in self.context_map.items():
                heading_key = chapter + "_heading"
                if heading_key in self.context_map:
                    heading = self.context_map[heading_key]
                else:
                    heading = chapter

                html_content += f"<h1>{heading}</h1>"
                html_content += markdown.markdown(str(content))

        html_content += "</body></html>"

        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)

        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(html_content)

        return self

    #
    # Functions to support inclusion in other chat instances. Only implemented by returning strings
    #
    def return_latest_to_text(self):
        return self.context_map["latest"]

    def return_context_to_text(self, label="latest"):
        return self.context_map[label]

    def return_reduce_messages_to_text(self, func):
        if func is not None:
            return func(self.chat_messages)
        return self

    def return_latest_as_md(self):
        return display(Markdown(self.context_map["latest"]))

    def return_context_as_md(self, label="latest"):
        return display(Markdown(self.context_map[label]))

    #
    # Saving state
    #
    def save_state(self, filename=""):
        if filename == "" and self.latest_state_filename == "":
            print("Error - no state filename provided")
            return self

        if filename == "":
            filename = self.latest_state_filename

        self.latest_state_filename = filename

        state_to_save = self.__dict__.copy()
        state_to_save.pop("client", None)
        with open(filename, "w") as f:
            json.dump(state_to_save, f, indent=4)

        return self

    def load_state(self, filename="state.json"):
        self.latest_state_filename = filename
        try:
            with open(filename, "r") as f:
                state = json.load(f)
            self.__dict__.update(state)
        except FileNotFoundError:
            print(f"File '{filename}' not found.")
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

        if self.verbose:
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

        # increase the tokens using completion.usage
        self.add_token_usage(response)

        if self.verbose:
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

        # increase the tokens using completion.usage
        self.add_token_usage(response)

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

        if self.verbose:
            print(moderation.to_json())

        self.context_map["latest_moderation"] = moderation.to_json()
        self.context_map[label] = moderation.to_json()

        return self

    def add_token_usage(self, usage):
        # increase the tokens using completion.usage
        self.completion_tokens += usage.completion_tokens
        self.prompt_tokens += usage.prompt_tokens
        self.total_tokens += usage.total_tokens

        self.context_map["_completion_tokens"] = self.completion_tokens
        self.context_map["_prompt_tokens"] = self.prompt_tokens
        self.context_map["_total_tokens"] = self.total_tokens

        return
