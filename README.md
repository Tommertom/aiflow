# AIFlow class

Something to simplify your AI pipelines using the builder pattern - see `aiflow.py`.
All other files and folders in the repo are not necessary to run your flows.

## Optimization

The code has been optimized using the Aider tool.

## Demo

[AIFlow demo](aiflow_demo.ipynb) - Jupyter workbook showing the works

[Empty book to start](aiflow_start.ipynb) - Jupyter workbook to start your own project

[Generating a real book](42book.ipynb) - A project I did to generate a book inspired on The Hitchikers Guide (fun project no real business goals) [PDF version after edits](42-illustrated.pdf) - the version after manual edits on layout and adding images from dalle3.

## AIFlow Class

### Initialization
- `__init__(self, api_key, model="gpt-4", temperature=0, max_tokens=150)`: Initialize the AIFlow class with API key, model, temperature, and max tokens.

### Model Configuration
- `set_temperature(self, temperature=0)`: Set the temperature for the model.
- `set_model(self, model="gpt-4")`: Set the model to be used.
- `set_max_tokens(self, max_tokens=150)`: Set the maximum number of tokens.
- `set_json_output(self, json_mode=False)`: Set the output format to JSON.
- `show_model_config(self)`: Display the current model configuration.
- `get_token_usage(self)`: Get the token usage statistics.

### Output Configuration
- `set_output_folder(self, folder="")`: Set the default folder for output.
- `set_verbose(self, level=True)`: Set the verbosity level.
- `set_step_save(self, step=False)`: Enable or disable saving state per step.

### Debugging Tools
- `display_internal_data(self)`: Display internal data for debugging.
- `clear_internal_data(self)`: Clear internal data.

### Chat Methods
- `pretty_print_messages(self)`: Pretty print chat messages.
- `pretty_print_messages_to_file(self, file_name="output.txt", html=True)`: Pretty print chat messages to a file.
- `set_system_prompt(self, prompt="")`: Set the system prompt.
- `add_user_chat(self, prompt, label="latest")`: Add a user chat message and get a response.
- `filter_messages(self, func)`: Filter chat messages using a function.
- `reduce_messages_to_text(self, func)`: Reduce chat messages to text using a function.

### Completion Methods
- `generate_completion(self, prompt, label="latest")`: Get a completion for a given prompt.

### Context Management
- `replace_tags_with_content(self, input_string="")`: Replace tags in the input string with context content.
- `copy_latest_to(self, label="latest")`: Copy the latest context to a specified label.
- `transform_context(self, label="latest", func=lambda x: x)`: Transform the context using a function.
- `set_context_of(self, content="", label="latest")`: Set the context for a specified label.
- `delete_context(self, label="latest")`: Delete the context for a specified label.
- `show_context_of(self, label="latest")`: Show the context for a specified label.
- `show_context_keys(self)`: Show all context keys.
- `return_context_keys(self)`: Return all context keys.
- `load_to_context(self, filename, label="latest_file")`: Load content from a file into the context.
- `dump_context_to_file(self, label="latest", filename="")`: Dump the context to a file.
- `dump_context_to_files(self)`: Dump all contexts to files.
- `dump_context_to_markdown(self, output_filename="content.md")`: Dump the context to a markdown file.
- `generate_headings_for_contexts(self, labels=[], prompt="Generate a short 10 word summary of the following content:\n", replace=True)`: Generate headings for multiple contexts.
- `generate_heading_for_context(self, label="latest", prompt="Generate a short 10 word summary of the following content:\n", replace=True)`: Generate a heading for a single context.
- `save_context_to_docx(self, output_filename, chapters_to_include=[])`: Save the context to a DOCX file.
- `save_context_to_html(self, output_filename, chapters_to_include=[])`: Save the context to an HTML file.

### Image Generation
- `generate_image(self, model="dall-e-2", style="vivid", response_format="url", prompt="A white siamese cat", size="1024x1024", quality="standard", n=1, label="latest_image", html=False)`: Generate an image.
- `save_image_to_file(self, label="latest_image", filename="")`: Save the generated image to a file.

### Image Analysis
- `analyze_image(self, image="", prompt="What's in this image?", model="gpt-4o", label="latest", detail="low", max_tokens=300)`: Analyze an image.

### Speech Generation
- `generate_speech(self, model="tts-1", voice="alloy", response_format="mp3", prompt="A white siamese cat", speed=1, filename="", label="latest_speech", html=False)`: Generate speech from text.

### Audio Transcription
- `transcribe_audio(self, filename="", model="whisper-1", language="en", prompt="", response_format="text", temperature=0, label="latest")`: Transcribe audio to text.

### Moderation
- `moderate_content(self, prompt="", label="latest_moderation")`: Create a moderation for a given prompt.

### State Management
- `save_internal_state(self, filename="")`: Save the internal state to a file.
- `load_internal_state(self, filename="state.json")`: Load the internal state from a file.

### Utility Methods
- `get_latest_context_as_text(self)`: Get the latest context as text.
- `get_context_as_text(self, label="latest")`: Get the context as text for a specified label.
- `get_reduced_chat_messages_as_text(self, func)`: Get reduced chat messages as text using a function.
- `display_latest_context_as_markdown(self)`: Display the latest context as markdown.
- `display_context_as_markdown(self, label="latest")`: Display the context as markdown for a specified label.
- `execute_function(self, func=lambda: "", label="")`: Run a function that may return something or nothing.
