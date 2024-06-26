# AIFlow class

Something to simplify your AI pipelines using the builder pattern - see `aiflow.py`.
All other files and folders in the repo are not necessary to run your flows.

## Methods

Here is a list of methods in the `AIFlow` class with one-line descriptions (in `aiflow.py`):
Here's a one-line description for each method in the `AIFlow` class that you can use in your `README.md`:

## Class `AIFlow` Method Descriptions

### Initialiser and config

- **`__init__(self, api_key, model="gpt-4", temperature=0, max_tokens=150)`**: Initializes the AIFlow instance with given API key and model parameters.
- **`set_temperature(self, temperature=0)`**: Sets the temperature for the model's responses.
- **`set_model(self, model="gpt-4")`**: Configures the AI model to be used.
- **`set_max_tokens(self, max_tokens=150)`**: Sets the maximum number of tokens for model completions.
- **`set_json_output(self, json_mode=False)`**: Configures the output format to JSON if specified.
- **`show_model_config(self)`**: Displays the current model configuration settings.
- **`get_token_usage(self)`**: Returns a summary of token usage.
- **`set_output_folder(self, folder="")`**: Sets the default folder for output files.
- **`set_verbose(self, level=True)`**: Enables or disables verbose output.

### Dumping and some utility

- **`show_self_data(self)`**: Prints the current chat messages, context map, images map, and audio map.
- **`clear_self_data(self)`**: Clears all stored chat messages, context, images, and audio data.
- **`run(self, func=lambda: "", label="")`**: Runs a specified function within the AIFlow context.

### Chat interface

- **`pretty_print_messages(self)`**: Prints all chat messages in a readable format.
- **`pretty_print_messages_to_file(self, file_name="output.txt", html=True)`**: Saves chat messages to a file and optionally returns an HTML download link.
- **`set_system_prompt(self, prompt="")`**: Sets a system prompt for the chat session.
- **`add_user_chat(self, prompt, label="latest")`**: Adds a user chat message and generates an AI response.
- **`filter_messages(self, func)`**: Filters chat messages using a specified function.
- **`reduce_messages_to_text(self, func)`**: Reduces chat messages to text using a specified function.
- **`completion(self, prompt, label="latest")`**: Generates a completion for a given prompt.

### System call

- **`call_openai_chat_api(self, messages=[])`**: Calls the OpenAI chat API with the provided messages.

### Operating on context variables

- **`replace_tags_with_content(self, input_string="")`**: Replaces tags in the input string with content from the context map.
- **`copy_latest_to(self, label="latest")`**: Copies the latest context to a specified label.
- **`transform_context(self, label="latest", func=lambda x: x)`**: Transforms context content using a specified function.
- **`set_context_of(self, content="", label="latest")`**: Sets the content of a specified context label.
- **`delete_context(self, label="latest")`**: Deletes the content of a specified context label.
- **`show_context_of(self, label="latest")`**: Prints the content of a specified context label.
- **`show_context_keys(self)`**: Prints all keys in the context map.
- **`load_to_context(self, filename, label="latest_file")`**: Loads content from a file into the context map.
- **`dump_context_to_file(self, label="latest", filename="")`**: Dumps context content to a specified file.
- **`dump_context_to_files(self)`**: Dumps all context content to individual files.
- **`dump_context_to_markdown(self, output_filename="content.md")`**: Dumps context content to a markdown file.
- **`generate_heading_for_context(self, label="latest", prompt="Generate a short 10 word summary of the following content:\n")`**: Generates a heading for the specified context content.
- **`dump_context_to_docx(self, output_filename)`**: Dumps context content to a DOCX file.
- **`return_latest_to_text(self)`**: Returns the latest context content as text.
- **`return_context_to_text(self, label="latest")`**: Returns the content of a specified context label as text.
- **`return_reduce_messages_to_text(self, func)`**: Returns reduced chat messages as text using a specified function.

### State management so you don't need to rerun prompts

- **`save_state(self, filename="state.json")`**: Saves the current state to a JSON file.
- **`load_state(self, filename="state.json")`**: Loads the state from a JSON file.

### Media operations

- **`generate_image(self, model="dall-e-2", style="vivid", response_format="url", prompt="A white siamese cat", size="1024x1024", quality="standard", n=1, label="latest_image", html=False)`**: Generates an image using the specified model and parameters.
- **`dump_image_to_file(self, label="latest_image", filename="")`**: Saves a generated image to a file.
- **`interpret_image(self, image="", prompt="What's in this image?", model="gpt-4o", label="latest", detail="low", max_tokens=300)`**: Interprets the content of an image using the specified model.
- **`generate_speech(self, model="tts-1", voice="alloy", response_format="mp3", prompt="A white siamese cat", speed=1, filename="", label="latest_speech", html=False)`**: Generates speech from text using the specified model and parameters.
- **`generate_transcription(self, filename="", model="whisper-1", language="en", prompt="", response_format="text", temperature=0, label="latest")`**: Generates a transcription from an audio file using the specified model.
- **`generate_moderation(self, prompt="", label="latest_moderation")`**: Generates a moderation response for a given prompt.

### System method

- **`add_token_usage(self, usage)`**: Updates the token usage statistics.

### Chroma supporting function

**chroma_query_result_to_text(obj)**: Converts a query result to a string for use in the class.

### AIFlow Class Methods

#### Initialization and Configuration

2. \***\*init**(self, api_key, model="gpt-4", temperature=0, max_tokens=150)\*\*: Initializes the AIFlow class with API key and model parameters.
3. **set_temperature(self, temperature=0)**: Sets the temperature for the model.
4. **set_model(self, model="gpt-4")**: Sets the model to be used.
5. **set_max_tokens(self, max_tokens=150)**: Sets the maximum number of tokens for the model.
6. **set_json_output(self, json_mode=False)**: Sets the output mode to JSON.
7. **show_model_config(self)**: Prints the current model configuration.
8. **set_output_folder(self, folder="")**: Sets the default folder for output files.

#### Debugging and Data Management

9. **show_self_data(self)**: Prints internal data such as chat messages and context maps.
10. **clear_self_data(self)**: Clears all internal data.

#### Chat Methods

11. **pretty_print_messages(self)**: Prints chat messages in a readable format.
12. **pretty_print_messages_to_file(self, file_name="output.txt", html=True)**: Saves chat messages to a file and optionally returns an HTML link for download.
13. **set_system_prompt(self, prompt="")**: Sets the system prompt for the chat.
14. **add_user_chat(self, prompt, label="latest")**: Adds a user chat message and gets a response from the model.
15. **filter_messages(self, func)**: Filters chat messages based on a provided function.
16. **reduce_messages_to_text(self, func)**: Reduces chat messages to text using a provided function.

#### Completion Methods

17. **completion(self, prompt, label="latest")**: Generates a completion for the given prompt.

#### OpenAI API Interaction

18. **call_openai_chat_api(self, messages=[])**: Calls the OpenAI chat API with provided messages.

#### Context Management

19. **replace_tags_with_content(self, input_string="")**: Replaces tags in a string with corresponding context content.
20. **copy_latest_to(self, label="latest")**: Copies the latest context to a specified label.
21. **transform_context(self, label="latest", func=lambda x: x)**: Transforms context using a provided function.
22. **set_context_of(self, content="", label="latest")**: Sets the content for a specified context label.
23. **delete_context(self, label="latest")**: Deletes a specified context label.
24. **show_context_of(self, label="latest")**: Prints the content of a specified context label.
25. **show_context_keys(self)**: Prints all context keys.
26. **read_textfile_to_context(self, filename, label="latest_file")**: Reads a text file into context.
27. **dump_context_to_files(self)**: Dumps all context data to text files.
28. **dump_context_to_markdown(self, output_filename="content.md")**: Dumps context data to a Markdown file.
29. **generate_heading_for_context(self, label="latest", prompt="Generate a short 10 word summary of the following content:\n")**: Generates a heading for context content.
30. **dump_context_to_docx(self, output_filename)**: Dumps context data to a DOCX file.

#### Chat-in-chat Support (not return self, but returning the last output)

31. **return_latest_to_text(self)**: Returns the latest context as text.
32. **return_context_to_text(self, label="latest")**: Returns the context of a specified label as text.
33. **return_reduce_messages_to_text(self, func)**: Returns reduced chat messages as text using a provided function.

#### State Management

34. **save_state(self, filename="state.json")**: Saves the current state to a JSON file.
35. **load_state(self, filename="state.json")**: Loads state from a JSON file.

#### Image Generation and Handling

36. **generate_image(self, model="dall-e-2", style="vivid", response_format="url", prompt="A white siamese cat", size="1024x1024", quality="standard", n=1, label="latest_image", html=False)**: Generates an image based on a prompt.
37. **dump_image_to_file(self, label="latest_image", filename="")**: Saves an image to a file.

#### Vision Model

38. **interpret_image(self, image="", prompt="What's in this image?", model="gpt-4o", label="latest", detail="low", max_tokens=300)**: Interprets an image using a vision model.

#### Audio Generation and Handling

39. **generate_speech(self, model="tts-1", voice="alloy", response_format="mp3", prompt="A white siamese cat", speed=1, filename="", label="latest_speech", html=False)**: Generates speech from a prompt.
40. **generate_transcription(self, filename="", model="whisper-1", language="en", prompt="", response_format="text", temperature=0, label="latest")**: Transcribes audio from a file.

#### Moderation

41. **generate_moderation(self, prompt="", label="latest_moderation")**: Generates moderation output for a given prompt.

## Demo

[AIFlow demo](aiflow_demo.ipynb)
