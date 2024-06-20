# AIFlow class

Something to simplify your AI pipelines using the builder pattern

## Methods

Here is a list of methods in the `AIFlow` class with one-line descriptions:

1. **`__init__(self, api_key, model='gpt-4', temperature=0, max_tokens=150)`**: Initializes the AIFlow instance with given API key, model, temperature, and max tokens.
2. **`set_temperature(self, temperature=0)`**: Sets the temperature for the model.
3. **`set_model(self, model='gpt-4')`**: Sets the model to use.
4. **`set_max_tokens(self, max_tokens=150)`**: Sets the maximum number of tokens for responses.
5. **`show_model_config(self)`**: Prints the current model configuration.
6. **`show_self_data(self)`**: Prints the current state of chat messages, context map, images map, and audio map.
7. **`pretty_print_messages(self)`**: Prints chat messages in a formatted manner.
8. **`pretty_print_messages_to_file(self, file_name='output.txt', html=True)`**: Writes formatted chat messages to a file and optionally returns a download link.
9. **`set_system_prompt(self, prompt='')`**: Sets a system prompt message at the beginning of the chat messages.
10. **`add_user_chat(self, prompt, label='latest')`**: Adds a user chat message, gets a response from the API, and updates the context map.
11. **`filter_messages(self, func)`**: Filters chat messages based on a provided function.
12. **`reduce_messages_to_text(self, func)`**: Reduces chat messages to a text string using a provided function.
13. **`completion(self, prompt, label='latest')`**: Generates a completion for a given prompt and updates the context map.
14. **`call_openai_chat_api(self, messages=[])`**: Calls the OpenAI API to get a response for given messages.
15. **`replace_tags_with_content(self, input_string='')`**: Replaces tags in a string with corresponding context map values.
16. **`latest_to_context(self, label='latest')`**: Copies the latest context to a specified label.
17. **`transform_context(self, label='latest', func=lambda x: x)`**: Transforms a specified context using a provided function.
18. **`set_context(self, content='', label='latest')`**: Sets a specified context to the given content.
19. **`delete_context(self, label='latest')`**: Deletes a specified context from the context map.
20. **`show_context(self, label='latest')`**: Prints the content of a specified context.
21. **`read_textfile_to_context(self, filename, label='latest_file')`**: Reads a text file and sets its content to a specified context.
22. **`return_latest_to_text(self)`**: Returns the latest context as a text string.
23. **`return_context_to_text(self, label='latest')`**: Returns a specified context as a text string.
24. **`return_reduce_messages_to_text(self, func)`**: Reduces chat messages to a text string using a provided function and returns it.
25. **`dump_context_to_files(self)`**: Dump the context in a file, one file per context.

## Demo

[AIFlow demo](aiflow_demo.ipynb)
