import os
import openai
from openai import OpenAI


class GPTModel:
    """
    Interacting with the OpenAI GPT API, handling system messages and error handling.
    Attributes:
        client (OpenAI): An instance of the OpenAI API client.
        model_name (str): The name of the GPT model to use.
        sys_msg (str): The system message loaded from the prompt file.
    """
    
    def __init__(self, api_key, model_name="gpt-4o-mini"):
        """
        Initializes the GPTModel with the provided API key and model name.
        Args:
            api_key (str): The API key for authenticating with the OpenAI API.
            model_name (str, optional): The model name to use for the api requests.
        """
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.sys_msg = self.load_sys_msg()

    def load_sys_msg(self):
        """
        Loads the system message from the prompt.txt file.
        Returns:
            str: The system message read from the file.
        """
        # Path to prompt file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(script_dir, "prompt", "prompt.txt")
        
        try:
            # Open and read the system message from the prompt file
            with open(prompt_path, "r", encoding="utf-8") as f:
                sys_msg = f.read()
        except FileNotFoundError:
            print(f"Error: File not found at {prompt_path}")
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to load system message from {prompt_path}.") from e
        
        # Return the system message text
        return sys_msg

    def api_caller(self, input, max_tries=3):
        """
        Calls the OpenAI GPT API with the given user input.
        Args:
            user_input (str): The text input to be processed by the model.
            max_tries (int, optional): The number of times to retry in case of API failures.
        Returns:
            str | None: The model's generated response if successful, or None if all attempts fail.
        """

        # Construct the message
        message = [
                {"role": "system", "content": self.sys_msg},
                {"role": "user", "content": [{"type": "text", "text": input},]}
        ]

        # Initialize retry counter
        attempt = 0

        while attempt < max_tries:
            try:
                # Send the request to OpenAI API
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=message,
                    max_tokens=300) # Limit response length
                
                # Return the generated output
                return response.choices[0].message.content
            
            # Handle different API-related errors
            except openai.InternalServerError:
                print("OpenAI API service is temporarily unavailable. Please try again later.")
                
            except openai.AuthenticationError:
                print("There was an issue with API authentication. Please check your API key.")
                break # Stop retrying if API key is invalid
            except openai.RateLimitError:
                print("You have exceeded your rate limit or run out of credits. Please check your usage.")
                break # Stop retrying if out of credits
            except openai.APITimeoutError:
                print("Request timed out.")

            except openai.APIConnectionError:
                print("Network error: Unable to connect to the OpenAI API. Please check your internet connection.")

            except Exception as e:
                print(f"An unexpected error occurred: {e}")
            
            attempt += 1

        print("Max retries reached. API call failed.")
        return None # Return None if all attempts fail

