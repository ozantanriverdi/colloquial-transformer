import os
from dotenv import load_dotenv
from model import GPTModel
from evaluator import Evaluator

load_dotenv()

def main(input: str):
    """
    Main function for processing input text using a GPT model and evaluating its output.
    
    Steps:
        - Loads the API key from the environment.
        - Transforms the input text using the GPT model.
        - Evaluates the similarity between the input and output.
        - Prints the transformed output and similarity score.

    Args:
        input_text (str): The input text to be transformed.
    """
    
    # Load API key from environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Error: API key is missing. Please check your .env file.")
    
    # Initialize GPT model and Evaluator
    model = GPTModel(api_key=api_key)
    evaluator = Evaluator()
    
    output = model.api_caller(input)
    print("Transformed Output:")
    print(output)
    
    # Evaluate similarity
    similarity_result = evaluator.evaluate_output(input, output)
    print("Similarity Evaluation Result:")
    print(similarity_result)


if __name__ == '__main__':
    input_text = "Heute ist das Wetter sehr schön."
    main(input=input_text)