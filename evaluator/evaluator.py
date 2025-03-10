import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

nltk.download('punkt')

class Evaluator:
    """
    Evaluates similarity between input and output sentences and determines validity based on thresholds.
    Attributes:
        model (SentenceTransformer): The model used for embedding and similarity calculation.
        threshold (float): The similarity threshold to determine whether an output is accepted.
    """
    def __init__(self, threshold=0.7):
        """
        Initializes the SentenceTransformer model and sets the acceptance threshold.
        Args
            threshold (float, optional): The similarity threshold for acceptance.
        """
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.threshold = threshold

    def evaluate_output(self, input_text, output_text):
        """
        Computes similarity between input and output sentences and determines acceptance.
        Args:
            input_text (str): The original input text.
            output_text (str): The transformed output text.
        Returns:
            dict: A dictionary containing:
                - "similarities" (list): List of similarity scores for each sentence pair.
                - "final_score" (float): The averaged similarity score.
                - "decision" (str): Either "Accepted" or "Rejected".
        """
        # Chunk input/output into sentences
        input_sentences = sent_tokenize(input_text)
        output_sentences = sent_tokenize(output_text)

        # Ensure equal sentence count
        if len(input_sentences) != len(output_sentences):
            raise ValueError("Error: Sentence count mismatch between input and output!")

        # Encode input and output sentences
        input_embed = self.model.encode(input_sentences)
        output_embed = self.model.encode(output_sentences)

        # Compute pairwise similarity
        similarities = self.model.similarity(input_embed, output_embed)
        similarities = np.diag(similarities)
        # Extract only the diagonal values from the similarity matrix
        score = float(np.mean(similarities))

        # Determine if output is Accepted or Rejected
        decision = "Accepted" if score >= self.threshold else "Rejected"

        result = {
            "similarities": similarities.tolist(),
            "final_score": score,
            "decision": decision
        }
        return result
    
