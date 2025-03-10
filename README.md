# Colloquial Transformer

This project uses OpenAI's GPT API to transform input text into a more colloquial style while ensuring that the meaning remains unchanged. It supports both single sentences and multi-sentence text, allowing flexible input. The output is then evaluated using Sentence Transformers to compare the similarity between the input and output.

## Project Structure

```
│── model/                 # GPT Model interaction
│   ├── model.py           # Handles OpenAI API requests
│   ├── prompt/            # Stores prompt text for GPT model
│   │   ├── prompt.txt     # System message for text transformation
│── evaluator/             # Sentence similarity evaluation
│   ├── evaluator.py       # Computes similarity scores
│── main.py                # Main script
│── .env                   # API key storage (ignored in Git)
│── requirements.txt       # Required dependencies for installation
│── README.md              # Project documentation
```

## Installation

1. Clone the Repository

    ```
    git clone https://github.com/ozantanriverdi/colloquial-transformer.git
    cd colloquial-transformer
    ```

2. Install Dependencies

    ```
    pip install -r requirements.txt
    ```

3. Set Up Environment Variables: 
    
    Create a ```.env``` file and add your OpenAI API key:

    ```
    OPENAI_API_KEY=your-api-key-here
    ```

## Usage

1. Add the text you want to transform in ```main.py```:

    Modify main.py and set your input text:

    ```
    input_text = "Heute ist das Wetter sehr schön."
    ```

2. Run the main script:

    ```
    python main.py
    ```

3. Expected Output:

    ```
    Transformed Output:
    Heute ist das Wetter echt super!
    Similarity Evaluation Result:
    {'similarities': [0.8185423612594604], 'final_score': 0.8185423612594604, 'decision': 'Accepted'}
    ```