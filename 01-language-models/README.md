# Bigram Language Model

## Overview

This project implements a simple statistical Bigram Language Model
from scratch using Python.

The purpose of the project is to understand the fundamental process
of language modeling, including tokenization, word-transition
probabilities, next-token prediction, sampling, and text generation.

## Objective

To understand how a language model can learn patterns from training
data and estimate the probability of the next token.

## Concepts Covered

- Language Models
- Tokenization
- Bigrams
- Frequency Counting
- Conditional Probability
- Probability Distribution
- Greedy Decoding
- Probability-Based Sampling
- Autoregressive Text Generation
- Sentence Boundary Tokens
- Basic Model Evaluation

## Architecture

Training Text
     ↓
Tokenization
     ↓
Sentence Boundary (<END>)
     ↓
Bigram Generation
     ↓
Frequency Counting
     ↓
Probability Calculation
     ↓
Next-Word Prediction
     ↓
Text Generation

## Example

Training data contains:

I love Python
I love Java
I love Python
I love Python

The model learns:

love → Python = 3
love → Java = 1

Therefore:

P(Python | love) = 3/4 = 0.75
P(Java | love) = 1/4 = 0.25

Using greedy decoding:

love → Python

## Greedy Decoding

The model selects the next token with the highest probability.

Example:

Python:
- is = 0.62
- <END> = 0.38

Therefore:

Python → is

## Probability-Based Sampling

The model can also sample the next token according
to the learned probability distribution.

For example:

love:
- Python = 0.75
- Java = 0.25

Sampling may produce different outputs from the same
starting word.

## Sample Output

I love Python is powerful

I love Python

I love Python is interesting

I love Java

## Evaluation

A small manually selected test set was used to verify
next-word prediction.

Test Accuracy: 100%

Note: This is not a generalization benchmark because
the test cases are based on patterns represented in the
training data.

## Key Observations

1. The model learns word-transition patterns from the training data.

2. Multiple next tokens can have different probabilities.

3. Greedy decoding selects the highest-probability token.

4. Probability sampling can produce different outputs.

5. Sentence boundaries are important during preprocessing.

6. A Bigram model only considers the immediately preceding token.

## Limitations

- Very small training dataset
- Word-level tokenization
- Only one previous token is considered
- Cannot capture long-range dependencies
- Cannot understand semantics
- Cannot generalize like modern Transformer-based language models

## Technologies

- Python
- Dictionaries
- Probability
- Random Sampling

## Future Improvements

- Trigram Language Model
- Neural Language Model
- Transformer-based Language Model
- Modern Tokenization
- Larger Dataset
- Better Evaluation