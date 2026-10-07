import random


# ============================================================
# 1. BUILD THE BIGRAM LANGUAGE MODEL
# ============================================================

def build_model(sentences):
    """
    Build a Bigram Language Model from the given sentences.

    Returns:
        counts       -> frequency of each next-word transition
        probabilities -> probability of each next-word transition
    """

    # --------------------------------------------------------
    # Tokenization + sentence boundary
    # --------------------------------------------------------

    tokenized_sentences = []

    for sentence in sentences:
        tokens = sentence.split()

        # Special token that represents the end of a sentence
        tokens.append("<END>")

        tokenized_sentences.append(tokens)

    # --------------------------------------------------------
    # Create bigrams
    # --------------------------------------------------------

    bigrams = []

    for sentence in tokenized_sentences:

        for i in range(len(sentence) - 1):
            pair = (sentence[i], sentence[i + 1])
            bigrams.append(pair)

    # --------------------------------------------------------
    # Count bigram frequencies
    # --------------------------------------------------------

    counts = {}

    for word1, word2 in bigrams:

        if word1 not in counts:
            counts[word1] = {}

        if word2 not in counts[word1]:
            counts[word1][word2] = 0

        counts[word1][word2] += 1

    # --------------------------------------------------------
    # Convert counts into probabilities
    # --------------------------------------------------------

    probabilities = {}

    for word1 in counts:

        probabilities[word1] = {}

        total = sum(counts[word1].values())

        for word2 in counts[word1]:

            probability = counts[word1][word2] / total

            probabilities[word1][word2] = probability

    return counts, probabilities


# ============================================================
# 2. GREEDY NEXT-WORD PREDICTION
# ============================================================

def predict_next_word(word, probabilities):
    """
    Predict the most probable next word.

    Uses greedy decoding:
    choose the word with the highest probability.
    """

    if word not in probabilities:
        return None

    next_words = probabilities[word]

    predicted_word = max(
        next_words,
        key=next_words.get
    )

    return predicted_word


# ============================================================
# 3. PROBABILITY-BASED SAMPLING
# ============================================================

def sample_next_word(word, probabilities):
    """
    Select the next word according to its probability.
    """

    if word not in probabilities:
        return None

    next_words = list(probabilities[word].keys())
    probs = list(probabilities[word].values())

    selected_word = random.choices(
        next_words,
        weights=probs,
        k=1
    )[0]

    return selected_word


# ============================================================
# 4. GENERATE SENTENCE
# ============================================================

def generate_sentence(start_word, probabilities, max_words=10):
    """
    Generate a sentence starting from start_word
    using probability-based sampling.
    """

    words = [start_word]

    current_word = start_word

    for _ in range(max_words - 1):

        next_word = sample_next_word(
            current_word,
            probabilities
        )

        # Stop if word is not known
        if next_word is None:
            break

        # Stop at the end of sentence
        if next_word == "<END>":
            break

        words.append(next_word)

        current_word = next_word

    return " ".join(words)


# ============================================================
# 5. EVALUATE MODEL
# ============================================================

def evaluate_model(test_cases, probabilities):
    """
    Evaluate next-word prediction using test cases.

    test_cases format:
        [("current_word", "expected_next_word"), ...]
    """

    correct = 0

    print("\nTEST RESULTS:")
    print("-" * 55)

    for word, expected_word in test_cases:

        predicted_word = predict_next_word(
            word,
            probabilities
        )

        print(
            f"Input: {word:<15}"
            f"Expected: {expected_word:<15}"
            f"Predicted: {str(predicted_word):<15}"
        )

        if predicted_word == expected_word:
            correct += 1

    accuracy = correct / len(test_cases)

    return accuracy


# ============================================================
# 6. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Training Data
    # --------------------------------------------------------

    training_sentences = [
        "I love Python",
        "I love Java",
        "I love Python",
        "I love Python",

        "Python is powerful",
        "Python is powerful",
        "Python is powerful",

        "Java is powerful",
        "Java is powerful",

        "Python is popular",

        "Machine learning is interesting",
        "Machine learning is interesting",

        "Python is interesting"
    ]

    # --------------------------------------------------------
    # Build Model
    # --------------------------------------------------------

    counts, probabilities = build_model(
        training_sentences
    )

    # --------------------------------------------------------
    # Display Counts
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("BIGRAM COUNTS")
    print("=" * 60)

    for word, next_words in counts.items():
        print(f"{word} -> {next_words}")

    # --------------------------------------------------------
    # Display Probabilities
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("BIGRAM PROBABILITIES")
    print("=" * 60)

    for word, next_words in probabilities.items():

        print(f"\n{word}")

        for next_word, probability in next_words.items():

            print(
                f"    {next_word:<12} "
                f"{probability:.2f}"
            )

    # --------------------------------------------------------
    # Greedy Predictions
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("GREEDY PREDICTIONS")
    print("=" * 60)

    test_words = [
        "I",
        "love",
        "Python",
        "Java",
        "is"
    ]

    for word in test_words:

        prediction = predict_next_word(
            word,
            probabilities
        )

        print(
            f"After '{word}' -> {prediction}"
        )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    test_cases = [
        ("I", "love"),
        ("love", "Python"),
        ("Python", "is"),
        ("Java", "is"),
        ("is", "powerful")
    ]

    accuracy = evaluate_model(
        test_cases,
        probabilities
    )

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print(f"Accuracy: {accuracy:.2%}")

    # --------------------------------------------------------
    # Generate Sentences
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("GENERATED SENTENCES")
    print("=" * 60)

    for i in range(5):

        sentence = generate_sentence(
            "I",
            probabilities,
            max_words=10
        )

        print(f"{i + 1}. {sentence}")