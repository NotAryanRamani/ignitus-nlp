# Text Generation

The goal of the project is to generate text using look up tables at character level and word level embeddings.

## 1. Markov Chain (Probabilistic Approach)

Steps:
1. Import and preprocess data.
2. Split data into bigrams (pair of 2 characters) and get unique characters.
3. Create character to index and index to character mappings.
4. Create look up table using Bigram counts and normalize them to find probabilities.
5. Generate text using [torch.multinomial](https://pytorch.org/docs/stable/generated/torch.multinomial.html) function.

## 2. Single Layer Network

Steps:
1. Create bigrams
2. Compile a single layer neural network.
3. Forward pass using one hot encodings of characters to produce logits, and backpass to change weights of the network.
4. Generate text using [torch.multinomial](https://pytorch.org/docs/stable/generated/torch.multinomial.html) function.

## 3. Multilayer Network

Steps:
1. Create bigrams or n-grams inputs(recommended 3-gram inputs).
2. Compile a 3 layer neural network:
    * A Embedding layer  
    * Hidden Linear layer with Tanh activation
    * Output Linear layer with Softmax activation
3. Forward pass using embedding table of characters to produce logits, and backpass to change the embeddings, weights and biases of the network.
4. Generate text using [torch.multinomial](https://pytorch.org/docs/stable/generated/torch.multinomial.html) function.


## 3. Word Level Embedding

Steps:
1. Preprocess text to produce vocabulary size.
2. Generate word to index and index to word mappings
3. Create bigrams or n-grams inputs on word level(recommended 3-gram inputs).
4. Compile a 3 layer neural network:
    * A Embedding layer of n-dimensions  
    * Hidden Linear layer with Tanh activation
    * Output Linear layer with Softmax activation
5. Forward pass using embedding table of words to produce logits, and backpass to change the embeddings, weights and biases of the network.
6. Generate text using [torch.multinomial](https://pytorch.org/docs/stable/generated/torch.multinomial.html) function.

## 4. Wave Net

*Coming Soon :)*