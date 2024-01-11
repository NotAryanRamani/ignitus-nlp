# <center> Customer Sentiment  </center>

This initiative focuses on the examination of customer sentiment within Amazon reviews through the application of diverse analytical models. Specifically, our approach incorporates logistic regression, multinomial naive Bayes, and advanced transformer-based sentiment analysis models. By leveraging these sophisticated methodologies, we aim to derive nuanced insights into customer opinions, thereby facilitating a comprehensive understanding of the prevailing sentiments expressed in Amazon reviews. This project aligns with the broader goal of enhancing customer experience analysis and contributing valuable insights for strategic decision-making processes.

---

# Dataset  Used

Dataset Used: [amazon reviews dataset](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews)  

The dataset features:
1. `target` : the sentiment (positive or negative) given by customer to the product (target variable for sentiment)
2. `text` : the text review given by customer

---

# Models Trained

### 1. Logistic Regression

Parameters: `{'C': 0.001, 'max_iter': 100, 'multi_class': 'multinomial'}`

### 1. Multinomial Naive Bayes

*No parameters were used*

---

# Results

### 1. Logistic Regression

The metric for model evaluation was accuracy

- Accuracy Score on Train set: 0.876
- Accuracy Score on Test set: 0.8757

The model was regularized using the 'C' parameter(mentioned in the [Models Trained](#models-trained) section)  

### Confusion Matrix and F1 scores

    Confusion Matrix:

        [[4398  643]
        [ 600 4359]]

    F1 score = 0.8752133319947797

>Conclusion: **A decent model with approximately 87% accuracy.**  

### 2. Multinomial Naive Bayes

The metric for model evaluation was accuracy

- Accuracy Score on Train set: 0.8199
- Accuracy Score on Test set: 0.835

### Confusion Matrix and F1 scores

    Confusion Matrix:

        [[4194  847]
        [ 803 4156]]

    F1 score = 0.8343706083115839

>Conclusion: **Perfoming better on test set.**  

---

## Contributed by

Aryan Ramani
1. [LinkedIn](linkedin.com/in/aryan-ramani-a516b5212/)
2. [email](mailto:aryanramani67@gmail.com)
3. [Twitter/X](https://twitter.com/AryanRamani_DS) 







