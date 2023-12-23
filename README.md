# <center> Customer Sentiment  </center>

This initiative focuses on the examination of customer sentiment within Amazon reviews through the application of diverse analytical models. Specifically, our approach incorporates logistic regression, multinomial naive Bayes, and advanced transformer-based sentiment analysis models. By leveraging these sophisticated methodologies, we aim to derive nuanced insights into customer opinions, thereby facilitating a comprehensive understanding of the prevailing sentiments expressed in Amazon reviews. This project aligns with the broader goal of enhancing customer experience analysis and contributing valuable insights for strategic decision-making processes.

---

# Dataset  Used

Dataset Used: [amazon reviews dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)  

Not all reviews were used. Only a few files were selected:
1. All_Beauty
2. AMAZON_FASHION
3. Appication
4. Gift_Cards
5. Luxury_Beauty
6. Magazine_Subscriptions
7. Software

The datasets were cleaned and merged into single dataset which only contains relevant features:
1. `overall` : the rating given by customer to the product (target variable for sentiment)
2. `reviewText` : the review given by customer
3. `category` : created feature; the category of the product

---

# Models Trained

### 1. Logistic Regression

Conducted a Grid Search Cross Validation using following parameters:

    param_grid = {  
        'max_iter': [100, 200],   
        'multi_class': ['ovr', 'multinomial'],  
        'C': [0.001, 0.01]    
    }

Best Parameters: `{'C': 0.001, 'max_iter': 100, 'multi_class': 'multinomial'}`

*Note:  Parameters are less because of limited computational resources.*

> ***Stay tuned for more updates***

---

# Results

### 1. Logistic Regression

The metric for model evaluation was accuracy

- Accuracy Score on Train set: 0.6117
- Accuracy Score on Test set: 0.6195

The model was regularized using the 'C' parameter(mentioned in the [Models Trained](#models-trained) section)

*Stay tuned*

---

## Contributed by

Aryan Ramani
1. [LinkedIn](linkedin.com/in/aryan-ramani-a516b5212/)
2. [email](mailto:aryanramani67@gmail.com)
3. [Twitter/X](https://twitter.com/AryanRamani_DS) 







