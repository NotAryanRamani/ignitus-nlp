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

### 1. Multinomial Naive Bayes

Conducted a Grid Search Cross Validation using following parameters:

    param_grid = {
        'alpha': [0.1, 0.5, 1.0, 1.5],
        'fit_prior': [True, False]
    }


Best Parameters: `{'alpha': 0.1, 'fit_prior': True}`

*Note:  Parameters are less because of limited computational resources.*

---

# Results

### 1. Logistic Regression

The metric for model evaluation was accuracy

- Accuracy Score on Train set: 0.6117
- Accuracy Score on Test set: 0.6195

The model was regularized using the 'C' parameter(mentioned in the [Models Trained](#models-trained) section)  

### Confusion Matrix and F1 scores

    Confusion Matrix:

        [[ 43   8  14   3  31]
        [ 18   8  11  20  48]
        [  7   8  55  73  80]
        [  3   3  44 183 251]
        [  4   3  15 117 950]]

    F1 score of Class 0 = 0.4942528735632184
    F1 score of Class 1 = 0.11851851851851852
    F1 score of Class 2 = 0.30386740331491713
    F1 score of Class 3 = 0.4159090909090909
    F1 score of Class 4 = 0.7758268681094324

    Average F1 score = 0.5876566846188007

>Conclusion: **Very underperforming model.**  The model struggles with accurate predictions, particularly for classes 1 and 2, as indicated by the higher counts of false predictions in these classes.

### 2. Multinomial Naive Bayes

The metric for model evaluation was accuracy

- Accuracy Score on Train set: 0.5657
- Accuracy Score on Test set: 0.564

### Confusion Matrix and F1 scores

    Confusion Matrix:

        [[ 45   0   1   2  51]
        [ 15   0   1  19  70]
        [ 21   0   7  54 141]
        [ 33   0   2  82 367]
        [ 42   0   0  53 994]]

    F1 score of Class 0 = 0.3529411764705882
    F1 score of Class 1 = 0.0
    F1 score of Class 2 = 0.05982905982905984
    F1 score of Class 3 = 0.23631123919308358
    F1 score of Class 4 = 0.7330383480825958

    Average F1 score = 0.48046822882193396

>Conclusion: **Worse model than Logistic Regression model.**  The model appears to struggle with accurate predictions, particularly for classes 1 and 4, as indicated by the higher counts of false predictions in these classes. Further analysis and model improvement may be needed.

---

## Contributed by

Aryan Ramani
1. [LinkedIn](linkedin.com/in/aryan-ramani-a516b5212/)
2. [email](mailto:aryanramani67@gmail.com)
3. [Twitter/X](https://twitter.com/AryanRamani_DS) 







