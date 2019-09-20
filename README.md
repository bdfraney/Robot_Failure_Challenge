# Reducing Maintenance Costs with Predictive Analytics
---
By: Blake Franey

## Table of Contents:
1. [Problem Statement](#problem-statement)
2. [Notebooks](#notebooks)
3. [Methods](#methods)
4. [Results](#results)
5. [Conclusions/Discussions](#conclusions-and-discussion)


## Problem Statement

Cloud BOT has a fleet of devices transmitting daily aggregated telemetry. Standard maintenance procedures, ie routine
 and time-based preventative methods, can be costly.  In order to better manage maintenance costs, implementing
  machine learning-based predictive maintenance techniques could save time and money by allowing technicians to only
   work on devices that are in need of repairs.  A standard classification model utilizing the daily aggregated
    telemetry that is already collected would be the preferable way to tackle this since it doesn't require
     additional data collection and a large database of historical telemetry data is already available.
        

## Notebooks

The notebooks are organized in the order best recommended to go through the repo.  There are also comments to try and
 explain my reasoning and how I decided my next steps.

1. [Exploratory Data Analysis](./notebooks/1_EDA.ipynb)
2. [Basic Logistic Regression Model](./notebooks/2_Basic_LogReg_Model.ipynb)
3. [Tuned Logistic Regression Model](./notebooks/3_Tuned_LogReg_Model.ipynb)
4. [Tuned and Feature Engineering/Selection](./notebooks/4_Tuned_LogReg_Features_Selection.ipynb)
5. [Random Forest Classifier Model](./notebooks/5_Random_Forest_Model.ipynb)


## Methods

I first started exploring the data and trying to understand it.  First thing of note was that there were a lot of
 samples compared to the number of features.  Another issue was that the features (attributes) didn't have any units
 , the attributes were on different magnitude scales (10e0 for some and 10e8 for others),
  nor was there any context as to what they were aggregating.  The baseline accuracy was also 99.9% because the
   classes were so heavily imbalanced which would be an issue when modeling.  I used `qgrid`, an interactive Jupyter
    Notebook
   extension for
   exploring data frames, to look at the data in full as well as compare devices that failed and devices that didn't
   .  I realized that since the telemetry was aggregated daily until a device failed, the change over time for each
    attribute could be
    important to the modeling process.  After plotting the the telemetry data as a pseudo-time series (pseudo because
     the time axis was imperfect) for a device that failed and one that didn't, I didn't see any clear pattern to the
      data.

Once finished exploring the data, I fit a baseline logistic regression model.  I then tried to improve the logistic
 regression model by regularization, hyper parameter tuning, and feature selection/engineering.  I used the
  coefficients from the regularized logistic regression model to fit another model with just those features then used
   `PolynomialFeatures` from the Scikit-learn API to engineer features.  Since I didn't know much about the
    attributes I couldn't make any educated guesses on which interaction terms would be relevant which is why I
     decided this was the best approach.  I used a `saga` solver because Stochastic Average Gradient Descent is good
      for larger data sets and the `saga` variant allows for both l1 and l2 regularization.  I chose `elasticnet` in
       order to take advantage of the two regularization methods.  In order to handle the class imbalance I set
        `class_weight` to `balanced`.  I set the `l1_ratio` to 0.65 and the strength to `C=0.0001` to drive some of
         the coefficients to 0.  Finally, `max_iter` was set to 500 so that the model went through enough iterations
          to converge.

After optimizing the logistic regression the best I could, I moved on to a slightly more complicated model.  I didn't
 want to go with something to computationally expensive so I chose a Random Forest Classifier.  I used all the
  attributes and tuned the model; specifically I set the number of estimators `n_estimators = 15`, minimum samples
   per leaf `min_samples_leaf = 450`, maximum features `max_features = None` and the class weight `class_weight
    = 'balanced_subsample` to
    address the class
    imbalance.     

## Results

The baseline accuracy was already 99.9% however the goal was to predict when devices would fail so the model
 needed to effectively differentiate between the positive (failed) and negative (not-failed) classes.  Therefore the
  metrics used to evaluate the model were: 1) Specificity, 2) Sensitivity, and 3) ROC AUC score.  The ROC AUC score
   was important because it measures how well the model can differentiate between the two classes; a high specificity
    wasn't difficult to achieve because it could easily predict the negative class and I needed to ensure it actually
     predicted when devices would fail so that maintenance could be performed on those. 
     
The baseline logistic regression model did terrible with a ROC AUC score of 0.5, meaning it did not differentiate
 between the two classes at all.  The specificity was 0.999 but the sensitivity was 0.  The other three logistic
  regression models performed about the same with a specificity of 0.98, sensitivity of 0.42, and an ROC AUC score of
   0.845.  
   
 The Random Forest Classifier did significantly better with a specificity of 0.96, sensitivity of 0.69, and an ROC
  AUC score of 0.92.  


## Conclusions and Discussion

I had moderate success in being able to build a model that would effectively predict device failure.  There still
 were a lot of false positives which would add to maintenance costs however the amount of devices that would be
  inspected are significantly less than just doing routine or time-based maintenance at random.  In order to improve
  , a more sophisticated modeling technique should be explored such as support vector machines or even an artificial
   neural network.  An ensemble method including the random forest classifier and tuned logistic regression model
    could be a good next step since it would be guaranteed to be at least as good as the two separately and would be
     less computationally expensive than a support vector machine or neural network.  Using an SVM alone or in the
      ensemble would be my next step after the initial ensemble.
      
Regardless of which model is chosen or next steps taken, a more robust approach to the class imbalance is absolutely
 crucial.  If collecting more data is unfeasible or too expensive, techniques such as oversampling and/or
  undersampling should be implemented at the very least.  
  
Ultimately, the predictive maintenance approach seems promising and would greatly reduce the amount of unnecessary
 maintenance inspections however no model will be perfect so a combination of routine or time-based maintenance with
  predictive maintenance would be most effective.