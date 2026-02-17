# Risk Based Insurance Premium Modeling

1. Develop a predictive regression model to accurately estimate individual medical insurance charges based on demographic and lifestyle factors in order to support data-driven underwriting decisions.
2. Identify and quantify the key drivers influencing insurance premium costs (e.g., smoking status, BMI, age) to better understand risk factors and inform pricing strategy.

# Project Description

This project explores the relationship between demographic and lifestyle attributes and individual medical insurance charges. Using a dataset of 1,338 insured individuals, the objective is to develop predictive regression models that estimate insurance premiums based on features such as age, BMI, smoking status, number of children, sex, and region.

Because medical cost data is highly right-skewed, this project also investigates target transformation techniques to improve model assumptions and predictive performance. The analysis combines exploratory data analysis, statistical hypothesis testing, and regression modeling to generate both predictive accuracy and interpretable business insights.

The final model aims to provide accurate premium predictions while identifying the key drivers of insurance risk.

# Initial Questions

1. Is smoking status a statistically significant predictor of insurance charges?
2. Is BMI positively correlated with insurance costs?
3. Do insurance charges increase linearly with age?
4. Does the number of children significantly influence premium pricing?
5. Are there statistically significant differences in charges across regions?

# The Plan

 - Create README with project goals, project description, initial hypotheses, planning of project, data dictionary, and come up with recommedations/takeaways

### Acquire Data
 - Acquire data from Kaggle and create a function to later import the data into a juptyer notebook. (acquire.py)

### Prepare Data
 - Clean and prepare the data creating a function that will give me data that is ready to be explored upon. Within this step we will also write a function to split our data into train, validate, and test. (prepare.py) 
 
### Explore Data
 - Create at least two hypotheses, set an alpha, run the statistical tests needed, reject or fail to reject the Null Hypothesis, document any findings and takeaways that are observed.
 
### Model Data 
 - Establish a baselin. The baseline model will consist of predicting the average insurance charge for all individuals. Model performance to be measured using RMSE.
 
 - Create and train at least four regression models
 
 - Evaluate models on train and validate datasets
 
 - Evaluate which model performs the best and on that model use the test data subset.
 
### Delivery  
 - Create CSV file with the clients that are most likely to subscribe to a term deposit.
 
 - Create a Final Report Notebook to document conclusions, takeaways, and next steps in recommadations for clients to suscribe. Also, inlcude visualizations to help explain why the model that was selected is the best to better help the viewer understand. 


## Data Dictionary


| Target Variable |     Definition     |
| --------------- | ------------------ |
|      subscribed      | yes(1) or no(0) |

| Feature  | Definition |
| ------------- | ------------- |
| age  | Age of primary beneficiary (numeric)  |
| sex | Insurance contractor gender, female / male (binary) |
| bmi | Body mass index, providing an understanding of body, weights that are relatively high or low (numeric) |
| children | Number of children covered by health insurance / Number of dependents (numeric) |
| smoker | (binary: smoker, no-smoker)  |
| region | The beneficiary's residential area in the US, northeast, southeast, southwest, northwest. (categorical) | 
| charges | Individual medical costs billed by health insurance. (numeric) |


## Steps to Reproduce

 - You will need the csv file from kaggle 

- Clone my repo including the acquire.py, prepare.py, and explore.py

- Put the data in a file containing the cloned repo.

- Run notebook.

## Conclusion (TBD)

- Approximately 13% of contacted clients subscribe to term deposits, indicating a highly imbalanced dataset.

-  Students and retired clients show higher subscription rates relative to other occupational groups.

- Age is a meaningful predictor; when grouped into lifecycle segments, younger adults and clients aged 65+ demonstrate higher likelihood of subscription.

- Call duration strongly correlates with subscription probability, but was excluded from modeling due to data leakage (recorded post-contact).

## Best Model's performance:

- A Random Forest model (max_depth = 9) achieved:
        73.33% precision on the test set 
    Compared to a baseline conversion rate of ~13%

- This represents a 5.6x lift in targeting efficiency, significantly reducing wasted telemarketing calls.

## Recommendations:

- Deploy the model to prioritize high-probability customers in telemarketing campaigns.

- Adjust classification thresholds depending on campaign budget and desired call volume.

- Monitor model precision over time to ensure consistent targeting efficiency.

## Next Steps:

- Perform threshold optimization to maximize expected profit based on cost per call and revenue per subscription.

- Evaluate alternative models (e.g., Gradient Boosting, XGBoost) for potential precision gains.
