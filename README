# Risk Based Insurance Premium Modeling

# Project Goals

1. Develop a predictive model to identify customers most likely to subscribe to a term deposit in order to optimize telemarketing campaign efficiency.
2. Identify the key factors influencing customer subscription behavior to inform marketing strategy and targeting decisions.

# Project Description

A Portuguese banking institution would like to know if a client is likely to subscribe to a term deposit based on their marketing campaigns. Term deposits are a major source of income for a bank. A term deposit is a cash investment held at a financial institution. Your money is invested for an agreed rate of interest over a fixed amount of time, or term. The bank has various outreach plans to sell term deposits to their customers such as email marketing, advertisements, telephonic marketing, and digital marketing.

Telephonic marketing campaigns are one of the most effective ways to reach out to people. However, they require huge investment as large call centers are hired to actually execute these campaigns. It is crucial to identify the customers most likely to convert beforehand so that they can be specifically targeted via call.

# Initial Questions

 1. Does the type of job have a relationship if a client will subscribe to a term deposit?
 2. Does the type of education a client have make it more likely to subscribe?
 3. Are clients older than average age more likely to subscribe? 
 4. Does the duration of contact have a relationship with subscription to a term deposit?

# The Plan

 - Create README with project goals, project description, initial hypotheses, planning of project, data dictionary, and come up with recommedations/takeaways

### Acquire Data
 - Acquire data from Kaggle and create a function to later import the data into a juptyer notebook. (acquire.py)

### Prepare Data
 - Clean and prepare the data creating a function that will give me data that is ready to be explored upon. Within this step we will also write a function to split our data into train, validate, and test. (prepare.py) 
 
### Explore Data
 - Create at least two hypotheses, set an alpha, run the statistical tests needed, reject or fail to reject the Null Hypothesis, document any findings and takeaways that are observed.
 
### Model Data 
 - Establish a baseline accuracyusing Logoistic Regression 
 
 - Create and train at least four classification models
 
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
| age  | age for each client (numeric)  |
| job | admin,unknown,unemployed,management,housemaid,entrepreneur,student, blue-collar,self-employed,retired,technician,services (categorical) |
| marital | married,divorced,single; note: "divorced" means divorced or widowed) (categorical) |
| education | unknown,secondary,primary,tertiary(categorical) |
| default | has credit in default? (binary: yes,no)  |
| balance | average yearly balance, in euros (numeric) | 
| housing | has housing loan? (binary: yes or no) |
| loan | has personal loan?(binary: yes or no) |
| contact | related with the last contact of the current campaign:  contact communication type unknown,telephone,cellular (categorical) |
| day | last contact day of the month (numeric) |
| month | last contact month of year (categorical) |
| duration | last contact duration, in seconds (numeric) |
| campaign | number of contacts performed during this campaign and for this client (numeric, includes last contact) |
| pdays | number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted) |
| previous | number of contacts performed before this campaign and for this client (numeric) |
| poutcome | outcome of the previous marketing campaign unknown,other,failure,success (categorical) |

## Steps to Reproduce

 - You will need both csv files from kaggle 

- Clone my repo including the acquire.py, prepare.py, and explore.py

- Put the data in a file containing the cloned repo.

- Run notebook.

## Conclusion

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
