# TLDR
I built a machine learning pipeline to predict PGE power outage duration (in hours) that beats PGE's estimates handily. PGE has a RMSE of 34.69 hours for their outage estimates, while the ML model has a RMSE of 20.72 hours. In plain English, if there is an outage, on average, PGE is wrong by ~35 hours. If my machine learning model makes a prediction, on average, it is wrong by ~21 hours. If you were impacted by a power outage, wouldn't you want accurate estimates of when your power would be back on?

This entire repo took about 6 hours of focused work.


## Description
This work was completed as a part of a hackathon hosted by my employer, PwC. We annually host an internal hackathon where we can choose from a handful of publicly available datasets and solve some interesting problem.

This year, the PGE outages dataset was one of our hackathon dataset choices. In short, it captures all the outages that have occurred in California, USA over the past few months (Q4 of 2019) at 10 minute intervals.

With this data, we wanted to try to answer a few basic questions: 1) could we predict where there would be outages based on weather? (ran out of time for this one), 2) could we predict how long an outage would last? (this github repo addresses that), and 3) how do you prioritize repairs given limited manpower, remote electrical grid infrastructure, and number of people affected (no time for this one).

As mentioned, we wanted to predict how long an outage would last once it occurred. We went through the typical machine learning workflow of preparing the dataset, conducting exploratory data analysis (EDA), and building a machine learning model.

However, we didn't want to cut corners just because we only had 1 day to work on this. We still 1) defined our baseline (PGE time estimates) to beat before ever even writing a single line of ML code, set random seeds for reproducibility, defined a test set according to our domain, not randomly (in our case, we needed to test a holdout time period), grid searched for the best paramaters, and we started simple (no need for a deep learning model on your first go-round).

## Data
The data can be [found here](https://github.com/simonw/pge-outages). Essentially, all the Github commits store the data so you would have to clone the repo and run the `build_database.py` script. You can convert the sqlite tables to .csv files if you like. The whole process takes about 4-5 hours.