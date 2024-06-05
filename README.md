# Commodity Trading and Risk Modelling Project

## Project Overview

This project involves four primary tasks related to commodity trading and risk modelling:

1. **Natural Gas Price Extrapolation**
2. **Commodity Storage Contract Pricing Model**
3. **Loan Default Probability Prediction**
4. **FICO Score Quantisation for Mortgage Default Prediction**

## Task 1: Natural Gas Price Extrapolation

### Objective
Enhance the granularity of available market data for natural gas storage contracts by estimating historical prices and extrapolating future prices.

### Approach
- **Data Acquisition**: Collected monthly natural gas price data from October 2020 to September 2024.
- **Data Analysis**: Identified seasonal trends affecting natural gas prices.
- **Extrapolation**: Used statistical interpolation to estimate prices for any date and extrapolated future prices for one year.

## Task 2: Commodity Storage Contract Pricing Model

### Objective
Develop a prototype pricing model for natural gas storage contracts, considering various cash flows and costs.

### Approach
- **Inputs**: Injection/withdrawal dates, prices, injection/withdrawal rates, storage costs.
- **Model**: Calculated the value of the contract by considering purchase/sale prices, storage fees, and other associated costs.
- **Output**: Provided a fair estimate of the contract value.

## Task 3: Loan Default Probability Prediction

### Objective
Predict the probability of default (PD) for personal loans using borrower characteristics.

### Approach
- **Data**: Utilised a sample loan book with borrower details and default history.
- **Model**: Employed machine learning techniques (e.g., regression, decision trees) to predict PD.
- **Outcome**: Estimated the expected loss for loans based on predicted PD and recovery rate.

## Task 4: FICO Score Quantisation for Mortgage Default Prediction

### Objective
Develop a method for quantising FICO scores to predict mortgage default probabilities.

### Approach
- **Data**: Used FICO scores from the bank’s mortgage portfolio.
- **Quantisation**: Mapped FICO scores into buckets to simplify prediction models.
- **Optimisation**: Minimised mean squared error and maximised log-likelihood for bucket boundaries.

## Conclusion

This project provides comprehensive tools and models for improving commodity trading strategies and predicting loan defaults. The methodologies and models developed can be used for further validation and integration into production systems.
