# Robo-Stock-Portfolio-Advisor
## About the Project
This project implements a comprehensive strategy for selecting and optimizing a portfolio from any list of random stocks. The core objective is to design and implement an algorithm that utilizes different financial concepts to evaluate the performance of each stock and choose 12 stocks with the highest performance to maximize the chance of beating the market, which we defined to be the average between the return of each index for the TSX 60 and the S&P 500 over a period of time. In the end, the program outputs a fully constructed portfolio with allocated weights and generates a CSV file detailing the selected stocks and their respective investments.

## Prerequisites
Ensure you have the following installed:
- Python 3.8 or later
  Install Python from python.org.
- pip (comes with Python)
  Verify pip is installed by running:

      pip --version
- Required Libraries
  Install the required Python packages:

      pip install pandas numpy numpy-financial yfinance matplotlib ipython

## Workflow
- Financial data extraction
  - Read ticker symbols from a CSV file
  - Retrieve historical stock data using yfinance
- Filtering algorithms
  - Remove stocks that aren't denominated in CAD/USD, didn't IPO before the time interval used for analysis, and don't have sufficient monthly volumes or trading history
- Indicator algorithms that each evaluate stocks based on different financial concepts and compute performance scores
  - Market correlation analysis
  - Implied Volatility(IV) calculations
  - Exponentional Moving Average(EMA) calculations
  - Sharpe ratios analysis
- Overall evaluation
  - Apply weighted indicators to rank stocks
- Portfolio generation
  - Select the top 12 stocks based on scores
  - Allocate weights and calculate fees for each stock
  - Output the final result
