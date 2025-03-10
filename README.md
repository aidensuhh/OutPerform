# Outperform - Robo Stock Portfolio Advisor
## About the Project
This project implements a comprehensive strategy for selecting and optimizing a portfolio from any list of random stocks. The core objective is to design and implement an algorithm that utilizes different financial concepts to evaluate the performance of each stock and choose 12 stocks with the highest performance to maximize the chance of beating the market, which we defined to be the average between the return of each index for the TSX 60 and the S&P 500 over a period of time. In the end, the program outputs a fully constructed portfolio with allocated weights and generates a CSV file detailing the selected stocks and their respective investments.

## Workflow
- Financial data extraction
  - Read ticker symbols from a CSV file
  - Retrieve historical stock data using yfinance
    
- Filtering algorithms
  - Remove stocks that aren't denominated in CAD/USD, didn't IPO before the time interval used for analysis, and don't have sufficient monthly volumes or trading history
    
- Indicator algorithms; each evaluates stocks based on different financial concepts and compute performance scores
  - Market correlation analysis
    - Evaluates the correlation between individual stocks and the market average (TSX and S&P 500)
    - Scores stocks based on their ability to diversify away from market trends
  - Implied Volatility(IV) calculations
    - Calculates implied volatility for stock options
    - Scores stocks based on their options data for optimal strike prices
  - Exponential Moving Average(EMA) calculations
    - Utilizes EMA indicators (12-day, 26-day, and 200-day) to assess stock price trends
    - Computes the MACD (Moving Average Convergence Divergence) to evaluate momentum and trend reversals
    - Scores stocks based on recent performance, with emphasis on those showing positive momentum and price stability over the last 10 trading days
  - Sharpe ratios analysis
    - Simulates 1,000,000 portfolios using random weight distributions
    - Identifies the best portfolio and computes individual stock scores based on Sharpe ratios
      
- Overall evaluation
  - Apply weighted indicators to rank stocks
    
- Portfolio generation
  - Select the top 12 stocks based on scores
  - Allocate weights and calculate fees for each stock
  - Output the final result

## Example Outputs
- Portfolio Summary:
  - List of 12 selected stocks with weights and investment value
  - Total portfolio value and weight
- CSV Export:
  - A CSV file containing the tickers and shares for the final portfolio.

## Run Our Program
Ensure you have the following installed:
- Python 3.8 or later
- pip (comes with Python)

<br>1. Create a virtual environment:
  ``` 
  python -m venv .venv
  ```

<br>2. Activate the virtual environment:
- On Windows:
  ``` 
  .venv\Scripts\activate
  ```
- On macOS/Linux:
  ```
  source .venv/bin/activate
  ```
  
<br>3. Install the packages from requirements.txt:
   ```
   pip install -r requirements.txt
   ```
   
<br>4. Test the application:
- Run portfolio_advisor.py to use our program!
