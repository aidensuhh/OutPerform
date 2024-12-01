# Libraries used

from IPython.display import display, Math, Latex

import pandas as pd
import numpy as np
import numpy_financial as npf
import yfinance as yf
import matplotlib.pyplot as plt
import random
from datetime import datetime


# extract data
data = pd.read_csv('Tickers.csv')

# if the first ticker is incorrectly made into a column, we need to append that ticker to first
extra_ticker = data.columns[0]
if extra_ticker not in data.iloc[:, 0].tolist():
    data = pd.concat([pd.DataFrame([[extra_ticker]], columns=data.columns), data], ignore_index=True)

# initialization of interval variables for data analysis later on
start_date = '2024-01-01'
end_date = '2024-11-23'

# Aiden's Algorithm to filter stocks
def filter_data(data):
    filtered_data = []
    ticker_symbols = data.iloc[:, 0].tolist()  # puts all ticker symbols into a list first

    for symbol in ticker_symbols:
        try:
            ticker = yf.Ticker(symbol)

            if ticker.info and ticker.info.get('currency') in ['CAD', 'USD']: # checks if currency is either in CAD or USD
                ticker_history = ticker.history(start='2023-10-01', end='2024-10-01')

                # if no data is available for the ticker, filter that symbol
                if ticker_history.empty:
                    continue

                # check if the stock has IPO'd, or if it did, check if the stock IPO'd after our start date for the analysis
                first_valid_date = ticker_history.first_valid_index()
                if first_valid_date is None:
                    continue
                else:
                    if pd.Timestamp(first_valid_date).tz_localize(None) > pd.to_datetime(start_date).tz_localize(None):
                        continue

                # check if the ticker data includes its volume info
                if 'Volume' in ticker_history.columns:
                    ticker_history.index = pd.to_datetime(ticker_history.index)
                    monthly_data = ticker_history['Volume'].resample('ME').agg(sum='sum', count='count')
                    valid_months = monthly_data[monthly_data['count'] >= 18] # filters any month with less than 18 trading days

                    if not valid_months.empty:
                        monthly_avg_volume = valid_months['sum'] / valid_months['count']

                        if monthly_avg_volume.mean() >= 100000: # checks if the stock has an average monthly volume of at least 100,000 shares
                            filtered_data.append(symbol) # only when every requirement is fulfilled, add the symbol to filtered list

        except Exception as e:
            print(f"An error occurred: {e}")

    return filtered_data


# Vincent's Algorithm to find the 12 most correlated stocks in the csv file

# Filter the stocks
data_filter = filter_data(data)

# Initialization
ticker_name = ""
data_close = pd.DataFrame()

for i in range (0,len(data_filter)):
  ticker_name = data_filter[i]

  ticker1 = yf.Ticker(ticker_name)
  ticker1_hist = ticker1.history(start=start_date, end=end_date)

  # if the ticker is delisted or not available, then don't add it to data_close
  if (not ticker1_hist.empty):
    ticker1_hist.index = pd.to_datetime(ticker1_hist.index).strftime('%Y-%m-%d')

    data_close[ticker_name]=ticker1_hist['Close']

# Find the average percent change for the S%P500 and TSX which represent the market
data_market = yf.download(["^GSPC", "^GSPTSE"], start=start_date, end=end_date)['Close']
data_market.index = data_market.index.strftime('%Y-%m-%d')

# Calculate the average of S&P 500 and TSX closing prices, and create a one-column dataframe
market_close = (data_market["^GSPC"] + data_market["^GSPTSE"])/2
market_close = pd.DataFrame(market_close, columns=["Average"])
market_close_pct = market_close.pct_change()
market_close_pct.drop(index=market_close_pct.index[0], inplace=True)

# Creating a dataframe for percent returns
data_pct = data_close.pct_change()
# drop the NaN values
data_pct.drop(index=data_pct.index[0], inplace=True)

# Add the market average to the data_pct
data_pct['Average'] = market_close_pct['Average']


# Determining the correlation between the market and each stock, with the lowest correlation being the best
# Initialization
corr = 0
data_corr_score = pd.DataFrame()

best_score_so_far = 0
worst_score_so_far = 1

best_stock_so_far = ""
worst_stock_so_far = ""

# Loop through all columns, except for the 'Average' column
for col in data_pct.columns:
  if col != 'Average':
    corr = data_pct['Average'].corr(data_pct[col])

    # correlation is adjusted to a scale of 0 to 1, then subtracted by 1
    # since we want the lowest correlation in order to beat the market
    corr_score = (1-((corr + 1)/2))
    data_corr_score.at[0, col] = corr_score

    # Keep track of the best and worst stocks for graphing purposes
    if corr_score >= best_score_so_far:
      best_stock_so_far = col
      best_score_so_far = corr_score

    if corr_score <= worst_score_so_far:
      worst_stock_so_far = col
      worst_score_so_far = corr_score


# We can use best_stock_so_far, worst_stock_so_far to graph the data

# Convert to datatime
data_close_dt = data_pct
data_close_dt.index = pd.to_datetime(data_close_dt.index)

# Convert to weekly data, so it fits into a graph, allowing for easier analysis
weekly_pct = data_close_dt.resample('W').mean()
weekly_pct = weekly_pct * 100 # changing to percentage


# Size
plt.figure(figsize=(20, 6))

# Create Plot
plt.plot(weekly_pct.index,weekly_pct[best_stock_so_far], color='g', label="Best Stock: "+best_stock_so_far)
plt.plot(weekly_pct.index,weekly_pct[worst_stock_so_far], color='r', label="Worst Stock: " +worst_stock_so_far)
plt.plot(weekly_pct.index,weekly_pct['Average'], color='y', label= "Market Percent Change: S&P500 + TSX")

#Labels and Title
plt.legend(loc='best')
plt.title('Best Stock V.S Worst Stock Correlation Scores Against Market Average from '+ start_date + ' to '+ end_date)
plt.xlabel('Date')
plt.ylabel('Weekly Percent Change (%)')

# Output
display(data_corr_score)
plt.show()


# Samuel's Algorithm to calculate IVs.

# Produces the strike prices for 3 options as detailed above in the markdown.
def get_strike_prices(ticker, underlying_price, expiration_date, is_call):
    # Separate calls and puts.
    if is_call:
        options = ticker.option_chain(expiration_date).calls
    else:
        options = ticker.option_chain(expiration_date).puts

    # Take the option's strike prices.
    strike_prices = options["strike"].tolist()
    strike_prices.sort()
    min_list = []

    # Take the distance between each strike and the underlying stock's price to find
    #   the option closest to being at the money.
    for strike_price in strike_prices:
        min_list.append(abs(underlying_price - strike_price))

    position = min_list.index(min(min_list))
    atm_strike = strike_prices[position]

    # We take the options one price level above and below our ATM option.
    below_atm_strike = strike_prices[position - 1]
    above_atm_strike = strike_prices[position + 1]

    return [below_atm_strike, atm_strike, above_atm_strike]

# Produces a DataFrame that scores each stock's IV data.
def get_iv_scores(ticker_list):
    data_iv_score = pd.DataFrame()

    call_iv = []
    put_iv = []

    # Loop through each ticker and get the options data for each strike price we get from
    #   the get_strike_prices function.
    for ticker_symbol in ticker_list:
        expiration_date = "2024-11-29"
        ticker = yf.Ticker(ticker_symbol)

        # If the stock doesn't have options data, give it a None value
        if ticker.options:
            last_quote = ticker.info.get("currentPrice")

            # If the stock doesn't have any options expiring on our optimal date we take the
            #   earliest expiration.
            if expiration_date not in ticker.options:
                expiration_date = ticker.options[0]

            call_strikes = get_strike_prices(ticker, last_quote, expiration_date, is_call=True)
            put_strikes = get_strike_prices(ticker, last_quote, expiration_date, is_call=False)
            options = ticker.option_chain(expiration_date)

            for call_strike, put_strike in zip(call_strikes, put_strikes):
                cur_iv = options.calls[options.calls['strike'] == call_strike]["impliedVolatility"]
                call_iv.append(cur_iv)

                cur_iv = options.puts[options.puts['strike'] == put_strike]["impliedVolatility"]
                put_iv.append(cur_iv)

            # Take the average of our IVs and add it into our DataFrame.
            avg_iv = np.mean(call_iv + put_iv)
            data_iv_score.loc[0, ticker_symbol] = avg_iv

        else:
            data_iv_score.loc[0, ticker_symbol] = None

    # Use min-max normalization to set the values to be in between 0 and 1, 0 being the
    # worst performing stock and 1 being the highest.
    min_score = data_iv_score.iloc[0].min()
    max_score = data_iv_score.iloc[0].max()

    data_iv_score = (data_iv_score - min_score) / (max_score - min_score)

    return data_iv_score


data_iv_score = get_iv_scores(data_filter)
data_iv_score


# Samuel's Algorithm to produce a score for each stock based on how their technicals perform in the strategy outlined above.
def get_ema_scores():
    data_ema_score = pd.DataFrame()
    ticker_hist = pd.DataFrame()

    # For every valid ticker, get the historical data and construct the technical indicators.
    for ticker_symbol in data_close.columns.values.tolist():
        ticker_hist["Close"] = data_close.loc[:, ticker_symbol]

        ticker_hist['EMA_12'] = ticker_hist["Close"].ewm(span=12).mean()
        ticker_hist['EMA_26'] = ticker_hist["Close"].ewm(span=26).mean()
        ticker_hist["MACD"] = ticker_hist['EMA_12'] - ticker_hist['EMA_26']
        ticker_hist["EMA_200"] = ticker_hist["Close"].ewm(span=200).mean()

        # Consider the last 10 trading days to target stocks in a recent upturn
        ticker_hist_last_10 = ticker_hist.iloc[-10:].copy()
        for date in ticker_hist_last_10.index.tolist():
            # Calculate the score by adding the 200 day EMA to the product of 200 day EMA and MACD, keeping into
            #   account the negatives.
            ema_200_diff = (ticker_hist.loc[date, "Close"] - ticker_hist.loc[date, "EMA_200"])

            if ticker_hist.loc[date, "MACD"] < 0 and ema_200_diff < 0:
                ticker_ema_score = (ema_200_diff - (ema_200_diff * ticker_hist.loc[date, "MACD"])) / ticker_hist.loc[date, "Close"]
            else:
                ticker_ema_score = (ema_200_diff + ema_200_diff * ticker_hist.loc[date, "MACD"]) / ticker_hist.loc[date, "Close"]

            data_ema_score.loc[0, ticker_symbol] = ticker_ema_score

    # Standardize the data to between 0 and 1 using min-max again.
    min_score = data_ema_score.iloc[0].min()
    max_score = data_ema_score.iloc[0].max()

    data_ema_score = (data_ema_score - min_score) / (max_score - min_score)

    return data_ema_score

data_ema_score = get_ema_scores()
data_ema_score


# Aiden's Algorithm to produce performance scores for each stock based on the most optimal sharpe ratios
def get_sharp_scores(stocks):

    starting_cash = 1000000
    risk_free_rate = 0.04
    num_tested_portfolios = 1000000  # set the number of random portfolios to be tested high(one million) to improve the accuracy of random sampling by CLT
    # Central Limit Theorem(CLT) says that the sampling distribution of the mean will always be normally distributed, as long as the sample size is large enough

    # retrieve closing datas for all stocks
    stock_data = yf.download(stocks, start=start_date, end=end_date)['Close']

    # NEW CODE: filter only the stocks with valid data
    stocks = stock_data.columns.tolist()

    # create portfolios for individual stocks & calculate their sharpe ratios to measure individual performances
    individual_portfolios = (starting_cash / stock_data.iloc[0]) * stock_data
    individual_pct_return = ((individual_portfolios - individual_portfolios.iloc[0]) /  individual_portfolios.iloc[0]) * 100
    individual_exp_return = individual_pct_return.mean(axis=0)
    individual_std_devs = individual_pct_return.std(axis=0)
    individual_sharpe_ratios = (individual_exp_return - risk_free_rate) / individual_std_devs

    # generate random weights for each stocks which add up to 1(100%)
    weights = np.random.random((num_tested_portfolios, len(stocks)))
    weights /= weights.sum(axis=1)[:, np.newaxis]

    # using individual portfolios and random weights to generate test portfolios & calculate their sharpe ratios
    combined_portfolios = individual_portfolios.dot(weights.T)
    combined_pct_returns = ((combined_portfolios - combined_portfolios.iloc[0]) / combined_portfolios.iloc[0]) * 100
    combined_expected_returns = combined_pct_returns.mean(axis=0)
    combined_std_devs = combined_pct_returns.std(axis=0)
    combined_sharpe_ratios = (combined_expected_returns - risk_free_rate) / combined_std_devs

    # find the combined portfolio with the best sharpe ratio & store weights of each stocks for the highest ratio
    best_idx = combined_sharpe_ratios.argmax()
    best_weights = weights[best_idx] * 100
    best_sharpe = combined_sharpe_ratios[best_idx]

    # using individual portfolios' sharpe ratios and weightings of stocks in the
    # combined portfolio with highest sharpe ratio to get each stock's overall "score"
    performance_scores = pd.Series(best_weights, index=stocks) * (pd.Series(individual_sharpe_ratios, index=stocks))

    # standardize the performance scores between 0 and 1
    min_score = performance_scores.min()
    max_score = performance_scores.max()
    standardized_scores = (performance_scores - min_score) / (max_score - min_score)

    # create a df with standardized scores
    optimal_weights_df = pd.DataFrame([standardized_scores], columns=stocks)
    optimal_weights_df.index = [0]

    return optimal_weights_df # return the final df

data_sharpe_score = get_sharp_scores(data_filter)


# Vincent's Alogrithm for adding the weightings together and output

# Weights of each indicator are as follows, adding up to 100%
options_weight = 0.2
corr_weight = 0.4
ema_weight = 0.2
sharpe_weight = 0.2

# New dataframe constants for our indicators, improving readability
indicator_options = data_iv_score
indicator_corr = data_corr_score
indicator_ema = data_ema_score
indicator_sharpe = data_sharpe_score
final_scores = pd.DataFrame()

# Apply the weights to each indicator

# Since some stocks do not have options, in the case where a stock does not have data for options, the weight will
# be transferred to the correlation
corr_weight_no_options = corr_weight + options_weight

for col in indicator_options.columns:
  if np.isnan(indicator_options.loc[0, col]):
    final_scores.loc[0,col]= indicator_corr.loc[0,col] * corr_weight_no_options

  else:
    final_scores.loc[0,col]= indicator_options.loc[0,col] * options_weight + indicator_corr.loc[0,col] * corr_weight

# Finally, add the weight adjusted indicators from EMA and sharpe ratio
final_scores = final_scores + (indicator_ema * ema_weight) + (indicator_sharpe * sharpe_weight)

# Find the best 12 stocks to invest in, ordered from best to worst
best_twelve_stocks = final_scores.iloc[0].nlargest(12).index.tolist()

# Using yfinance to find exchange data
exchange = yf.Ticker('USDCAD=x')
exchange_data = exchange.history(start="2024-11-01",end="2024-11-23")
exchange_data.index = exchange_data.index.strftime('%Y-%m-%d')
exchange_data = exchange_data['Close']


# We will choose 12 stocks, the minimum, for the highest potential returns
# 15% of our investment will be put into the 4 highest ranked stocks in final_scores, with 5% into the rest
initial_funds = 1000000 # CAD
current_ticker = ""
current_currency = ""
current_close = 0
current_investment = 0
current_weight = 0
current_fee =0
fixed_fee = 3.95 # CAD
fee_per_share = 0.001 # CAD
shares_bought = 0
stock_price = 0
purchase_date = "2024-11-22"
Portfolio_Final = pd.DataFrame(columns=["Ticker", "Price", "Currency", "Shares", "Value", "Weight"],index=range(1, 13))


# Loop repeats for each stock, applying greater weighting to the first 4 only
for i in range(0,len(best_twelve_stocks)):
  if i <= 3:
    current_weight = 0.15
  else:
    current_weight = 0.05
  current_investment = current_weight * initial_funds

  # Find the current ticker in the loop and scrape the data
  current_ticker = best_twelve_stocks[i]
  ticker_data = yf.Ticker(current_ticker)

  # Changing currency to CAD if in USD
  current_close = data_close.loc[purchase_date, current_ticker]
  current_currency = ticker_data.fast_info.currency

  # Convert USD prices to CAD
  if (str(current_currency)) == "USD":
    current_close = current_close * exchange_data.loc[purchase_date]

  # Considers both the flat fee and the variable fee, but chooses the one that yields the most shares
  shares_bought1 = (current_investment - fixed_fee) / current_close
  shares_bought2 = (current_investment)/(current_close+fee_per_share)

  # Determine the actual fee
  if shares_bought1 >= shares_bought2:
    current_fee = fixed_fee
  else:
    current_fee = shares_bought2 * fee_per_share


  shares_bought = max(shares_bought1, shares_bought2)

  # Create the final portfolio with all our information
  Portfolio_Final.loc[i+1] = [current_ticker, current_close, "CAD", shares_bought, current_investment-current_fee, current_weight]

# Output the final portfolio
display(Portfolio_Final)

# Illustrate that the portfolio value is nearly equal to our initial_funds minus the current fees, or near $1,000,000 in our case
print("The total value of our portfolio is: $",round(Portfolio_Final["Value"].sum(),2), " CAD.", sep="")

# Illustrate that the total portfolio weight is nearly 1 or 100%
print("The total weight of our portfolio is: ", round(Portfolio_Final["Weight"].sum(),2), " or ", round(Portfolio_Final["Weight"].sum()*100,2) ,"%", sep="")


# Stocks_Final created for final CSV export
Stocks_Final = Portfolio_Final[["Ticker", "Shares"]]
print("Stocks_Final:")
display(Stocks_Final)

# Output to CSV file in the same folder the current .ipynb one is in
Stocks_Final.to_csv('Stocks_Group_15.csv')
print("A CSV file titled 'Stocks_Group_15.csv' has been created.")