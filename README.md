# stock-scorer

My first attempt at quant, scoring stocks based on congressional insider trading, geometric brownian motion + monte carlo simulations, and basic sentiment analysis. The goal was to build a Python script that systematically evaluates and ranks stocks based on a combination of publicly available data.

## How It Works

The script's logic is built on a rudimentary multi-factor analysis. Instead of relying on a single metric, it gathers data from several sources to form a more holistic view of each stock.

The process for each stock is as follows:

1.  **Insider Trading Analysis:** The script begins by identifying stocks with significant insider trading activity from U.S. politicians. It scrapes [CapitolTrades](https://www.capitoltrades.com) for tickers that have been recently traded by three or more politicians, using this as a proxy for "smart money" interest.

2.  **Price Forecasting with GBM + Monte Carlo:** To gauge potential price movement, the script employs a Geometric Brownian Motion (GBM) model, a common method for simulating stock price paths. This is run through a Monte Carlo simulation with 125,000 scenarios (5 iterations of 25,000) to forecast the price in 2 weeks and calculate the probabilities of three outcomes:
    *   **Win:** The stock price increases by more than 1%.
    *   **Lose:** The stock price decreases by more than 1%.
    *   **Same:** The price stays within a ~1% band of its current value.

    The model is designed to leverage a GPU (PyTorch on CUDA or Apple's MPS) if available.

3.  **News Sentiment Analysis:** As numbers dont tell the whole story, the script also assesses market sentiment by scraping the latest news headlines for each ticker from [Finviz](https://www.finviz.com). It then analyzes these headlines using NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner) to determine if the recent news flow is positive, negative, or neutral.

4.  **The Scoring Model:** Finally, the data from the previous steps are aggregated into a single score. The final score is a weighted average of the different factors:
    *   **Outlook (30%):** The probability of a "win" minus the probability of a "lose" from the Monte Carlo simulation.
    *   **Sentiment (30%):** The average compound sentiment score from the news analysis.
    *   **Value (25%):** The percentage difference between the model's predicted price and the current price.
    *   **Popularity (15%):** The number of politicians trading the stock in the last 2 months.

The script then outputs a ranked list of the analyzed stocks, from highest to lowest score.

The list should tell you which stocks would be ideal to invest in the medium term (next few weeks or months), as it gets recent sentiment & congressional data, which will hopefully be slightly ahead of the curve, that data is the bolstered by the GBM + Monte Carlo model. This is based on my intuitive understanding of the program, and is also supported by my backtesting

## Getting Started

To get the script running on your local machine, you'll need to install the necessary Python libraries:
```sh
pip install pandas torch yfinance numpy matplotlib requests beautifulsoup4 nltk
```

Then run script. It will print its progress as it analyzes each stock and will display the final ranked list in the console.

## Acknowledgments
A special thanks to **Matteo Bottacini**. The Geometric Brownian Motion function in this script is a updated and modified version (Uses torch+GPU processing instead of cpu, works beyond Python 3.8, calculates probability of winning/losing/stagnation), but still based on his original [stochastic-asset-pricing-in-continuous-time](https://github.com/bottama/stochastic-asset-pricing-in-continuous-time) repository from 2021.
