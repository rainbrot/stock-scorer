import pandas as pd
import torch
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import warnings
import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer


warnings.simplefilter(action="ignore", category=FutureWarning)

base_date = datetime.datetime.now().date()
insider_url = f"https://www.capitoltrades.com/issuers?sortBy=-countPoliticians&pageSize=96&txDate={base_date - datetime.timedelta(weeks=8)}%2C{base_date}"

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


"""
 gbm_monte function is Based on https://github.com/bottama/stochastic-asset-pricing-in-continuous-time by Matteo Bottacini, but with a few key changes:
   - Migrated to GPU for increased calculation speed/efficiency 
   - Output has been changed from a graph to simply the future price and likelihood of certain events (either win, lose, or staying the same within a ~1% margin)
   - Old repo was from 2021 and used Python 3.8, leading it to be broken and unusable in 2025. I tweaked a few key things so that it works in Python 3.11 without issue
   Regardless, thanks Matteo!
"""


def gbm_monte(
    stock_name,
    start_date=base_date - datetime.timedelta(days=(365 * 10)),
    end_date=None,
    scen_size=25000,
    pred_start_date=base_date,
    pred_end_date=base_date + datetime.timedelta(weeks=2),
    iterations=5,
):
    end_date = pred_start_date
    def run_iteration():
        prices = yf.download(
            tickers=stock_name, start=start_date, end=base_date, progress=False
        )["Close"]
        train_set = prices.loc[:end_date]
        daily_returns = ((train_set / train_set.shift(1)) - 1)[1:]
        So = torch.tensor(train_set.iloc[-1], device=device, dtype=torch.float32)

        dt = 1
        n_of_wkdays = (
            pd.date_range(
                start=pd.to_datetime(pred_start_date, format="%Y-%m-%d")
                + pd.Timedelta("1 days"),
                end=pd.to_datetime(pred_end_date, format="%Y-%m-%d"),
            )
            .to_series()
            .map(lambda x: 1 if x.isoweekday() in range(1, 6) else 0)
            .sum()
        )
        T = n_of_wkdays
        N = T / dt
        t = torch.arange(1, int(N) + 1, device=device, dtype=torch.float32)
        mu = torch.tensor(daily_returns.mean(), device=device, dtype=torch.float32)
        sigma = torch.tensor(daily_returns.std(), device=device, dtype=torch.float32)
        b = torch.randn(scen_size, int(N), device=device, dtype=torch.float32)
        W = b.cumsum(axis=1)
        drift = (mu - 0.5 * sigma**2) * t
        diffusion = sigma * W
        S = So * torch.exp(drift + diffusion)

        S_flat = S.flatten()
        total_elements = S_flat.numel()
        winners = (S_flat > (So * 1.01)).sum().item() / total_elements
        losers = (S_flat < (So * 0.99)).sum().item() / total_elements
        same = 1 - winners - losers

        S = torch.cat(
            (
                torch.full(
                    (scen_size, 1), So.item(), device=device, dtype=torch.float32
                ),
                S,
            ),
            1,
        )
        S_max = torch.max(S, 0)[0]
        S_min = torch.min(S, 0)[0]
        S_pred = 0.5 * S_max + 0.5 * S_min
        pred_dates = pd.date_range(start=pred_start_date, end=pred_end_date)

        final_df = pd.DataFrame(data=[S_pred.cpu().numpy()], index=["pred"]).T
        final_df.index = pred_dates[: len(S_pred)]
        return {
            "predicted_value": final_df["pred"].to_list()[-1:][0],
            "win": winners,
            "lose": losers,
            "same": same,
            "current_price": float(So),
        }

    chances = {
        "predicted_value": [],
        "win": [],
        "lose": [],
        "same": [],
        "current_price": [],
    }
    iterations = range(0, iterations)
    for i in iterations:
        metrics = run_iteration()
        [chances[key].append(value) for key, value in list(metrics.items())]
    output = {key: np.mean(value) for key, value in chances.items()}
    return output


"""
 This function does the following:
    - Scrapes capitoltrades.com in order to gain info on how much insider trading has been done with a stock in the last 2 months
    - Isolates the ticker and how many politicians are currently insider trading with it
    - Returns the data
"""


def insider_trades():
    raw_html = requests.get(
        insider_url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
        },
    ).text
    html = BeautifulSoup(raw_html, "html.parser")
    raw_trades = html.find_all("span", {"class": "issuer-ticker"})
    stocks = []
    for trade in raw_trades:
        if (
            len(trade.text) >= 1
            and not "BTC" in trade.text
            and not "BHC" in trade.text
            and not "ETH" in trade.text
        ):
            ticker = trade.text.replace(":US", "").replace("$", "").replace("/", "-")
            popularity = list(trade.parent.parent.parent.parent.parent.parent.children)[
                4
            ].text  # Messy, but I found it to be simpler this way
            if int(popularity) >= 3:
                stocks.append({"ticker": ticker, "popularity": popularity})
    return stocks


"""
 This function does the following:
    - Scrapes finviz.com to gain the latest news headlines on a stock
    - Uses the ntlk VADER model to do sentiment analysis
    - Gets the average score and returns it
"""


def sentiment_analysis(ticker):
    url = f"https://www.finviz.com/quote.ashx?t={ticker.upper()}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None
    html = BeautifulSoup(response.text, "html.parser")
    news_headlines = html.find_all("a", class_="tab-link-news")
    if not news_headlines:
        print(f"No news headlines found for {ticker}.")
        return None

    analyzer = SentimentIntensityAnalyzer()
    scores = []
    parsed_news = []

    for headline in news_headlines:
        text = headline.text
        # Get sentiment polarity scores
        sentiment = analyzer.polarity_scores(text)
        scores.append(sentiment["compound"])
        parsed_news.append(
            [
                text,
                sentiment["neg"],
                sentiment["neu"],
                sentiment["pos"],
                sentiment["compound"],
            ]
        )

    average_score = sum(scores) / len(scores) if scores else 0
    return average_score


tickerlist = insider_trades()
scores = []
for stockinfo in tickerlist:
    gbm = gbm_monte(stock_name=stockinfo["ticker"])
    data = {**stockinfo, **gbm, "sentiment": sentiment_analysis(stockinfo["ticker"])}
    weights = {"popularity": 0.15, "outlook": 0.30, "sentiment": 0.30, "value": 0.25}
    score = 10 * (
        (weights["popularity"] * (float(data["popularity"]) / 10))
        + (weights["outlook"] * (data["win"] - data["lose"]))
        + (weights["sentiment"] * data["sentiment"])
        + (
            weights["value"]
            * (
                (data["predicted_value"] - data["current_price"])
                / data["current_price"]
            )
        )
    )
    print(
        f"{ stockinfo['ticker'] } ({tickerlist.index(stockinfo)+1}/{len(tickerlist)} Done)"
    )
    scores.append({"ticker": stockinfo["ticker"], "score": score})
scores = sorted(scores, key=lambda x: (1 - x["score"]))
print("\n")
for i, item in enumerate(scores, 1):
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(i if i < 20 else i % 10, "th")
    print(f"{i}{suffix} Place: {item['ticker']} with a score of {item['score']:.2f}")
