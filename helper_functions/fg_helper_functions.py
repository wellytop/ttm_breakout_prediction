import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
from finta import TA
from sklearn import linear_model

from bs4 import BeautifulSoup
import re
from requests import get
import requests


regr = linear_model.LinearRegression()


def bbands(df, window_size=20, num_of_std=2):

    """
        Calculate the Bollinger Bands®
    """
    price = df["close"]
    rolling_mean = pd.Series(
        price.rolling(window=window_size).mean(), name="Rolling_Mean"
    )
    rolling_std = price.rolling(window=window_size).std()
    upper_band = pd.Series(rolling_mean + (rolling_std * num_of_std), name="bb_upper")
    lower_band = pd.Series(rolling_mean - (rolling_std * num_of_std), name="bb_lower")

    df = df.join(rolling_mean, lsuffix="_left")
    df = df.join(upper_band, lsuffix="_left")
    df = df.join(lower_band, lsuffix="_left")

    return df


def save_figure(stock_target, name, label):

    """
        Create figure for each set up day.
    """

    fig = plt.figure(figsize=(16, 10))

    close = stock_target["close"]
    lower_band = stock_target["bb_lower"]
    kc_lower = stock_target["kc_lower"]
    day_diff = stock_target.index
    upper_band = stock_target["bb_upper"]
    kc_upper = stock_target["kc_upper"]

    plt.plot(close, color="black", linewidth=3, label="Closing Price")
    plt.plot(lower_band, color="red", linestyle="dashed", label="Keltner Channels")
    plt.plot(upper_band, color="red", linestyle="dashed", label="_nolegend_")

    plt.plot(kc_lower, color="blue", linestyle="dashed", label="Bollinger Bands")
    plt.plot(kc_upper, color="blue", linestyle="dashed", label="_nolegend_ ")

    plt.title(name, fontsize=20)

    plt.xlabel("Days Before Squeeze", fontsize=18)
    plt.ylabel("Closing Price (USD)", fontsize=18)

    plt.xticks(np.arange(day_diff[0], day_diff[-1], step=15), fontsize=16)
    plt.yticks(fontsize=16)

    plt.axvline(x=0, color="green", label="TTM Squeeze")

    plt.grid()
    plt.legend(loc="best")

    if label == "BUY":
        plt.text(
            0.95,
            0.01,
            "BUY",
            style="oblique",
            verticalalignment="bottom",
            horizontalalignment="right",
            color="green",
            fontsize=25,
            transform=plt.gcf().transFigure,
        )

    elif label == "SELL":
        plt.text(
            0.95,
            0.01,
            "SELL",
            style="oblique",
            verticalalignment="bottom",
            horizontalalignment="right",
            color="Red",
            fontsize=25,
            transform=plt.gcf().transFigure,
        )
    else:
        plt.text(
            0.95,
            0.01,
            "ambiguous",
            style="italic",
            verticalalignment="bottom",
            horizontalalignment="right",
            color="black",
            fontsize=22,
            transform=plt.gcf().transFigure,
        )

    fname = os.getcwd() + "/stock_png_plot/" + name + ".png"

    plt.savefig(fname)
    plt.close()

    return plt


def days_difference(stock_df, set_up_day):

    """
        Get the days relative to the set up days
    """
    day_diff = []
    for i, r in stock_df.iterrows():
        days = i - set_up_day
        day_diff.append(days.days)
    stock_df["days_difference"] = day_diff
    stock_df.set_index("days_difference", inplace=True)

    return stock_df


def get_ttm_dates(stock):

    """
        Get the dates for a TTM squeeze. Dates must also be within a 30 day high
    """

    closing_price = stock["close"]
    stock["rolling_max"] = pd.Series(
        closing_price.rolling(window=365).max(), name="rolling_max"
    )
    stock_bb = bbands(stock, 20)
    stock_kc = stock_bb
    stock_kc["kc_upper"] = TA.KC(stock_bb, atr_period=20, kc_mult=1.5)["KC_UPPER"]
    stock_kc["kc_lower"] = TA.KC(stock_bb, atr_period=20, kc_mult=1.5)["KC_LOWER"]
    target_dates = []
    for i, r in stock_kc.iterrows():
        if r["bb_lower"] > r["kc_lower"] and r["bb_upper"] < r["kc_upper"]:
            if (
                r["close"] > 0.9 * r["rolling_max"]
                or r["close"] < 1.1 * r["rolling_max"]
            ):
                target_dates.append(i)

    return stock_kc, target_dates


def separate_dates(target_dates):

    """
     This function takes a set of dates in a TTM Squueze for one stock 
     and seperates them based on a 30 day time frame.
    """
    date_sets = dict()
    squeeze_num = 1
    tmp = target_dates[0]
    date_sets["set" + str(squeeze_num)] = []
    for date in target_dates:
        if (date - tmp).days > 30:
            squeeze_num += 1
            date_sets["set" + str(squeeze_num)] = []
        date_sets["set" + str(squeeze_num)].append(date)

        tmp = date

    return date_sets


def get_label(stock_df):

    """
    This function determines the label of the time series based on the 
    trajector of the prices after the set up day

    """

    close_price = stock_df.loc[0:]["close"].values.reshape(-1, 1)
    days_after_setup_day = stock_df.loc[0:].index.values.reshape(-1, 1)

    regr.fit(close_price, days_after_setup_day)
    r_value = regr.coef_[0][0]
    if r_value > 0.6:
        label = "BUY"
    elif r_value < -0.5:
        label = "SELL"
    else:
        label = "ambiguous"

    return label, r_value


def get_filing_dates_quarterly(ticker):
    """
        Find the filing dates for each ticker and return as list.
    """

    URL = "http://www.sec.gov/cgi-bin/browse-edgar?CIK={}&Find=Search&owner=exclude&action=getcompany"
    CIK_RE = re.compile(r".*CIK=(\d{10}).*")
    results = CIK_RE.findall(str(get(URL.format(ticker)).content))
    if len(results):
        cik = results[0]
    else:
        return []
    # base URL for the SEC EDGAR browser
    endpoint = r"https://www.sec.gov/cgi-bin/browse-edgar"

    # define our parameters dictionary
    param_dict = {
        "action": "getcompany",
        "CIK": cik,
        "type": "10-Q",
        "dateb": "20190101",
        "owner": "exclude",
        "start": "",
        "output": "",
        "count": "100",
    }

    # request the url, and then parse the response.
    response = requests.get(url=endpoint, params=param_dict)
    soup = BeautifulSoup(response.content, "html.parser")

    # find the document table with our data
    doc_table = soup.find_all("table", class_="tableFile2")

    # define a base url that will be used for link building.
    base_url_sec = r"https://www.sec.gov"

    filing_date_list = []
    # loop through each row in the table.
    for row in doc_table[0].find_all("tr"):
        # find all the columns
        cols = row.find_all("td")
        # if there are no columns move on to the next row.
        if len(cols) != 0:
            filing_date = cols[3].text.strip()
            filing_date_list.append(filing_date)

    return filing_date_list
