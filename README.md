Periods of consolidation in the stock market are times where there is indecisivness amongst traders overs a stock's price. Very often this will lead a breakout to the upside or downside. The goal of this project is to predict these breakouts for big and medium market capitalization stocks from NYSE and NASDAQ using only technical analysis. 

The raw data was filtered by taking tickers that had an average volume of over 500,000  and average price of over 30$ since 2010. In order to generate features, TTM squeeze indicators were identified. This occurs when the [Bollinger Bands](https://www.investopedia.com/terms/b/bollingerbands.asp) fall within the [Keltner Channels](https://www.investopedia.com/terms/k/keltnerchannel.asp). 

A `set_up_day` is defined as the first day of the squeeze. A feature is 30 days  leading up to a `set_up_day`.  Multiple features within one ticker are identified such that the last day of the squeeze is atleast 30 days before the next `set_up_day`. 

The lables are made by taking 21 days after the `set_up_day` and perform linear regression. If the correlation coefficient (r value) is greater than 0.6 the label a BUY. If it is less than -0.5, thelabel is a SELL. if it is between -0.5 and 0.6 label this as ambiguous and discarded. 

![enter image description here](https://raw.githubusercontent.com/HanadS/ilaf_algos/master/imgs/AAPL_set3_08-01-2018.png?token=AGJBZXFKUAACPVQSHRBDCTS63EO2M)
![enter image description here](https://raw.githubusercontent.com/HanadS/ilaf_algos/master/imgs/AAPL_set4_11-02-2018.png?token=AGJBZXALITNXG63SESSNTVC63EPDE)

Once the features and labels were generated, an SVM, Logistic Regression and LSTM were trained to classify BUY and SELL. The SVM and logisitc Regression were trained by a grid search over a parameter grid. The LSTM was trained with several lstm cells, a nonlinear layer and e cyclic learning rate scheduler by [L.Smith](https://arxiv.org/abs/1506.01186). 


