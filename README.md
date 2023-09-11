# Intraday patterns of the price impact coefficient in the Bitcoin market
We use Binance data for Bitcoin analysis. We followed the methods proposed by Cont et al. (2014) - The Price Impact of Order Book Events. We aimed to gain insights into how the price impact coefficient of Bitcoin, a key metric in understanding market dynamics, varies throughout the trading day. Our findings enlighten how different times of the day may have distinct levels of liquidity, trading volume, and price sensitivity in the Bitcoin market. The project includes functions for loading and processing data, performing statistical analysis, and visualizing the results. It aims to extract valuable insights from financial data, such as order flow imbalance, beta coefficients, and average depth of the order book.

For the purposes of this project, we propose working with already processed data from '2020-11-15' to '2020-11-30'. Therefore, only the functions from Analysis.py file are used in this project, but we also provide functions for the processing stage (DataPreparation). The data is resampled at 10-second intervals to capture intraday patterns effectively.

## How to run the project
Copy the repository, run the Presentation.ipynb and observe numerical and graphical results.
