# Non-parametric, arbitrage free, robust option implied densities for bitcoin
This module computes option implied densities for bitcoin options that are traded on Deribit (for now I only support optiondata from optionsDX). Notably, this module aims to compute pdfs with the following desirable qualities. (1) Non-parametric (2) Arbitrage free (3) Robust to market noise, and 'realistic' (4) contains tails and (5) uses all the data.
I've combined a blend of techniques from the literature, as well as a few of my own invention to create a program that calcualtes a time series of pdfs, without any arbitrary parameterizations from the user. 

## Description
The primary component of this module is the class "Option_Market". The Option_Market class is initialized with variables: 

(1) option_data_prefix, name of csv files from OptionsDX, end of day qoutes free from: https://www.optionsdx.com/product/btc-option-chains/

(2) par_yield_curve_prefix, name of csv files from: https://home.treasury.gov/interest-rates-data-csv-archive  

(3) stock_price_data_prefix. 
## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation
Follow these steps to install the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/username/project.git
   ```
## Usage

1. **Import functions and initialize `option_market`**
   ```python
   from bitcoin_pdf import *
   option_data_prefix = "btc_eo"
   stock_price_data_prefix = 'bitcoin_10'
   par_yield_curve_prefix = 'daily-'
   option_market = OptionMarket(option_data_prefix, par_yield_curve_prefix, stock_price_data_prefix)
   
2. **Set master dataframe, and specifiy final cleaning parameters**
   ```python
   ###We keep all options with a volume greater than -1 (keeps all data). Keeps all options
   ###that expire in between 85 and 90 days inclusive. 
   option_market.set_master_dataframe(volume=-1, maturity=[85, 90], rolling=False)
3. **Choose procedure to calculate option implied density of every date-expiration slice in option_market**
   ```python
   results=option_market.compute_option_pdfs(bw_setting="cv_ml",kde_setting="ISJ",kde_scale=1,plotting=True,truncate=False,plot_raw=True)



