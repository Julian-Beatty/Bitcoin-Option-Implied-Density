# Non-parametric, arbitrage free, robust option implied densities for bitcoin
This module computes option implied densities for bitcoin options that are traded on Deribit (for now I only support optiondata from optionsDX). Notably, this module aims to compute pdfs with the following desirable qualities. (1) Non-parametric (2) Arbitrage free (3) Robust to market noise, and 'realistic' (4) contains tails and (5) uses all the data.
I've combined a blend of techniques from the literature, as well as a few of my own invention to create a program that calcualtes a time series of pdfs, without any arbitrary parameterizations from the user. 
Special thanks to authors of arbitrage repair
Samuel N. Cohen and Christoph Reisinger and Sheng Wang
https://github.com/vicaws/arbitragerepair
## Description
The primary component of this module is the class "Option_Market". The Option_Market class is initialized with variables: 

(1) option_data_prefix, name of csv files from OptionsDX, end of day qoutes free from: https://www.optionsdx.com/product/btc-option-chains/

(2) par_yield_curve_prefix, name of csv files from: https://home.treasury.gov/interest-rates-data-csv-archive  

(3) stock_price_data_prefix. 
## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Results](#Results)
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
   from main_bitcoin_pdf import*
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
   ## Chose your IV interpolation method using setting bw_setting to the following:
   ##"spline"-piece-wise cubic interpolation
   ##"cv_ml" kernell regression with bandwidth selection determined by cross validation maximum likelihood. Recommended, default setting.
   ##"cv_ls" kernel regression with bandwidth selection determined by cross validation least squares
   ## [float] eg [1000] manually set the bandwidth to a fixed number. Not recommended.

   ##Chose your KDE method
   ## Improved Sheather jones "ISJ", fits the "raw" pdf most closely. Recommended, especially for CV_ML.
   ## Scott or silverman "scott", "silvermann", generally produces smooth guassian densities. Can produced very biased results at the cost of lower variance. Not recommended
   ## KDE_scale: divides the bandwidth selected by Scott or silverman by this number. Can be used to encourage the algorithmn to pick a bandwidth between ISJ and scott. If ISJ is selected this argument does nothing. Default to 1
   results=option_market.compute_option_pdfs(bw_setting="cv_ml",kde_setting="ISJ",kde_scale=1,plotting=True,truncate=False,plot_raw=True)
## Results
The black dotted curve represents the raw pdf. Depending on the noise in the market, as well as the smoothness level in the IV interpolation, this may have lots of noise. We use Kernel Density estimation to extract a reasonable density from it. I show below that even when the raw btc density is noisy, we can extract some meaningfully realistic density from it without having to go back and oversmooth the IV curve.
1. **extracting 90 day bitcoin pdfs**
<img width="720" alt="2021_06_26_2021_09_24" src="https://github.com/user-attachments/assets/3134c738-8f4a-41f8-95d6-72c6674a4090">

2. **successful extraction of even "noisy" 90 day bitcoin pdfs**
<img width="720" alt="2021_10_02_2021_12_31" src="https://github.com/user-attachments/assets/439018cc-5e2a-4dbd-8916-97e2b8c5da73">

3. **Extraction of 1 day btc pdf**
<img width="720" alt="2021_06_22_2021_06_23" src="https://github.com/user-attachments/assets/adfbbb3b-2f3a-49a5-ab4b-07a8f29f2dc4">

