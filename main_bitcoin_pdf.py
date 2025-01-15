import os
import pandas as pd
from scipy.interpolate import CubicSpline
import warnings
import random
import py_vollib.black_scholes_merton.implied_volatility
from arbitragerepair import constraints, repair
import matplotlib.pyplot as plt
import pickle
import numpy as np
from statsmodels.nonparametric.kernel_regression import KernelReg
import py_vollib
from scipy.stats import gaussian_kde
from py_vollib_vectorized import implied_volatility
from KDEpy import*
import time  # Import the time module
import sys

class OptionMarket:
    def __init__(self, option_data_prefix, stock_price_data_prefix,spot_curve_df,forward_curve_df):
        """
        Initializes the OptionMarket class with option data, par yield curve, and stock price data.

        Parameters:
        option_data_prefix (str): Prefix for the option data files.
        par_yield_curve_prefix (str): Prefix for the yield curve data files.
        stock_price_data_prefix (str): Prefix for the csv file containing stock price data.
        """
        self.option_data_prefix = option_data_prefix  # Store the prefix for option data files
       # self.par_yield_curve_prefix = par_yield_curve_prefix  # Store par yield curve data
        self.stock_price_data_prefix = stock_price_data_prefix  # Store stock price data
        self.spot_curve_df=spot_curve_df
        self.forward_curve_df=forward_curve_df
        self.option_df = self.load_option_data()  # Load the option data
        #self.yield_curve_df = self.load_fed_sheet()
        self.stock_df = self.load_stock_price_data()

        self.original_market_df = self.merge_option_stock_yields()
        
    def load_option_data(self):
        """
        Loads option data from CSV files in the current directory.

        Returns:
        DataFrame containing combined option data.
        """
        option_data_prefix=self.option_data_prefix
        try:
            current_directory_str = os.getcwd()  # Current working directory

            # List all files starting with the option_data_prefix
            options_csv_files = [f for f in os.listdir(current_directory_str) if f.startswith(option_data_prefix)]

            # Create full paths for the files
            options_filepaths_list = [os.path.join(current_directory_str, f) for f in options_csv_files]

            # Read the files into a DataFrame
            option_df = pd.concat(map(pd.read_csv, options_filepaths_list), ignore_index=True)
            print(f"We have successfully loaded and merged all CSV files beginning with '{option_data_prefix}' in your working directory. Storing in object.")
            return option_df
        except Exception as e:
            print(f"An error occurred while loading option data: {e}. Check that files beginning with {option_data_prefix} are in {current_directory_str}")
    def load_fed_sheet(self):
        fed_sheet_name=self.par_yield_curve_prefix
        fed_sheet=fed_yield_curve_maker(fed_sheet_name)
        
        yield_curve = {}
        group=fed_sheet.groupby(["Date"])
        group.apply(lambda x: yield_curve.update(compute_yields(x,plotting=False)))
        return yield_curve
    
    def load_stock_price_data(self):
        """
        Loads the stock data into a dataframe.
        Returns:
        Dataframe containing the stock price.
        """
        stock_price_data_prefix=self.stock_price_data_prefix
        current_directory_str = os.getcwd()  # Current working directory
            
            # Attempt to load stock data files
        try:
                # List all files starting with the stock_price_data_prefix
            stock_csv_files = [f for f in os.listdir(current_directory_str) if f.startswith(stock_price_data_prefix)]
                
            if not stock_csv_files:
                raise FileNotFoundError(f"No files found starting with '{stock_price_data_prefix}'.")
    
                # Create full paths for the files and read them into a DataFrame
            stock_filepaths_list = [os.path.join(current_directory_str, f) for f in stock_csv_files]
            stock_df = pd.concat(map(pd.read_csv, stock_filepaths_list), ignore_index=True)
            
            stock_header=stock_df.columns
            if any("ate" in column for column in stock_header): #finds date column name
                date_column_name = list(filter(lambda word: "ate" in word, stock_header))
            else:
                raise FileNotFoundError(f"The stock data file must contain a column labeled date. We could not find it")


                # Convert 'dates' column to datetime
            stock_df['Date_column'] = pd.to_datetime(stock_df[date_column_name[0]])
            print("--" * 20)  # Print a line of dashes for separation
            print(f"We have successfully loaded the stock data files beginning with '{stock_price_data_prefix}' in your working directory.")

                # Check for the required 'price' column
            if 'price' not in stock_df.columns:
                raise ValueError("The stock data file must contain a column labeled 'price'.")
            if 'price' in stock_df.columns:
                stock_df=stock_df[['Date_column','price']]
                
                all_dates = pd.date_range(start=stock_df['Date_column'].min(), 
                                          end=stock_df['Date_column'].max(), freq='D')
        
                # Find missing dates (i.e., days that should have data but don't)
                missing_dates = all_dates.difference(stock_df['Date_column'])
        
                # Reindex the DataFrame to include missing dates, then fill with the previous day's yield curve
                stock_df = stock_df.set_index('Date_column').reindex(all_dates).ffill().reset_index()
        
                # Rename index column back to Date
                stock_df = stock_df.rename(columns={'index': 'Date_column'})
            return stock_df
    
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")
            return None  # Return None if there's an error
    
        except Exception as e:
            print(f"An error occurred while loading the stock data: {e}. Check that files beginning with '{stock_price_data_prefix}' are in {current_directory_str}")
            return None  # Return None if there's a different error

    def merge_option_stock_yields(self):
        """
        Merges the stock data and the option dataframe together, and appends the risk free rate, as calibrated from niegel-svenson model to another column beside the maturity.
        Additionally, only keeps OTM options by replacing ITM calls with OTM puts using the put-call parity.
        Returns:
        Dataframe containing options, stock data and risk free rate.
        
        """
        print("--"*20)
        print("We are now merging option, yield curve and stock data")
        option_df=self.load_option_data()
        #stock_df=self.load_stock_price_data()
        #self.stock_df=stock_df
        stock_df=self.stock_df
        spot_curve_df=self.spot_curve_df
        forward_curve_df=self.forward_curve_df

        #yield_curve_df=self.load_par_yield_curve()
        #self.yield_curve=yield_curve_df
        option_df.columns = option_df.columns.str.strip()
        header_list=option_df.columns.tolist()
        
        if "[BASE_CURRENCY]" in header_list:
            source_identification="Options_dx_bitcoin_eod_quotes"
            option_df=clean_btc_optionsdx(option_df,stock_df,spot_curve_df,forward_curve_df)

            print("--" * 20)  # Print a line of dashes for separation
            print("We believe this data is end of day quotes for bitcoin options from Options_DX")
        if "cp_flag" in header_list:
            option_df=clean_optionmetrics(option_df,stock_df,spot_curve_df,forward_curve_df)
            print("Options_metrics")
        
        option_df=option_df.reset_index(drop=True)
        #self.original_market_df=option_df
        return option_df
    def set_master_dataframe(self,volume=-1,maturity=[-1,5000],moneyness=[-1000,1000],rolling=False):
        """
        

        Parameters
        ----------
        volume : float
            Removes all options with volume less than or equal to this number.
        maturity : list
            removes all options with maturity less than maturity[0] or greater than maturity[1]. If list has only 1 entry, only options with maturity exactly equal to maturity[0] are kept. Maturity is in DAYs.
            i.e maturity=[1,2] keeps all options with maturities of 1 or 2 days.
        Rolling: float
            if True Keeps all options with the maturity closest to float.
            
        Returns
        -------
        None.

        """
        original_dataframe=self.original_market_df.copy()
        
        original_dataframe=original_dataframe[original_dataframe["volume"]>volume]
        original_dataframe=original_dataframe[original_dataframe["bid_size"]>volume]
        original_dataframe=original_dataframe[original_dataframe["offer_size"]>volume]

        ###
        original_dataframe=original_dataframe[(original_dataframe["rounded_maturity"]>=maturity[0]/365) & (original_dataframe["rounded_maturity"]<=maturity[1]/365)]
        ##
        original_dataframe=original_dataframe[(original_dataframe["moneyness"]>=moneyness[0]) & (original_dataframe["moneyness"]<=moneyness[1])]
        ##
        if rolling != False:
            date_group=original_dataframe.groupby(["date"])
            original_dataframe = date_group.apply(lambda x: rolling_window(x,rolling))
        
        original_dataframe=original_dataframe.reset_index(drop=True)
        
        self.master_dataframe=original_dataframe
        return None
    def compute_option_pdfs(self,bw_setting="cv_ml",kde_setting="ISJ",kde_scale=1,plot_pdf=True,truncate_pdf=False,plot_raw_pdf=True,foldername="figureplots"):
        """
        Computes option implied PDFs of the master-dataframe.

        Parameters
        ----------
        bw_setting : STR or [float], optional
            Interpolation setting for IV curve: Pick either "spline","cv_ml","cv_ls" or enter a manual number [float]. The default is "cv_ml".
        kde_setting : STR, optional
            BW method for KDE. Pick either "ISJ","scott","silverman. The default is "ISJ".
        kde_scale : float, optional
            Divides silverman or scotts bandwidth by kde_scale. The default is 1.
        plotting : True/False boolean, optional
            Saves plots. The default is True.
        truncate : True/False boolean, optional
            Truncates until CDF is 99.5%, starting from center. The default is False.
        plot_raw : true/False boolean, optional
            Also plots the raw pdf. The default is True.

        Returns
        -------
        result_dict : dict
            dictionary containing the dates/expiration as keys, and as a value, a dataframe containing strike,return and pdf axes.

        """
        option_market=self.master_dataframe
        date_group=option_market.groupby(["date",'exdate'])
        result_dict = {}
        date_group.apply(lambda x: result_dict.update(compute_pdf(x,bw_setting,kde_setting,kde_scale,plotting=plot_pdf,truncate=truncate_pdf,
                                                                  plot_raw=plot_raw_pdf,foldername=foldername)))
        
        return result_dict
        
def clean_btc_optionsdx(option_df,stock_df,spot_curve_df,forward_curve_df):
    """
    Description of function: Merges stock data, yield data and option data into one dataframe, and does basic data cleaning.

    Parameters
    ----------
    option_df : dataframe
        dataframe of options
    stock_df : dataframe
        dataframe of stock prices.
    yield_curve_df : dataframe
        dataframe of yield curve.

    Returns
    option_df, a dataframe containing merged information and cleaning.
    None.

    """
    ###extracting header list and removing brackets
    header_list=option_df.columns.tolist()
    header_list = [entry.replace("[", "").replace("]", "") for entry in header_list]
    option_df.columns=header_list
    
    ##Renaming column headers to a standard convention, and creating date-time values        
    option_df=option_df.loc[:,["QUOTE_DATE","EXPIRY_DATE","ASK_PRICE","BID_PRICE","STRIKE",'MARK_IV',"VOLUME","BID_SIZE","ASK_SIZE","UNDERLYING_INDEX","UNDERLYING_PRICE","OPTION_RIGHT"]]
    option_df.columns=["date","exdate","best_offer","best_bid","strike","mid_iv","volume","bid_size","offer_size","underlying_future","underlying_price","option_right"]
    option_df['option_right'] = option_df['option_right'].str.strip()
    option_df['date'] = option_df['date'].str.strip()
    option_df['exdate'] = option_df['exdate'].str.strip()
    option_df['qoute_dt'] = pd.to_datetime(option_df['date'])# + pd.to_timedelta(8, unit='h') #qoute 8:00UTC
    option_df['expiry_dt'] = pd.to_datetime(option_df['exdate'])# + pd.to_timedelta(22, unit='h') #Expires 10PM central
            
    ###Scaling mid_iv to decimal
    option_df["mid_iv"]=option_df["mid_iv"]/100        
    ####Pasting stock data into option frame, and converts the options from BTC into US dollars.
    date_group=option_df.groupby(["date"])
    option_df = option_df.merge(stock_df, how='left', left_on='qoute_dt', right_on='Date_column')
    option_df = option_df.rename(columns={'price': 'stock_price'})
    
    #Adjusting datetimevalues

    
    option_df["best_offer"]=option_df["best_offer"]*option_df['stock_price']
    option_df["best_bid"]=option_df["best_bid"]*option_df['stock_price']
            
    ####Addressing deribit future issues See helper function
    date_group=option_df.groupby(['date','exdate'])
    print("--" * 20)  # Print a line of dashes for separation
    print("We are averaging bitcoin futures")    
    option_df=date_group.apply(lambda x: average_futures(x))
    option_df=option_df.reset_index(drop=True)

    ####Cleaning volume. Any missing data is replaced with zero. Leading space removed.
    option_df["volume"]=option_df["volume"].astype(str)
    option_df['volume'] =option_df['volume'].replace(' ', "0")
    option_df["volume"] =option_df["volume"].str.lstrip()
    option_df["volume"]=option_df["volume"].astype(float)
    option_df=option_df.reset_index(drop=True)
        
    #### Converting "calls" and "puts" to "c" and "p" respectively.
    option_df['option_right'] = option_df['option_right'].replace({'call': 'c', 'put': 'p'})
###############################################################General Cleaning

    ###Creating maturity in year fraction. For using black scholes formula I replace 0 with 0.001 for numerical reasons.
    option_df['maturity']= ((pd.to_datetime(option_df['exdate']) - pd.to_datetime(option_df['date'])).dt.days)/365
    option_df['maturity']=option_df['maturity']+(20/24)/365 #expires 10pm, qouted 2am.
    #option_df['maturity'] =option_df['maturity'].replace(0, 0.001)
    ##rounded maturity for grouping purposes
    option_df['rounded_maturity']= ((pd.to_datetime(option_df['exdate']) - pd.to_datetime(option_df['date'])).dt.days)/365
    ###Creating mid price. Removing any instance in which the mid price is negative
    option_df['mid_price']=(option_df['best_bid']+option_df['best_offer'])/2
    option_df=option_df[option_df['mid_price']>0]
    ##Creating Log-moneyness column
    option_df.loc[option_df['option_right']=='p','log moneyness'] = np.log(option_df['strike']/option_df['stock_price'])
    option_df.loc[option_df['option_right']=='c','log moneyness'] = np.log(option_df['stock_price']/option_df['strike'])
    
    ###################################################### Yield Curve Interpolation with Cubic Splines
    #######
    #option_df["risk_free_rate"]=0
    #option_df = attach_risk_free_rate(option_df, yield_curve)
    option_df=pasting_yields_option_df(option_df, spot_curve_df, "risk_free_rate")
    option_df=pasting_yields_option_df(option_df, forward_curve_df, "forward_risk_free_rate")
    ######
    # ##merging yield curve onto main option dataframe
    # option_df = option_df.merge(yield_curve_df, how='left', left_on='qoute_dt', right_on='Date')
    # option_df.drop(columns=['Date'], inplace=True)
    # date_group=option_df.groupby(['date'])
    # yield_curve_headers=yield_curve_df.columns.tolist()[1:]
    
    # print("--" * 20)  # Print a line of dashes for separation
    # print("We are interpolating the yield curve to match your options. This may take some time.")
    # ###Performing Yield Curve interpolation
    # option_df=option_df.reset_index(drop=True)
    # option_df=date_group.apply(lambda x: paste_rate(x,yield_curve_headers)) 
    # print("We are done interpolating Yield curve")
    # option_df=option_df.reset_index(drop=True)
    
    
    ################################################   Use only OTM options. Converts ITM puts to (OTM) calls via put-call parity. Remove ITM calls. 
    print("--" * 20)  # Print a line of dashes for separation
    print("Converting puts to calls. If both are present we use only OTM options.")
    date_group=option_df.groupby(['date','exdate'])
    option_df=date_group.apply(lambda x: use_only_OTM(x))
    option_df=option_df.reset_index(drop=True)

    ##############################################  Aggregates duplicate option prices if they exist
    #Aggregates duplicates mid prices *See remove_duplicates function
    
    date_group=option_df.groupby(['date','exdate'])
    print("--" * 20)  # Print a line of dashes for separation
    print("We are Aggregating duplicate option entries")    
    option_df=date_group.apply(lambda x: remove_duplicates(x))
    option_df=option_df.reset_index(drop=True)
    option_df['option_right'] = option_df['option_right'].replace({'call parity': 'c'})
    ###
    option_df["moneyness"]=option_df["strike"]/option_df['stock_price']-1
    
    ####keep only neccessary columns
    option_df=option_df[["date","exdate","maturity","rounded_maturity","risk_free_rate","forward_risk_free_rate","strike","best_bid","mid_price","best_offer","mid_iv","underlying_price","stock_price",
                        "volume","offer_size","bid_size","option_right","moneyness","qoute_dt","expiry_dt"]]
    option_df=option_df.reset_index(drop=True)
    ##Computes BS IV

    
    return option_df
def clean_optionmetrics(option_df,stock_df,spot_curve_df,forward_curve_df):
    """
    Description of function: To clean other option_metrics data. Coming soon.

    Parameters
    ----------
    option_df : TYPE
        DESCRIPTION.
    stock_df : TYPE
        DESCRIPTION.
    yield_curve_df : TYPE
        DESCRIPTION.

    Yields
    ------
    option_df : TYPE
        DESCRIPTION.

    """
    header_list=option_df.columns.tolist()
    try:
        option_df=option_df[["date","exdate","best_offer","best_bid","strike_price","cp_flag","volume"]]
    except Exception as e:
        print("Missing information. Check file has columns labeled,date,exdate,cp_flag,best_bid,best_offer")
    
    ###Renaming Columns
    option_df.columns=["date","exdate","best_offer","best_bid","strike","option_right","volume"]
    
    ##Quotes
    option_df['qoute_dt'] =pd.to_datetime(option_df['date'])
    option_df['expiry_dt'] =pd.to_datetime(option_df['exdate'])
    option_df = option_df.merge(stock_df, how='left', left_on='qoute_dt', right_on='Date_column')
    option_df = option_df.rename(columns={'price': 'stock_price'})
    option_df = option_df.sort_values(by=['qoute_dt', 'expiry_dt', 'strike'], ascending=[True, True, True])

    ###Strike
    option_df["strike"]=option_df["strike"]/1000
    #### Converting "calls" and "puts" to "c" and "p" respectively.
    option_df['option_right'] = option_df['option_right'].replace({'C': 'c', 'P': 'p'})

    ###Creating maturity in year fraction. For using black scholes formula I replace 0 with 0.001 for numerical reasons.
    option_df['maturity']= ((pd.to_datetime(option_df['exdate']) - pd.to_datetime(option_df['date'])).dt.days)/365
    option_df['maturity'] =option_df['maturity'].replace(0, 0.001)
    option_df['maturity']=option_df['maturity']+(15/24)/365 #expires 10pm, qouted 2am.
    option_df['rounded_maturity']= ((pd.to_datetime(option_df['exdate']) - pd.to_datetime(option_df['date'])).dt.days)/365

    ###Creating mid price. Removing any instance in which the mid price is negative
    option_df['mid_price']=(option_df['best_bid']+option_df['best_offer'])/2
    option_df=option_df[option_df['mid_price']>0]
    
    ##Creating Log-moneyness column
    option_df.loc[option_df['option_right']=='p','log moneyness'] = np.log(option_df['strike']/option_df['stock_price'])
    option_df.loc[option_df['option_right']=='c','log moneyness'] = np.log(option_df['stock_price']/option_df['strike'])
    #option_df=option_df[option_df["mid_price"]>0.5]
    ###
    
    
    option_df=pasting_yields_option_df(option_df, spot_curve_df, "risk_free_rate")
    option_df=pasting_yields_option_df(option_df, forward_curve_df, "forward_risk_free_rate")
    ##
    ##merging yield curve onto main option dataframe
    # option_df = option_df.merge(yield_curve_df, how='left', left_on='qoute_dt', right_on='Date')
    # option_df.drop(columns=['Date'], inplace=True)
    # date_group=option_df.groupby(['date'])
    # yield_curve_headers=yield_curve_df.columns.tolist()[1:]
    
    # print("--" * 20)  # Print a line of dashes for separation
    # print("We are interpolating the yield curve to match your options. This may take some time.")
    # ###Performing Yield Curve interpolation
    # option_df=option_df.reset_index(drop=True)
    # option_df=date_group.apply(lambda x: paste_rate(x,yield_curve_headers)) 
    # print("We are done interpolating Yield curve")
    # option_df=option_df.reset_index(drop=True)
    
    print("--" * 20)  # Print a line of dashes for separation
    print("Converting puts to calls. If both are present we use only OTM options.")
    date_group=option_df.groupby(['date','exdate'])
    option_df=date_group.apply(lambda x: use_only_OTM(x))
    option_df=option_df.reset_index(drop=True)
    
    date_group=option_df.groupby(['date','exdate'])
    print("--" * 20)  # Print a line of dashes for separation
    print("We are Aggregating duplicate option entries")    
    option_df=date_group.apply(lambda x: remove_duplicates(x))
    option_df=option_df.reset_index(drop=True)
    option_df['option_right'] = option_df['option_right'].replace({'call parity': 'c'})  
    
    ###Repair arbitrage
    option_df["underlying_price"]=option_df["stock_price"]
    option_df=option_df.reset_index(drop=True)
    date_group=option_df.groupby(['date','exdate'])
    option_df["repair_price"]=0
    option_df=date_group.apply(lambda x: pre_arbitrage_repair(x))
    ##
    
    
    mid_iv=py_vollib.black.implied_volatility.implied_volatility(option_df["mid_price"], 
                                                                 option_df["stock_price"], option_df["strike"], option_df["risk_free_rate"], option_df["maturity"], 'c', return_as='numpy')
    option_df["mid_iv"]=mid_iv
    option_df=option_df.reset_index(drop=True)
    option_df["bid_size"]=option_df["volume"]
    option_df["offer_size"]=option_df["volume"]
    option_df["moneyness"]=option_df["strike"]/option_df['stock_price']-1
    option_df=option_df[["date","exdate","maturity","rounded_maturity","risk_free_rate","strike","best_bid","mid_price","best_offer","mid_iv","underlying_price","stock_price",
                        "volume","offer_size","bid_size","option_right","moneyness","qoute_dt","expiry_dt"]]
    
    return option_df


def use_only_OTM(option_df):
    """
    Description of function: Converts puts into calls via put call parity, and replaces puts with calls.

    Parameters
    ----------
    option_df : dataframe
        option dataframe (day-expiration slice).

    Returns
    -------
    option_df : dataframe
        Dataframe with puts replaced with calls.

    """
    option_df=option_df.reset_index(drop=True)

    contains_only_c = all(x == 'c' for x in option_df['option_right'].to_list())
    contains_only_p = all(x == 'p' for x in option_df['option_right'].to_list())

    if contains_only_c==True:
        #print("Keeping all Calls.")
        return option_df
    if contains_only_p==True:
        #print("Converting puts into calls via put call parity")
        condition = (option_df['option_right'] == "p")
        option_df.loc[condition,'mid_price']=option_df["mid_price"]-option_df["strike"]*np.exp(-option_df["risk_free_rate"]*option_df["maturity"])+option_df["stock_price"]
        option_df.loc[condition,'best_bid']=option_df["best_bid"]-option_df["strike"]*np.exp(-option_df["risk_free_rate"]*option_df["maturity"])+option_df["stock_price"]
        option_df.loc[condition,'best_offer']=option_df["best_offer"]-option_df["strike"]*np.exp(-option_df["risk_free_rate"]*option_df["maturity"])+option_df["stock_price"]
        
        option_df.loc[condition,'option_right']='c'
    if ((contains_only_c==False) & (contains_only_p==False)):
        #print("Replacing ITM calls with puts via put call parity")
        condition = (option_df['option_right'] == "p") & (option_df['strike'] < option_df['stock_price']) ##otm puts
        option_df.loc[condition,'mid_price']=option_df["mid_price"]-option_df["strike"]*np.exp(-option_df["risk_free_rate"]*option_df["maturity"])+option_df["stock_price"] ###convert puts into calls
        option_df.loc[condition,'best_bid']=option_df["best_bid"]-option_df["strike"]*np.exp(-option_df["risk_free_rate"]*option_df["maturity"])+option_df["stock_price"]
        option_df.loc[condition,'best_offer']=option_df["best_offer"]-option_df["strike"]*np.exp(-option_df["risk_free_rate"]*option_df["maturity"])+option_df["stock_price"]
        
        option_df.loc[condition,'option_right']='call parity' #otm puts relabedl as call parity
        option_df = option_df[~(option_df['option_right'] == 'p')] #ITM puts removed
        condition=(option_df['option_right']=='c') & (option_df['strike']<option_df['stock_price']) ##ITM calls
        option_df=option_df.loc[~condition] ##ITM calls removed
        option_df.loc[condition,'option_right']='call parity' #otm puts relabedl as call parity

    return option_df

def paste_rate(df,yield_curve_headers,plot_curve=True):
    """
    Description of function: This function finds the risk free rate for each maturity in the option dataframe, and appends it in the main
    option dataframe. I use cubic interpolation, however I fix a knot at maturity=0, of 0. Thus effectively putting a lower bound of the rate to zero.

    Parameters
    ----------
    df : dataframe day-slice
        dataframe of options.
    yield_curve_headers : list
        list of yield curve headers. 
    plot_curve : True/False boolean, optional
        plots the yield curve. The default is False.

    Yields
    ------
    df : dataframe
        option dataframe with risk free rate appended..

    """
    df=df.reset_index(drop=True)
    date_str=df['date'].tolist()[0]
    date = date_str.replace('-', '_')
    
    ###Extracts the yield curve from the dataframe and drops nan values.
    yield_curve=df.loc[0,yield_curve_headers]
    yield_curve = yield_curve.dropna()
    
    ###Inserts a in the np array representing that the yield for 0 days to maturity is zero.
    yield_curve_rate=np.array(yield_curve.to_list())
    yield_curve_rate = np.insert(yield_curve_rate, 0, 0)
    
    ###extracts the maturities list from the dataframe
    yield_curve_maturities = np.array(yield_curve.index.tolist())
    yield_curve_maturities = np.insert(yield_curve_maturities, 0, 0)

    ###extracts the option maturities and computes the risk free rate using cubic interpolation
    option_maturities=np.array(df['maturity'].unique().tolist())
    cubic_spline = CubicSpline(yield_curve_maturities, yield_curve_rate)
    interpolated_rate = cubic_spline(option_maturities)

    ###appends the interpolated rate to the main option
    rate_df=pd.DataFrame({'option_maturities':option_maturities,'risk_free_rate':interpolated_rate})
    df = df.merge(rate_df, how='left', left_on='maturity', right_on='option_maturities')
    
    if plot_curve==True:
        ##Saves yield curve plot in a folder called Yield Curve Folder, which is automatically created in the users directory
        
        current_directory = os.getcwd()
        yield_folder = os.path.join(current_directory, "Yield Curve Folder")
        
        if not os.path.exists(yield_folder):
            os.makedirs(yield_folder)
            print(f"Created folder: {yield_folder}")
        
        smooth_yield_curve_maturities=np.arange(0,max(yield_curve_maturities),0.01)
        smooth_yield_curve_rate=cubic_spline(smooth_yield_curve_maturities)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(yield_curve_maturities, yield_curve_rate, label='Par Yield Curve Points', color='red')  # Original data points
        plt.scatter(option_maturities, interpolated_rate, label='Option Maturities Points', color='blue')  # Interpolated curve
        plt.plot(smooth_yield_curve_maturities,smooth_yield_curve_rate,label='Interpoalted Rate')
        plt.title('Cubic Spline Interpolation of Yield Curve on ' + date)
        plt.xlabel('Maturity (Years)')
        plt.ylabel('Yield')
        plt.axhline(0, color='grey', lw=0.5, ls='--')  # Horizontal line at yield = 0
        plt.grid()
        plt.legend()
        
        titlename_1="Yield Curve on "+date
        save_path = os.path.join(yield_folder, titlename_1)
        plt.savefig(save_path)
        plt.close()

    
    ###indicator text that program is working
    if random.randint(0, 100)<2:
        print(f"Done with yield on {date}, moving to next yield.")
    return df

def remove_duplicates(df):
    """
    Helper function that removes prices in the option dataframe. Prices should be monotonically decreasing/decreasing with respect to strike. In deribit data sometimes option prices will have the same price
    even as the strike prices increases/decreases. This usually only happens at the end of the strike domain. I truncate the data at the first occurence of the price repeating.

    Parameters
    ----------
    df : Dataframe
        Option dataframe. Should be a Date-exdate slice.

    Returns
    -------
    df : Dataframe
        Original Dataframe bu with mid prices truncated.

    """
    df=df.reset_index(drop=True)
    df = df[~df.duplicated(subset=['mid_price'], keep='first')]
    strike_list=df.loc[:,'strike'].values
    
    unique_values, counts = np.unique(strike_list, return_counts=True)
    has_multiple_duplicates = np.any(counts > 1)

    if has_multiple_duplicates:
        def custom_agg_func(series):
            """
            Helper function that handles DF where there are duplicate strikes by averaging.
    
            Parameters
            ----------
            series : TYPE
                DESCRIPTION.
    
            Returns
            -------
            TYPE
                DESCRIPTION.
    
            """
            if pd.api.types.is_numeric_dtype(series):
                return series.mean()
            else:
                return series.iloc[0]
        
        # Apply the custom aggregation function
        df = df.groupby('strike', as_index=False).agg(custom_agg_func)
    #print('working remove')
    return df
def average_futures(df):
    """
    Helper function that averages the futures prices on a given day, for each maturity. This is a known issue with deribit, where for some reason they quote multiple futures prices for a single maturity.
    I take the mean of these and use it as the underlying.

    Parameters
    ----------
    df : Dataframe date-expiry slice
        Dataframe with date-expiry slice.

    Returns
    -------
    df : Dataframe
        Original dataframe with futures price averaged.

    """
    df=df.reset_index(drop=True)
    mean_underlying_future=np.mean(df['underlying_price'].values)
    df.loc[:,'underlying_price']=mean_underlying_future
    return df


def rolling_window(df,nearest_maturity):
    """
    Parameters
    ----------
    df : Date-slice expiration dataframe
        Date_group:
    nearest_maturity : float
        Keeps only options with maturity closest to this number.

    Returns
    -------
    df : dataframe
        contains only options with maturies nearest to the nearest maturity.

    """
    
    df=df.copy()
    unique_maturity_list = df["rounded_maturity"].unique()*365
    closest_maturity = round(unique_maturity_list[np.abs(unique_maturity_list - nearest_maturity).argmin()])/365
    
    df=df[df["rounded_maturity"]==closest_maturity]
    
    return df
    
def compute_pdf(option_market,bw_setting="cv_ml",kde_setting="ISJ",kde_scale=1,plotting=True,truncate=False,plot_raw=True,foldername="figure1plot"):
    """
    Main function for computing the PDFs. 
    (1) User choses an interpolation type for the IV curve (spline, Kernel methods (least squares or maximum likelihood),
    or manually fixing bandwidth to a number. Similair to Ait-saihallia
    (2) After IV curve is interpolated, I remove any possible arbitrage using (Cohen,Reisinger,Wang) arbitrage repair.
    (3) To estimate the tails I apply Jing and Tiang (2007) and extrpolate by fixing the IV slope at the furthest OTM option. I resmooth the attachment points using
    kernel regression with bandwidth=50.
    (4) Numerically differentiate twice to achieve the option-implied PDF. Normalize to 1. In some cases this may be very noisy and produce unreasonable PDFs. 
    See step 5
    (5) User choses a bandwidth procedure (Improved Seather Jones, Scott, silverman, or a scaled variant of Scott/Silverman), and then I apply
    weighted kernel density estimation to back out possible noise/spurious spikes and obtain a valid density.
    (6 optional) The user may chose to truncate the data based on the when the CDF reaches 99.5%.
    (7) Plots the IV curve, transforms the X axis from stock price to return/relative strike (K/S-1) and plots the return distribution.
    
    My recommendation is cross validation maximimum likelihood with improved sheather jones.
    Parameters
    ----------
    option_market : dataframe
        Dataframe containing option data.
    bw_setting : string or list[], optional
        Interpolation setting for the IV curve phase (2).
        Your options are "spline","cv_ml","cv_ls", or [number]" Which are cubic splines cross validation maximum likeilihood, cross validation least squares,
        or a manually bandwidth number (enter it as an array eg. bw_setting=[1000] uses bandwidth of 1000 (not recommended to arbitrarily fix bw)).
        The default is "cv_ml". 
    kde_setting : String, optional
        Choses the BW setting for kernel density estimation. Chose between ISJ, scott or silverman. ISJ will produce densities closer to the raw
        density. The others will produce more "guassian" densities with less noise (but possibly much more bias). The default is "ISJ".
    kde_scale : float, optional
        Divides the scott/silverman bandwidth by this number. Can be used if you a bandwidth between ISJ and silverman/scott. The default is 1.
        If ISJ is chosen, this parameter does nothing.
    plotting : True/False, optional
        Plots IV-curve/density. The default is True.
    truncate : True/False, optional
        Truncates the density until CDF is 99.5% (starting from the center working outwards). The default is False.
    plot_raw : True/False, optional
        Plots the raw density as well. The default is True.

    Returns
    -------
    dict
        Dictionary containing the date,expiration date as keys, and as values a pandas dataframe containing strikes and the PDF after KDE estimation.

    """
    #############################################Extracting useful information from option market (0)
    option_market=option_market.copy().reset_index(drop=True)
    date_str=option_market['date'].tolist()[0]
    date = date_str.replace('-', '_')
    date_str=option_market['exdate'].tolist()[0]
    exdate = date_str.replace('-', '_')
    option_market['log moneyness']=np.abs(np.log(option_market["strike"]/option_market['stock_price']))
    option_market['log moneyness']=abs(option_market['log moneyness'])
    option_market['moneyness']=abs(np.exp(option_market['log moneyness']))
    stock_price=option_market['stock_price'].values[0]
    
    maturity=option_market['maturity'].values[0]
    R=option_market['risk_free_rate'].values[0]
    
    
    ####################################### Creating IV curve  (1-3)
    ###Uses 250 points for interpolation for speed. Will be re-interpolated for even finer points at step 5.
    print(bw_setting)
    iv_curve_strike,iv_curve,raw_iv_curve_strike,raw_iv_curve,error_ml,bw=create_iv_curve(option_market,bw_setting,500)
    iv_curve=arbitrage_repair(iv_curve, iv_curve_strike, option_market).values.reshape(-1)
    #plt.plot(iv_curve_strike,iv_curve)    
    
    full_iv_strike,test,full_iv_curve=extrapolation(iv_curve_strike,iv_curve,stock_price)
    #full_iv_strike,full_iv_curve,test=extrapolation(iv_curve_strike,iv_curve,stock_price)

    # plt.plot(full_iv_strike,full_iv_curve)
    # plt.scatter(full_iv_strike,test)
    dense_calls = py_vollib.black_scholes.black_scholes('c', option_market['underlying_price'].values[0],
                                                            full_iv_strike, maturity, R,full_iv_curve).values.reshape(-1)
     
    
    ##################################### Deriving option implied pdf (4)
    __,pdf=numerical_differentiation(np.array(full_iv_strike), dense_calls)
    pdf=pdf*np.exp(-R*maturity)
    pdf=normalize_pdf(full_iv_strike, pdf,stock_price)
    pdf_strike=full_iv_strike
        

    
    ####################################################### Estimated realistic density with kernel density (5)
    ###Uses 4000 points in final estimation.
    KDE_grid=np.linspace(0, max(full_iv_strike)+stock_price*0.25,1000*4)
    #kde_setting="ISJ"
    if kde_setting=="ISJ":
        kde = NaiveKDE(bw="ISJ").fit(pdf_strike/stock_price-1,weights=pdf)
        option_pdf=kde.evaluate(KDE_grid/stock_price-1)
        #option_pdf=normalize_pdf(KDE_grid, option_pdf)
    else:
        ##kde options scott,silverman
        kde=gaussian_kde(full_iv_strike/stock_price-1,kde_setting,pdf)
        kde_bw=float(kde.__getattribute__("factor"))/kde_scale
        kde=gaussian_kde(full_iv_strike/stock_price-1,kde_bw,pdf)
        option_pdf=kde(KDE_grid/stock_price-1)
        
        #option_pdf=normalize_pdf(KDE_grid, option_pdf)

        kde_setting=kde_setting+"/"+str(kde_scale)
        
    ######################################################### Truncates pdf (6)
    if truncate==True:    
        KDE_grid,option_pdf=truncate_pdf_iteratively(KDE_grid,option_pdf,stock_price)
        #option_pdf=normalize_pdf(KDE_grid, option_pdf)

    ####################################################Plotting (7)
    ###saves in a folder called BTC figure pdf in directory.
    if plotting==True:
        current_directory = os.getcwd()
        figure1_folder = os.path.join(current_directory, foldername)
        if not os.path.exists(figure1_folder):
            os.makedirs(figure1_folder)
            print(f"Created folder: {figure1_folder}")
            
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        ax1.plot(full_iv_strike/stock_price-1, full_iv_curve, color='green', linewidth=2, label='IV curve bw='+str(bw_setting))
        ax1.scatter(raw_iv_curve_strike/stock_price-1,raw_iv_curve, color='black',marker="x", s=50, label='Market IV')
    
        
        ax1.set_xlim(-1, 1.5) # Set x-axis limits from 0 to 20

        #ax1.set_ylim(0.0, 1.0)     # Set y-axis limits from 0 to 1.8
        ax1.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
        ax1.set_xlabel('Relative Strike (K/S - 1)', fontsize=14)
        ax1.set_ylabel('Implied Volatility', fontsize=14)
        ax1.set_title('Implied Volatility Curve', fontsize=16)
        ax1.legend()
    
        if plot_raw==True:
            test=1
            ax2.plot(pdf_strike/stock_price-1, pdf, color='black',linestyle='--', linewidth=2, label='Raw PDF')
        ax2.plot(KDE_grid/stock_price-1, option_pdf, color='green', linewidth=2, label='PDF method='+str(kde_setting))
            # Add grid, labels, and title for Plot 2
        ax2.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
        ax2.set_xlabel('Relative Strike (K/S - 1)', fontsize=14)
        ax2.set_ylabel('Density', fontsize=14)
        ax2.set_title('Density from '+date+" to "+ exdate, fontsize=16)
        ax2.legend()
        ax2.set_xlim(-1, 2.0) # Set x-axis limits from 0 to 20

        if truncate==False:
            ax2.set_xlim(-1, 1.5) # Set x-axis limits from 0 to 20
    
        # Adjust layout to prevent overlap
        plt.tight_layout()
            
        # Show the combined plot
        titlename_1=date+"_"+exdate
        save_path = os.path.join(figure1_folder, titlename_1)
        plt.savefig(save_path)
        plt.close()

    ####Final packaging into dictionary. Delivers strike axis, return axis, and final pdf in dataframe.
    return_axis=KDE_grid/stock_price-1
    density_df=pd.DataFrame({"Return":return_axis,"strike":KDE_grid,"PDF":option_pdf,"maturity":maturity})
    dictionary_pdf[date+exdate]=density_df
    return {date +","+ exdate: density_df}





def truncate_pdf_iteratively(x_axis, y_axis, stock_price, cdf_threshold=0.99):
    """
    Iteratively truncates the PDF starting from the given stock price. The truncation stops after 10 seconds.

    Parameters:
    x_axis (array-like): Array of X-axis values (e.g., strikes).
    y_axis (array-like): Array of Y-axis values (probabilities).
    stock_price (float): The central value (e.g., current trading price) to start truncating around.
    cdf_threshold (float): CDF threshold for truncation (default is 0.99 for 99%).

    Returns:
    tuple: Truncated x_axis and y_axis arrays around the stock price up to the specified CDF threshold.
    """
    # Create a DataFrame for easy manipulation
    pdf_df = pd.DataFrame({"x": x_axis, "y": y_axis})
    
    # Calculate interval widths (difference between consecutive x values)
    pdf_df['width'] = pdf_df['x'].diff().fillna(0)
    
    # Find the index closest to the stock price
    stock_index = (pdf_df['x'] - stock_price).abs().idxmin()
    
    # Initialize CDF and start iterating from the stock price outward
    cumulative_mass = 0.0
    total_mass = (pdf_df['y'] * pdf_df['width']).sum()
    threshold_mass = cdf_threshold * total_mass
    
    # Start accumulating CDF in both directions from the stock_index
    left_index = stock_index
    right_index = stock_index
    
    start_time = time.time()
    while cumulative_mass < threshold_mass:
        # Expand to the left if possible
        if left_index > 0:
            left_index -= 1
            cumulative_mass += pdf_df.loc[left_index, 'y'] * pdf_df.loc[left_index, 'width']
        
        # Expand to the right if possible
        if right_index < len(pdf_df) - 1 and cumulative_mass < threshold_mass:
            right_index += 1
            cumulative_mass += pdf_df.loc[right_index, 'y'] * pdf_df.loc[right_index, 'width']
        
        if time.time() - start_time > 10:
            print("Stopping pdf truncation because 10 seconds have elapsed.")
            break
    # Truncate the DataFrame within the computed range and return truncated arrays
    truncated_pdf_df = pdf_df.loc[left_index:right_index]
    return truncated_pdf_df['x'].values, truncated_pdf_df['y'].values

def normalize_pdf(strike,pdf,stock_price=True):
    """
    Description function: Normalizes PDF to add to 1

    Parameters
    ----------
    strike : numpy array
        array containing strikes.
    pdf : numpy array
        array containing pdf.

    Returns
    -------
    pdf : array
        pdf normalied to 1.

    """
    if stock_price!=False:
        x_axis=strike/stock_price-1
    else:
        x_axis=strike
    pdf = np.clip(pdf, a_min=0, a_max=None)
    bin_width =x_axis[1,]-x_axis[0,]  # Assuming uniform bin width
    normalized_pdf=pdf/(np.sum(pdf)*bin_width)
    
    #####    
    
    
    # # pdf = np.clip(pdf, a_min=0, a_max=None)    
    #integral = np.trapz(pdf, strike)
    
    #normalized_pdf=pdf/integral
    ###Does not normalize
    return normalized_pdf
    
def arbitrage_repair(iv_curve,iv_strike,option_market):
    """
    Description of function: Wrapper function for the arbitrage repair package. I convert IV to call prices, 
    remove arbitrage with L1 repair (does not use bid/ask prices), then convert back to IV.

    Parameters
    ----------
    iv_curve : numpy array
        iv curve.
    iv_strike : numpy array
        array containing stirke.
    option_market : dataframe
        dataframing containing option data.

    Returns
    -------
    iv_curve : numpy array
        iv_curve with options removed..

    """
    R=option_market["risk_free_rate"].values[0]
    T=option_market["maturity"].values[0]
    strike=iv_strike
    underlying_price=option_market["underlying_price"].values[0]
    dense_calls=py_vollib.black_scholes.black_scholes('c', underlying_price, iv_strike, T, R,iv_curve).values.reshape(-1)
    
    c_fv=np.exp(R*T)*dense_calls
    F=np.array([option_market["underlying_price"].values[0]*np.exp(R*T) for i in range(0,len(dense_calls))])
    Tau_vect=np.array([T for i in range(0,len(dense_calls))])
    
    warnings.filterwarnings("ignore")

    normaliser = constraints.Normalise()
    normaliser.fit(Tau_vect, strike, c_fv, F)
    
    T1, K1, C1 = normaliser.transform(Tau_vect, strike, c_fv)
    
    mat_A, vec_b, _, _ = constraints.detect(T1, K1, C1, verbose=False)
    epsilon1 = repair.l1(mat_A, vec_b, C1)
    K0, C0 = normaliser.inverse_transform(K1, C1 + epsilon1)
    
    cleaned_calls=C0*np.exp(-R*T)
    iv = implied_volatility.vectorized_implied_volatility(cleaned_calls,underlying_price, strike,T,R,'c')
    warnings.resetwarnings()

    print("cleaned successfully")
    return iv
#option_df=test
def pre_arbitrage_repair(option_df):
    """
    Description of function: Wrapper function for the arbitrage repair package. I convert IV to call prices, 
    remove arbitrage with L1 repair (does not use bid/ask prices), then convert back to IV.

    Parameters
    ----------
    iv_curve : numpy array
        iv curve.
    iv_strike : numpy array
        array containing stirke.
    option_market : dataframe
        dataframing containing option data.

    Returns
    -------
    iv_curve : numpy array
        iv_curve with options removed..

    """
    option_df=option_df.copy()
    R=option_df["risk_free_rate"].values[0]
    T=option_df["maturity"].values[0]
    strike=option_df["strike"].values
    underlying_price=option_df["stock_price"].values[0]
    calls=option_df["mid_price"].values
    c_fv=np.exp(R*T)*calls
    F=np.array([underlying_price*np.exp(R*T) for i in range(0,len(option_df["strike"].values))])
    Tau_vect=np.array([T for i in range(0,len(option_df["strike"].values))])
    
    warnings.filterwarnings("ignore")

    normaliser = constraints.Normalise()
    normaliser.fit(Tau_vect, strike, c_fv, F)
    
    T1, K1, C1 = normaliser.transform(Tau_vect, strike, c_fv)
    
    mat_A, vec_b, _, _ = constraints.detect(T1, K1, C1, verbose=False)
    epsilon1 = repair.l1(mat_A, vec_b, C1)
    K0, C0 = normaliser.inverse_transform(K1, C1 + epsilon1)
    
    cleaned_calls=C0*np.exp(-R*T)
    #error=cleaned_calls-C0
    iv = implied_volatility.vectorized_implied_volatility(cleaned_calls,underlying_price, strike,T,R,'c')
    mid_iv=iv.values.reshape(-1,1)
    warnings.resetwarnings()
    option_df["mid_price"]=cleaned_calls
    option_df["mid_iv"]=0
    option_df["mid_iv"]=mid_iv
    print("cleaned successfully")
    return option_df


def numerical_differentiation(strike_vector, points):
    """
    Description of function: Finds the 2nd derivative of IV curve.

    Parameters
    ----------
    strike_vector : numpy array
        Strike array.
    points : numpy array
        IV curve array.

    Returns
    -------
    first_derivative : array
        first derivative.
    second_derivative : array
        second derivative.

    """
    # Ensure that the inputs are numpy arrays
    strike_vector = np.array(strike_vector, dtype=float)
    points = np.array(points, dtype=float)
    
    # Calculate the first derivative (dy/dx)
    first_derivative = np.gradient(points, strike_vector)
    
    # Calculate the second derivative (d^2y/dx^2)
    second_derivative = np.gradient(first_derivative, strike_vector)
    
    return first_derivative, second_derivative   


def find_error_at_steepest(dense_iv,dense_strike,option_market):
    """
    Archived function. Was designed to compare fit of IV curves ATM. Not used.

    Parameters
    ----------
    dense_iv : TYPE
        DESCRIPTION.
    dense_strike : TYPE
        DESCRIPTION.
    option_market : TYPE
        DESCRIPTION.

    Returns
    -------
    difference_iv : TYPE
        DESCRIPTION.

    """
    strike=option_market['strike'].values
    ivs=option_market['mid_iv'].values
    
    ####Find the Lowest IV quote
    lowest_iv_index=ivs.argmin()
    lowest_iv_strike=strike[lowest_iv_index]
    lowest_iv=min(ivs)
    ### Find corresponding point on the interpolated IV curve and compute the difference
    index_lowest_iv_strike=(np.abs(dense_strike-lowest_iv_strike)).argmin()
    dense_iv_lowest=dense_iv[index_lowest_iv_strike]
    difference_iv=np.abs(lowest_iv-dense_iv_lowest)
    
    return difference_iv


def extrapolation(iv_strike, iv,stock_price):
    """
    Description of function: Extrapolate the IV curve by fixing the slope of the furthest out of the money options (Jiang and Tian -2007). 
    The extrapolation is done out to deepest OTM option strike+0.8*current trading price. The left extrapolation is bounded at $0.1. 
    To smooth out the attachment points, I run kernel smoothing again with bandwidth of 50.
    

    Parameters
    ----------
    iv_strike : numpy array
        Strike prices of IV curve.
    iv : numpy array
        Implied volatilties curve.
    stock_price : float
        Current price of asset.

    Returns
    -------
    combined_strike : numpy array
        Array of strike prices after extrapolation and smoothing.
    combined_iv : numpy array
        implied volatilties that have been extrapolated.
    smooth_iv : TYPE
        implied volatilities after extrapolation and smoothing. Just in case to check if there was oversmoothing..

    """    
    # Ensure iv_strike and iv are numpy arrays
    iv = np.array(iv)
    iv_strike = np.array(iv_strike)
    
    # Calculate the gradient
    gradient = np.gradient(iv, iv_strike)
    m_right = gradient[-1]  # Slope from the last segment for right extrapolation
    m_left = gradient[0]    # Slope from the first segment for left extrapolation
    

    
    
    # Intercepts for extrapolation
    b_right = iv[-1] - m_right * iv_strike[-1]  # Intercept for the last point
    b_left = iv[0] - m_left * iv_strike[0]      # Intercept for the first point
    
    # Calculate the step size
    strike_step = abs(iv_strike[-1] - iv_strike[-2])
    
    # Generate extrapolated strike prices to the right
    extrapolation_strike_right = np.arange(iv_strike[-1] + strike_step, iv_strike[-1] +0.8*stock_price, strike_step)
    extrapolation_right = m_right * extrapolation_strike_right + b_right
    
    # Generate extrapolated strike prices to the left
    extrapolation_strike_left = np.arange(max(iv_strike[0]-stock_price*0.8,0.1), iv_strike[0]-strike_step,strike_step)
    extrapolation_left = m_left * extrapolation_strike_left + b_left
    
    # Combine original and extrapolated strike prices and implied volatility
    combined_strike = np.concatenate((extrapolation_strike_left, iv_strike, extrapolation_strike_right))
    combined_iv = np.concatenate((extrapolation_left, iv, extrapolation_right))
    
    
    kernel_obj = KernelReg(endog=combined_iv, exog=combined_strike, reg_type="ll", var_type='c',bw=[5])
    smooth_iv,_=kernel_obj.fit(combined_strike)
    ####


    return combined_strike, combined_iv,smooth_iv

def create_iv_curve(option_market,bw_method,num_points):
    """
    Description of Function: This function interpolates the IV curve using either Cubic piecewise splines, or kernel regressions.

    Parameters
    ----------
    option_market : dataframe
        dataframe containing risk free rate, stock price, maturity.
    bw_method : String or array [number]
        Determines whether interpolation is done using splines, or kernel smoothing. Options are: "spline", "cv_ls","cv_ml", or if you want to
        manually fix the bandwidth of the kernel [number] eg [1000] which sets the BW to be 1000.
    num_points : float
        Determine the number of points in the newly interpolated curve.

    Returns
    -------
    dense_strike : numpy array
        interpolated strikes.
    dense_iv : numpy array
        interpolated IV curve,.
    raw_strike : numpy array
        raw strike prices from markey.
    raw_iv : TYPE
        raw market iv qoutes.
    iv_error_at_steepest : float
        Not used. But calculates the absolute error from the interpolated IV curve to the market quote with lowest IV.
    bw : float
        Not used, but returns bandwidth from kernel regression if used..

    """
    option_market=option_market.copy()
    maturity=option_market['maturity'].values[0]
    risk_free_rate=option_market['risk_free_rate'].values[0]
    strike=option_market['strike'].values
    mid_iv=option_market['mid_iv'].values
    #strike,mid_iv=interpolate_large_gaps(strike, mid_iv)
    stock_price=option_market["stock_price"].values[0]
    noption_market=pd.DataFrame({"strike":strike,"mid_iv":mid_iv})
    noption_market["stock_price"]=stock_price
    noption_market['log moneyness']=np.abs(np.log(noption_market["strike"]/noption_market['stock_price']))

    noption_market['maturity']=option_market['maturity'].values[0]
    noption_market['risk_free_rate']=option_market['risk_free_rate'].values[0]
    bw=0
    print(bw_method)
    acceptable_density=False
    N=1
    warnings.filterwarnings("ignore")
    if bw_method=="poly":
        quart_poly = np.poly1d(np.polyfit(strike, mid_iv, 4))
        dense_strike=np.linspace(max(min(strike)-stock_price*0.5,4),max(strike)+stock_price*0.5,num_points)
        dense_iv=quart_poly(dense_strike)
        iv_error_at_steepest=0
        raw_iv=noption_market['mid_iv'].values
        raw_strike=noption_market['strike'].values
        
        
    if bw_method=="pilot":
        dense_strike=np.linspace(min(noption_market["strike"].values),max(noption_market["strike"].values),num_points)
        h = noption_market["strike"].diff().dropna().values
        h=np.mean(h)
        kernel_obj = KernelReg(endog=noption_market['mid_iv'].values, exog=noption_market['strike'], reg_type="ll", var_type='c',bw=[h])
        dense_iv,_=kernel_obj.fit(dense_strike)
        
        
        if np.any(dense_iv < 0.03):
            dense_strike=np.linspace(min(noption_market["strike"].values),max(noption_market["strike"].values),num_points)
            h = noption_market["strike"].diff().dropna().values
            h=1.5*np.mean(h)
            kernel_obj = KernelReg(endog=noption_market['mid_iv'].values, exog=noption_market['strike'], reg_type="ll", var_type='c',bw=[h])
            dense_iv,_=kernel_obj.fit(dense_strike)
        
        raw_iv=noption_market['mid_iv'].values
        raw_strike=noption_market['strike'].values
        
        bw=str(h)
        iv_error_at_steepest=0

    if bw_method=="spline":
        cs = CubicSpline(strike, mid_iv)
        dense_strike=np.linspace(min(noption_market["strike"].values),max(noption_market["strike"].values),num_points)
        dense_iv = cs(dense_strike)
        #dense_calls = py_vollib.black_scholes.black_scholes('c', option_market['underlying_price'].values[0], dense_strike, maturity, risk_free_rate, dense_iv).values.reshape(-1)  # 12.111581435
        raw_iv=noption_market['mid_iv'].values
        raw_strike=noption_market['strike'].values
        iv_error_at_steepest=0
        print("spline")
        bw="cubic"
    if bw_method == "cv_ml" or bw_method == "cv_ls":
        ##Note that "CV_ML" has some convergence issues. In this event, if convergence fails, I remove the deepest OTM option and rerun the algorithmn.
        ## In general this isn't an issue because the slope of the IV curve is linear deep OTM anyway.
        while acceptable_density==False: 
            #if bw_method=="cv_ml":
                #print("--" * 20)  # Print a line of dashes for separation
            try:
                print(f"Attempting kernel regression with {bw_method}. Pass {N}")    
                dense_strike=np.linspace(min(noption_market["strike"].values),max(noption_market["strike"].values),num_points)
                kernel_obj = KernelReg(endog=noption_market['mid_iv'].values, exog=noption_market['strike'], reg_type="ll", var_type='c',bw=bw_method)
    
                bw=kernel_obj.bw
                #dense_strike=np.linspace(min(noption_market["strike"].values)-0.1*stock_price,max(noption_market["strike"].values)+0.1*stock_price,num_points)

                # kernel_obj = KernelReg(endog=noption_market['mid_iv'].values, exog=noption_market['strike'], reg_type="ll", var_type='c',bw=[bw])
                
                dense_iv,_=kernel_obj.fit(dense_strike)
                
                dense_calls = py_vollib.black_scholes.black_scholes('c', option_market['underlying_price'].values[0], dense_strike, maturity, risk_free_rate, dense_iv).values.reshape(-1)  # 12.111581435
                _,pdf=numerical_differentiation(dense_strike,dense_calls)
                        
            except:
                bw=0
                print('ML Kernel Regression cannot converge. We removing are the next Deepest OTM option and trying again.')
            ##I check if the bandwidth is unreasonably small, of the IV produces some extremely small IV values. Generally not an issue. It will remove deepest
            ## OTM option and try again.
            if (bw > 0.1) and not np.any(dense_iv < 0.03):
                acceptable_density=True
                iv_error_at_steepest=find_error_at_steepest(dense_iv,dense_strike,noption_market)
                print(f'We have an acceptable density after removing {N-1} deepest otm options')
                
            else:
            ###Code for removing deepest OTM option
                moneyness_threshold = noption_market['log moneyness'].nlargest(len(noption_market)).iloc[0]
                noption_market=noption_market[noption_market['log moneyness']<moneyness_threshold]
                N=N+1
                noption_market=noption_market.reset_index(drop=True)
            raw_iv=noption_market['mid_iv'].values
            raw_strike=noption_market['strike'].values
        

        warnings.filterwarnings("default")
        
    # kernel_obj = KernelReg(endog=dense_iv, exog=dense_strike, reg_type="ll", var_type='c', bw=[bw])
    # dense_strike=np.arrange(dense_strike[0],dense_strike[-1],10)
    # dense_iv=kernel_obj.fit(dense_strike)

        
    return dense_strike,dense_iv,raw_strike,raw_iv,iv_error_at_steepest,bw 
    
def fed_yield_curve_maker(fed_yield_curve,date_truncation="2021-06-01"):
        
    fed_sheet=pd.read_csv(fed_yield_curve)
        
    date_row_index = fed_sheet[fed_sheet.isin(["Date"])].stack().index[0][0]
    useful_information_list=["Date","BETA0","BETA1","BETA2","BETA3","TAU1","TAU2","SVENY01","SVENY02","SVENY03","SVENY04","SVENY05","SVENY06",
                                 "SVENY07"]
        ##
    fed_sheet_truncated=fed_sheet.iloc[8:,].reset_index(drop=True)
    fed_sheet_truncated.columns = fed_sheet_truncated.iloc[0]  # Assign the first row as the column names
        
        ##Remove unnecessary information
        # useful_information_list=["Date","BETA0","BETA1","BETA2","BETA3","TAU1","TAU2","SVENF01","SVENF02","SVENF03","SVENF04","SVENF05","SVENF06",
        #                          "SVENF07"]
    useful_information_list=["Date","BETA0","BETA1","BETA2","BETA3","TAU1","TAU2","SVENY01","SVENY02","SVENY03","SVENY04","SVENY05","SVENY06",
                                 "SVENY07"]
    fed_sheet=fed_sheet_truncated[useful_information_list]
    fed_sheet=fed_sheet[1:]
        
        ##keep only after 1980
    fed_sheet["Date"]=pd.to_datetime(fed_sheet["Date"])
    fed_sheet = fed_sheet[fed_sheet["Date"] >= "1980-01-02"]  # Filter for the specific date
    fed_sheet = fed_sheet[fed_sheet["Date"] >= date_truncation]  # Filter for the specific date
    
    fed_sheet = fed_sheet.dropna()  # Drop rows with any NaN values
    return fed_sheet
        
    
def svensson_model(t, beta0, beta1, beta2, beta3, tau1, tau2):
    """
    Svensson interest rate model function.

    Parameters:
    t (float or np.ndarray): Time to maturity (can be a scalar or an array).
    beta0 (float): Long-term level parameter.
    beta1 (float): Short-term slope parameter.
    beta2 (float): Medium-term curvature parameter.
    beta3 (float): Second medium/long-term curvature parameter.
    tau1 (float): Decay factor for the first exponential term.
    tau2 (float): Decay factor for the second exponential term.

    Returns:
    float or np.ndarray: Instantaneous forward rate(s) at time t.
    """
    term1 = beta0
    term2 = beta1 * np.exp(-t / tau1)
    term3 = beta2 * (t / tau1) * np.exp(-t / tau1)
    term4 = beta3 * (t / tau2) * np.exp(-t / tau2)
    return term1 + term2 + term3 + term4

def svensson_spot_rate(T, beta0, beta1, beta2, beta3, tau1, tau2):
    """
    Calculate spot rate for a given maturity T using the Svensson interest rate model.
    
    Parameters:
    T (float or np.ndarray): Maturity (in years) for which the spot rate is calculated.
    beta0, beta1, beta2, beta3, tau1, tau2 (float): Svensson model parameters.

    Returns:
    float or np.ndarray: Spot rate for the given maturity T.
    """
    # Create a grid of t values from 0 to T (1000 points)
    t_values = np.linspace(0, T, 1000)  # Correct use of T (not t)
    
    # Compute forward rate at each t in the interval [0, T]
    f_values = svensson_model(t_values, beta0, beta1, beta2, beta3, tau1, tau2) 
    
    # Compute the integral of f(t) over [0, T] using trapezoidal integration
    integral = np.trapz(f_values, t_values)  
    
    # Calculate the spot rate r(T) as the average forward rate
    spot_rate = integral / T
    
    return spot_rate
    
def compute_yields(fed_sheet,plotting=True):
    
    fed_sheet=fed_sheet.reset_index(drop=True)

    #collect parameters
    param_list = np.array(fed_sheet.iloc[0, 1:7])
    param_list = pd.to_numeric(param_list, errors='coerce')  # Replace non-numeric values with NaN
    
    
    #Predicting maturities
    predicted_maturities = np.array([i / 365 for i in range(0, 2555)])  # Daily maturities (1 day to ~7 years)
    #predicted_spot_rates = [svensson_model(T, *param_list) for T in predicted_maturities]
    spot_rates = np.array([svensson_spot_rate(T, *param_list) for T in predicted_maturities])
    spot_rates=np.clip(spot_rates, 0, a_max=None)
    spot_rates[0]=spot_rates[1] ##Assume the zero day risk free rate is just the 1 day risk free rate
    
    ##
    observed_spot=fed_sheet.iloc[0,7:]
    observed_spot =np.array( pd.to_numeric(observed_spot, errors='coerce'))  # Replace non-numeric values with NaN

    observed_maturities=np.array([1,2,3,4,5,6,7])
    date = fed_sheet.loc[0, "Date"].strftime("%Y_%m_%d")
    if plotting==True:
        current_directory = os.getcwd()
        figure1_folder = os.path.join(current_directory, "yield_curves")
        if not os.path.exists(figure1_folder):
            os.makedirs(figure1_folder)
            print(f"Created folder: {figure1_folder}")
        

        
        plt.figure(figsize=(10, 6))
        plt.plot(predicted_maturities, spot_rates, label=f"Svensson curve on {date}", linestyle="--")
        plt.scatter(observed_maturities, observed_spot, label="Observed spot Yield Curve", color="red", s=10)
        plt.xlabel("Maturity (Years)")
        plt.ylabel("Yield (%)")
        plt.title(f"Svensson Model Fit vs Observed Yield Curve {date}")
        plt.legend()
        
        save_path = os.path.join(figure1_folder, f"yield_curve on {date}")
        plt.savefig(save_path)
        plt.close()
    date=date.replace("_","-")
    yield_df=pd.DataFrame({"spot_rate":spot_rates/100,"time(days)":predicted_maturities})
    dictionary_yield={}
    #dictionary_yield[date]=yield_df
    if np.any(spot_rates)>10:
        print(date)
    if np.any(spot_rates)<0:
        print(date)
    return {date: yield_df}


def attach_risk_free_rate(option_df, yield_curve):
    """
    Appends the risk-free rate to the given option dataframe based on the yield curve.

    Parameters
    ----------
    option_df : pandas.DataFrame
        Contains option information. Must include:
        - "date": date of the option quote in "YYYY-MM-DD" format.
        - "rounded_maturity": maturity of the option (in year fractions, e.g., 1 day = 1/365).
    
    yield_curve : dict
        A dictionary where:
        - Keys are dates in "YYYY-MM-DD" format.
        - Values are DataFrames with columns:
            - "time(days)": maturity (in years).
            - "spot_rate": spot risk-free rate for that maturity.

    Returns
    -------
    pandas.DataFrame
        The updated option_df with an additional column "risk_free_rate" containing the matched rates.
    """
    option_df = option_df.reset_index(drop=True)
    option_df["risk_free_rate"] = np.nan  # Initialize risk-free rate column

    for i in range(len(option_df)):
        # Extract the date and maturity for the current option
        current_date = option_df.loc[i, "date"]
        maturity = option_df.loc[i, "rounded_maturity"]
        
        try:
            # Fetch the yield curve for the current date
            if current_date in yield_curve:
                rate_curve_df = yield_curve[current_date]
            else:
                raise KeyError(f"Date {current_date} not found in yield_curve")
            
            # Match the maturity in the yield curve
            risk_free_rate_row = rate_curve_df[rate_curve_df["time(days)"] == maturity]
            if not risk_free_rate_row.empty:
                risk_free_rate = risk_free_rate_row.iloc[0]["spot_rate"]
                option_df.loc[i, "risk_free_rate"] = risk_free_rate
            else:
                raise ValueError(f"Maturity {maturity} not found in yield curve for {current_date}")
        
        except Exception as e:
            # If exact date or maturity is not found, search for the previous available yield curve
            print(f"Error for row {i} ({current_date}): {e}. Searching for previous date.")
            previous_date = pd.to_datetime(current_date)
            
            while True:
                previous_date -= pd.Timedelta(days=1)
                previous_date_str = previous_date.strftime("%Y-%m-%d")
                
                if previous_date_str in yield_curve:
                    rate_curve_df = yield_curve[previous_date_str]
                    risk_free_rate_row = rate_curve_df[rate_curve_df["time(days)"] == maturity]
                    
                    if not risk_free_rate_row.empty:
                        risk_free_rate = risk_free_rate_row.iloc[0]["spot_rate"]
                        option_df.loc[i, "risk_free_rate"] = risk_free_rate
                        break
                else:
                    print(f"No yield curve found for {previous_date_str}. Continuing search...")

    return option_df

def pasting_yields_option_df(option_df,yields_df,new_column_name):
    """
    Description of function: Pastes interpolated risk free rates from the yields_df onto the options df

    Parameters
    ----------
    option_df : TYPE
        DESCRIPTION.
    yields_df : TYPE
        DESCRIPTION.
    new_column_name : TYPE
        DESCRIPTION.

    Returns
    -------
    option_df : TYPE
        DESCRIPTION.

    """
    # Reshape spot_curves_df to long format
    yields_df_long = yields_df.reset_index().melt(
        id_vars=["index"], var_name="rounded_maturity", value_name=new_column_name
    )
    yields_df_long.rename(columns={"index": "qoute_dt"}, inplace=True)
    
    # Ensure the data types align for merging
    yields_df_long["qoute_dt"] = pd.to_datetime(yields_df_long["qoute_dt"])
    yields_df_long["rounded_maturity"] = yields_df_long["rounded_maturity"].astype(float)/365  # Convert maturity to float for matching
    
    # Assuming 'option_df' is your option dataframe
    option_df["qoute_dt"] = pd.to_datetime(option_df["qoute_dt"])  # Ensure 'date' is datetime
    option_df["maturity"] = option_df["maturity"].astype(float)  # Ensure 'maturity' is float for matching
    
    # Merge the option dataframe with the reshaped spot curves
    option_df = option_df.merge(
        yields_df_long,
        how="left",  # Use 'left' join to keep all rows in the option dataframe
        on=["qoute_dt", "rounded_maturity"]
    )
    return option_df
