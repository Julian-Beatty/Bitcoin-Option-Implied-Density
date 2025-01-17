from main_option_market import*
from create_yield_curve import*
###If manually want to run yield curves##
spot,forward=create_yield_curves("fed_yield_curve.csv","2024-11-01")
###Otherwise just load mine

with open("yields_dict.pkl", "rb") as file:
    loaded_dict = pickle.load(file)
    
    
spot=loaded_dict["spot"]  
forward=loaded_dict["forward"] 
###Datafiles
option_data_prefix="btc_eod"
stock_price_data_prefix='bitcoin_10'

option_data_prefix="GLD_options.csv"
stock_price_data_prefix='GLD price.csv'
#################Test 
####
option_market=OptionMarket(option_data_prefix,stock_price_data_prefix,spot,forward)
option_df=option_market.original_market_df
option_market.set_master_dataframe(volume=-1, maturity=[90,90],moneyness=[-50.5,50.5],rolling=False)
option_df=option_market.master_dataframe
group=option_df.groupby(["date","exdate"])
option_df=group.get_group(("2023-12-30", "2024-01-26"))
###Computation of PDFs
argument_dict={"order":4,"folder":"option_poly"}
argument_dict={"cv":9,"folder":"kernel ridge 90"}

result=option_market.estimate_option_pdfs("kernel ridge",argument_dict)
