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

# option_data_prefix="options_gld.csv"
# stock_price_data_prefix='GLD price.csv'
#################Test 
####
option_market=OptionMarket(option_data_prefix,stock_price_data_prefix,spot,forward)
option_df=option_market.original_market_df
option_market.set_master_dataframe(volume=-1, maturity=[10,150],moneyness=[-50.5,50.5],rolling=60)
option_df=option_market.master_dataframe
group=option_df.groupby(["date","exdate"])
# option_df=group.get_group(("2021-07-27", "2021-09-24"))

###Computation of PDFs
#argument_dict={"order":4,"folder":"option_poly"}
argument_dict={"cv":9,"folder":" kernel ridge"}
argument_dict={"cv":12,"folder":"SVR"}
argument_dict={"folder":"local_smoother"} ##scikitfda
argument_dict={"folder":"local_poly","cv":"cv_ls"} ##statsmodel implementation

GLD_90_9=option_market.estimate_option_pdfs("kernel ridge",argument_dict)
###(1) first argument: "kernel ridge" (2) "SVR" (3) "local_smoother" (4) "local_poly














option_market.set_master_dataframe(volume=0, maturity=[20,90],moneyness=[-50.5,50.5],rolling=60)
option_df=option_market.master_dataframe
group=option_df.groupby(["date","exdate"])
argument_dict={"cv":"loo","folder":" kernel ridge 6 GLD"}

btc_60_loo=option_market.estimate_option_pdfs("kernel ridge",argument_dict)



btc_30_dict={"BTC 30": result}
filename = "GLD_90_9dict.pkl"

# Save the dictionary to a pickle file
with open(filename, "wb") as file:
    pickle.dump(GLD_90_9, file)
    

btc_90_60_loo_dict={"btc_90_loo":btc_90_loo,"btc_60_loo":btc_60_loo}
