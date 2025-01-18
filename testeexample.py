from main_option_market import*
from create_yield_curve import*
from knn import*
from local_linear import *
from SVR_model import *
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


#################Test 
####
option_market=OptionMarket(option_data_prefix,stock_price_data_prefix,spot,forward)
option_df=option_market.original_market_df
#option_market.set_master_dataframe(volume=-1, maturity=[30,30],moneyness=[-50.5,50.5],rolling=30)
option_market.set_master_dataframe(volume=-1, maturity=[15,15],moneyness=[-50.5,50.5],rolling= False)
option_df=option_market.master_dataframe
date_group=option_df.groupby(["date","exdate"])
#option_df=group.get_group(("2023-12-29", "2024-01-26"))
###Computation of PDFs
#argument_dict={"order":4,"folder":"option_poly"}
#argument_dict={"cv":9,"folder":"knn"} # if use knn
argument_dict={"folder":"local_linear"} # if use Locally linear
argument_dict={"cv":9,"folder":"SVR"} # if use knn
#result=option_market.estimate_option_pdfs("kernel ridge",argument_dict)
result_dict={}
date_group.apply(lambda x: result_dict.update(SVR_model(x,argument_dict,interpolate_large =True)))



#result=option_market.estimate_option_pdfs("polynomial",argument_dict)
#result=option_market.estimate_option_pdfs("knn",argument_dict) 

#date_group.apply(lambda x: result_dict.update(kernel_ridge_pdf(x,argument_dict)))
