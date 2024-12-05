from main_bitcoin_pdf import*


option_data_prefix="btc_eo"
stock_price_data_prefix='bitcoin_10'
par_yield_curve_prefix='daily-'


option_market=OptionMarket(option_data_prefix,par_yield_curve_prefix,stock_price_data_prefix)
option_df=option_market.original_market_df
#option_market.set_master_dataframe(-1, [90,90])


#results_90=option_market.compute_option_pdfs(bw_setting="cv_ls")
option_market.set_master_dataframe(volume=-1, maturity=[1,1],rolling=False)

results_30=option_market.compute_option_pdfs(bw_setting="cv_ml",kde_setting="ISJ",kde_scale=2)


option_picke={"results":results_30}
with open('rolling_results.pkl', 'wb') as file:
    pickle.dump(option_picke, file)