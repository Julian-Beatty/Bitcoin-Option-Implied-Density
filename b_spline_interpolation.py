from main_bitcoin_pdf import*
from scipy.interpolate import BSpline, make_interp_spline

#####Preloading all the data needed

with open("yields_dict.pkl", "rb") as file:
    loaded_dict = pickle.load(file)

spot=loaded_dict["spot"]  
forward=loaded_dict["forward"] 

option_data_prefix="btc_eod"
stock_price_data_prefix='bitcoin_10'


######Initializing Option market
option_market=OptionMarket(option_data_prefix,stock_price_data_prefix,spot,forward)
option_market.set_master_dataframe(volume=-1, maturity=[30,30],moneyness=[-50.5,50.5],rolling=False)

master_df=option_market.master_dataframe
date_group=master_df.groupby(["date","exdate"])



########################Example option dataframe for your testing
option_df=date_group.get_group(("2021-06-30","2021-07-30"))

###Your function
def B_spline_interpolation(option_df,foldername,order):
    """
    Description of function: This function takes in an option dataframe, which contains the option quotes on a particular day, with a specific maturity.
    It will use B-spline interpolation to smooth out the implied volatilities, then convert them to calls, numerically differentiate twice and save the plots.
    Do not change the inputs of this function, and do not change the output.

    Parameters
    ----------
    option_df : Pandas dataframe containing market information
        Dataframe  containing market information.
    foldername : Str
        Name of the folder created.

    Returns
    -------
    dictionary:
        Key is date and expiry date, value is a pandas dataframe containing the pdf

    """
    #### Some code to get you started.
    
    ##Extract Implied volatility qoutes and strikes
    iv=option_df["mid_iv"].values.reshape(-1)
    strikes=option_df["strike"].values.reshape(-1)
    
    
    ##Visualize IV smile
    plt.scatter(strikes,iv)
    
    ##Define interpolation grid
    interpolated_strikes=np.linspace(strikes[0], strikes[-1],1000)
    
    ##Interpolation via Cubic splines
    spl=make_interp_spline(strikes,iv,order)
    interpolated_iv = spl(interpolated_strikes)

    ##Visualize interpolated smile
    plt.scatter(strikes,iv)
    plt.plot(interpolated_strikes,interpolated_iv)

    ##Convert implied volatilities back to call prices
    R=option_df["risk_free_rate"].values[0] ##Risk free rate
    T=option_df["maturity"].values[0]   ##Option maturity
    underlying_price=option_df["underlying_price"].values[0] ##price of the underlying
    
    
    ##Black scholes formula to convert IV to prices
    interpolated_prices = py_vollib.black_scholes.black_scholes('c', underlying_price,interpolated_strikes, T, R,interpolated_iv).values.reshape(-1)
    
    ##visualize calls
    plt.plot(interpolated_strikes,interpolated_prices)
    
    ##Numerically differentiate
    CDF,PDF=numerical_differentiation(interpolated_strikes, interpolated_prices)
    
    ##Normalize PDF (making sure it integrates to 1, clip any negative values to zero)
    PDF=normalize_pdf(interpolated_strikes,PDF)
    ##visualize PDF and CDF
    plt.plot(interpolated_strikes,PDF)
    plt.plot(interpolated_strikes,CDF)

    ##Extracting date
    date=option_df["date"].values[0]
    exdate=option_df["exdate"].values[0]

    ##Save plots in a folder in the user directory
    current_directory = os.getcwd()
    figure1_folder = os.path.join(current_directory, foldername)
    if not os.path.exists(figure1_folder):
        os.makedirs(figure1_folder)
        print(f"Created folder: {figure1_folder}")
        
        
    fig, (ax1, ax2,ax3) = plt.subplots(3, 1, figsize=(10, 15))
    ax1.scatter(strikes, iv,color="red")
    ax1.plot(interpolated_strikes, interpolated_iv)
    ax1.set_title(f"Order {order} Spline interpolation on {date} to {exdate}")
    ax1.set_xlabel("Strikes")
    ax1.set_ylabel("Implied Volatility")
    
    ax2.plot(interpolated_strikes,CDF)
    ax2.set_title("Option implied CDF")
    ax2.set_xlabel("Strikes")
    ax2.set_ylabel("CDF")
    
    ax3.plot(interpolated_strikes, PDF)
    ax3.set_title("Option implied PDF")
    ax3.set_xlabel("Strikes")
    ax3.set_ylabel("PDF")
    # Show the combined plot
    titlename_1=f"{date} on {exdate}"
    save_path = os.path.join(figure1_folder, titlename_1)
    plt.savefig(save_path)
    plt.close()
    
    ##final packaging
    density_df=pd.DataFrame({"strike":interpolated_strikes,"PDF":PDF,"IV":interpolated_iv})

    return {date +","+ exdate: density_df}

##Test out your by looping through the groups plz use this code
result_dict = {}
date_group.apply(lambda x: result_dict.update(B_spline_interpolation(x,foldername="My foldername",order=5)))
