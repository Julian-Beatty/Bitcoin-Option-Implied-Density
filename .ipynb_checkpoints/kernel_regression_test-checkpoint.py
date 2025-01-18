from main_option_market import*

###For your convenience
with open("yields_dict.pkl", "rb") as file:
    loaded_dict = pickle.load(file)
        
spot=loaded_dict["spot"]  
forward=loaded_dict["forward"] 

##Initialize class
option_market=OptionMarket(option_data_prefix,stock_price_data_prefix,spot,forward)
##If you are curious what the original dataframe looks like
option_df=option_market.original_market_df
##We can filter out the option frame by only considering options within a certain days to expiration (maturity). Rolling filters for options with maturity closest to 30. 
##Or you can set it to false and it will do nothing.
option_market.set_master_dataframe(volume=-1, maturity=[10,50],moneyness=[-50.5,50.5],rolling=30) #keep only options with maturities close to 30 days

option_df=option_market.master_dataframe ##extract from class
group=option_df.groupby(["date","exdate"])

##example
option_df=group.get_group(("2023-12-30", "2024-01-26")) ##retrieve a specific option-chain.




def kernel_regression(option_df,argument_dict):
    """
    This function will interpolate the IV curve using either (1)Locally Linear (2) Nadara watson (3)Nearest Neighbor. Then plot the corresponding CDF.
    Use sci-kit learns implementation here: https://fda.readthedocs.io/en/latest/auto_examples/plot_kernel_regression.html
    
    Please read: Before fitting the model, please augment the IV curve with synthetic points (interpolate_large_gaps). Then AFTER, augment the IV curve with extrapolated points
    using tail_extrapolation. These functions are found in main_options_market.py.
    Parameters
    ----------
    option_df : Pandas Dataframe
        dataframe.
    argument_dict : Dictionary
        contains arguments as key-value pairs.

    Returns
    -------
    Dictionary date:dataframe containing IV/pdf/strikes.

    """
    ###########################################################Starter code
    ####################Extracting arguments, depending on your approach you may need more arguments, so feel free to add more. 
    foldername=argument_dict["folder"]
    # example_argument=argument_dict["argument_key"]
    # method=argument_dict["Nadara watson"]
    # method=argument_dict["locally Linear"]
    # method=argument_dict["Nearest Neighbor"]

    ####################Extracting basic option information
    option_df=option_df.dropna()
    stock_price=option_df["stock_price"].values[0]
    R=option_df["risk_free_rate"].values[0]
    maturity=option_df["maturity"].values[0]
    ##date information
    date=option_df["date"].values[0]
    exdate=option_df["exdate"].values[0]
    
    #Extract strikes and IV
    original_iv=option_df["mid_iv"].values.reshape(-1)
    original_strikes=option_df["strike"].values.reshape(-1)
    
    ###check iv curve
    plt.scatter(original_strikes,original_iv)
    
    ###Interpolate large gaps
    synthetic_used = False 
    if (interpolate_large):
        strikes, iv,synthetic_strikes,synthetic_ivs= interpolate_large_gaps(original_strikes.reshape(-1),original_iv.reshape(-1))   
        if (len(synthetic_strikes))> 0:
            synthetic_used = True
    ##Check iv curve
    plt.scatter(synthetic_strikes,synthetic_ivs)
    plt.scatter(original_strikes,original_iv)
    
    ##Now augment with extrapolated points ( I use linear regression of the last 3 points to extrpolate)
    strikes,iv,extrapolated_strikes,extrapolated_iv=tail_extrapolation(strikes, iv)
    ##Check iv curve
    plt.figure(figsize=(10, 6))  # Set figure size for better visibility
    plt.scatter(synthetic_strikes, synthetic_ivs, label="Synthetic", color="blue", alpha=0.7)
    plt.scatter(original_strikes, original_iv, label="Original", color="red", alpha=0.7)
    plt.scatter(extrapolated_strikes, extrapolated_iv, label="Extrapolated", color="green", alpha=0.7)
    
    # Add legend, labels, and title
    plt.legend(loc="best")  # Automatically place the legend at the best location
    plt.title("IV Scatter Plot: Synthetic, Original, and Extrapolated Data", fontsize=14)
    plt.xlabel("Strikes", fontsize=12)
    plt.ylabel("Implied Volatility", fontsize=12)
    
    #Which should be the same as strikes/iv
    plt.scatter(strikes,iv)
    
    
    
    
    #####################################################To do list
    ###Instead of fitting polynomials, we fit using scikitlearns implementations. Make sure to use cross validation on the bandwidth or NNN parameters.
    ##Priortize readibility of code. If you can get them in 1 function sure, but if the code is convloluted then just make 3 functions, one for each method.
    interpolated_strikes = np.linspace(min(strikes), max(strikes), 1000).reshape(-1, 1)  
    
    ###implement scikit learn stuff. You train on the strikes/iv, which has everything already augmented. Then test on interpolated strikes.
    ###scikit_fda has this weird datastructure functional analysis, so feel free to chat-gpt to figure it out. Good luck.

    
    
    ###Plotting stage
    ##Plot the IV curve as before, the PDF, and CDF. Don't worry about the KDE.
    
    if plotting==True:
        current_directory = os.getcwd()
        figure1_folder = os.path.join(current_directory, foldername)
        if not os.path.exists(figure1_folder):
            os.makedirs(figure1_folder)
            print(f"Created folder: {figure1_folder}")
            
            
        fig, (ax1, ax2,ax3) = plt.subplots(3, 1, figsize=(10, 15))
        ax1.scatter(original_strikes, original_iv,color="red",label="IV Qoutes")
        ax1.scatter(extrapolated_strikes,extrapolated_iv,color="blue",label="Extrapolated IV")
        if synthetic_used:
            ax1.scatter(synthetic_strikes, synthetic_ivs, color="orange",label="Synthetic IV")
        ax1.plot(interpolated_strikes, interpolated_iv,label="Kernel Ridge IV")
        ax1.set_title(f"Kernel Ridge on {date} to {exdate}")
        ax1.set_xlabel("Strikes")
        ax1.set_ylabel("Implied Volatility")
        ax1.legend()
        ax1.set_title("IV curve")
        
        #plot pdf, make sure to delete the 
        ax2.plot( , ,label="PDF")
        ax2.set_title("PDF")
        ax2.legend()

        ax3.plot(pdf_strike,cdf)
        ax3.set_title("CDF")
        fig.suptitle(f"BTC PDF for {date} to {exdate}", fontsize=16)

        # Show the combined plot
        titlename_1=date+"_"+exdate
        save_path = os.path.join(figure1_folder, titlename_1)
        plt.savefig(save_path)
        plt.close()
        
        
    ##You may use your own name_convention for the dataframe below. Just put it in a dataframe like before    
    density_df=pd.DataFrame({"Strikes":interpolated_strikes,"PDF":PDF,"IV":interpolated_iv})

    return {date +","+ exdate: density_df}

    
