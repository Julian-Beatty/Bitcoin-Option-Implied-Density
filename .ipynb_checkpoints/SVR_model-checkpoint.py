import numpy as np
np.float_ = np.float64
from skfda.misc.hat_matrix import LocalLinearRegressionHatMatrix
from skfda.ml.regression._kernel_regression import KernelRegression
from sklearn.model_selection import GridSearchCV
from main_option_market import*
import pdb
    

def local_linear(option_df,argument_dict,interpolate_large = True):
    """
    Description of function: This function takes in an option dataframe, which contains the option quotes on a particular day, with a specific maturity.
    It will fit a N order polynomial to this IV curve to smooth out the implied volatilities, then convert them to calls, numerically differentiate twice and save the plots.
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

    foldername=argument_dict["folder"]
    
    ##Extract Implied volatility qoutes and strikes
    original_iv=option_df["mid_iv"].values.reshape(-1)
    original_strikes=option_df["strike"].values.reshape(-1)


    #######TO DO: 
    synthetic_used = False 
    if (interpolate_large):
        strikes, iv,synthetic_strikes,synthetic_iv= interpolate_large_gaps(original_strikes,original_iv)   
        if (len(synthetic_strikes))> 0:
            synthetic_used = True

    
    ### Extrapolation
    combined_strikes,combined_iv,extrapolated_strikes,extrapolated_iv=tail_extrapolation(strikes,iv)


    # Create an array of bandwidth for GridSearchCV
    bandwidth = np.logspace(0.3, 1, num=len(original_strikes))


    # Set up GridSearchCV with Kernel Regression using LocalLinearRegressionHatMatrix
    llr = GridSearchCV(
        KernelRegression(kernel_estimator=LocalLinearRegressionHatMatrix()),
        param_grid={'kernel_estimator__bandwidth': bandwidth})


    # Fit the model
    llr.fit(combined_strikes,combined_iv)  

    ##Define interpolation grid
    interpolated_strikes=np.linspace(combined_strikes[0], combined_strikes[-1],1000)
    # Evaluate the polynomial at the interpolated strike prices
    interpolated_iv = llr.predict(interpolated_strikes)
    
    
    ##Convert implied volatilities back to call prices
    R=option_df["risk_free_rate"].values[0] ##Risk free rate
    T=option_df["maturity"].values[0]   ##Option maturity
    underlying_price=option_df["underlying_price"].values[0] ##price of the underlying
    
    
    ##Black scholes formula to convert IV to prices
    interpolated_prices = py_vollib.black_scholes.black_scholes('c', underlying_price,interpolated_strikes, T, R,interpolated_iv).values.reshape(-1)
    
    ##visualize calls
    #plt.plot(interpolated_strikes,interpolated_prices)
    
    ##Numerically differentiate
    CDF,PDF=numerical_differentiation(interpolated_strikes, interpolated_prices)

  
    ##Normalize PDF (making sure it integrates to 1, clip any negative values to zero)
    PDF=normalize_pdf(interpolated_strikes,PDF)
    ##visualize PDF and CDF
    # plt.plot(interpolated_strikes,PDF)
    # plt.plot(interpolated_strikes,CDF)


    
    interpolated_strikes = interpolated_strikes[2:-2]
    interpolated_iv = interpolated_iv[2:-2]

    interpolated_prices = interpolated_prices[2:-2]
    CDF = CDF[2:-2]
    PDF = PDF[2:-2]
    
    
    
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
    ax1.scatter(original_strikes, original_iv,color="red")
    ax1.scatter(extrapolated_strikes,extrapolated_iv,label="Extrapolated points")
    # Add synthetic points if interpolate_large is True
    if synthetic_used:
        ax1.scatter(synthetic_strikes, synthetic_iv, color="blue", label="Synthetic Points")
    ax1.scatter(strikes, iv,color="red",label="Original Points")
    ax1.plot(interpolated_strikes, interpolated_iv)
    ax1.set_title(f"Local linear interpolation on {date} to {exdate}")
    ax1.set_xlabel("Strikes")
    ax1.set_ylabel("Implied Volatility")
    ax1.legend()

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
    density_df=pd.DataFrame({"strike":interpolated_strikes.reshape(-1),"PDF":PDF.reshape(-1),"IV":interpolated_iv.reshape(-1)})

    return {date +","+ exdate: density_df}
