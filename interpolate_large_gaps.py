


def interpolate_large_gaps(strikes,iv):
    """
    This function is designed to handle the unusual behavior when we try to fit a curve to a function that is discontious. When there is a large gap between points,
    the fitted curve will do strange behavior. This function will first 
    (1) Calculate the average distance between strikes
    (2) If the gap between two consecutive IV points is more than say 5 times the average, linearly interpolate between those points until the average distance
    between those points less than 3 times the average distance.
    (3) Use numpy.interp() to interpolate between points.
    
    Take the Union of the original IVs and their respective strikes, with the interpolated strikes. 
    
    Parameters
    ----------
    strikes : N, np array
        DESCRIPTION.
    iv : N, np array
        DESCRIPTION.

    Returns
    Combined strikes: N, np array
    Combined IV: N,np array
    
    synthetic IV, N, Np array
    synthetic strike, N np array

    """
    ###Program goes here
    average_step=np.mean(np.diff(strikes.reshape(-1)))
    
    
    
    
    
    ###Check with plots
    plt.scatter(combined_strikes,combined_iv,color="red")
    plt.scatter(synthetic_strikes,synthetic_iv,color="blue")
    
    return combined_strikes,combined_iv,synthetic_strikes,synthetic_iv