option_market=df.copy()
from scipy.interpolate import CubicSpline

def spline(option_market):
    strike=option_market["strike"].values
    stock_price=option_market["stock_price"].values
    iv=option_market["mid_iv"].values
    maturity=optionmarket["maturity"].values[0]
    risk_free_rate=option_market["risk_free_rate"].values
    plt.scatter(strike,iv)
    cs = CubicSpline(strike, iv)
    x_new = np.linspace(min(strike), max(strike), 400)
    y_new = cs(x_new)
    
    plt.plot(x_new,y_new)
    plt.scatter(strike,iv)
    dense_calls = py_vollib.black_scholes.black_scholes('c', option_market['underlying_price'].values[0], x_new, maturity, risk_free_rate[0], y_new).values.reshape(-1)  # 12.111581435
    plt.plot(x_new,dense_calls)
    _,pdf=numerical_differentiation(x_new,dense_calls)
    Pdf=pdf*np.exp(-risk_free_rate[0]*maturity)
    plt.plot(x_new,Pdf)
    pdf_ml_strike=x_new[Pdf>0]
    pdf_ml=Pdf[Pdf>0]
    plt.plot(pdf_ml_strike,pdf_ml)
    KDE_grid=np.linspace(min(x_new), max(x_new),1000)
    gkde_ml=gaussian_kde(pdf_ml_strike,'silverman',pdf_ml)
    gkdebw_ml=float(gkde_ml.__getattribute__("factor"))/2
    gkde_ml=gaussian_kde(pdf_ml_strike,gkdebw_ml,pdf_ml)
    kpdf_ml=gkde_ml(KDE_grid)
    plt.plot(KDE_grid,kpdf_ml)
    plt.plot(pdf_ml_strike,pdf_ml)
    

