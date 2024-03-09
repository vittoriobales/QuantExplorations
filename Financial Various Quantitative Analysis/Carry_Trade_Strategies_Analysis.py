import pandas as pd
import numpy as np
from statistics import stdev,mean
from scipy.stats import kurtosis
from scipy.stats import skew
import matplotlib.pyplot as plt
import seaborn as sns
import math
import yfinance as yf

#import data from excel
fwd_rates_df = pd.read_excel(r"C:\Users\User\Desktop\203\International Finance\Midterm assignment\fwd_rates.xlsm", sheet_name="fwd_clean")
spot_rates_df = pd.read_excel(r"C:\Users\User\Desktop\203\International Finance\Midterm assignment\spot_rates.xlsm",sheet_name="spot_clean_monthly")



#Cleaning data
spot_rates_df = spot_rates_df.drop([0,188,189,190,191])
cols = [1,27]
spot_rates_df.drop(spot_rates_df.columns[cols],axis=1,inplace=True)
spot_rates_df=spot_rates_df.rename(columns={"Unnamed: 0":"Date"})
fwd_rates_df=fwd_rates_df.rename(columns={"Name":"Date"})
spot_rates_df = spot_rates_df.set_index("Date")
fwd_rates_df = fwd_rates_df.set_index("Date")
fwd_rates_df = fwd_rates_df.drop(["Code"])

fwd_rates_df.index = pd.date_range('2004-03-10','2019-10-01', freq='MS').strftime("%m-%Y").tolist()
spot_rates_df.index = pd.date_range('2004-03-10','2019-10-01', freq='MS').strftime("%m-%Y").tolist()
### calculate spot return and excess return

##spot returns
spot_returns = np.log(spot_rates_df.astype('float')).diff()

##excess returns

 # COMPLETE FUNCTION EXCESS RETURNS
def excess_returns_fun(spot,fwd,rows,columns):
    excess_return=pd.DataFrame(index=range(0,rows),columns=range(0,columns)).fillna(value=0)
    df1=spot.copy()
    df2=fwd.copy()
    j=0
    i=0
    for j in list(range(0,columns)):   
        for i in list(range(0,rows-1)):
            excess_return.iloc[i+1,j]=np.log(df1.iloc[i+1,j]/(df2.iloc[i,j]))
            i+1
        j+1
    excess_return.index= spot_returns.index
    excess_return.columns = spot_returns.columns
    return excess_return

excess_returns = excess_returns_fun(spot_rates_df,fwd_rates_df,187,25)

###calculate forward premiums

 # COMPLETE FUNCTION FORWARD PREMIUMS
def forward_premiums(spot,fwd,rows,columns):
    forward_premiums=pd.DataFrame(index=range(0,rows),columns=range(0,columns)).fillna(value=0)

    df1=spot.copy()
    df2=fwd.copy()
    j=0
    i=0
    for j in list(range(0,columns)):   
        for i in list(range(0,rows-1)):
            forward_premiums.iloc[i,j]=np.log(df2.iloc[i,j]) - np.log(df1.iloc[i,j])
            i+1
        j+1
    forward_premiums.index= spot_returns.index
    forward_premiums.columns = spot_returns.columns
    return forward_premiums

forward_premiums = forward_premiums(spot_rates_df,fwd_rates_df,187,25)


### Carry trade strategies 


##function to rank and create the 5 portfolios
def rank_portfolios(forward_premiums,n_assets):
    df1 = forward_premiums
    x = pd.DataFrame({n: df1.T[col].nlargest(n_assets).index.tolist() 
                  for n, col in enumerate(forward_premiums.T)}).T
    x.index = forward_premiums.index

    portfolio1 = x.iloc[:,range(0,5)]
    portfolio2 = x.iloc[:,range(5,10)]
    portfolio3 = x.iloc[:,range(10,15)]
    portfolio4 = x.iloc[:,range(15,20)]
    portfolio5 = x.iloc[:,range(20,25)]
    
    return portfolio1,portfolio2,portfolio3,portfolio4,portfolio5    

ranked_portfolio = rank_portfolios(forward_premiums,25)


# Function to calculate the portfolios' excess returns

def portfolio_returns(excess_returns,forward_premiums,index_portfolio,rows,columns,n_asset_portfolios):
    returnportfolio=pd.DataFrame(index=range(0,rows+1),columns=range(0,n_asset_portfolios)).fillna(value=0)
    df1=excess_returns.copy()
    df2 = forward_premiums.copy()
    df1.index = list(range(0,rows))
    df1.columns = list(range(0,columns))
    df2.index = list(range(0,rows))
    df2.columns = list(range(0,columns))
    xy= rank_portfolios(df2,25)
    portfolio = xy[index_portfolio]
    i=
    j=0
    for j in list(range(0,n_asset_portfolios)):
            for i in list(range(0,rows-1)):
                returnportfolio.iloc[i,j] = df1.iloc[i+1,portfolio.iloc[i+1,j]]
                i+1
            j+1
    returnportfolio.index= excess_returns.index        
    returns= returnportfolio.mean(axis=1)
    return returns

returns_portfolio_1 = portfolio_returns(excess_returns,forward_premiums,0,187,25,5)
returns_portfolio_2 = portfolio_returns(excess_returns,forward_premiums,1,187,25,5)
returns_portfolio_3 = portfolio_returns(excess_returns,forward_premiums,2,187,25,5)
returns_portfolio_4 = portfolio_returns(excess_returns,forward_premiums,3,187,25,5)
returns_portfolio_5 = portfolio_returns(excess_returns,forward_premiums,4,187,25,5)

 #Function to calculate returns od the carry trade strategy 
  #portfolio 5 -> highest performance  / portfolio 1 -> lowest performance
def carry_trade_strategy(returns_portfolio_1,returns_portfolio_5): 
    x = []
    i=0
    n =list(range(0,len(returns_portfolio_1)))
    for i in n:  
        x.append(returns_portfolio_5[i] - returns_portfolio_1[i])
        i+1
    carry_trade_returns = pd.Series(x,index=returns_portfolio_1.index)
    return carry_trade_returns

returns_carry_trade=carry_trade_strategy(returns_portfolio_1,returns_portfolio_5)    
  


# descriptive statistics

statistics_portfolios = pd.DataFrame(index=['Portfolio 1','Portfolio 2','Portfolio 3', 'Portfolio 4', 'Portfolio 5','Carry Trade'])
statistics_portfolios['expected_return']= [sum(returns_portfolio_1),sum(returns_portfolio_2),mean(returns_portfolio_3),sum(returns_portfolio_4),sum(returns_portfolio_5),sum(returns_carry_trade)]
statistics_portfolios['std_dev'] = [stdev(returns_portfolio_1),stdev(returns_portfolio_2),stdev(returns_portfolio_3),stdev(returns_portfolio_4),stdev(returns_portfolio_5),stdev(returns_carry_trade)]
statistics_portfolios['kurtosis'] = [kurtosis(returns_portfolio_1),kurtosis(returns_portfolio_2),kurtosis(returns_portfolio_3),kurtosis(returns_portfolio_4),kurtosis(returns_portfolio_5),kurtosis(returns_carry_trade)]
statistics_portfolios['skeweness'] = [skew(returns_portfolio_1),skew(returns_portfolio_2),skew(returns_portfolio_3),skew(returns_portfolio_4),skew(returns_portfolio_5),skew(returns_carry_trade)]



###calculate annualized returns and volatility

# annualized returns 
def annualized_returns(returns_portfolio):
    x=pd.DataFrame(returns_portfolio)
    x['years']=pd.DatetimeIndex(returns_portfolio.index).year
    x['months']=pd.DatetimeIndex(returns_portfolio.index).month
    table = pd.pivot_table(x, values=0, index=['years'], columns=['months'])
    annual_returns = table.sum(axis=1)
    return annual_returns
   
# annualized volatility
def annualized_volatility(returns_portfolio):
    x=pd.DataFrame(returns_portfolio)
    x['years']=pd.DatetimeIndex(returns_portfolio.index).year
    x['months']=pd.DatetimeIndex(returns_portfolio.index).month
    table = pd.pivot_table(x, values=0, index=['years'], columns=['months'])
    annual_vol = table.std(axis=1)
    return annual_vol   

annualized_std_portfolio_1 = annualized_volatility(returns_portfolio_1)
annualized_std_portfolio_2 = annualized_volatility(returns_portfolio_2)
annualized_std_portfolio_3 = annualized_volatility(returns_portfolio_3)
annualized_std_portfolio_4 = annualized_volatility(returns_portfolio_4)
annualized_std_portfolio_5 = annualized_volatility(returns_portfolio_5)
annualized_std_carry_trade = annualized_volatility(returns_carry_trade)
annualized_std = pd.concat([annualized_std_portfolio_1,annualized_std_portfolio_2,annualized_std_portfolio_3,annualized_std_portfolio_4,annualized_std_portfolio_5,annualized_std_carry_trade],axis=1)
annualized_std.columns = statistics_portfolios.index
    
annualized_returns_portfolio_1 = annualized_returns(returns_portfolio_1)
annualized_returns_portfolio_2 = annualized_returns(returns_portfolio_2)
annualized_returns_portfolio_3 = annualized_returns(returns_portfolio_3)
annualized_returns_portfolio_4 = annualized_returns(returns_portfolio_4)
annualized_returns_portfolio_5 = annualized_returns(returns_portfolio_5)
annualized_returns_carry_trade = annualized_returns(returns_carry_trade)




### GRAPHS


#histogram portfolios' returns
colors = ['#E69F00', '#56B4E9', '#F0E442', '#009E73', '#D55E00']
names = ['returns_portfolio_1', 'returns_portfolio_2', 'returns_portfolio_3', 'returns_portfolio_4', 'returns_portfolio_5']
plt.hist([returns_portfolio_1, returns_portfolio_2, returns_portfolio_3, returns_portfolio_4, returns_portfolio_5],
         bins = int(180/15), color = colors, label=names)
plt.legend()
plt.title('Portfolios Returns')


# plot portfolios' returns
portfolios_cum_returns= pd.concat([returns_portfolio_1, returns_portfolio_2, returns_portfolio_3, returns_portfolio_4, returns_portfolio_5,returns_carry_trade],axis=1)
portfolios_cum_returns = portfolios_cum_returns.cumsum()
plt.plot(portfolios_cum_returns)
plt.legend(['returns_portfolio_1', 'returns_portfolio_2', 'returns_portfolio_3', 'returns_portfolio_4', 'returns_portfolio_5','carry_trade_returns'],fancybox= True,framealpha=0.4 ,loc='upper left')
plt.title('Portfolios Returns vs carry trade returns')

#plot carry trade returns
carry_trade_cum_returns=returns_carry_trade.cumsum()
plt.plot(carry_trade_cum_returns)
plt.title('Carry trade returns')


# Density Plot and Histogram of carry trade returns
sns.distplot(returns_carry_trade, hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
plt.title('Carry Trade returns distribution')
#statistics portofolios
sns.heatmap(statistics_portfolios,annot=True,cmap="YlGnBu")
plt.title('Portfolios and carry trade statistics')

#volatility per year
sns.heatmap(annualized_std,annot=True,cmap="YlGnBu")
plt.title('Standard deviation per year')

# histogram annual carry trade returns
plt.hist(annualized_returns_carry_trade.index,weights=annualized_returns_carry_trade, color = 'blue', edgecolor = 'black',
         bins = int(180/5))
plt.title('Carry Trade returns per year')



### cumulative wealth

cumulative_wealth_carry_trade = carry_trade_cum_returns.apply(lambda x: 1*math.exp(x))


spy = yf.download("SPY", start="2004-04-01", end="2019-10-10",interval = "1mo")
spy=spy.dropna()
spy=spy['Adj Close']

spy_cum_ret=np.log(spy).diff().cumsum()
cumulative_wealth_SPY = spy_cum_ret.apply(lambda x: 1*math.exp(x))

data=pd.concat([cumulative_wealth_carry_trade,cumulative_wealth_SPY],axis=1)

plt.plot(data)
plt.legend(['carry trade','SPY'])
plt.title('cumulative wealth - carry trade vs SPY - 1$ invested')