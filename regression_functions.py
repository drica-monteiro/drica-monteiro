import numpy as np    
import matplotlib.pyplot as plt # type: ignore
from sklearn.ensemble import GradientBoostingRegressor # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from classification_functions import new_dataset, quant, eps, comp_lambda

def plot_multiplemean_reg(x_train, x_test, y_train, n_tau, alpha):
    """
        Plots mean prediction versus tau when multiple means are stressed.
        Compute projection of x_test and train models in x_train, y_train. 
        Plots for all models: 'GB', 'RF'.

        Inputs:
            x_train: dataframe of training features
            x_test: dataframe of test featues
            y_train: dataframe of training targets
            n_tau: int value number of taus you want to compute 
            alpha: float value that partitions the observations
            col_name: str name of the column you want to stress
            model: str model to be used
    """

    col_names = ['lstat', 'dis', 'rm', 'nox', 'crim']
    plt.style.use("seaborn")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for col_name in col_names:
        ###compute lambdas
        list_lbd, _ = comp_lambda(x_test, col_name, n_tau, alpha)
        
        ###project the dataframe
        dfs = []
        for i in range(len(list_lbd)):
            df = new_dataset(x_test, col_name, list_lbd[i])
            dfs.append(df)

        ###instantiate the model
        reg_GB=GradientBoostingRegressor(max_depth=2)
        reg_GB.fit(x_train,y_train)

        ##calcula average prediction
        means = []
        for i in range(len(dfs)):
            prev_GB = reg_GB.predict(dfs[i])
            mean = np.mean(prev_GB)
            means.append(mean)      

        ###plot average prediction versus tau
        taus = np.linspace(-1, 1, n_tau)
        ax1.plot(taus, means, label = col_name)
        ax1.set_title('Gradient Boosting')
        ax1.set_ylabel('Average price')
        ax1.set_xlabel(r'$\tau$')
    ax1.legend(loc ='best', frameon = True, shadow=True, fontsize = 'x-small')
    plt.legend()
    #plt.show()

    for col_name in col_names:
        ###compute lambdas
        list_lbd, _ = comp_lambda(x_test, col_name, n_tau, alpha)
    
        ###project the dataframe
        dfs = []
        for i in range(len(list_lbd)):
            df = new_dataset(x_test, col_name, list_lbd[i])
            dfs.append(df)

        ###instantiate the model
        reg_RF = RandomForestRegressor(random_state=0, n_estimators=100)
        reg_RF.fit(x_train, y_train)

        ##calcula average predction
        means = []
        for i in range(len(dfs)):
            prev_RF = reg_RF.predict(dfs[i])
            mean = np.mean(prev_RF)
            means.append(mean)
        
        ###plot average prediction versus tau
        taus = np.linspace(-1, 1, n_tau)
        ax2.plot(taus, means)#, label = col_name)
        ax2.set_title('Random Forest')
        #ax1.set_ylabel('Average price')
        ax2.set_xlabel(r'$\tau$')
    #ax2.legend(loc ='best', frameon = True, shadow=True, fontsize = 'x-small')
    #plt.legend()
    fig.tight_layout()
    plt.show()

def plot_multiplemean_reg_var(x_train, x_test, y_train, n_tau, alpha):
    """
        Plots variance prediction versus tau when multiple means are stressed.
        Compute projection of x_test and train models in x_train, y_train. 
        Plots for all models: 'GB', 'RF'.

        Inputs:
            x_train: dataframe of training features
            x_test: dataframe of test featues
            y_train: dataframe of training targets
            n_tau: int value number of taus you want to compute 
            alpha: float value that partitions the observations
            col_name: str name of the column you want to stress
    """

    col_names = ['lstat', 'dis', 'rm', 'nox', 'crim']
    plt.style.use("seaborn")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for col_name in col_names:
        ###compute lambdas
        list_lbd, _ = comp_lambda(x_test, col_name, n_tau, alpha)
        
        ###project the dataframe
        dfs = []
        for i in range(len(list_lbd)):
            df = new_dataset(x_test, col_name, list_lbd[i])
            dfs.append(df)

        ###instantiate the model
        reg_GB=GradientBoostingRegressor(max_depth=2)
        reg_GB.fit(x_train,y_train)

        ##calcula variance prediction
        variances = []
        for i in range(len(dfs)):
            prev_GB = reg_GB.predict(dfs[i])
            var = np.var(prev_GB)
            variances.append(var)      

        ###plot average prediction versus tau
        taus = np.linspace(-1, 1, n_tau)
        ax1.plot(taus, variances, label = col_name)
        ax1.set_title('Gradient Boosting')
        ax1.set_ylabel('Price variance')
        ax1.set_xlabel(r'$\tau$')
    ax1.legend(loc ='best', frameon = True, shadow=True, fontsize = 'x-small')
    plt.legend()
    #plt.show()

    for col_name in col_names:
        ###compute lambdas
        list_lbd, _ = comp_lambda(x_test, col_name, n_tau, alpha)

        ###project the dataframe
        dfs = []
        for i in range(len(list_lbd)):
            df = new_dataset(x_test, col_name, list_lbd[i])
            dfs.append(df)

        ###instantiate the model
        reg_RF = RandomForestRegressor(random_state=0, n_estimators=100)
        reg_RF.fit(x_train, y_train)

        ##calcula variance predction
        variances = []
        for i in range(len(dfs)):
            prev_RF = reg_RF.predict(dfs[i])
            var = np.var(prev_RF)
            variances.append(var)
        
        ###plot average prediction versus tau
        taus = np.linspace(-1, 1, n_tau)
        ax2.plot(taus, variances)#, label = col_name)
        ax2.set_title('Random Forest')
        #ax1.set_ylabel('Average price')
        ax2.set_xlabel(r'$\tau$')
    #ax2.legend(loc ='best', frameon = True, shadow=True, fontsize = 'x-small')
    
    fig.tight_layout()
    plt.show()



def plot_mean_reg(x_train, x_test, y_train, n_tau, alpha, col_name, model):
    """
        Plots mean prediction versus mean of a stressed feature (col_name). 
        Compute projection of x_test and train models in x_train, y_train. 
        Plots for one model Gradient Boosting or Random Forest.

        Inputs:
           x_train: dataframe of training features
           x_test: dataframe of test featues
           y_train: dataframe of training targets
           n_tau: int value number of taus you want to compute 
           alpha: float value that partitions the observations
           col_name: str name of the column you want to stress
           model: str 'GB' or 'DT' regression model to be used
    """
    ###computa os lbds
    list_lbd, list_t = comp_lambda(x_test, col_name, n_tau, alpha)
    
    ###computa os novos dfs
    dfs = []
    for i in range(len(list_lbd)):
        data = new_dataset(x_test, col_name, list_lbd[i])
        dfs.append(data)  
        

    ##instantiate the model
    if model == 'GB':
        reg_GB=GradientBoostingRegressor(max_depth=2)
        reg_GB.fit(x_train,y_train)
    

        ##calcula portions of 1
        means = []
        for i in range(len(dfs)):
            prev_GB = reg_GB.predict(dfs[i])
            mean = np.mean(prev_GB)
            means.append(mean)

    if model == 'RF':
        ###instantiate the model
        reg_RF = RandomForestRegressor(random_state=0, n_estimators=100)
        reg_RF.fit(x_train, y_train)

        ##calcula variance predction
        means = []
        for i in range(len(dfs)):
            prev_RF = reg_RF.predict(dfs[i])
            mean = np.mean(prev_RF)
            means.append(mean)

            
    plt.style.use("seaborn")
    plt.plot(list_t, means, label = col_name)
    plt.title(f'{model}')
    plt.ylabel('Average price')
    plt.xlabel(f'Mean {col_name}')
    #plt.legend()
    plt.show()


def quatro_figs(x_train, x_test, y_train, n_tau, alpha):
    """
        Plots mean prediction versus tau when multiple means are stressed.
        Compute projection of x_test and train models in x_train, y_train. 
        Plots for all models: 'GB', 'RF'.

        Inputs:
            x_train: dataframe of training features
            x_test: dataframe of test featues
            y_train: dataframe of training targets
            n_tau: int value number of taus you want to compute 
            alpha: float value that partitions the observations
            col_name: str name of the column you want to stress
            model: str model to be used
    """

    col_names = ['lstat', 'dis', 'rm', 'nox', 'crim']
    plt.style.use("seaborn")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 4))
    for col_name in col_names:
        ###compute lambdas
        list_lbd, _ = comp_lambda(x_test, col_name, n_tau, alpha)
        
        ###project the dataframe
        dfs = []
        for i in range(len(list_lbd)):
            df = new_dataset(x_test, col_name, list_lbd[i])
            dfs.append(df)

        ###instantiate the model
        reg_GB=GradientBoostingRegressor(max_depth=2)
        reg_GB.fit(x_train,y_train)

        ##calcula average prediction
        means = []
        variances_gb = []
        for i in range(len(dfs)):
            prev_GB = reg_GB.predict(dfs[i])
            mean = np.mean(prev_GB)
            means.append(mean)
            var = np.var(prev_GB)
            variances_gb.append(var)      

        ###plot average prediction versus tau
        taus = np.linspace(-1, 1, n_tau)
        ax1.plot(taus, means, label = col_name)
        ax1.set_title('Gradient Boosting')
        ax1.set_ylabel('Average price')
        ax3.set_xlabel(r'$\tau$')
        ax3.plot(taus, variances_gb)#, label = col_name)
        ax3.set_ylabel('Price Variance')
    ax1.legend(loc ='best', frameon = True, shadow=True, fontsize = 'x-small')
    plt.legend()
    #plt.show()

    for col_name in col_names:
        ###compute lambdas
        list_lbd, _ = comp_lambda(x_test, col_name, n_tau, alpha)
    
        ###project the dataframe
        dfs = []
        for i in range(len(list_lbd)):
            df = new_dataset(x_test, col_name, list_lbd[i])
            dfs.append(df)

        ###instantiate the model
        reg_RF = RandomForestRegressor(random_state=0, n_estimators=100)
        reg_RF.fit(x_train, y_train)

        ##calcula average predction
        means = []
        variances_rf = []
        for i in range(len(dfs)):
            prev_RF = reg_RF.predict(dfs[i])
            mean = np.mean(prev_RF)
            means.append(mean)
            var = np.var(prev_RF)
            variances_rf.append(var) 
        
        ###plot average prediction versus tau
        taus = np.linspace(-1, 1, n_tau)
        ax2.plot(taus, means)#, label = col_name)
        ax2.set_title('Random Forest')
        #ax1.set_ylabel('Average price')
        ax4.set_xlabel(r'$\tau$')
        ax4.plot(taus, variances_rf)#, label = col_name)
    #ax2.legend(loc ='best', frameon = True, shadow=True, fontsize = 'x-small')
    #plt.legend()
    fig.tight_layout()
    plt.show()

        
        #ax3.set_title('Random Forest')
    #     ax3.set_ylabel('Price variance')
    #     ax3.set_xlabel(r'$\tau$')
    # #ax2.legend(loc ='best', frameon = True, shadow=True, fontsize = 'x-small')
    # #plt.legend()
    # fig.tight_layout()
    #plt.show()

        
    #     #ax3.set_title('Random Forest')
    #     #ax4.set_ylabel('Price variance')
    #     ax4.set_xlabel(r'$\tau$')
    # #ax2.legend(loc ='best', frameon = True, shadow=True, fontsize = 'x-small')
    # #plt.legend()
    # fig.tight_layout()
    # plt.show()




        
        

        