############# 
### Functions to compute projected datasets

import numpy as np    
import matplotlib.pyplot as plt # type: ignore
from sklearn.tree import DecisionTreeClassifier # type: ignore
from sklearn.ensemble import GradientBoostingClassifier # type: ignore
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA # type: ignore
from sklearn.naive_bayes import GaussianNB as NB # type: ignore
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

###computing quatiles
def quant(alpha, col_name, df):
    """
        Computes the alpha-quantile of a column. Returns a scalar.

        Inputs:
            alpha: float value that partitions the observations
            df: dataframe of features to be used
            col_name: str name of the column for which the quantile will be computed 
    """

    df = df[col_name]
    quant = np.quantile(df, alpha)
    return quant 

### computing epsilons
def eps(df, col_name, tau, alpha):
    """
        Computes the epsilon value used for stressing the mean of a variable. 
        Returns the eps and the true mean of the column 'col_name'.

        Inputs:
            df: dataframe of features to be used
            col_name: str name of the column you want to stress
            tau: int value in [-1,1] 
            alpha: float value that partitions the observations

    """
    
    t0 = np.mean(df[col_name])
    if tau < 0:
        eps = tau * (t0 - quant(alpha, col_name, df))
    if tau > 0:
        eps = tau * (quant(1 - alpha, col_name, df) - t0)
    if tau == 0:
        eps = 0
    return eps, t0

### computing lambdas
def comp_lambda(df, col_name, n_tau, alpha):
    """
        Computes the Lagrange multipliers lambdas and the t values, which are the stressed means.
        Returns a list of lambdas and a list of t's.
        
        Inputs:
            df: dataframe of features to be used
            col_name: str name of the column you want to stress
            n_tau: int value number of taus you want to compute
            tau_size: int value multiplying the interval of tau
            alpha: float value that partitions the observations
    """
    taus = np.linspace(-1, 1, n_tau)
    list_lbd = []
    list_t = []
    for tau in taus:    
        ep, t0 = eps(df, col_name, tau, alpha)
        t = t0 + ep
        lambda_star = 2*t - 2*t0
        list_lbd.append(lambda_star)
        list_t.append(t)
        
    return list_lbd, list_t
    
##finding new observations
def new_dataset(df, col_name, lambda_star):
    """
        Computes new observations. Given a dataset, it computes its tranlated version (the projected dataset). 
        It is a translation in the column 'col_name' given by the optimal lambda. Returns a dataframe.

        Inputs:
           df: dataframe of features to be used
           col_name: str name of the column you want to stress
           lambda_star: int Lagrange multiplier
    """
    X_new = df.copy()
    X_newcol= X_new[col_name] + lambda_star/2
    X_new[col_name] = X_newcol
    return X_new


def plot_mean_pp1(x_train, x_test, y_train, n_tau, alpha, col_name):
    """
        Plots Portion of 1's versus mean of a stressed feature (column). 
        Compute projection of x_test and train models in x_train, y_train.
        Plots for all models Decision Tree, GradientBoosting, LDA and Naive Bayes.

        Inputs:
           x_train: dataframe of training features
           x_test: dataframe of test featues
           y_train: dataframe of training targets
           n_tau: int value number of taus you want to compute 
           alpha: float value that partitions the observations
           col_name: str name of the column you want to stress
    """

    ###compute lambdas
    list_lbd, list_t = comp_lambda(x_test, col_name, n_tau, alpha)
    
    ###compute new dataframes
    dfs = []
    for i in range(len(list_lbd)):
        data = new_dataset(x_test, col_name, list_lbd[i])
        dfs.append(data)

    ###instantiate the model
    clf_GB = GradientBoostingClassifier()
    clf_GB.fit(x_train, y_train)

    ##compute portions of 1
    portions_GB = []
    for i in range(len(dfs)):
        pred_GB = clf_GB.predict_proba(dfs[i])[:,1]
        Y_pred_GB=1*(pred_GB>0.5)
        n = Y_pred_GB.shape[0]
        portion = np.sum(Y_pred_GB)/n
        portions_GB.append(portion) 
    
    ##instantiate the model
    clf_DT=DecisionTreeClassifier(max_depth=5)
    clf_DT.fit(x_train, y_train)

    ##compute portions of 1
    portions_DT = []
    for i in range(len(dfs)):
        pred_DT = clf_DT.predict_proba(dfs[i])[:,1]
        Y_pred_DT=1*(pred_DT>0.5)
        n = Y_pred_DT.shape[0]
        portion = np.sum(Y_pred_DT)/n
        portions_DT.append(portion)
   
    ##instantiate the model
    sklearn_lda = LDA()
    lda = sklearn_lda.fit(x_train, y_train)
            
    ##compute portions of 1
    portions_LDA = []
    for i in range(len(dfs)):
        pred_lda = lda.predict_proba(dfs[i])[:,1]
        Y_pred_lda=1*(pred_lda>0.5)
        n = Y_pred_lda.shape[0]
        portion = np.sum(Y_pred_lda)/n
        portions_LDA.append(portion)


    ##instantiate the model
    clf_NB = NB()
    clf_NB.fit(x_train, y_train)
            
    ##compute portions of 1
    portions_NB = []
    for i in range(len(dfs)):
        X_test_prob_NB = clf_NB.predict_proba(dfs[i])[:,1]
        Y_pred_NB=1*(X_test_prob_NB>0.5)
        n = Y_pred_NB.shape[0]
        portion = np.sum(Y_pred_NB)/n
        portions_NB.append(portion)
           
    ### plots portion 1's versus mean of column for all 4 models
    plt.style.use("seaborn")
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 5))
    ax1.plot(list_t, portions_GB, label = col_name)
    ax1.set_title('GradientBoosting')
    ax1.set_xlabel(f'Mean {col_name}')
    ax1.set_ylabel('Portion of 1s')

    ax2.plot(list_t, portions_DT, label = col_name)
    ax2.set_title('Decision Tree')
    ax2.set_xlabel(f'Mean {col_name}')
    #ax2.set_ylabel('Portion of 1s')
    
    ax3.plot(list_t, portions_LDA, label = col_name)
    ax3.set_title('LDA')
    ax3.set_xlabel(f'Mean {col_name}')
    #ax3.set_ylabel('Portion of 1s')

    ax4.plot(list_t, portions_NB, label = col_name)
    ax4.set_title('Naive Bayes')
    ax4.set_xlabel(f'Mean {col_name}')
    #ax4.set_ylabel('Portion of 1s')
    
    #plt.legend(loc ='best', frameon = True, shadow=True)
    fig.tight_layout()
    plt.show()        


def plot_mean_pp1_onemodel(x_train, x_test, y_train, n_tau, alpha, col_name, model):
    """
        Plots Portion of 1's versus mean of a stressed feature (column).
        Compute projection of x_test and train models in x_train, y_train. 
        Plots for one model in 'GB', 'DT', 'LDA' or 'NB'.

        Inputs:
            x_train: dataframe of training features
            x_test: dataframe of test featues
            y_train: dataframe of training targets
            n_tau: int value number of taus you want to compute 
            alpha: float value that partitions the observations
            col_name: str name of the column you want to stress
            model: str model to be used
    """

    if model == 'GB':
        ###compute lambdas
        list_lbd, list_t = comp_lambda(x_test, col_name, n_tau, alpha)
        
        ###compute new dataframes
        dfs = []
        for i in range(len(list_lbd)):
            df = new_dataset(x_test, col_name, list_lbd[i])
            dfs.append(df)

        ##instantiate the model
        clf_GB=GradientBoostingClassifier()
        clf_GB.fit(x_train,y_train)

        ##compute portions of 1
        portions = []
        for i in range(len(dfs)):
            pred_GB = clf_GB.predict_proba(dfs[i])[:,1]
            Y_pred_GB=1*(pred_GB>0.5)
            n = Y_pred_GB.shape[0]
            portion = np.sum(Y_pred_GB)/n
            portions.append(portion)

    if model == 'DT':
        ###compute lambdas
            list_lbd, list_t = comp_lambda(x_test, col_name, n_tau, alpha)
        
        ###compute new dataframes
            dfs = []
            for i in range(len(list_lbd)):
                df = new_dataset(x_test, col_name, list_lbd[i])
                dfs.append(df)

            ##instantiate the model
            clf_DT=DecisionTreeClassifier(max_depth=5)
            clf_DT.fit(x_train,y_train)

            ##compute portions of 1
            portions = []
            for i in range(len(dfs)):
                pred_DT = clf_DT.predict_proba(dfs[i])[:,1]
                Y_pred_DT=1*(pred_DT>0.5)
                n = Y_pred_DT.shape[0]
                portion = np.sum(Y_pred_DT)/n
                portions.append(portion)

    if model == 'LDA':
            ###compute lambdas
            list_lbd, list_t = comp_lambda(x_test, col_name, n_tau, alpha)
        
            ###compute new dataframes
            dfs = []
            for i in range(len(list_lbd)):
                df = new_dataset(x_test, col_name, list_lbd[i])
                dfs.append(df)

            ##instantiate the model
            sklearn_lda = LDA()
            lda = sklearn_lda.fit(x_train, y_train)
                
            ##compute portions of 1
            portions = []
            for i in range(len(dfs)):
                pred_lda = lda.predict_proba(dfs[i])[:,1]
                Y_pred_lda=1*(pred_lda>0.5)
                n = Y_pred_lda.shape[0]
                portion = np.sum(Y_pred_lda)/n
                portions.append(portion)


    if model == 'NB':
        ###compute lambdas
        list_lbd, list_t = comp_lambda(x_test, col_name, n_tau, alpha)
    
        ###computa os novos dfs
        dfs = []
        for i in range(len(list_lbd)):
            df = new_dataset(x_test, col_name, list_lbd[i])
            dfs.append(df)

        ##instantiate the model
        NB_class = NB()
        NB_class.fit(x_train, y_train)
            
        ##compute portions of 1
        portions = []
        for i in range(len(dfs)):
            X_test_prob_NB = NB_class.predict_proba(dfs[i])[:,1]
            Y_pred_NB=1*(X_test_prob_NB>0.5)
            n = Y_pred_NB.shape[0]
            portion = np.sum(Y_pred_NB)/n
            portions.append(portion)
        
    ### plot portion 1's versus means
    plt.style.use("seaborn")
    plt.plot(list_t, portions)
    plt.title(model)
    plt.ylabel('Portion of 1s')
    plt.xlabel(f'Mean {col_name}')
    plt.show()

def plot_multiplemean(x_train, x_test, y_train, n_tau, alpha):
    """
        Plots Portion of 1's versus tau when multiple means are stressed.
        Compute projection of x_test and train models in x_train, y_train. 
        Plots for all models: 'GB', 'DT', 'LDA' or 'NB'.

        Inputs:
            x_train: dataframe of training features
            x_test: dataframe of test featues
            y_train: dataframe of training targets
            n_tau: int value number of taus you want to compute 
            alpha: float value that partitions the observations
            col_name: str name of the column you want to stress
            model: str model to be used
    """

    col_names = ['Education-Num', 'Age', 'Capital Gain', 'Capital Loss', 'Hours per week']
    plt.style.use("seaborn")
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 4))
    for col_name in col_names:
        ###compute lambdas
        list_lbd, _ = comp_lambda(x_test, col_name, n_tau, alpha)
        
        ###project the dataframe
        dfs = []
        for i in range(len(list_lbd)):
            df = new_dataset(x_test, col_name, list_lbd[i])
            dfs.append(df)

        ###instantiate the model
        clf_GB=GradientBoostingClassifier()
        clf_GB.fit(x_train,y_train)

        ###compute pp1
        portions = []
        for i in range(len(dfs)):
            pred_GB = clf_GB.predict_proba(dfs[i])[:,1]
            Y_pred_GB=1*(pred_GB>0.5)
            n = Y_pred_GB.shape[0]
            portion = np.sum(Y_pred_GB)/n
            portions.append(portion)

        ###plot pp1 versus tau
        taus = np.linspace(-1, 1, n_tau)
        ax1.plot(taus, portions, label = col_name)
        ax1.set_title('Gradient Boosting')
        ax1.set_ylabel('Portion of 1s')
        ax1.set_xlabel(r'$\tau$')
    ax1.legend(loc ='best', frameon = True, shadow=True, fontsize = 'x-small')
    #plt.legend()
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
        clf_DT=DecisionTreeClassifier(max_depth=5)
        clf_DT.fit(x_train,y_train)

        ###compute pp1
        portions = []
        for i in range(len(dfs)):
            pred_DT = clf_DT.predict_proba(dfs[i])[:,1]
            Y_pred_DT=1*(pred_DT>0.5)
            n = Y_pred_DT.shape[0]
            portion = np.sum(Y_pred_DT)/n
            portions.append(portion)

        ###plot pp1 versus tau
        taus = np.linspace(-1, 1, n_tau)
        ax2.plot(taus, portions, label = col_name)
        ax2.set_title('Decision Tree')
        ax2.set_xlabel(r'$\tau$')
    
    for col_name in col_names:
        ###compute lambdas
        list_lbd, _ = comp_lambda(x_test, col_name, n_tau, alpha)

        ###project the dataframe
        dfs = []
        for i in range(len(list_lbd)):
            df = new_dataset(x_test, col_name, list_lbd[i])
            dfs.append(df)

        ###instantiate the model
        sklearn_lda = LDA()
        lda = sklearn_lda.fit(x_train, y_train)
        
    
        ###compute pp1
        portions = []
        for i in range(len(dfs)):
            pred_lda = lda.predict_proba(dfs[i])[:,1]
            Y_pred_lda=1*(pred_lda>0.5)
            n = Y_pred_lda.shape[0]
            portion = np.sum(Y_pred_lda)/n
            portions.append(portion)
            

        ###plot pp1 versus tau
        taus = np.linspace(-1, 1, n_tau)
        ax3.plot(taus, portions, label = col_name)
        ax3.set_title("LDA")
        ax3.set_xlabel(r'$\tau$')
   
    for col_name in col_names:
        ###compute lambdas
        list_lbd, _ = comp_lambda(x_test, col_name, n_tau, alpha)

        ###project the dataframe
        dfs = []
        for i in range(len(list_lbd)):
            df = new_dataset(x_test, col_name, list_lbd[i])
            dfs.append(df)

        ###instantiate the model
        NB_class = NB()
        NB_class.fit(x_train, y_train)
    
        ###compute pp1
        portions = []
        for i in range(len(dfs)):
            X_test_prob_NB = NB_class.predict_proba(dfs[i])[:,1]
            Y_pred_NB=1*(X_test_prob_NB>0.5)
            n = Y_pred_NB.shape[0]
            portion = np.sum(Y_pred_NB)/n
            portions.append(portion)

        ###plot pp1 versus tau
        taus = np.linspace(-1, 1, n_tau)
        ax4.plot(taus, portions, label = col_name)
        ax4.set_title('Naive Bayes')
        ax4.set_xlabel(r'$\tau$')
        

    #plt.legend(loc ='best', frameon = True, shadow=True, fontsize = 'x-small')
    fig.tight_layout()
    plt.show()  

def stress_twomeans(x_test, col_name1, col_name2, n_tau, alpha):
    """
        Stress two means simultaneously.
        Returns a list of projected dataframes, a list of new mean col_name1 and a list of new mean col_name2.

        Inputs:
            x_test: dataframe of test featues
            col_name1: str name of the first column you want to stress
            col_name2: str name of the second column you want to stress
            n_tau: int value number of taus you want to compute 
            alpha: float value that partitions the observations
    """
    ###computing lambda opt for each column:
    #print("chegou0")
    list_lbd1, list_t1 = comp_lambda(x_test, col_name1, n_tau, alpha)
    list_lbd2, list_t2 = comp_lambda(x_test, col_name2, n_tau, alpha)
    lbds = np.array([list_lbd1, list_lbd2]).T
    
    ###creating a list of datasets with new columns
    dfs = []
    for i in range(lbds.shape[0]):
        df_new = x_test.copy()
        df_new_col1 = df_new[col_name1] + lbds[i,0]/2
        df_new[col_name1] = df_new_col1
        
        df_new_col2 = df_new[col_name2] + lbds[i,1]/2
        df_new[col_name2] = df_new_col2
        
        dfs.append(df_new)
    return dfs, list_t1, list_t2

def plot_twomeans(x_train, x_test, y_train, col_name1, col_name2, n_tau, alpha, model):
    """
        Stress two means simultaneously then plots mean col_name1 vs mean col_name2 with colorbar representing portion of 1's.
        Choose one of the models in Decision Tree, GradientBoosting, Linear Discriminant Analysis or Naive Bayes.

        Inputs:
            x_train: dataframe of training features
            x_test: dataframe of test featues
            y_train: dataframe of training targets
            col_name1: str name of the first column you want to stress
            col_name2: str name of the second column you want to stress
            alpha: float value that partitions the observations
            n_tau: int value number of taus you want to compute
            model: str 'DT', 'GB', 'LDA' or 'NB' classification model to be used
            
    """
    dfs, list_t1, list_t2 = stress_twomeans(x_test, col_name1, col_name2, n_tau, alpha)
    
    if model == 'GB':
        ###instantiate the model
        clf_GB=GradientBoostingClassifier()
        clf_GB.fit(x_train,y_train)

        ###compute pp1
        portions = []
        for i in range(len(dfs)):
            pred_GB = clf_GB.predict_proba(dfs[i])[:,1]
            Y_pred_GB=1*(pred_GB>0.5)
            n = Y_pred_GB.shape[0]
            portion = np.sum(Y_pred_GB)/n
            portions.append(portion)
        
        ###instantiate the model
    if model == 'DT':
        clf_DT=DecisionTreeClassifier(max_depth=5)
        clf_DT.fit(x_train, y_train)

        ###compute pp1
        portions = []
        for i in range(len(dfs)):
            pred_DT = clf_DT.predict_proba(dfs[i])[:,1]
            Y_pred_DT=1*(pred_DT>0.5)
            n = Y_pred_DT.shape[0]
            portion = np.sum(Y_pred_DT)/n
            portions.append(portion)
 
    if model == 'LDA':
        ###instantiate the model
        sklearn_lda = LDA()
        lda = sklearn_lda.fit(x_train, y_train)
         
        ###compute pp1
        portions = []
        for i in range(len(dfs)):
            pred_lda = lda.predict_proba(dfs[i])[:,1]
            Y_pred_lda=1*(pred_lda>0.5)
            n = Y_pred_lda.shape[0]
            portion = np.sum(Y_pred_lda)/n
            portions.append(portion)
                    
    if model == 'NB':
        ###instantiate the model
        NB_class = NB()
        NB_class.fit(x_train, y_train)
        
        ###compute pp1
        portions = []
        for i in range(len(dfs)):
            X_test_prob_NB = NB_class.predict_proba(dfs[i])[:,1]
            Y_pred_NB=1*(X_test_prob_NB>0.5)
            n = Y_pred_NB.shape[0]
            portion = np.sum(Y_pred_NB)/n
            portions.append(portion)
            
    ###plot with colorbar
    plt.style.use("seaborn")
    fig, ax = plt.subplots(1, 1, figsize=(5,3), dpi=150)
    sc=ax.scatter(list_t1, list_t2, c=portions, marker='o', edgecolor='k', cmap=plt.cm.RdBu_r)
    cbar=fig.colorbar(sc, ax=ax, label='Portion of 1s')
    ax.set_xlabel(f'Average {col_name1}')
    ax.set_ylabel(f'Average {col_name2}')
    ax.set_title(model)        


    