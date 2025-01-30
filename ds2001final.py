# -*- coding: utf-8 -*-
#imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency as chi2
from scipy.stats import pointbiserialr as pbsr

'''
consulted https://www.pythonfordatascience.org for running a chi-squared test
consulted https://www.statology.org for running a point biserial test
'''

#complete data
DATA = r"C:\Users\arand\Desktop\NEU\ds2001\data_files\clean_dataset.csv"
#quantitative data with approvals
DATA_ABRV = r"C:\Users\arand\Desktop\NEU\ds2001\data_files\cleaned_dataset(abrv).csv"


def get_qual_data(df, col1, col2):
    """ gets the number of induviduals in every unique combination of two variables
        from the data frame
    
    params:
        df: pandas dataframe with the relevant data
        col1: string representing first dataframe column
        col2: string representing second dataframe column
        
    returns:
        data: pandas dataframe of every unique combination of the two variable 
              values and the corresponding number of times that combination has 
              appeared
    """
    data = {col1 : []}
    y_list = []
    
    for ind in df.index:
        if "?" not in [df[col1][ind], df[col2][ind]]:
            if df[col1][ind] not in data[col1]: data[col1].append(df[col1][ind])
            if df[col2][ind] not in y_list: y_list.append(df[col2][ind])
            
    for val in y_list: data[val] = [0 for i in range(len(data[col1]))]
    
    for ind in df.index:
        if "?" not in [df[col1][ind], df[col2][ind]]:
            data[df[col2][ind]][data[col1].index(df[col1][ind])] += 1
            
    return pd.DataFrame(data)


def plot_variables_vs_approved(df, variables):
    """ for each variable, plots the number of approvals and rejections for each
        value of the variable based on dataframe data
    
    params:
        df: pandas dataframe with relevant data
        variables: list of strings representing qualitative variables from dataframe
                   to be plotted
               
    returns:
        None
    """
    for var in variables:
        data = get_qual_data(df, var, "Approved")
        data.set_index(var).plot(kind='bar', stacked=True)
        plt.xticks(rotation=75)
        plt.title("Number of Approvals vs " + var)
        plt.ylabel("# of applicants")
    
    
def qual_corr_w_approved(df, variables, alpha=0.05):
    """ separates qualitative variables into good and bad predictors of approval
        based on a chi^2 contingency test from dataframe data
        
    params:
        df: pandas dataframe with relevant data
        variables: list of strings representing qualitative variables from dataframe
        alpha: significance level for chi^2 test, (default of 0.05)
        
    returns:
        gp: dictionary of variables and corresponding p values deemed 'good' 
            predictors of approval
        bp: dictionary of variables and corresponding p values deemed 'bad' 
            predictors of approval 
    """
    gp, bp = {}, {}
    
    for i in variables:
        a = np.array(pd.crosstab(total_df['Approved'], total_df[i]))
        stats, p, dof, freq = chi2(a, correction=False)
        if p < alpha: gp[i] = p
        else: bp[i] = p
            
    return gp, bp


def quant_corr_w_approved(df, variables, thresh=0.3):
    """ separates quantitative variables into good and bad predictors of approval
        based on a point biserial test from dataframe data
        
    params:
        df: pandas dataframe with relevant data
        variables: list of strings representing quantitative variables from dataframe
        thresh: threshold for point biserial correlation, (default of 0.30)
        
    returns:
        sp: dictionary of variables and corresponding p values deemed 'strong/good' 
            predictors of approval
        wp: dictionary of variables and corresponding p values deemed 'weak/moderate' 
            predictors of approval 
    """
    sp, wp = {}, {}
    
    apr_bools = [0 if x == "+" else 1 for x in df['Approved']]
    
    for i in variables:
        r, p = pbsr(apr_bools, df[i])
        if abs(r) >= thresh: sp[i] = r
        else: wp[i] = r
    
    return sp, wp
    
          
if __name__ == "__main__":
    #dataframes of total and quantitative data
    quant_df = pd.read_csv(DATA_ABRV)
    total_df = pd.read_csv(DATA)

    print(total_df.columns, "\n")
    print(total_df.head(), "\n\n-----\n")
    
    #generates pairplot of all quantitative variables
    sns.pairplot(quant_df, hue="Approved")
    
    total_vars = ['Gender', 'Age', 'Debt', 'Married', 'BankCustomer', 'Industry',
                  'Ethnicity', 'YearsEmployed', 'PriorDefault', 'Employed', 
                  'CreditScore', 'DriversLicense', 'Citizen', 'ZipCode', 'Income', 
                  'Approved']
    qual_vars = ['Gender', 'Married', 'BankCustomer', 'Industry', 'Ethnicity', 
                 'PriorDefault', 'Employed', 'DriversLicense', 'Citizen']
    quant_vars = ['Age', 'Debt', 'YearsEmployed', 'CreditScore', 'Income']
    
    #generates plots of all qualitative variables
    plot_variables_vs_approved(total_df, qual_vars)
    
    #separates qualitative variables into good and bad predictors
    
    gp, bp = qual_corr_w_approved(total_df, qual_vars)
    print("qualitative variables:\n")
    print("\tgood predictors at alpha=0.05:", gp, "\n")
    print("\tbad predictors at alpha=0.05:", bp, "\n\n-----\n")
    
    #seperates qualitative variables into good and bad predictors
    sp, wp = quant_corr_w_approved(total_df, quant_vars)
    print("quantitave variables:\n")
    print("\tgood/strong predictors:", sp, "\n")
    print("\tweak/moderate predictors:", wp, "\n\n")                              
    
    print("finished")
