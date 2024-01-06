import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.impute import KNNImputer


def load_data1():
    data = pd.read_csv(f"data/Dataset1.csv")
    data["P"] = pd.to_numeric(data["P"], errors="coerce")
    data.dropna(inplace=True)
    return data
def parse_date(row,date_column="Start date"):
    date_str = str(row[date_column])
    time_period = row['time_period']
    month_mapping = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
    'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
    'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}
    
    if '/' in date_str:
        # Format with month/day/year
        return pd.to_datetime(date_str, format='%m/%d/%Y')
    else:
        # Extract month and day from the format like '5-Apr'
        day_str,month_str = date_str.split('-')
        month = month_mapping[month_str.lower()]
        day = int(day_str)

        # Determine the year based on the time_period
        if 18 == time_period:
            return pd.to_datetime(f'2019-{month:02d}-{day:02d}')
        elif 18 <= time_period <= 35:
            return pd.to_datetime(f'2020-{month:02d}-{day:02d}')
        elif 36 <= time_period <= 53:
            return pd.to_datetime(f'2021-{month:02d}-{day:02d}')
        elif 54 <= time_period <= 155:
            return pd.to_datetime(f'2022-{month:02d}-{day:02d}')
        else:
            # Handle other cases as needed
            return pd.NaT  # Not a Time (missing value)
def load_data2():
    data = pd.read_csv(f"data/Dataset2.csv")
    #replace null values
    for col in ["positive tests","case count","test count"]:
       imputer = KNNImputer(n_neighbors=3)
       data[col] = imputer.fit_transform(data[[col]])
    
    #fix dates   
    data2_copy = data.copy(deep=True)
    data2_copy["Start date"] =pd.to_datetime(data2_copy["Start date"],errors="coerce",format='%m/%d/%Y', infer_datetime_format=True)
    data['Start date'] = data.apply(parse_date, axis=1,date_column="Start date")
    data['end date'] = data.apply(parse_date, axis=1,date_column="end date")
    data['zcta'] = data['zcta'].astype(object)
    data = process_outliers(data)
    
    return data
def load_data3():
    data3 = pd.read_csv(f"data/Dataset3.csv")
    data3["Temperature"]=data3["Temperature"].str.replace(',', '.').astype(float)
    data3["Humidity"]=data3["Humidity"].str.replace(',', '.').astype(float)
    data3["Rainfall"]=data3["Rainfall"].str.replace(',', '.').astype(float)
 
    return data3  

def normalization(data,method="MinMax"):
    
    """
    data : Dataframe
    
    method : "MinMax" or "Zscore"
    
    return : normalized Dataframe
    """
    if method.lower() == "zscore": 
       scaler = StandardScaler()
    else: 
        scaler = MinMaxScaler()
    
    return pd.DataFrame(scaler.fit_transform(data),columns=data.columns) 



def process_outliers(data,rep_med=True):
    
    data_copy = data.copy(deep=True)
    
    if not rep_med:
        med = np.nan
    
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
           Q1 = data_copy[column].quantile(0.25)
           Q3 = data_copy[column].quantile(0.75)
           
           if rep_med:
              med = data_copy[column].median()
           
           IQR = Q3 - Q1
           
           max_threshold = Q3 + (IQR*1.5)
           min_threshold = Q1 - (IQR*1.5)
           
           # data_copy[column] = np.where(((data_copy[column]<= max_threshold)&(data_copy[column]>= min_threshold)),data_copy[column],np.nan)
           data_copy[column] = np.where(((data_copy[column]<= max_threshold)&(data_copy[column]>= min_threshold)),data_copy[column],med)
       
    return data_copy    



def moustache(data,plots_per_row=3,outliers=True,figure_size=(15,10)):
    
    cols=len(data.columns)
    rows= ceil(cols/plots_per_row)
    
    
    
    # fig, axes = plt.subplots(nrows=1, ncols=len(data.columns), figsize=(15, 5))
    fig, axes = plt.subplots(nrows=rows, ncols=plots_per_row, figsize=figure_size)
    
    axes = axes.flatten()
    
    # Create boxplots for each column
    for i, col in enumerate(data.columns):
        if pd.api.types.is_numeric_dtype(data[col]):
          sns.boxplot(x=data[col], ax=axes[i],showfliers=outliers)
        
        # axes[i].set_title(col)
    
    # Adjust layout
    plt.tight_layout()
    # plt.show()

def plot_scatter(data,col1:str , col2:str):
    
    """plot pair plot 
    """
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.scatter(data[col1],data[col2])
    # plt.show()    
    
def pairplot_custom(data):
    sns.pairplot(data, markers=["o"], diag_kind="kde")    

def plot_heat(data):
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')    
    
import plotly.express as px  
def plot_tree(data):
    return px.treemap(data, path=['zcta'], values='case count', color='positive tests',
                 color_continuous_scale='viridis', title='Total and Positive Cases by Zone',
                 hover_data={'case count': ':,.0f', 'positive tests': ':,.0f','zcta':':,.0f'})    
    
def evolution(data,selected_zone):      
    
           df_selected_zone = data[data['zcta'] == selected_zone]
           # df_selected_zone['Start date'] = pd.to_datetime(df_selected_zone['Start date'])
           
           # Set the 'periode' column as the index for time-based analysis
           df_selected_zone.set_index('Start date', inplace=True)
           
           # Resample data for weekly and monthly frequency
           df_weekly = df_selected_zone.resample('W-Mon').sum()
           df_monthly = df_selected_zone.resample('M').sum()
           df_yearly = df_selected_zone.resample('Y').sum()
           
           # print(df_weekly,'here')
           
           
           # Set up the figure and axes
           plt.figure(figsize=(12, 8))
           
           # Line plot for weekly data
           plt.subplot(3, 1, 1)
           sns.lineplot(x=df_weekly.index, y='test count', data=df_weekly, label='Tests', marker='o')
           sns.lineplot(x=df_weekly.index, y='positive tests', data=df_weekly, label='Positive Tests', marker='o')
           sns.lineplot(x=df_weekly.index, y='case count', data=df_weekly, label='Total Cases', marker='o')
           plt.title(f'Weekly Evolution of Tests, Positive Tests, and Total Cases in {selected_zone}')
           plt.xlabel('Week')
           plt.ylabel('Count')
           plt.legend()
           
           # Line plot for monthly data
           plt.subplot(3, 1, 2)
           sns.lineplot(x=df_monthly.index, y='test count', data=df_monthly, label='Tests', marker='o')
           sns.lineplot(x=df_monthly.index, y='positive tests', data=df_monthly, label='Positive Tests', marker='o')
           sns.lineplot(x=df_monthly.index, y='case count', data=df_monthly, label='Total Cases', marker='o')
           plt.title(f'Monthly Evolution of Tests, Positive Tests, and Total Cases in {selected_zone}')
           plt.xlabel('Month')
           plt.ylabel('Count')
           plt.legend()
           
           
           plt.subplot(3, 1, 3)
           sns.lineplot(x=df_yearly.index.year, y='test count', data=df_yearly, label='Tests', marker='o')
           sns.lineplot(x=df_yearly.index.year, y='positive tests', data=df_yearly, label='Positive Tests', marker='o')
           sns.lineplot(x=df_yearly.index.year, y='case count', data=df_yearly, label='Total Cases', marker='o')
           
           plt.title(f'Yearly Evolution of Tests, Positive Tests, and Total Cases in {selected_zone}')
           plt.xlabel('Year')
           plt.ylabel('Count')
           plt.legend()
           
           plt.tight_layout()
        #    plt.show()    
        
def plot_stack(data):
    
     df_selected_zone = data.copy(deep=True)
     
     # Extract the year from the 'periode' column
     df_selected_zone['Start date'] = df_selected_zone['Start date'].dt.year
     
     # Group by zone and year, summing the positive tests
     df_grouped = df_selected_zone.groupby(['zcta', 'Start date'])['positive tests'].sum().reset_index()
     
     # Pivot the DataFrame for easier plotting
     df_pivot = df_grouped.pivot(index='Start date', columns='zcta', values='positive tests').fillna(0)
     
     # Set up the figure and axes
     plt.figure(figsize=(12, 8))
     
     # Stacked bar chart for positive tests by zone and year
     df_pivot.plot(kind='bar', stacked=True, colormap='viridis')
     plt.title(f'Distribution of COVID-19 Positive Cases by Zone and Year  ')
     plt.xlabel('Year')
     plt.ylabel('Positive Cases')
     plt.legend(title='Zone', loc='upper left', bbox_to_anchor=(1, 1))
     
    #  plt.show()        


def pop_tests(data):
    
    sns.barplot(x='population', y='test count', data=data ,palette='viridis')
    plt.title('Population vs. Tests')
    plt.xlabel('Population')
    plt.ylabel('Number of Tests')
    
def positif_zone(data):
    sns.barplot(x='zcta', y='positivity rate', data=data ,palette='viridis')
    plt.title('Zones vs. positivity rate ')
    plt.xlabel('Zone')
    plt.ylabel('positivity rate')    
    
def rapport(dataframe,selected_period):
    df = dataframe.copy(deep=True)

    df =df[df['time_period'] == selected_period]
    
    df['positive_count_test_count_ratio'] = 100*df['positive tests']/df['test count'] 
    df['case_count_positive_count_ratio'] = 100*df['case count']/df['positive tests']
    df['case_count_test_count_ratio'] = 100*df['case count']/ df['test count']
    fig = plt.figure(figsize=(10, 10))

    # Bar chart for test/case ratio
    plt.subplot(3, 1, 1)
    sns.barplot(x="zcta", y='positive_count_test_count_ratio', data=df, color='skyblue')
    plt.title('positive/Test Ratio')
    
    # Bar chart for positive test ratio
    plt.subplot(3, 1, 2)
    sns.barplot(x="zcta", y='case_count_positive_count_ratio', data=df, color='salmon')
    plt.title('case/positive Ratio')
    
    plt.subplot(3, 1, 3)
    sns.barplot(x="zcta", y='case_count_test_count_ratio', data=df, color='lightgreen')
    plt.title('case/test Ratio')
    
    # plt.tight_layout()
    # plt.show()    
def equal_width(data,column,k=None,labels=None):
        
        if k == None:
           k = round(1 + (10/3) * np.log10(len(data)))
        if labels ==None: 
           labels = [f'{column}_cat{i}' for i in range(1, k + 1)]

        min_value = data[column].min()
        max_value = data[column].max()
        interval_width = (max_value - min_value) / k

        cuts = [min_value + i * interval_width for i in range(1, k)]

        # labels = [f'{column}_cat{i}' for i in range(1, k + 1)]

        data[column+"_cat"] = pd.cut(data[column], bins=[min_value] + cuts + [max_value], labels=labels, include_lowest=True)

        # means = data.groupby(column+"_cat").mean()[column]
        #Mean
        data[column] = data.groupby(column+"_cat")[column].transform('mean')

        return data    
def equal_freq(data, column,k=None,labels=None):
    
    if k == None:
        k = round(1 + (10/3) * np.log10(len(data)))
    if labels ==None: 
        labels = [f'{column}_cat{i}' for i in range(1, k + 1)]
        
    data[column+"_cat"], _ = pd.qcut(data[column], q=k, labels=labels, retbins=True)

    data[column] = data.groupby(column+"_cat")[column].transform('mean')

    return data
def discretisize(data,method="Equal_Width")  :
    
    if method == "Equal_Width":
        method = equal_width
    else:
        method =  equal_freq  
    
    data = method(data.copy(deep=True),"Temperature",k=3,labels=[f'Temperature_{i}' for i in ("low","average","high")])
    data = method(data.copy(deep=True),"Humidity",3,labels=[f'Humidity_{i}' for i in ("low","average","high")])
    data = method(data.copy(deep=True),"Rainfall",3,labels=[f'Rainfall_{i}' for i in ("low","average","high")])  
    
    data["Temperature"]=data["Temperature"].astype("str")#for compatibility purposes later
    data["Humidity"]=data["Humidity"].astype("str")#for compatibility purposes later
    data["Rainfall"]=data["Rainfall"].astype("str")#for compatibility purposes later
    
    return data


def generate_transactions(data):
    return data[["Soil","Crop","Fertilizer","Temperature_cat","Humidity_cat","Rainfall_cat"]].values.tolist()

#AKA APRIORI
def get_frequent_itemsets(transactions, min_support):
    def Support(transactions, itemset,length):
        count = 0
        for transaction in transactions:
            if set(itemset).issubset(transaction):
                count += 1
        return (count/length)*100
        # return count
    def generate_combinations(remaining, k):
    
        candidates = []
    
        for i in range(len(remaining)-1):
          
           for j in range(i + 1, len(remaining)):
               
               for item in remaining[j]:
                     
                     x= remaining[i].union(set([item]))
                     
                     candidates.append(x)
        # Convert sets to tuples and then create a set
        return [set(t) for t in set(tuple(sorted(s)) for s in [item for item in candidates if len(item)==k])] #since we are leveraging the use of sets ,we should check the length of the generated itemsets
        # return [set(t) for t in set(tuple(s) for s in [item for item in candidates if len(item)==k])] #since we are leveraging the use of sets ,we should check the length of the generated itemsets
    
    """APRIORI
       transactions is a list of transactions
       min_support : is the minimum support in percents
    Returns:
        _type_: _description_
    """
    
    #get unique items
    unique_items = set(item for transaction in transactions for item in transaction)
    #list to save all itemsets that verifies the min_support
    frequent_itemsets = []
    
    k = 1
    
    #first iteration candidates
    candidate_itemsets = [set([item]) for item in unique_items]
    
    while True:
    
        # print(f"iteration{k} candidates : {candidate_itemsets}")
        
        #list to save each iteration's frequent itemsets with their support value
        frequent_itemsets_k = []
        
        #list to save each iteration's frequent itemsets to be used in next iteration candidates
        remaining_candidates = []
        
        #calculate support
        for itemset in candidate_itemsets:
            
            support = Support(transactions, itemset,len(transactions)) 
            # support = frequency(transactions, itemset) / len(transactions)
            
            if support >= min_support:
                
                frequent_itemsets_k.append((itemset, support))
                
                remaining_candidates.append(itemset)
                
        #no itemsets satisfies the min_support
        if not frequent_itemsets_k:
            # print("End of search and scan\n")
            break
        
        #save the new itemsets 
        frequent_itemsets.extend(frequent_itemsets_k)
        
        # print(f"iteration{k} remaining candidates  :{remaining_candidates}\n\n")
        
        k += 1
        
        #generate candidates for next iteration
        candidate_itemsets = generate_combinations(remaining_candidates,k)
        # candidate_itemsets = list(generate_combinations(remaining_candidates,k))
      

    return frequent_itemsets

from itertools import combinations
def generate_association_rules(Lk):
    
    #this line just converts LK to the appropriate datasructure 
    Lk = {tuple(item):support for item,support in Lk}

    rules = []
    for itemset in Lk:
        for i in range(1, len(itemset)):
            for subset in combinations(itemset, i):
                antecedent = set(subset)
                consequent = set(itemset) - antecedent
                rules.append((tuple(sorted(antecedent)), tuple(sorted(consequent))))
                # rules.append((tuple(antecedent), tuple(consequent)))
    return rules

def forte_regles_association_conf(associations,itemsets,min_conf=0):
  
    """"
    Extracts fortly correlated associations based on confidence
    """
  
    #this line just converts LK to the appropriate datasructure 
    
    itemsets = {tuple(sorted(item)):support for item,support in itemsets}
    
    associations_confidence=dict()
    
    for association in associations:
            antecedent , consequent = association
            
            # confiance = 100 * (itemsets[tuple(sorted(consequent+antecedent))])/(itemsets[tuple(sorted(antecedent))])
            confiance = 100*(itemsets[tuple(sorted(consequent+antecedent))])/(itemsets[tuple(sorted(antecedent))])
            
            if confiance >= min_conf:
              
              associations_confidence[association] = confiance
    
    return associations_confidence                                                     

def forte_regles_association_lift(associations,itemsets,min_conf):
  
    """"
    Extracts fortly correlated associations based on lift
    """
  
    #this line just converts LK to the appropriate datasructure 
    
    itemsets = {tuple(sorted(item)):support for item,support in itemsets}
    
    associations_confidence=dict()
    
    for association in associations:
            antecedent , consequent = association
  
            lift = (itemsets[tuple(sorted(consequent+antecedent))])/(itemsets[tuple(sorted(antecedent))]*itemsets[tuple(sorted(consequent))])
            # lift = (itemsets[tuple(sorted(consequent+antecedent))])/(itemsets[tuple(sorted(consequent))])
            if lift >= min_conf:
              associations_confidence[association] = lift
            associations_confidence[association] = lift
    
    return associations_confidence                                                     
from math import sqrt
def forte_regles_association_cosine(associations,itemsets,min_cosine):
  
    """"
    Extracts fortly correlated associations based on cosine
    """
  
    #this line just converts LK to the appropriate datasructure 
    
    itemsets = {tuple(sorted(item)):support for item,support in itemsets}
    
    associations_confidence=dict()
    
    for association in associations:
            antecedent , consequent = association

            cosine = (itemsets[tuple(sorted(consequent+antecedent))])/sqrt(itemsets[tuple(sorted(antecedent))]*itemsets[tuple(sorted(consequent))])
            if cosine >= min_cosine:
                associations_confidence[association] = cosine
    
    return associations_confidence                                                     
def forte_regles_association_jaccard(associations,itemsets,min_cosine):
  
    """"
    Extracts fortly correlated associations based on Jaccard
    """
  
    #this line just converts LK to the appropriate datasructure 
    
    itemsets = {tuple(sorted(item)):support for item,support in itemsets}
    
    associations_confidence=dict()
    
    for association in associations:
            antecedent , consequent = association
            
           
            cosine = (itemsets[tuple(sorted(consequent+antecedent))])/(itemsets[tuple(sorted(antecedent))]+itemsets[tuple(sorted(consequent))]-itemsets[tuple(sorted(consequent+antecedent))])
        #     associations_confidence[association] = cosine
            if cosine >= min_cosine:
                associations_confidence[association] = cosine 
    return associations_confidence     
def Recommendation_extractor(associations_fortes,recommendation_values=['rice', 'Coconut', 'DAP', 'Good NPK', 'MOP', 'Urea']):
    """extract the most interesting associations

    Args:
        associations_fortes (list): associations
        recommendation_values (list, optional): values where we are interested. Defaults to ['rice', 'Coconut', 'DAP', 'Good NPK', 'MOP', 'Urea'].

    Returns:
        dict: key is antecedent+Consequent and value is the used metric (support or confidence etc..)
    """
    return {key:value for key,value in associations_fortes.items() if len(set(key[1]).intersection(recommendation_values))!=0}
def Recommend(antecedent,associations,metric="confidence"):
    #make recommendation
    
    # associations=Recommendation_extractor(association_conf)
    return [key[1]for key,value in associations.items() if set(key[0]).issubset(antecedent)]    

unique_input_recommendations = ['Clayey','laterite','silty clay','sandy','coastal','clay loam','alluvial','Temperature_low','Temperature_high','Temperature_average','Rainfall_high','Rainfall_low','Rainfall_average','Humidity_low','Humidity_high','Humidity_average']