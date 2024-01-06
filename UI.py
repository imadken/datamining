   
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from contextlib import redirect_stdout

from io import StringIO
import plotly.express as px

from part1back import *
from part2back import *


st.set_page_config(page_title="DataMining", page_icon="👨‍💻⛏📊", layout="wide", initial_sidebar_state="expanded")
st.set_option('deprecation.showPyplotGlobalUse', False)


def capture_output(df):
    info_output = StringIO()
    with redirect_stdout(info_output):
            df.info()
    return info_output.getvalue()
    
    
# # Initialize session_state
if 'data1' not in st.session_state:
    st.session_state.data1 = load_data1()
    
if 'data2' not in st.session_state:
    st.session_state.data2 = load_data2()  
    
if 'data3' not in st.session_state:
    st.session_state.data3 = load_data3()       

if 'transactions' not in st.session_state:
    st.session_state.transactions = None

if 'frequents' not in st.session_state:
    st.session_state.frequents = None    

if 'associations' not in st.session_state:
    st.session_state.associations = None
if 'strong_rules' not in st.session_state:
    st.session_state.strong_rules = None

if 'Recommendation_rules' not in st.session_state:
    st.session_state.Recommendation_rules = None    

if 'knn' not in st.session_state:
    st.session_state.knn = None  

if 'dt' not in st.session_state:
    st.session_state.dt = None  

if 'rf' not in st.session_state:
    st.session_state.rf = None 

if 'evaluation_table' not in st.session_state:
    st.session_state.evaluation_table = None               

# if 'data2' not in st.session_state:
#     st.session_state.data2 = load_data2()    
# if 'data3' not in st.session_state:
#     st.session_state.data3 = load_data3()


st.markdown("### - Dataset1 : Part1")
st.markdown("### - Dataset2 : Part1")
st.markdown("### - Dataset3 : Part1")
st.markdown("### - Clustering : Part2")

# Filters Tab
with st.expander("Dataset1"):
    
    col1, col2 = st.columns(2,gap="large")
    ##COLUMN2
    
    col2.subheader("Data:")
    df_table1 = col2.dataframe(st.session_state.data1)
    # df_table1 = col2.dataframe()
    col2.subheader("Data Description:")
    desc_table1 = col2.table(st.session_state.data1.describe())
    # desc_table1 = col2.table()
    col2.subheader("Data info:")
    info_table1 = col2.text(capture_output(st.session_state.data1))
    # info_table1 = col2.text("")
    
    ##COLUMN1
    col1.subheader("Preprocessing",divider=True)
    
    if col1.button("load",use_container_width=True):
        
        # data1 = pd.read_csv(f"data/Dataset1.csv")
        # data1["P"]=pd.to_numeric(data1["P"],errors="coerce")
        # data1.dropna(inplace=True)
        # data1=load_data1()
        st.session_state.data1 = load_data1()
        df_table1.dataframe(st.session_state.data1)
        desc_table1.table(st.session_state.data1.describe())
        info_table1.text(capture_output(st.session_state.data1))
        
        
    outliers_check = col1.checkbox("Replace outliers with median",value=True)
    if col1.button("Outliers",use_container_width=True):
        st.session_state.data1= process_outliers(st.session_state.data1,rep_med=outliers_check)
        st.session_state.data1.reset_index(inplace=True,drop=True)
        df_table1.dataframe(st.session_state.data1)
        desc_table1.table(st.session_state.data1.describe())
        info_table1.text(capture_output(st.session_state.data1))
        
    
    if col1.button("Duplicates",use_container_width=True):
        st.session_state.data1.drop_duplicates(inplace=True)
        st.session_state.data1.drop(columns=["OM","Fertility"],inplace=True)
        st.session_state.data1.reset_index(inplace=True,drop=True)
        df_table1.dataframe(st.session_state.data1)
        desc_table1.table(st.session_state.data1.describe())
        info_table1.text(capture_output(st.session_state.data1))
    
    normalization1 = col1.selectbox("Normalization",["Minmax","Zscore"])
    
    if col1.button("Normalize",use_container_width=True):
        
        st.session_state.data1=normalization(st.session_state.data1,method=normalization1)
        df_table1.dataframe(st.session_state.data1)
        desc_table1.table(st.session_state.data1.describe())
        info_table1.text(capture_output(st.session_state.data1))
        
    
    col1.subheader("Visualization",divider=True)
    
    visualize1 = col1.selectbox("Plots",["Boxplot","Histogram","Pairplot","Heatmap","Scatter"])
    feature1 = col1.selectbox("Feature1",st.session_state.data1.columns)
    feature2 = col1.selectbox("Feature2",st.session_state.data1.columns)
    plot_space1 = col1.pyplot()
    if col1.button("Visualize",use_container_width=True):
        
        if visualize1=="Boxplot": moustache(st.session_state.data1,outliers=outliers_check)##outliers!!
        elif visualize1=="Histogram": 
            st.session_state.data1[feature1].hist()
            plt.title(feature1)
        elif visualize1=="Pairplot": pairplot_custom(st.session_state.data1)
        elif visualize1=="Heatmap": plot_heat(st.session_state.data1)
        elif visualize1=="Scatter": plot_scatter(st.session_state.data1,feature1,feature2)
        else:raise ValueError("Type of plot not found")
        plot_space1.pyplot(plt)
 
    
############################################################################################################################
  
        

with st.expander("Dataset2"):
    col21, col22 = st.columns(2,gap="large")
    ##COLUMN2
    
    col22.subheader("Data:")
    df_table2 = col22.dataframe(st.session_state.data2)
    col22.subheader("Data Description:")
    desc_table2 = col22.table(st.session_state.data2.describe())
    col22.subheader("Data info:")
    info_table2 = col22.text(capture_output(st.session_state.data2))
    
    
    ##COLUMN1
    # col21.subheader("Preprocessing",divider=True)
    
    # if col21.button("load2",use_container_width=True):
    #     st.session_state.data2
    
    # if col21.button("Preprocess",use_container_width=True):
    #     pass
  

    col21.subheader("Visualization",divider=True)
    
    visualize2 = col21.selectbox("Plots",["Boxplot","Treemap","Evolution","Stackedbars","Population/tests","positivity/zone","Rates"])
    # feature21 = col21.selectbox("Feature",st.session_state.data2.columns)
    zone = col21.selectbox("Zone",st.session_state.data2["zcta"].unique())
    period = col21.selectbox("Period",st.session_state.data2["time_period"].unique())
    plot_space2 = col21.pyplot()
    
    if col21.button("Visualiser",use_container_width=True):
        
        if visualize2=="Boxplot": 
            moustache(st.session_state.data2)
            plot_space2.pyplot(plt)
        elif visualize2=="Treemap":
            fig = plot_tree(st.session_state.data2) 
            fig.show()
            plot_space2.pyplot(fig)
        elif visualize2=="Evolution": 
            evolution(st.session_state.data2,zone)
            plot_space2.pyplot(plt)
        elif visualize2=="Stackedbars": 
            plot_stack(st.session_state.data2)
            plot_space2.pyplot(plt)
        elif visualize2=="Population/tests": 
            pop_tests(st.session_state.data2)
            plot_space2.pyplot(plt)
        elif visualize2=="positivity/zone": 
            positif_zone(st.session_state.data2)
            plot_space2.pyplot(plt)
        elif visualize2=="Rates": 
            rapport(st.session_state.data2,period)
            plot_space2.pyplot(plt)
        else:raise ValueError("Type of plot not found")
        # plot_space2.pyplot(plt)
    

############################################################################################################################
with st.expander("Dataset3"):
    col31 ,col32, col33 = st.columns(3,gap="medium")
    
    
    ##COLUMN2
    col32.subheader("Data:")
    df_table3 = col32.dataframe(st.session_state.data3.head(5))
    
    # col32.subheader("Data Description:")
    # desc_table3 = col32.table(data1.describe())
    
    col32.subheader("Transactions")
    # transactions_table = col32.dataframe()
    transactions_table = col32.dataframe(pd.DataFrame(st.session_state.transactions))
    
    col33.subheader("Frequent Items")
    Frequent_items = col33.dataframe(pd.DataFrame(st.session_state.frequents))
    
    col33.subheader("Association Rules")
    Association_rules = col33.dataframe(pd.DataFrame(st.session_state.associations))
    
    col32.subheader("Strong Association Rules")
    if st.session_state.strong_rules is not None:
        strong_ass_df = pd.DataFrame(st.session_state.strong_rules.keys(),columns=["Antecedent","Consequent"])
        strong_ass_df["support"] = st.session_state.strong_rules.values()
    else : strong_ass_df = None  
    strong_Association_rules = col32.dataframe(strong_ass_df)
    # strong_Association_rules = col32.dataframe(pd.DataFrame(st.session_state.strong_rules))
    
    # col32.subheader("Recommendation Rules")
    
    # recommendation_table= col32.dataframe(pd.DataFrame(st.session_state.Recommendation_rules))
    col33.subheader("Recommendation Rules")
    if st.session_state.Recommendation_rules is not None:
        strong_ass_df = pd.DataFrame(st.session_state.Recommendation_rules.keys(),columns=["Antecedent","Consequent"])
        strong_ass_df["support"] = st.session_state.Recommendation_rules.values()
    else : strong_ass_df = None  
    recommendation_table = col33.dataframe(strong_ass_df)
    ##COLUMN1
    if col31.button("Load_data"):
        st.session_state.data3 = load_data3()
        df_table3.dataframe(st.session_state.data3.head())
        #clear
        st.session_state.transactions = None 
        st.session_state.frequents = None 
        st.session_state.associations = None 
        st.session_state.strong_rules = None 
        st.session_state.Recommendation_rules = None 
        transactions_table.dataframe()
        Association_rules.dataframe()
        Frequent_items.dataframe(use_container_width=True)
        strong_Association_rules.dataframe(use_container_width=True)
        recommendation_table.dataframe(use_container_width=True)
        
    
    discretisation = col31.selectbox("Discretisation",["Equal_Width","Equal_Freq"])
    if col31.button("Discretesize"):
        st.session_state.data3 = discretisize(st.session_state.data3,discretisation)
        df_table3.dataframe(st.session_state.data3.head(5))
        
    if col31.button("Transactions"):
        st.session_state.transactions = generate_transactions(st.session_state.data3)
        # transactions_table.table(st.session_state.transactions)
        transactions_table.dataframe(pd.DataFrame(st.session_state.transactions))
        
    
    Min_support = col31.slider("Min_Support %",min_value=0,max_value=100)
    
    if col31.button("Apriori"):
        st.session_state.frequents = get_frequent_itemsets(st.session_state.transactions,Min_support)
        st.session_state.associations = generate_association_rules(st.session_state.frequents)
        Frequent_items.dataframe(pd.DataFrame(st.session_state.frequents,columns=["ItemSet","Support"]),use_container_width=True)
        Association_rules.dataframe(pd.DataFrame(st.session_state.associations,columns=["Antecedent","Consequent"]),use_container_width=True)
    
    
    criteria = col31.selectbox("Association Criteria",["Confidence","Lift","Cosine","Jaccard"])
    Minimum = col31.number_input("Min",min_value=0.0,max_value=100.0)
    
    if col31.button("Rules"):
        if criteria=="Confidence": st.session_state.strong_rules = forte_regles_association_conf(st.session_state.associations,st.session_state.frequents,Minimum)
        elif criteria=="Lift":st.session_state.strong_rules = forte_regles_association_lift(st.session_state.associations,st.session_state.frequents,Minimum)
        elif criteria=="Cosine":st.session_state.strong_rules = forte_regles_association_cosine(st.session_state.associations,st.session_state.frequents,Minimum)
        else :st.session_state.strong_rules = forte_regles_association_jaccard(st.session_state.associations,st.session_state.frequents,Minimum)
        strong_ass_df = pd.DataFrame(st.session_state.strong_rules.keys(),columns=["Antecedent","Consequent"])
        strong_ass_df[criteria] = st.session_state.strong_rules.values()
        strong_Association_rules.dataframe(strong_ass_df)
        
        st.session_state.Recommendation_rules = Recommendation_extractor(st.session_state.strong_rules)
        strong_ass_df = pd.DataFrame(st.session_state.Recommendation_rules.keys(),columns=["Antecedent","Consequent"])
        strong_ass_df[criteria] = st.session_state.Recommendation_rules.values()
        recommendation_table.dataframe(strong_ass_df)
    
    
    antecedent = col31.multiselect("Recommendation",unique_input_recommendations)
    Recommendation = col31.text("")
    if col31.button("Recommend"):
        # st.session_state.Recommendation_rules = Recommendation_extractor(st.session_state.strong_rules)
        # strong_ass_df = pd.DataFrame(st.session_state.Recommendation_rules.keys(),columns=["Antecedent","Consequent"])
        # strong_ass_df[criteria] = st.session_state.Recommendation_rules.values()
        # recommendation_table.dataframe(strong_ass_df)
        print(antecedent)
        print(Recommend(tuple(antecedent),st.session_state.Recommendation_rules,criteria))
        Recommendation.text(Recommend(tuple(antecedent),st.session_state.Recommendation_rules,criteria))
        
    


with st.expander("Clustering"):
    
    col41 ,col42 =st.columns(2,gap="large")
    col42.subheader("Evaluation")
    evaluation_df = col42.dataframe(st.session_state.evaluation_table)
    col42.subheader("Plot")
    plot2(X,y,"Before")
    plot4 = col42.pyplot(plt)
    
    
    algo = col41.selectbox("algo",["KNN","Random Forest","Decision Tree","Kmeans","Dbscan"])
    
    k = col41.number_input("K",min_value=2,max_value=20,value=3)
    max_depth = col41.number_input("max depth",min_value=1,max_value=20,value=2)
    nbr_trees = col41.number_input("number of trees",min_value=1,max_value=20,value=5)
    # max_features = col41.number_input("max features",min_value=1,max_value=20,value=5)
    rayon = col41.number_input("rayon",min_value=1.0,max_value=20.0,value=3.0)
    support = col41.number_input("support",min_value=1,max_value=20,value=3)
    
    if col41.button("Run"):
        # st.warning("warniong")
        if algo== "KNN":
            st.session_state.knn = init_knn(k)
            st.session_state.evaluation_table=classification_report_custom(y_test,st.session_state.knn.predict(X_test))
            evaluation_df.dataframe(st.session_state.evaluation_table)
           
            
        elif algo== "Decision Tree":
            st.session_state.dt = init_dt(max_depth)
            st.session_state.evaluation_table=classification_report_custom(y_test,st.session_state.dt.predict(X_test))
            evaluation_df.dataframe(st.session_state.evaluation_table)
        elif algo== "Random Forest":
            st.session_state.rf = init_rf(max_depth,nbr_trees,None)  
            st.session_state.evaluation_table=classification_report_custom(y_test,st.session_state.rf.predict(X_test))
            evaluation_df.dataframe(st.session_state.evaluation_table)
            
        elif algo== "Kmeans":
            kmeans= init_kmeans(k)
            labels = kmeans.predict(X)
            centroids = kmeans.centroids
            st.session_state.evaluation_table = clustering_report(X,labels,centroids)
            evaluation_df.dataframe(st.session_state.evaluation_table)
            plot2(X,labels,"After")
            plot4.pyplot(plt)
        
        elif algo== "Dbscan":
            dbscan = init_dbscan(rayon,support)   
            # labels = dbscan.predict(X_test)
            st.session_state.evaluation_table = clustering_report(X,dbscan.labels,get_dbscan_centroids(X,dbscan.labels))
            evaluation_df.dataframe(st.session_state.evaluation_table)
            plot2(X,dbscan.labels,"After")
            plot4.pyplot(plt)
        else:raise ValueError(" option not found")                  
        
    
    
    col41.subheader("Predict")
    col41.text("enter in this order and separate with comma :")
    col41.text("'N', 'P', 'K', 'pH', 'EC', 'OC', 'S', 'Zn', 'Fe', 'Cu', 'Mn', 'B'")
    
    instance_input = col41.text_area("instance",max_chars=70)
    prediction = ""
    if col41.button("predict"):
        # st.warning("warniong")
        entree = scale_input(instance_input)
        
        if algo== "KNN": prediction = st.session_state.knn.predict(entree)
        elif algo== "Decision Tree":prediction = st.session_state.dt.predict(entree)  
        elif algo== "Random Forest":prediction = st.session_state.rf.predict(entree)
        else:raise ValueError("you must choose a classification algorithm")
    col41.text(prediction)    
        