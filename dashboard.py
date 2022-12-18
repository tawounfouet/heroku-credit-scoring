# Core Packages
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
import math
from urllib.request import urlopen
import json
import requests
import plotly.graph_objects as go 

import os

import warnings
warnings.filterwarnings("ignore")


gender_dict = {"Male":1, "Female":2}
feature_dict = {"No":1, "Yes":2}

code_gender_dict = {"Male":1, "Female":0}
contract_type_dict = {'Cash loans':'Cash loans', 
                        'Revolving loans':'Revolving loans'}

education_type_dict = {'Academic degree': 'Academic degree',
                        'Higher education': 'Higher education',
                         'Incomplete higher': 'Incomplete higher',
		                'Lower secondary': 'Lower secondary',
                        'Secondary / secondary special': 'Secondary / secondary special',
                        }
family_status_dict = {'Married': 'Married',
                        'Civil marriage': 'Civil marriage',
                        'Single / not married': 'Single / not married',
                        'Separated': 'Separated ',
                        'Widow': 'Widow'
                    }

type_suite_dict = {'Children': 'Children',
                    'Family': 'Family',
                    'Spouse, partner': 'Spouse, partner',
                    'Group of people': 'Group of people',
                    'Other_A': 'Other_A',
                    'Other_B': 'Other_B',
                    'Unaccompanied': 'Unaccompanied',
                    'Unknown': 'Unknown'
                    }

housing_type_dict = {'House / apartment': 'House / apartment',
                        'With parents': 'Municipal apartment',
                        'Municipal apartment': 'Municipal apartment',
                        'Rented apartment': 'Rented apartment',
                        'Office apartment': 'Office apartment',
                        'Co-op apartment': 'Co-op apartment'
                    }

income_type_dict = {'Businessman': 'Businessman',
                    'Commercial associate': 'Commercial associate',
                    'Maternity leave': 'Maternity leave',
                    'State servant': 'State servant',
                    'Student': 'Student',
                    'Pensioner': 'Pensioner',
                    'Working': 'Working',
                    'Unemployed': 'Unemployed'
                    }


weekday_appr_process_dict = {'MONDAY': 'MONDAY',
                            'TUESDAY': 'TUESDAY',
                            'WEDNESDAY': 'WEDNESDAY',
                            'THURSDAY': 'THURSDAY',
                            'FRIDAY': 'FRIDAY',
                            'SATURDAY': 'SATURDAY',
                            'SUNDAY': 'SUNDAY'
                             }

occupation_type_dict = {'Accountants': 'Accountants',
                        'Cleaning staff ': 'Cleaning staff ',
                        'Cooking staff': 'Cooking staff',
                        'Core staff ': 'Core staff ',
                        'Drivers': 'Drivers',
                        'High skill tech staff': 'High skill tech staff',
                        'HR staff': 'HR staff',
                        'IT staff': 'IT staff',
                        'Laborers': 'Laborers',
                        'Low-skill Laborers': 'Low-skill Laborers',
                        'Managers': 'Managers',
                        'Medicine staff ': 'Medicine staff ',
                        'Realty agents': 'Realty agents',
                        'Sales staff': 'Sales staff',
                        'Secretaries': 'Secretaries',
                        'Private service staff': 'Private service staff',
                        'Waiters/barmen staff': 'Waiters/barmen staff',
                        'Unknown': 'Unknown'
                        }

organization_type_dict = {'XNA': 'XNA',
                        'Self-employed': 'Self-employed',
                        'Bank': 'Bank',
                        'Cleaning': 'Cleaning',
                        'Medicine': 'Medicine',
                        'Mobile': 'Mobile',
                        'Government':'Government',
                        'School': 'School',
                        'Kindergarten': 'Kindergarten',
                        'Construction': 'Construction',
                        'Security': 'Security',
                        'Housing': 'Housing',
                        'Military': 'Military',
                        'Police': 'Police',
                        'Agriculture': 'Agriculture',
                        'Postal': 'Postal',
                        'Restaurant': 'Restaurant',
                        'Services': 'Services',
                        'University': 'University',
                        'Hotel': 'Hotel',
                        'Telecom': 'Telecom',
                        'Emergency': 'Emergency',
                        'Legal Services': 'Legal Services',
                        'Religion': 'Religion'
		#('Business Entity Type 1': 'Business Entity Type 1'),
        #('Business Entity Type 2': 'Business Entity Type 2'),
       # ('Business Entity Type 3': 'Business Entity Type 3'),
        #('Industry: type 11': 'Industry: type 11'),
        #('Industry: type 2': 'Industry: type 2'),
        #('Industry: type 3': 'Industry: type 3'),
        #('Industry: type 4': 'Industry: type 4'),
        #('Industry: type 5': 'Industry: type 5'),
        #('Industry: type 6': 'Industry: type 6'),
        #('Industry: type 7': 'Industry: type 7'),
        #('Industry: type 8': 'Industry: type 8'),
        #('Industry: type 9': 'Industry: type 9'),
        #('Industry: type 10': 'Industry: type 10'),
        #('Industry: type 11': 'Industry: type 11'),
        #('Industry: type 12': 'Industry: type 12'),
        #('Industry: type 13': 'Industry: type 13'),
        #('Transport: type 1': 'Transport: type 1'),
        #('Transport: type 2': 'Transport: type 2'),
        #('Transport: type 3': 'Transport: type 3'),
        #('Trade: type 1': 'Trade: type 1'),
        #('Trade: type 2': 'Trade: type 2'),
        #('Trade: type 3': 'Trade: type 3'),
        #('Trade: type 4': 'Trade: type 4'),
        #('Trade: type 5': 'Trade: type 5'),
        #('Trade: type 6': 'Trade: type 6'),
        #('Trade: type 7': 'Trade: type 7'),
 }

#@st.cache
def plot_distribution(applicationDF,feature, client_feature_val, title):

        if (not (math.isnan(client_feature_val))):
            fig = plt.figure(figsize = (10, 4))

            t0 = applicationDF.loc[applicationDF['TARGET'] == 0]
            t1 = applicationDF.loc[applicationDF['TARGET'] == 1]

            if (feature == "DAYS_BIRTH"):
                sns.kdeplot((t0[feature]/-365).dropna(), label = 'Remboursé', color='g')
                sns.kdeplot((t1[feature]/-365).dropna(), label = 'Défaillant', color='r')
                plt.axvline(float(client_feature_val/-365), \
                            color="blue", linestyle='--', label = 'Position Client')

            elif (feature == "DAYS_EMPLOYED"):
                sns.kdeplot((t0[feature]/365).dropna(), label = 'Remboursé', color='g')
                sns.kdeplot((t1[feature]/365).dropna(), label = 'Défaillant', color='r')    
                plt.axvline(float(client_feature_val/365), color="blue", \
                            linestyle='--', label = 'Position Client')

            else:    
                sns.kdeplot(t0[feature].dropna(), label = 'Remboursé', color='g')
                sns.kdeplot(t1[feature].dropna(), label = 'Défaillant', color='r')
                plt.axvline(float(client_feature_val), color="blue", \
                            linestyle='--', label = 'Position Client')


            plt.title(title, fontsize='20', fontweight='bold')
            #plt.ylabel('Nombre de clients')
            #plt.xlabel(fontsize='14')
            plt.legend()
            plt.show()  
            st.pyplot(fig)
        else:
            st.write("Comparaison impossible car la valeur de cette variable n'est pas renseignée (NaN)")

    #@st.cache
def univariate_categorical(applicationDF,feature,client_feature_val,\
                               titre,ylog=False,label_rotation=False,
                               horizontal_layout=True):
        if (client_feature_val.iloc[0] != np.nan):

            temp = applicationDF[feature].value_counts()
            df1 = pd.DataFrame({feature: temp.index,'Number of contracts': temp.values})

            categories = applicationDF[feature].unique()
            categories = list(categories)

            # Calculate the percentage of target=1 per category value
            cat_perc = applicationDF[[feature,\
                                      'TARGET']].groupby([feature],as_index=False).mean()
            cat_perc["TARGET"] = cat_perc["TARGET"]*100
            cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)

            if(horizontal_layout):
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
            else:
                fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20,24))

            # 1. Subplot 1: Count plot of categorical column
            # sns.set_palette("Set2")
            s = sns.countplot(ax=ax1, 
                            x = feature, 
                            data=applicationDF,
                            hue ="TARGET",
                            order=cat_perc[feature],
                            palette=['g','r'])

            pos1 = cat_perc[feature].tolist().index(client_feature_val.iloc[0])
            #st.write(client_feature_val.iloc[0])

            # Define common styling
            ax1.set(ylabel = "Nombre de clients")
            ax1.set_title(titre, fontdict={'fontsize' : 15, 'fontweight' : 'bold'})   
            ax1.axvline(int(pos1), color="blue", linestyle='--', label = 'Position Client')
            ax1.legend(['Position Client','Remboursé','Défaillant' ])

            # If the plot is not readable, use the log scale.
            if ylog:
                ax1.set_yscale('log')
                ax1.set_ylabel("Count (log)",fontdict={'fontsize' : 15, \
                                                       'fontweight' : 'bold'})   
            if(label_rotation):
                s.set_xticklabels(s.get_xticklabels(),rotation=90)

            # 2. Subplot 2: Percentage of defaulters within the categorical column
            s = sns.barplot(ax=ax2, 
                            x = feature, 
                            y='TARGET', 
                            order=cat_perc[feature], 
                            data=cat_perc,
                            palette='Set2')

            pos2 = cat_perc[feature].tolist().index(client_feature_val.iloc[0])
            #st.write(pos2)

            if(label_rotation):
                s.set_xticklabels(s.get_xticklabels(),rotation=90)
            plt.ylabel('Pourcentage de défaillants [%]', fontsize=10)
            plt.tick_params(axis='both', which='major', labelsize=10)
            ax2.set_title(titre+" (% Défaillants)", \
                          fontdict={'fontsize' : 15, 'fontweight' : 'bold'})
            ax2.axvline(int(pos2), color="blue", linestyle='--', label = 'Position Client')
            ax2.legend()
            plt.show()
            st.pyplot(fig)
        else:
            st.write("Comparaison impossible car la valeur de cette variable n'est pas renseignée (NaN)")


def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value

def get_key(val):
    feature_dict = {"No":0, "Yes":1}
    for key, value in feature_dict.items():
        if val == key:
            return key

def get_feature_value(val):
    feature_dict = {"No":0, "Yes":1}
    for key, value in feature_dict.items():
        if val == key:
            return value

def main():
    """Mortalitty Prediction App"""
    st.markdown("<h1 style='text-align: center; color: green'>Credit Scoring App <h1>", 
                unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: left; color: red'>Credit Default Predictions <h2>", 
                unsafe_allow_html=True)

    @st.cache
    def load_data():
        #données nettoyés
        data_cleaned = pd.read_parquet('app_data_cleaned.parquet')
        #Données de test avant transformation 
        test_X = pd.read_parquet('test_X_NoTransformed.parquet')
        #Données de test après transformation
        test_X_transformed = pd.read_parquet('test_X_transformed.parquet')
        #description des features
        description = pd.read_csv('HomeCredit_columns_description.csv',
                                     usecols=['Row', 'Description'],
                                     index_col=0, encoding='unicode_escape')
        return data_cleaned, test_X, test_X_transformed, description

    @st.cache
    def load_model():
        '''loading the trained model'''
        return pickle.load(open('HistGB_Clf_model.pkl', 'rb'))

    @st.cache
    def get_client_info(df, id_client):
        client_info = df[df['SK_ID_CURR']==int(id_client)]
        return client_info

    
  
    #######################################
    # DATA
    #######################################

        # Charger les données clients
    data_cleaned, test_X, test_X_transformed, description = load_data()

    df = test_X

    seed = 123

    ignore_columns = ['SK_ID_CURR', 'TARGET']
    features = [col for col in df.columns if col not in ignore_columns]

    #Chargement du modèle
    model = load_model()
    
    #id_client = get_client_info(df, id_client=100001)

    #######################################
    # SIDEBAR
    #######################################
    with st.sidebar:
        st.header("💰 Prêt à dépenser")
        #id_list = df["SK_ID_CURR"].tolist()
        #id_client = st.selectbox("Sélectionner l'identifiant du client", id_list)

    if st.sidebar.checkbox("Collecter les informations du nouveau client", True):
        SK_ID_CURR = st.text_input("Entrer ID Nouveau Client")
        st.write("## Infos du Client", SK_ID_CURR)
        a, b, c= st.columns([1, 1, 1])
        first_name = a.text_input("Enter your name")
        last_name = b.text_input("Enter your last name")
        email_address = c.text_input("Enter your email adresse")
    
    
        a, b, c, d = st.columns([1, 1, 1, 1])
        AMT_INCOME_TOTAL = a.number_input("Amount of total income",1000,100000)
        AMT_CREDIT = b.number_input("Amount of Credit",500,100000)
        AMT_ANNUITY = c.number_input("Amount of Annuity",100,1000)
        AMT_GOODS_PRICE = d.number_input("Amount of Goods Price",500,100000)

        a, b, c = st.columns([1, 1, 1])
        EXT_SOURCE_1= a.number_input("External Source 1", 1.0, 5.0)
        EXT_SOURCE_2= b.number_input("External Source 2", 1.0, 5.0)
        EXT_SOURCE_3= c.number_input("External Source 3", 1.0, 5.0)
    
    
        a, b, c, d = st.columns([1, 1, 1, 1])
        FLAG_MOBIL = a.radio("Mobile phone ?", tuple(feature_dict.keys()))
        FLAG_EMP_PHONEL = b.radio("Employer phone ?", tuple(feature_dict.keys()))
        FLAG_WORK_PHONEL = c.radio("Work phone ?",tuple(feature_dict.keys()))
        FLAG_EMAILL = d.radio("Email adress ?", tuple(feature_dict.keys()))
    
        a, b, c, d = st.columns([1, 1, 1, 1])
   
        FLAG_CONT_MOBILEL=a.radio("Flag on Cont_Mobile ?", tuple(feature_dict.keys()))
        FLAG_PHONEL= b.radio("Flag on Phone ?", tuple(feature_dict.keys()))
        CODE_GENDER = c.radio("Code Gender", tuple(gender_dict.keys()))
        CNT_FAM_MEMBERS = age = d.slider("Family Members ?", 1, 20, 1)

        a, b, c, d= st.columns([1, 1, 1, 1])
        NAME_INCOME_TYPE = a.selectbox("Source of Income",tuple(income_type_dict.keys()))
        OCCUPATION_TYPE = b.selectbox("Occupation ",tuple(occupation_type_dict.keys()))
        ORGANIZATION_TYPE = c.selectbox("Type of Organization",tuple(organization_type_dict.keys()))
        WEEKDAY_APPR_PROCESS_START = d.selectbox("Process Start Day",tuple(weekday_appr_process_dict.keys()))

        a, b, c = st.columns([1, 1, 1])
        NAME_EDUCATION_TYPE = a.selectbox("Type of Education", tuple(education_type_dict.keys()))
        NAME_FAMILY_STATUS = b.selectbox("Family Status",tuple(family_status_dict.keys()))
        NAME_HOUSING_TYPE = c.selectbox("Housing Type",tuple(housing_type_dict.keys()))

        a, b, c, d= st.columns([1, 1, 1, 1])
        DAYS_BIRTH = a.date_input("Date of Birth")
        DAYS_EMPLOYED= b.date_input("Date Employed")
        DAYS_REGISTRATION= c.date_input("Date of Registration")
        DAYS_ID_PUBLISH= d.date_input("Date of publish")
        #DAYS_LAST_PHONE_CHANGE= st.date_input("Days of Birth")
        #HOUR_APPR_PROCESS_START= st.date_input("Days of Birth")

        a, b, c = st.columns([1, 1, 1])
        REGION_RATING_CLIENT= a.number_input("REGION_RATING_CLIENT", 1.0, 5.0)
        REG_CITY_NOT_LIVE_CITY= b.number_input("REG_CITY_NOT_LIVE_CITY", 1.0, 5.0)
        REG_CITY_NOT_WORK_CITY= c.number_input("REG_CITY_NOT_WORK_CITY", 1.0, 5.0)
    
        a, b, c,  = st.columns([1, 1, 1])
        REGION_RATING_CLIENT_W_CITY= a.number_input("REGION_RATING_CLIENT_W_CITY", 1.0, 5.0)
        REGION_POPULATION_RELATIVE= b.number_input("REGION_POPULATION_RELATIVE", 1.0, 5.0)
        REG_REGION_NOT_LIVE_REGION= c.number_input("REG_REGION_NOT_LIVE_REGION", 1.0, 5.0)
    
        a, b, c,  = st.columns([1, 1, 1])
        LIVE_CITY_NOT_WORK_CITY= a.number_input("LIVE_CITY_NOT_WORK_CITY", 1.0, 5.0)
        REG_REGION_NOT_WORK_REGION= b.number_input("REG_REGION_NOT_WORK_REGION", 1.0, 5.0)
        LIVE_REGION_NOT_WORK_REGION= c.number_input("LIVE_REGION_NOT_WORK_REGION", 1.0, 5.0)

        feature_list = [SK_ID_CURR, first_name, last_name, email_address,AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY,
                    AMT_GOODS_PRICE, EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3, DAYS_BIRTH, DAYS_EMPLOYED,
                    DAYS_REGISTRATION, DAYS_ID_PUBLISH, 
                    get_value(CODE_GENDER,gender_dict), get_value(NAME_EDUCATION_TYPE,education_type_dict),
                    get_value(NAME_FAMILY_STATUS,family_status_dict), get_value(NAME_HOUSING_TYPE,housing_type_dict),
                    get_value(NAME_INCOME_TYPE,housing_type_dict), get_value(NAME_HOUSING_TYPE,income_type_dict),
                    get_value(OCCUPATION_TYPE,occupation_type_dict), get_value(ORGANIZATION_TYPE,organization_type_dict),

                    get_feature_value(FLAG_MOBIL),get_feature_value(FLAG_EMP_PHONEL),
                    get_feature_value(FLAG_WORK_PHONEL),get_feature_value(FLAG_CONT_MOBILEL),
                    get_feature_value(FLAG_PHONEL),get_feature_value(FLAG_EMAILL),
                    REGION_RATING_CLIENT,REGION_RATING_CLIENT_W_CITY,REGION_POPULATION_RELATIVE,
                    REG_REGION_NOT_LIVE_REGION, int(CNT_FAM_MEMBERS), REG_CITY_NOT_LIVE_CITY,
                    REG_CITY_NOT_WORK_CITY, LIVE_CITY_NOT_WORK_CITY
                    ]
        #st.write(feature_list)

        pretty_result = {"SK_ID_CURR":SK_ID_CURR, "first_name": last_name, "last_name": last_name, "email_address": email_address, "AMT_INCOME_TOTAL": AMT_INCOME_TOTAL, 
                    "AMT_CREDIT": AMT_CREDIT, "AMT_ANNUITY": AMT_ANNUITY, "AMT_GOODS_PRICE": AMT_GOODS_PRICE, "CNT_FAM_MEMBERS":CNT_FAM_MEMBERS,
                    "EXT_SOURCE_1":EXT_SOURCE_1,"EXT_SOURCE_2":EXT_SOURCE_2, "EXT_SOURCE_3":EXT_SOURCE_3, "DAYS_BIRTH": DAYS_BIRTH,
                    "DAYS_EMPLOYED": DAYS_EMPLOYED, "DAYS_REGISTRATION": DAYS_REGISTRATION,  "DAYS_ID_PUBLISH": DAYS_ID_PUBLISH, 
                    "CODE_GENDER":CODE_GENDER,
                    "NAME_EDUCATION_TYPE": NAME_EDUCATION_TYPE, "NAME_FAMILY_STATUS":NAME_FAMILY_STATUS, "NAME_HOUSING_TYPE":NAME_HOUSING_TYPE,
                    "NAME_INCOME_TYPE": NAME_INCOME_TYPE, "NAME_HOUSING_TYPE": NAME_HOUSING_TYPE,  "OCCUPATION_TYPE": OCCUPATION_TYPE,
                    "ORGANIZATION_TYPE": ORGANIZATION_TYPE, "FLAG_MOBIL":FLAG_MOBIL, "FLAG_EMP_PHONEL":FLAG_EMP_PHONEL, "FLAG_WORK_PHONEL":FLAG_WORK_PHONEL,
                    "FLAG_WORK_PHONEL":FLAG_WORK_PHONEL, "FLAG_CONT_MOBILEL": FLAG_CONT_MOBILEL, "FLAG_PHONEL": FLAG_PHONEL, "FLAG_EMAILL": FLAG_EMAILL,
                    "REGION_RATING_CLIENT": REGION_RATING_CLIENT, "REGION_RATING_CLIENT_W_CITY": REGION_RATING_CLIENT_W_CITY,
                    "REGION_POPULATION_RELATIVE": REGION_POPULATION_RELATIVE, "REG_REGION_NOT_LIVE_REGION": REG_REGION_NOT_LIVE_REGION,
                    "REG_CITY_NOT_LIVE_CITY": REG_CITY_NOT_LIVE_CITY, "REG_CITY_NOT_WORK_CITY": REG_CITY_NOT_WORK_CITY, 
                    "LIVE_CITY_NOT_WORK_CITY": LIVE_CITY_NOT_WORK_CITY
                       }
        st.json(pretty_result)


    #id_list = df["SK_ID_CURR"].tolist()
    #id_client = st.selectbox("Sélectionner l'identifiant du client", id_list)
    
    try:
        if st.sidebar.checkbox("Client Existant dans la BD", True):
            st.write("## ID Client")
            id_list = df["SK_ID_CURR"].tolist()
            id_client = st.selectbox("Sélectionner l'identifiant du client", id_list)

            with st.sidebar:
                st.write("## Actions à effectuer")
                show_credit_decision = st.checkbox("Afficher la décision de crédit")
                show_client_details = st.checkbox("Afficher les informations du client")
                show_client_comparison = st.checkbox("Comparer aux autres clients")
                shap_general = st.checkbox("Afficher la feature importance globale")
                if(st.checkbox("Aide description des features")):
                    list_features = description.index.to_list()
                    list_features = list(dict.fromkeys(list_features))
                    feature = st.selectbox('Sélectionner une variable',\
                                   sorted(list_features))
            
                    desc = description['Description'].loc[description.index == feature][:1]
                    st.markdown('**{}**'.format(desc.iloc[0]))
    except UnboundLocalError:
        pass


    #######################################
    # HOME PAGE - MAIN CONTENT
    #######################################

    #Afficher l'ID Client sélectionné
    
    id_list = df["SK_ID_CURR"].tolist()

    
    if (int(id_client) in id_list):
        #st.write("## ID Client")
        client_info = get_client_info(df, id_client)
        model_choice = st.selectbox("Select Model",["HistBoost", "LR","DecisionTree"])
        #-------------------------------------------------------
        # Afficher la décision de crédit
        #-------------------------------------------------------
        if (show_credit_decision):
            st.header('‍⚖️ Scoring et décision du modèle')
            #st.header('‍⚖️ Scoring et décision du modèle')

            API_url = str(id_client)
            client_data = test_X_transformed[test_X_transformed["SK_ID_CURR"] == id_client]
            features = [col for col in client_data.columns if col not in ["SK_ID_CURR"]]
            
            single_sample = np.array(client_data[features]).reshape(1,-1)

            #loaded_model = load_model("models/knn_hepB_model.pkl")
            prediction = model.predict(single_sample)
            pred_prob = model.predict_proba(single_sample).round(4)

            with st.spinner('Chargement du score du client...'):
                if prediction == 1:
                    st.warning("❌ Mauvais prospect")
                    pred_probability_score = {"Crédit Refusé ❌":pred_prob[0][0]*100,
                                                "Crédit Accordé ✅":pred_prob[0][1]*100}
                    st.subheader("Prediction Probability Score using {}".format(model_choice))
                    st.json(pred_probability_score)
                    st.subheader("Prescriptive Analytics")
                    #st.markdown(prescriptive_message_temp,unsafe_allow_html=True)
                
                else:
                    st.success(" Bon prospect ")
                    pred_probability_score = {"Crédit Refusé ❌":pred_prob[0][0]*100,
                                                "Crédit Accordé ✅":pred_prob[0][1]*100}
                    st.subheader("Prediction Probability Score using {}".format(model_choice))
                    st.json(pred_probability_score)
        #-------------------------------------------------------
        # Afficher les informations du client
        #-------------------------------------------------------
        personal_info_cols = {
            'CODE_GENDER': "GENRE",
            'DAYS_BIRTH': "AGE",
            'NAME_FAMILY_STATUS': "STATUT FAMILIAL",
            'CNT_CHILDREN': "NB ENFANTS",
            'FLAG_OWN_CAR': "POSSESSION VEHICULE",
            'FLAG_OWN_REALTY': "POSSESSION BIEN IMMOBILIER",
            'NAME_EDUCATION_TYPE': "NIVEAU EDUCATION",
            'OCCUPATION_TYPE': "EMPLOI",
            'DAYS_EMPLOYED': "NB ANNEES EMPLOI",
            'AMT_INCOME_TOTAL': "REVENUS",
            'AMT_CREDIT': "MONTANT CREDIT", 
            'NAME_CONTRACT_TYPE': "TYPE DE CONTRAT",
            'AMT_ANNUITY': "MONTANT ANNUITES",
            'NAME_INCOME_TYPE': "TYPE REVENUS",
            #'EXT_SOURCE_1': "EXT_SOURCE_1",
            'EXT_SOURCE_2': "EXT_SOURCE_2",
            'EXT_SOURCE_3': "EXT_SOURCE_3",
        }
        default_list=["GENRE","AGE","STATUT FAMILIAL","NB ENFANTS","REVENUS","MONTANT CREDIT"]
        numerical_features = ['DAYS_BIRTH', 'CNT_CHILDREN', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','EXT_SOURCE_2','EXT_SOURCE_3']
        rotate_label = ["NAME_FAMILY_STATUS", "NAME_EDUCATION_TYPE"]
        horizontal_layout = ["OCCUPATION_TYPE", "NAME_INCOME_TYPE"]

        if (show_client_details):
            st.header('‍🧑 Informations relatives au client')

            with st.spinner('Chargement des informations relatives au client...'):
                personal_info_df = client_info[list(personal_info_cols.keys())]
                #personal_info_df['SK_ID_CURR'] = client_info['SK_ID_CURR']
                personal_info_df.rename(columns=personal_info_cols, inplace=True)
                personal_info_df["AGE"] = int(round(personal_info_df["AGE"]/365*(-1)))
                personal_info_df["NB ANNEES EMPLOI"] = int(round(personal_info_df["NB ANNEES EMPLOI"]/365*(-1)))

                filtered = st.multiselect("Choisir les informations à afficher", \
                                          options=list(personal_info_df.columns),\
                                          default=list(default_list))

                df_info = personal_info_df[filtered] 
                df_info['SK_ID_CURR'] = client_info['SK_ID_CURR']
                df_info = df_info.set_index('SK_ID_CURR')

                show_table_format = st.checkbox("Afficher les informations sous forme de Table ")
                if (show_table_format):
                    st.table(df_info.astype(str).T)

                show_json_format = st.checkbox("Afficher les informations au format json")
                result = df_info.to_json(orient="records")
                parsed = json.loads(result)
                if (show_json_format):
                    st.json(parsed)


                show_all_info = st.checkbox("Afficher toutes les informations (dataframe brute)")
                if (show_all_info):
                    st.dataframe(client_info)

        #-------------------------------------------------------
        # Comparer le client sélectionné à d'autres clients
        #-------------------------------------------------------
        if (show_client_comparison):
            st.header('‍👀 Comparaison aux autres clients')
            #st.subheader("Comparaison avec l'ensemble des clients")
            with st.expander("🔍 Explication de la comparaison faite"):
                st.write("Lorsqu'une variable est sélectionnée, un graphique montrant la distribution de cette variable selon la classe (remboursé ou défaillant) sur l'ensemble des clients (dont on connait l'état de remboursement de crédit) est affiché avec une matérialisation du positionnement du client actuel.") 

            with st.spinner('Chargement de la comparaison liée à la variable sélectionnée'):
                var = st.selectbox("Sélectionner une variable",\
                                   list(personal_info_cols.values()))
                feature = list(personal_info_cols.keys())\
                [list(personal_info_cols.values()).index(var)]    

                if (feature in numerical_features):                
                    plot_distribution(data_cleaned, feature, client_info[feature], var)   
                elif (feature in rotate_label):
                    univariate_categorical(data_cleaned, feature, \
                                           client_info[feature], var, False, True)
                elif (feature in horizontal_layout):
                    univariate_categorical(data_cleaned, feature, \
                                           client_info[feature], var, False, True, True)
                else:
                    univariate_categorical(data_cleaned, feature, client_info[feature], var)

        
    else:    
        st.markdown("**Identifiant non reconnu**")




if __name__ == '__main__':
    main()
