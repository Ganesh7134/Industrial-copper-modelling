import streamlit as st
import pandas as pd
import pickle
import numpy as np

def Regression():
    st.title(":orange[Predicting Selling Price of copper]")
    data = pd.read_csv("Regression_dataset.csv")
    # st.dataframe(data)    
    col1 , col2 = st.columns(spec=2)
    with col1:
        # Access the unique values for each column
        status_choices = data["status"].unique()
        item_type_choices = data["item type"].unique()
        application_choices = data["application"].unique()
        country_choices = data["country"].unique()
        product_ref_choices = data["product_ref"].unique()
        item_date_year_choices = data["item_date_year"].unique()

        # Create the selectboxes and assign the selected values to variables
        selected_status = st.selectbox("Status:", status_choices, key="status")
        selected_item_type = st.selectbox("Item Type:", item_type_choices, key="item_type")
        selected_application = st.selectbox("Application:", application_choices, key="application")
        selected_country = st.selectbox("Country ID:", country_choices, key="country")
        selected_product_ref = st.selectbox("Product Reference:", product_ref_choices, key="product_ref")
        selected_item_date_year = st.selectbox(
            "Item Date Year:", item_date_year_choices, key="item_date_year"
        )
    with col2:
        # Get min and max values
        min_quantity_tons = int(data["quantity tons"].min())
        max_quantity_tons = int(data["quantity tons"].max())
        min_thickness = int(data["thickness"].min())
        max_thickness = int(data["thickness"].max())
        min_width = int(data["width"].min())
        max_width = int(data["width"].max())
        min_customer = int(data["customer"].min())
        max_customer = int(data["customer"].max())

        quantity = st.text_input("Enter quantity value: ",key = "quantity",autocomplete="on")
        st.warning(f"**Note**: Min value {min_quantity_tons} and Max value {max_quantity_tons}")
        width = st.text_input("Enter width value: ", key="width")
        st.warning(f"**Note**: Min value: {min_width}, Max value: {max_width}")
        customer = st.text_input("Enter customer ID: ", key="customer")
        st.warning(f"**Note**: Min value: {min_customer}, Max value: {max_customer}")
        thickness = st.text_input("Enter thickness value: ", key="thickness")
        st.warning(f"**Note**: Min value: {min_thickness}, Max value: {max_thickness}")
    
    button = st.button(label="Predicting Selling Price", type="primary", key="center_button",use_container_width=True)
    if button:
        with open(r"regression_model.pkl",'rb') as file:
            loaded_model = pickle.load(file)   

        with open(r'scaler.pkl', 'rb') as f:
            scaler_loaded = pickle.load(f)

        with open(r"item.pkl", 'rb') as f:
            item_loaded = pickle.load(f)

        with open(r"status.pkl", 'rb') as f:
            status_loaded = pickle.load(f)
        with open(r"date.pkl", 'rb') as f:
            date_loaded = pickle.load(f)



        new_sample = np.array([[np.log(float(quantity)), selected_application, np.log(float(thickness)), float(width), selected_country,float(customer),int(selected_product_ref),selected_item_type,selected_status,selected_item_date_year]])
        new_sample_item = item_loaded.transform(new_sample[:, [7]]).toarray()
        new_sample_status = status_loaded.transform(new_sample[:, [8]]).toarray()
        new_sample_date = date_loaded.transform(new_sample[:,[9]]).toarray()
        new_sample = np.concatenate((new_sample[:, [0,1,2, 3, 4, 5, 6]], new_sample_item, new_sample_status , new_sample_date), axis=1)
        new_sample1 = scaler_loaded.transform(new_sample)
        new_pred = loaded_model.predict(new_sample1)
        st.write(f'## :green[Predicted selling price: {np.exp(new_pred)}]')

def classification():
    st.title(":orange[Predicting Status of copper]")
    data1 = pd.read_csv("classification_model.csv")
    col1 , col2 = st.columns(spec=2)
    with col1:
        min_class_thickness = int(data1["thickness"].min())
        max_class_thickness = int(data1["thickness"].max())

        item_type_choices = data1["item type"].unique()
        application_choices = data1["application"].unique()
        country_choices = data1["country"].unique()
        product_ref_choices = data1["product_ref"].unique()
        item_date_year_choices = data1["item_date_year"].unique()

        selected_item_type = st.selectbox("Item Type:", item_type_choices, key=20)
        selected_application = st.selectbox("Application:", application_choices, key=21)
        selected_country = st.selectbox("Country ID:", country_choices, key=22)
        selected_product_ref = st.selectbox("Product Reference:", product_ref_choices, key=23)
        selected_item_date_year = st.selectbox("Item Date Year:", item_date_year_choices, key=24)
        class_thickness = st.text_input("Enter thickness value: ", key=33)
        st.warning(f"**Note**: Min value: {min_class_thickness}, Max value: {max_class_thickness}")
    with col2:
        min_class_quantity_tons = int(data1["quantity tons"].min())
        max_class_quantity_tons = int(data1["quantity tons"].max())

        min_class_width = int(data1["width"].min())
        max_class_width = int(data1["width"].max())
        min_class_customer = int(data1["customer"].min())
        max_class_customer = int(data1["customer"].max())
        min_class_selling_price = int(data1["selling_price"].min())
        max_class_selling_price = int(data1["selling_price"].max())

        class_quantity = st.text_input("Enter quantity value: ",key = 30)
        st.warning(f"**Note**: Min value {min_class_quantity_tons} and Max value {max_class_quantity_tons}")
        class_width = st.text_input("Enter width value: ", key=31)
        st.warning(f"**Note**: Min value: {min_class_width}, Max value: {max_class_width}")
        class_customer = st.text_input("Enter customer ID: ", key=32)
        st.warning(f"**Note**: Min value: {min_class_customer}, Max value: {max_class_customer}")

        class_selling_price = st.text_input("Enter selling_price value: ", key=34)
        st.warning(f"**Note**: Min value: {min_class_selling_price}, Max value: {max_class_selling_price}")
    button = st.button(label="Predicting Status", type="primary", key=50,use_container_width=True)
    if button:
        with open(r"classification_model_2.pkl",'rb') as file:
            loaded_model = pickle.load(file)

        with open(r'class_scaler_2.pkl', 'rb') as f:
            scaler_loaded = pickle.load(f)

        with open(r"class_item_2.pkl", 'rb') as f:
            item_loaded = pickle.load(f)

        with open(r"class_date_2.pkl", 'rb') as f:
            date_loaded = pickle.load(f)

        new_sample = np.array([[np.log(float(class_quantity)), np.log(float(class_selling_price)), selected_application, np.log(float(class_thickness)),float(class_width),selected_country,int(class_customer),int(selected_product_ref),selected_item_type,selected_item_date_year]])
        new_item_ohe = item_loaded.transform(new_sample[:, [8]]).toarray()
        new_date_ohe = date_loaded.transform(new_sample[:,[9]]).toarray()
        new_sample = np.concatenate((new_sample[:, [0,1,2, 3, 4, 5, 6,7]], new_item_ohe,new_date_ohe), axis=1)
        new_sample = scaler_loaded.transform(new_sample)
        new_pred = loaded_model.predict(new_sample)
        if new_pred==1:
            st.write('## :green[The status is: Won]')
        else:
            st.write('## :red[The status is: Lost]')

tab1 , tab2 = st.tabs(["Regression","classification"])

# Call the selected function.
with tab1:
    Regression()
with tab2:
    classification()