import streamlit as st
import pandas as pd
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# ================= PAGE CONFIG ================= #
st.set_page_config(page_title="Food Wastage EDA & Prediction", layout="wide")
st.title("📊 Food Wastage Analysis & Prediction App")

st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to", ["EDA", "SQL Results", "Prediction"])

# ================= EDA SECTION ================= #
if section == "EDA":
    st.header("Exploratory Data Analysis")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Data Preview")
        st.dataframe(df.head())

        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

        st.subheader("Summary Statistics")
        st.write(df.describe())

        st.subheader("Missing Values")
        st.write(df.isnull().sum())

        # Correlation heatmap
        st.subheader("Correlation Heatmap (Numerical Features)")
        num_cols = df.select_dtypes(include=["number"]).columns
        if len(num_cols) > 1:
            fig, ax = plt.subplots()
            sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.write("Not enough numerical columns for correlation heatmap.")

        # Value counts for categorical columns
        st.subheader("Value Counts (Categorical Features)")
        cat_cols = df.select_dtypes(include=["object"]).columns
        for col in cat_cols:
            st.write(f"**{col}**")
            st.bar_chart(df[col].value_counts())

# ================= SQL RESULTS SECTION ================= #
elif section == "SQL Results":
    st.header("Results from Database Queries")

    def run_query(query, params=()):
        conn = sqlite3.connect("food_wastage.db")
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df

    queries = {
        "Providers & Receivers per City": """
            SELECT p.City AS City,
                   COUNT(DISTINCT p.Provider_ID) AS Providers,
                   COUNT(DISTINCT r.Receiver_ID) AS Receivers
            FROM Providers p
            LEFT JOIN Receivers r ON p.City = r.City
            WHERE (? = '' OR p.City LIKE ?)
            GROUP BY p.City;
        """,
        "Provider Type with Most Food": """
            SELECT f.Provider_Type, SUM(f.Quantity) AS Total_Quantity
            FROM Food_Listings f
            WHERE (? = '' OR f.Location LIKE ?)
            AND (? = '' OR f.Provider_Type LIKE ?)
            AND (? = '' OR f.Food_Type LIKE ?)
            AND (? = '' OR f.Meal_Type LIKE ?)
            GROUP BY f.Provider_Type
            ORDER BY Total_Quantity DESC;
        """,
        "Contact Info of Providers in City": """
            SELECT p.Name, p.Contact, p.City
            FROM Providers p
            WHERE (? = '' OR p.City LIKE ?);
        """,
        "Receivers with Most Food Claims": """
            SELECT r.Name, SUM(f.Quantity) AS Total_Claimed
            FROM Claims c
            JOIN Receivers r ON c.Receiver_ID = r.Receiver_ID
            JOIN Food_Listings f ON c.Food_ID = f.Food_ID
            WHERE (? = '' OR f.Location LIKE ?)
            AND (? = '' OR f.Provider_Type LIKE ?)
            AND (? = '' OR f.Food_Type LIKE ?)
            AND (? = '' OR f.Meal_Type LIKE ?)
            GROUP BY r.Name
            ORDER BY Total_Claimed DESC;
        """,
        "Total Food Quantity from All Providers": """
            SELECT SUM(f.Quantity) AS Total_Food_Quantity
            FROM Food_Listings f
            WHERE (? = '' OR f.Location LIKE ?)
            AND (? = '' OR f.Provider_Type LIKE ?)
            AND (? = '' OR f.Food_Type LIKE ?)
            AND (? = '' OR f.Meal_Type LIKE ?);
        """,
        "City with Most Food Listings": """
            SELECT f.Location AS City, COUNT(*) AS Listings
            FROM Food_Listings f
            WHERE (? = '' OR f.Location LIKE ?)
            AND (? = '' OR f.Provider_Type LIKE ?)
            AND (? = '' OR f.Food_Type LIKE ?)
            AND (? = '' OR f.Meal_Type LIKE ?)
            GROUP BY f.Location
            ORDER BY Listings DESC;
        """,
        "Most Common Food Types": """
            SELECT f.Food_Type, COUNT(*) AS Count
            FROM Food_Listings f
            WHERE (? = '' OR f.Location LIKE ?)
            AND (? = '' OR f.Provider_Type LIKE ?)
            AND (? = '' OR f.Food_Type LIKE ?)
            AND (? = '' OR f.Meal_Type LIKE ?)
            GROUP BY f.Food_Type
            ORDER BY Count DESC;
        """,
        "Claims per Food Item": """
            SELECT f.Food_Name, COUNT(c.Claim_ID) AS Claims_Count
            FROM Claims c
            JOIN Food_Listings f ON c.Food_ID = f.Food_ID
            WHERE (? = '' OR f.Location LIKE ?)
            AND (? = '' OR f.Provider_Type LIKE ?)
            AND (? = '' OR f.Food_Type LIKE ?)
            AND (? = '' OR f.Meal_Type LIKE ?)
            GROUP BY f.Food_Name
            ORDER BY Claims_Count DESC;
        """,
        "Provider with Most Successful Claims": """
            SELECT p.Name, COUNT(c.Claim_ID) AS Successful_Claims
            FROM Claims c
            JOIN Food_Listings f ON c.Food_ID = f.Food_ID
            JOIN Providers p ON f.Provider_ID = p.Provider_ID
            WHERE c.Status = 'Completed'
            AND (? = '' OR f.Location LIKE ?)
            AND (? = '' OR f.Provider_Type LIKE ?)
            AND (? = '' OR f.Food_Type LIKE ?)
            AND (? = '' OR f.Meal_Type LIKE ?)
            GROUP BY p.Name
            ORDER BY Successful_Claims DESC;
        """,
        "Claims Status Percentage": """
            SELECT c.Status,
                   ROUND((COUNT(*) * 100.0 / (SELECT COUNT(*) FROM Claims)), 2) AS Percentage
            FROM Claims c
            JOIN Food_Listings f ON c.Food_ID = f.Food_ID
            WHERE (? = '' OR f.Location LIKE ?)
            AND (? = '' OR f.Provider_Type LIKE ?)
            AND (? = '' OR f.Food_Type LIKE ?)
            AND (? = '' OR f.Meal_Type LIKE ?)
            GROUP BY c.Status;
        """,
        "Average Quantity Claimed per Receiver": """
            SELECT r.Name, AVG(f.Quantity) AS Avg_Quantity_Claimed
            FROM Claims c
            JOIN Receivers r ON c.Receiver_ID = r.Receiver_ID
            JOIN Food_Listings f ON c.Food_ID = f.Food_ID
            WHERE (? = '' OR f.Location LIKE ?)
            AND (? = '' OR f.Provider_Type LIKE ?)
            AND (? = '' OR f.Food_Type LIKE ?)
            AND (? = '' OR f.Meal_Type LIKE ?)
            GROUP BY r.Name;
        """,
        "Most Claimed Meal Type": """
            SELECT f.Meal_Type, COUNT(*) AS Claim_Count
            FROM Claims c
            JOIN Food_Listings f ON c.Food_ID = f.Food_ID
            WHERE (? = '' OR f.Location LIKE ?)
            AND (? = '' OR f.Provider_Type LIKE ?)
            AND (? = '' OR f.Food_Type LIKE ?)
            AND (? = '' OR f.Meal_Type LIKE ?)
            GROUP BY f.Meal_Type
            ORDER BY Claim_Count DESC;
        """,
        "Total Food Donated by Each Provider": """
            SELECT p.Name, SUM(f.Quantity) AS Total_Donated
            FROM Food_Listings f
            JOIN Providers p ON f.Provider_ID = p.Provider_ID
            WHERE (? = '' OR f.Location LIKE ?)
            AND (? = '' OR f.Provider_Type LIKE ?)
            AND (? = '' OR f.Food_Type LIKE ?)
            AND (? = '' OR f.Meal_Type LIKE ?)
            GROUP BY p.Name
            ORDER BY Total_Donated DESC;
        """
    }

    # Filters
    city_filter = st.sidebar.text_input("City Filter (optional)")
    provider_filter = st.sidebar.text_input("Provider Type Filter (optional)")
    food_type_filter = st.sidebar.text_input("Food Type Filter (optional)")
    meal_type_filter = st.sidebar.text_input("Meal Type Filter (optional)")

    selected_query = st.selectbox("Select a Query", list(queries.keys()))

    if selected_query in ["Providers & Receivers per City", "Contact Info of Providers in City"]:
        df_result = run_query(queries[selected_query], (city_filter, f"%{city_filter}%"))
    else:
        df_result = run_query(
            queries[selected_query],
            (
                city_filter, f"%{city_filter}%",
                provider_filter, f"%{provider_filter}%",
                food_type_filter, f"%{food_type_filter}%",
                meal_type_filter, f"%{meal_type_filter}%"
            )
        )

    st.dataframe(df_result)

# ================= PREDICTION SECTION ================= #
elif section == "Prediction":
    st.header("Food Wastage Prediction")

    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)

        st.sidebar.subheader("Prediction Inputs")
        quantity = st.sidebar.number_input("Quantity", min_value=1, value=10)
        provider_type = st.sidebar.selectbox("Provider Type", ["Restaurant", "Caterer", "Grocery"])
        food_type = st.sidebar.selectbox("Food Type", ["Vegetarian", "Non-Vegetarian", "Mixed"])
        meal_type = st.sidebar.selectbox("Meal Type", ["Breakfast", "Lunch", "Dinner"])

        input_df = pd.DataFrame({
            "Quantity": [quantity],
            "Provider_Type": [provider_type],
            "Food_Type": [food_type],
            "Meal_Type": [meal_type]
        })

        if st.button("Predict"):
            prediction = model.predict(input_df)
            st.success(f"Predicted Value: {prediction[0]}")

    except FileNotFoundError:
        st.error("No trained model found. Please train and save 'model.pkl' first.")




