import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import io

st.title("🚢 Titanic Survival Prediction - Logistic Regression")

# File upload
uploaded_file = st.file_uploader("📁 Upload titanic.csv", type="csv")

if uploaded_file is not None:
    # Read the uploaded file
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Dataset preview")
    st.dataframe(df.head())
    
    # Minimal preprocessing (same as original)
    df_processed = df[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]].dropna()
    df_processed["Sex"] = df_processed["Sex"].map({"male": 1, "female": 0})
    df_processed["Embarked"] = df_processed["Embarked"].map({"S": 2, "C": 0, "Q": 1})
    
    X = df_processed[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
    y = df_processed["Survived"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    st.success(f"✅ Train accuracy: {model.score(X_train, y_train):.3f}")
    st.success(f"✅ Test accuracy: {model.score(X_test, y_test):.3f}")
    
    # User input section
    st.subheader("🔮 Predict your survival")
    
    col1, col2 = st.columns(2)
    with col1:
        pclass = st.selectbox("Pclass", [1, 2, 3], index=2)
        sex = st.selectbox("Sex", ["male", "female"])
        age = st.slider("Age", 0, 80, 30)
    
    with col2:
        sibsp = st.number_input("Siblings/Spouses", 0, 10, 0)
        parch = st.number_input("Parents/Children", 0, 10, 0)
        fare = st.slider("Fare (£)", 0.0, 600.0, 32.0)
        embarked = st.selectbox("Embarked", ["S", "C", "Q"], index=0)
    
    sex_val = 1 if sex == "male" else 0
    emb_map = {"S": 2, "C": 0, "Q": 1}
    emb_val = emb_map[embarked]
    
    if st.button("🚀 Predict Survival", type="primary"):
        X_new = [[pclass, sex_val, age, sibsp, parch, fare, emb_val]]
        pred_prob = model.predict_proba(X_new)[0][1]
        pred = model.predict(X_new)[0]
        
        st.metric("Survival Probability", f"{pred_prob:.1%}")
        
        if pred == 1:
            st.balloons()
            st.success("🎉 YOU SURVIVED!")
        else:
            st.error("💀 Did not survive")
            


st.caption("Built with Streamlit + Scikit-learn")
