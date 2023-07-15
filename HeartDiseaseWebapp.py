import pickle
import streamlit as st
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression

def main():
    #path = 'finalized_model.sav'
    
    #diabetes_model = pickle.load(open(path, 'rb'))
    
    # page title
    st.title('Heart Disease Detection using ML')
        
        
    # getting the input data from the user
    col1, col2, col3, col4 = st.columns(4)
        
    with col1:
        Age = st.text_input('Age')
            
    with col2:
        Gender = st.text_input('Gender (0 female /1 male)')
        
    with col3:
        cp = st.text_input('Chest Pain level(0-3)')
        
    with col4:
        restingbp = st.text_input('Blood Pressure ')
        
    with col1:
        chol = st.text_input('Cholestrol Level')
        
    with col2:
        fbs = st.text_input('fbs value')

    with col3:
        restecg = st.text_input('restecg value')

    with col4:
        thalach = st.text_input('thalac value')

    with col1:
        exang = st.text_input('exangio value')

    with col2:
        oldpeak = st.text_input('oldpeak value')

    with col3:
        slope = st.text_input('slope value')

    with col4:
        ca = st.text_input('ca value')

    with col1:
        thal = st.text_input('thal value')
        
        
    # code for Prediction
    diab_diagnosis = ''
        
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        input_data = [int(Age),
                      int(Gender),
                      int(cp),
                      int(restingbp),
                      int(chol),
                      int(fbs),
                      int(restecg),
                      int(thalach),
                      int(exang),
                      float(oldpeak),
                      int(slope),
                      int(ca),
                      int(thal)]
        data = np.asarray(input_data)
        data_reshaped = data.reshape(1,-1)
        '''diab_prediction = diabetes_model.predict(data_reshaped)
        diab_percentage = diabetes_model.predict_proba(data_reshaped)
        prob = np.max(diab_percentage, axis=1)
        max_prob = np.round(prob, 3)'''
    
        if (diab_prediction[0] == 1):
            diab_diagnosis = 'The person is diabetic. Estimated risk: {} %'.format(float(max_prob) *100)
            
        else:
            diab_diagnosis = 'The person is not diabetic '
        
    st.success(diab_diagnosis)

if __name__ == '__main__':
    main()
