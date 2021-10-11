import streamlit as st
import pickle
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import time


def prediction(encode_vec):
    '''
    This function uses tfidf vector to predict the ticket-type. 
    And returns the prediction as well asprediction probabilty too.
    '''
    # open a file, where you stored the pickled data
    file = open('lightgbm_glove.pkl', 'rb')

    # loading model
    model = pickle.load(file)

    # close the file
    file.close()
    
    # predicting ticket-type
    pred = model.predict(encode_vec)[0]
    # prediction probability
    pred_prob = max(model.predict_proba(encode_vec)[0])
    
    with st.spinner(text = 'Predicting...'):
        time.sleep(3)
    
    # returning prediction with its probability
    return pred, pred_prob*100




def preprocess_text(text):
    '''
    This function allows to convert the text data into tf-idf vector and then returns it.
    '''
    with st.spinner(text = 'Analyzing ticket information...'):
        time.sleep(3)
    
    stop_words = set(stopwords.words('english'))
    clean_text = ' '.join(word.strip() for word in word_tokenize(str(re.sub('[^A-Za-z]+', ' ', text.lower()))) if word not in stop_words)
    
    # checker
    if clean_text=='':
        st.error('The information you have entered, does not contain any appropriate knowledge. Kindly reframe your query.')
        st.stop()
    
    # transforming text
    encode_vec = glove_en.encode(texts = [clean_text], pooling='reduce_mean')
    
    # returning tfidf text vector
    return encode_vec


def main():
    # app title
    st.header('Customer Ticket Classifier')
    
    html_temp = '''
    <div style="background-color:tomato; padding:20px; border-radius: 25px;">
    <h2 style="color:white; text-align:center; font-size: 30px;"><b>Customer Support Ticket Classification Model</b></h2>
    </div><br><br>
    '''
    st.markdown(html_temp, unsafe_allow_html=True)
    # input
    text = st.text_area('Please input ticket information:')
       
    # predicting ticket_type
    if st.button('Predict'):
        
        # necessary requirements
        
        # no empty text
        if text.strip()=='':
            st.warning('No information has been written! Kindly write your query again.')
            st.stop()
            
        # no punctuation
        if str(re.sub('[^A-Za-z]+', ' ', text)).strip()=='':
            st.warning('You have written punctuation only. Kindly write a proper query again.')
            st.stop()
        
        if len(text.split(' ')) < 5:
            st.warning('Ticket information provided is too low. Kindly write atleast five words in the query.')
            st.stop()
            
        # preprocessing of text
        encode_vec = preprocess_text(text)
            
        # predicting ticket-type
        pred, pred_prob = prediction(encode_vec)
                
        # result display
        ticket_type = {0 : 'a Bot. ', 1 : 'an Agent. '}
        result = 'The given user query will be resolve by ' + ticket_type[pred]
        acc = 'The model is ' + str(pred_prob) +'% confident about it.' 
        st.success(result + '\n' +acc)
        
    
if __name__=='__main__':
    main()