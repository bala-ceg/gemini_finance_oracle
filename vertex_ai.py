import os
from dotenv import load_dotenv
import streamlit as st
import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)
import base64


import yfinance as yf 
import pandas as pd
from GoogleNews import GoogleNews
import io
import sys
import matplotlib.pyplot as plt
import numpy as np
from streamlit_echarts import st_echarts
from google.cloud import storage
from io import BytesIO
from fpdf import FPDF





load_dotenv()


PROJECT_ID = os.environ.get("GCP_PROJECT")  
LOCATION = os.environ.get("GCP_REGION")  
vertexai.init(project=PROJECT_ID, location=LOCATION)
bucket_name = os.environ.get("GCP_BUCKET") 
destination_file_name = os.environ.get("MP3_File_Name") 
mp3_file_path = os.environ.get("MP3_File_Path") 



def upload_mp3_to_gcs(mp3_file_path, bucket_name, destination_file_name):
    
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(destination_file_name)
    blob.upload_from_filename(mp3_file_path)

    uri = f"gs://{bucket_name}/{destination_file_name}"

    return uri



st.set_page_config(page_title="Gemini Finance Oracle")


generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

sentiment_mapping = {
    "Bullish": 1,
    "Bearish": -1,
    "Neutral": 0
}



def fetch_financial_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info

    pe_ratio = info.get('trailingPE')
    pb_ratio = info.get('priceToBook')
    dividend_yield = info.get('dividendYield') * 100 if info.get('dividendYield') else 0
    roe = info.get('returnOnEquity') * 100 if info.get('returnOnEquity') else 0

    return pe_ratio, pb_ratio, dividend_yield, roe

def calculate_score(pe_ratio, pb_ratio, dividend_yield, roe):
    score = 0
    pe_weight = 0.25
    pb_weight = 0.25
    dividend_weight = 0.25
    roe_weight = 0.25

    pe_score = (1 / pe_ratio) * pe_weight if pe_ratio and pe_ratio > 0 else 0
    pb_score = (1 / pb_ratio) * pb_weight if pb_ratio and pb_ratio > 0 else 0
    dividend_score = dividend_yield * dividend_weight
    roe_score = roe * roe_weight
  
    score = pe_score + pb_score + dividend_score + roe_score

    return score
    

def draw_semi_circular_gauge(value):
    display_value = value * 100

    option = {
        "series": [
            {
                "type": "gauge",
                "startAngle": 180,
                "endAngle": 0,
                "radius": "100%",
                "center": ["50%", "75%"],
                "axisLine": {
                    "lineStyle": {
                        "width": 30,
                        "color": [
                            [0.5, "#ff0000"],  
                            [1, "#00ff00"],   
                        ],
                    },
                },
                "pointer": {
                    "length": "70%",
                    "width": 8,
                },
                "min": -100,
                "max": 100,
                "splitNumber": 10,
                "axisLabel": {
                    "show": False
                },
                "axisTick": {
                    "show": False
                },
                "splitLine": {
                    "show": False
                },
                "detail": {
                    "formatter": "{value}%",
                    "offsetCenter": [0, "30%"],
                    "fontSize": 20,
                },
                "data": [{"value": display_value}],
            }
        ],
        "title": {
            "show": True,
            "offsetCenter": [0, "100%"]
        },
    }
    st_echarts(options=option)


def is_valid_us_stock(symbol):
    try:
        stock = yf.Ticker(symbol)
        stock_info = stock.info
        if stock_info.get('exchange') in ['NMS','NYQ']:
            return True
        else:
            return False
    except Exception as e:
        return False

def extract_google_news(stock_ticker, num_articles=5):
    googlenews = GoogleNews(lang='en')
    googlenews.search(f"{stock_ticker} stock")
    
    googlenews.getpage(1)
    
    news_results = googlenews.results(sort=True)
    
    news = []
    for entry in news_results[:num_articles]:
        title = entry['title']
        link = entry['link']
        news_title = f"[{title}]({link})"
        news.append({'news_title': news_title})
    
    return news


def generate_sentiment_prompt(title):
    prompt = f"Analyse the market sentiment of the news title - {title} and return only one of the following sentiment Bullish, Bearish or Netural and don't provide any explanations"
    return prompt

def generate_pdf(stock_analysis):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size = 12)
    pdf.multi_cell(0, 10, txt=stock_analysis)
    pdf_output_path = "stock_analysis.pdf"
    pdf.output(pdf_output_path)
    return pdf_output_path


def download_max_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="max")
    
    data.index = data.index.strftime('%m/%d/%Y')
    
    filename = f"{symbol}_max_data.csv"
    data.to_csv(filename)
    st.write(f"Data for the stock {symbol} is  downloaded")



def audio_qa(gs_url,prompt):
    buffer = io.StringIO()
    sys.stdout = buffer
    model = GenerativeModel(
        "gemini-1.5-pro-001",
    )

    audio =  Part.from_uri(mime_type="audio/mpeg",uri=gs_url)
 
    qa_prompt = """{prompt}"""
    responses = model.generate_content(
        [audio,qa_prompt],
        generation_config=generation_config,
        stream=True,
    )
    try:
        for response in responses:
            print(response.text, end="")
    finally:
        sys.stdout = sys.__stdout__
    
    return buffer.getvalue()


def pdf_qa(gs_url,prompt):
    buffer = io.StringIO()
    sys.stdout = buffer
    model = GenerativeModel(
        "gemini-1.5-pro-001",
    )

    document = Part.from_uri(mime_type="application/pdf",uri=gs_url)
 
    qa_prompt = """{prompt}"""
    responses = model.generate_content(
        [document,qa_prompt],
        generation_config=generation_config,
        stream=True,
    )
    try:
        for response in responses:
            print(response.text, end="")
    finally:
        sys.stdout = sys.__stdout__
    
    return buffer.getvalue()

def predict_sentiment(prompt):
    buffer = io.StringIO()
    sys.stdout = buffer
    model = GenerativeModel(
        "gemini-1.5-flash-001",
    )

    responses = model.generate_content(
        [prompt],
        generation_config=generation_config,
        stream=True,
    )
    try:
        for response in responses:
            print(response.text, end="")
    finally:
        sys.stdout = sys.__stdout__
    
    return buffer.getvalue()


def qa_agent_df(user_prompt,symbol):
    buffer = io.StringIO()
    sys.stdout = buffer

    filename = f"{symbol}_max_data.csv"
    if not os.path.isfile(filename):
        download_max_data(symbol)

    df = pd.read_csv(f'{symbol}_max_data.csv')
    model = GenerativeModel(
        "gemini-1.5-pro-001",
    )

    prompt = f"""
        The following is a table of data:
        
        {df}
        
        Columns: {', '.join(df.columns)}
        
        Please answer the question based on the table data.
        
        Question: {user_prompt}
        Answer:
    """

    responses = model.generate_content(
        [prompt],
        generation_config=generation_config,
        stream=True,
    )
    try:
        for response in responses:
            print(response.text, end="")
    finally:
        sys.stdout = sys.__stdout__
    
    return buffer.getvalue()





def generate_advice(selected_expert,question):
    buffer = io.StringIO()
    sys.stdout = buffer

    prompt= f"""
        You are a public finance coach, channeling {selected_expert} and his style of talking and coaching!
        You are the best investor in the world, and CEOs are looking to get your advice on their investments.
        If the investor asks you to provide some thoughts about an investment, use the transcript and your observations to share your insights. 
        You will directly talk to the people, be encouraging towards them and help them improve on their investment patiently. 
        You will have access to the entire conversation with the people, as well as the speech expert's analysis in this prompt. 
        If there are multiple instances of the speech expert's analysis, then only consider the latest one for your feedback. When you are giving feedback, be specific and share details about which parts of their investment thesis they need to improve on, specifying the timestamps wherever available. 
        Add examples of {selected_expert}'s advice, speeches, jokes, and channel his personality in your responses. 
        Be brief in your responses, do not overwhelm the user with your answers. Do not mention that there is a speech expert or transcription expert working with you. 
        Feel free to make assumptions about the people and their investment, do not trouble them with many questions. 
        Remove all formatting from your response.
        """

    model = GenerativeModel(
        "gemini-1.5-flash-001",
    )
    responses = model.generate_content(
        [prompt,question],
        generation_config=generation_config,
        stream=True,
    )
    try:
        for response in responses:
            print(response.text, end="")
    finally:
        sys.stdout = sys.__stdout__
    
    return buffer.getvalue()


def generate_stock_analysis(symbol):
    buffer = io.StringIO()
    sys.stdout = buffer

    prompt= f"""
        Perform a comprehensive analysis of {symbol} Include:
        Overview: Primary business operations and market position.
        Financials: Revenue, profit margins, EPS, growth projections, and balance sheet.
        Valuation: P/E ratio, P/B ratio, and other relevant metrics.
        Market Performance: Stock price trends and market sentiment.
        Risks: Operational, market, and financial risks.
        Competition: Industry comparison and competitive strengths/weaknesses.
        ESG Factors: ESG ratings and sustainability initiatives.
        Outlook: Recent developments and future plans.
        Use the latest Market Analysis data and insights. Present information clearly.
    """

    model = GenerativeModel(
        "gemini-1.5-flash-001",
    )
    responses = model.generate_content(
        [prompt],
        generation_config=generation_config,
        stream=True,
    )
    try:
        for response in responses:
            print(response.text, end="")
    finally:
        sys.stdout = sys.__stdout__
    
    return buffer.getvalue()




def main_page():
    st.sidebar.title('Gemini Finance Oracle')

    mapping = {'company_analysis': '📈 Dashboard',
            'chat_on_audio': '💬 Ask Question on Earnings Call',
            'chat_on_pdf': '❓ Ask Question on Annual Report',
            'chat_with_expert': '✅ Chat with Expert',
            'chat_with_data': ' 📊 Chat with Data'
            }

    col = st.columns((6, 6), gap='large')

    selected_tab = None
    selected_tab = st.sidebar.radio(label='Go to', options=("company_analysis", "chat_on_audio", "chat_on_pdf","chat_with_expert","chat_with_data"), format_func=lambda x: mapping[x],
                                        label_visibility='hidden')

    if selected_tab == 'company_analysis':
        st.subheader('Company Data')
        symbol_input = st.text_input("Enter a stock symbol:")
        
        if st.button("Submit",key="stock_button"):

            if is_valid_us_stock(symbol_input):
                valid_stock = 1
                st.write(f"Please wait while we fetch the analysis of stock {symbol_input}")
            else:
                st.write(f"{symbol_input} is not a valid US stock symbol. Please enter a valid symbol.")
                return 

            tab1, tab2,tab3 = st.tabs(['Sentiment Analysis', 'Stock Score', 'Stock Report'])
            with tab1:
                if valid_stock: 

                    news = extract_google_news(symbol_input, num_articles=5)
                    df = pd.DataFrame(news)

                    df['sentiment'] = df['news_title'].apply(lambda title: predict_sentiment(generate_sentiment_prompt(title)))
                    df['sentiment'] = df['sentiment'].str.replace('\n', '').str.replace(' ', '')

                    df['sentiment_value'] = df['sentiment'].map(sentiment_mapping).fillna(0).astype(int)
                    print(df)
                    sentiment_score = df['sentiment_value'].mean()
                    value = 'Bullish' if sentiment_score > 0 else ('Bearish' if sentiment_score < 0 else 'Neutral')
                    st.info(f'Sentiment for the Stock {symbol_input} is {value}')

                    st.write(f'{value} Sentiment  for the Stock {symbol_input} is {int(sentiment_score * 100)} %')
    
                    
                    draw_semi_circular_gauge(sentiment_score)

                    st.title('Bullish/Bearish Sentiment Meter')

            with tab2:
                try:
                    pe_ratio, pb_ratio, dividend_yield, roe = fetch_financial_data(symbol_input)

                    print(f"Financial Data for {symbol_input}:")
                    print(f"  P/E Ratio: {pe_ratio}")
                    print(f"  P/B Ratio: {pb_ratio}")
                    print(f"  Dividend Yield: {dividend_yield}%")
                    print(f"  ROE: {roe}%")

                    score = calculate_score(pe_ratio, pb_ratio, dividend_yield, roe)
                    st.info(f'Stock Score: {score:.2f}')

                    if score > 1.0:
                        st.info('Recommendation: Buy')
                        print("Recommendation: Buy")
                    elif 0.5 <= score <= 1.0:
                        st.info('Recommendation: Hold')
                    else:
                        st.info('Recommendation: Sell')
                except Exception as e:
                    print(f"Error fetching data for {ticker}: {e}")

            with tab3:
                stock_analysis = generate_stock_analysis(symbol_input)
                print(stock_analysis)

              
                pdf_path = generate_pdf(stock_analysis)
                st.success("Generated stock_analysis.pdf")
                with open(pdf_path, "rb") as f:
                    st.download_button("Download PDF", f, "stock_analysis.pdf")


    if selected_tab == 'chat_on_audio':
        st.title('Chat with audio file')
        st.header("Enter the Audio File Path from google storage")

        gs_url = st.text_input("Enter the Audio File Path from google storage")
        if st.button("Submit"):
            st.write("Thank you for submitting the Audio File Path from google storage")


        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        try:
            if prompt := st.chat_input("Ask your questions related to the Quartely Conference call audio you added "):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    response = audio_qa(gs_url,prompt)
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"An error occurred while generating the response: {e}")
    
    if selected_tab == 'chat_on_pdf':
        st.title('Chat with data')
        st.header("Enter the PDF File Path from google storage")

        gs_url = st.text_input("Enter the PDF File Path from google storage")
        if st.button("Submit"):
            st.write("Thank you for submitting the Audio File Path from google storage")


        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        try:
            if prompt := st.chat_input("Ask your questions related to attached 10K document for the particular stock"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    response = pdf_qa(gs_url,prompt)
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"An error occurred while generating the response: {e}")
    
    if selected_tab == 'chat_with_expert':
        st.title('Chat with Expert')
        st.header("Select the Expert you want to chat about from the dropdown")

        experts = ["Warren Buffett", "Charlie Munger", "Peter Lynch"]
        selected_expert = st.selectbox("Choose an expert", experts)
    

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        try:
            if prompt := st.chat_input(f"Ask any  questions  you have to Mr. {selected_expert} "):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    response = generate_advice(selected_expert,prompt)
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"An error occurred while generating the response: {e}")
    
    if selected_tab == 'chat_with_data':
        st.title('Chat with data')
        with st.sidebar:
            symbol_input = st.text_input("Enter a stock symbol data that you want to chat on:")

            if st.button("Submit"):
                try:
                    if is_valid_us_stock(symbol_input):
                        download_max_data(symbol_input)    
                    else:
                        st.write(f"{symbol_input} is not a valid US stock symbol. Please enter a valid symbol.")
                except Exception as e:
                    st.error(f"An error occurred while processing the stock data: {e}")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        try:
            if user_prompt := st.chat_input("Ask your questions related to the stock data you submitted"):
                st.session_state.messages.append({"role": "user", "content": user_prompt})
                with st.chat_message("user"):
                    st.markdown(user_prompt)

                with st.chat_message("assistant"):
                    response = qa_agent_df(user_prompt, symbol_input)
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"An error occurred while generating the response: {e}")
        



def main():
    main_page()




if __name__ == "__main__":
    main()
