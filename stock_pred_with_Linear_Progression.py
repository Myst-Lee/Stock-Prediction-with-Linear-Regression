import yfinance as yf
import datetime 
from datetime import date, datetime, timedelta
import urllib.request
import json
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import streamlit as st

# Show header
st.title("Stock Prediction with Linear Regression")

# Show readme
readme = st.checkbox("readme first")

if readme:

    st.write("""This project uses realtime stock data which obtained using yfinance dependencies. It can predict the action whether sell or buy with the
    implementation of Linear Regression model""")

    st.write ("For more info, please contact:", "<a href='https://www.linkedin.com/in/mingyilee/'>Lee Ming Yi </a>", unsafe_allow_html=True)


# Creating sidebar
sideb = st.sidebar

sideb.header("User Input Parameter")
sb = sideb.selectbox(
    'Select a mini project',
     ['Buy Strategy','Sell Strategy', 'Update Stock']
     )
     
stock_symbol = sideb.text_input(
    "Enter Stock Symbol",
    value="AMZN"
).upper()

# declare function

def get_yahoo_shortname(symbol):
    response = urllib.request.urlopen(f'https://query2.finance.yahoo.com/v1/finance/search?q={symbol}')
    content = response.read()
    data = json.loads(content.decode('utf8'))['quotes'][0]['shortname']
    return data

def user_input_features():
    data = {'Stock_Symbol': stock_symbol,
            'Date': d}
    features = pd.DataFrame(data, index=[0])
    return features

# actual system

if sb=='Buy Strategy':
    
    d = sideb.date_input(
        "Select Predicted Date:", value = date.today())

    df = user_input_features()
    stockName = get_yahoo_shortname(stock_symbol)
    st.header("Buy Strategy - ("+stock_symbol+") "+stockName)
    dataframe = yf.download(stock_symbol, "2010-01-01", d, auto_adjust=True)
    dataframe = dataframe.dropna()
    st.write("Total Stock Data Downloaded: "+str(len(dataframe)))
    # st.write(dataframe.tail())

    dataframe = dataframe[["Close"]] #only require close for prediction
    chart_data=pd.DataFrame(dataframe)
    # pyplot.ylabel(stockName+" Stock Value")
    # pyplot.title(stockName+" ("+stock_symbol+") - 2010 - "+d.strftime("%Y"))
    st.subheader(stockName+" ("+stock_symbol+") - 2010 - "+d.strftime("%Y"))
    st.line_chart(chart_data)

    # Define variable
    dataframe["five_days_moving_avg"] = dataframe["Close"].rolling(window=5).mean()
    dataframe["twenty_days_moving_avg"] = dataframe["Close"].rolling(window=20).mean()
    dataframe = dataframe.dropna()
    X = dataframe[["five_days_moving_avg", "twenty_days_moving_avg"]]
    dataframe["value_next_day"] = dataframe["Close"].shift(-1)
    dataframe = dataframe.dropna()
    y = dataframe["value_next_day"]

    # Train test split
    split_index = 0.8 # Split data into 80:20
    split_index = split_index * len(dataframe)
    split_index = int(split_index)
    X_train = X[:split_index]
    y_train = y[:split_index]
    X_test = X[split_index:]
    y_test = y[split_index:]

    # Prepare for Linear Regression
    model = LinearRegression()
    model = model.fit(X_train, y_train)
    five_day_moving_avg = model.coef_[0]
    twenty_day_moving_avg = model.coef_[1]
    constant = model.intercept_

    # Make Predictions on tomorrow
    test_output = model.predict(X_test)
    y_test = y[(split_index - 1):]
    test_output = pd.DataFrame(test_output, index= y_test.index, columns = ["value"])
    compare = pd.concat([y_test, test_output], axis=1, join='inner')
    compare.columns = ['Actual_Value', 'Model_Predicted_Output']
    # st.write(compare.tail())
    st.subheader("Predict Value with Linear Regression")
    st.line_chart(compare)

    # Earning progress from start date
    stocks = pd.DataFrame()
    stocks["value"] = dataframe[split_index:]["Close"]
    stocks["predicted_tomorrow_value"] = test_output
    stocks["actual_tomorrow_value"] = y_test
    stocks["returns"] = stocks["value"].pct_change().shift(-1) # pct = percentage
    stocks["strategy"] = np.where(stocks.predicted_tomorrow_value.shift(1) < stocks.predicted_tomorrow_value, 1, 0)
    stocks["strategy_returns"] = stocks.strategy * stocks["returns"]
    cumulative_product = (stocks["strategy_returns"]+1).cumprod()
    st.write(stocks.tail())

    view = st.checkbox("View Cumulative Product")

    if view:
        st.subheader("Cumulative Product of adding Strategy Returns")
        st.line_chart(cumulative_product)

    # Make Prediction to buy or sell
    dataset = yf.download(stock_symbol, "2010-01-01", d, auto_adjust=True)
    dataset["five_days_avg"] = dataset["Close"].rolling(window=5).mean()
    dataset["twenty_days_avg"] = dataset["Close"].rolling(window=20).mean()
    dataset = dataset.dropna()
    dataset["predicted_stock_value"] = model.predict(dataset[["five_days_avg", "twenty_days_avg"]])
    dataset["strategy"] = np.where(dataset.predicted_stock_value.shift(1) < dataset.predicted_stock_value, "Buy", "Hold/Sell")
    st.subheader("Buy or Hold/Sell Strategy")
    st.write(dataset.tail())
    strategy = dataset.iloc[-1]['strategy']

    if strategy == "Buy":
        st.subheader("Strategy (Buy or Hold/Sell): ")
        msg = '<p style="font-family:sans-serif; color:Green; font-size: 18px;"><strong>Buy</strong></p>'
        st.markdown(msg, unsafe_allow_html=True)
    else:
        st.subheader("Strategy (Buy or Hold/Sell): ")
        msg = '<p style="font-family:sans-serif; color:Red; font-size: 18px;"><strong>Hold/Sell</strong></p>'
        st.markdown(msg, unsafe_allow_html=True)
    

elif sb =='Sell Strategy':
    stockName = get_yahoo_shortname(stock_symbol)
    st.header("Sell Strategy - ("+stock_symbol+") "+stockName)

    # stock_bought = sideb.number_input(
    #     label = 'Enter Amount of Stock Bought: ', 
    #     min_value=0,
    #     step = 1,
    #     value=0
    # )

    # buy_Prc = sideb.number_input(
    #     label = "Enter Current Stock Price (Buy Price): ",
    #     min_value=0.0000,
    #     step = 0.01,
    #     value=90.00,
    #     format = "%.5f"
    # )

    # sell_Prc = sideb.number_input(
    #     label = "Enter Current Stock Price (Sell Price): ",
    #     min_value=0.0000,
    #     step = 0.01,
    #     value=100.00,
    #     format = "%.5f"
    # )

    dl_template = st.checkbox("Download Sample Template")

    if dl_template:
        st.subheader("Sample Template")
        st.write("You may download the template and make amendments. Uplaod the file to see its updates")
        # Download sample csv
        template = {'Tickers': ['AMZN', 'AAPL', 'NFLX', 'GOOGL', 'MSFT'],
                    'Date': ['04/12/2022', '04/12/2022','04/12/2022', '04/12/2022', '04/12/2022'],
                    'Open': [94.47, 145.96, 310.49, 99.05, 249.82],
                    'High': [95.36, 148.00, 321.98, 100.77, 256.05],
                    'Low': [93.78, 145.65, 310.00, 98.90, 249.75],
                    'Close': [94.13, 147.81, 320.41, 98.90, 249.75],
                    'Volume': [72096414, 62231328, 12690602, 21480703, 21332290],
                    'Share Bought': [5.00, 2.00, 11.00, 1.00, 1.00],
                    'Current Share Price (Buy Price)': [100.00, 150.00, 350.00, 50.00, 200.00],
                    'Current Share Price (Sell Price)': [80.00, 100.00, 250.00, 200.00, 300.00],
                    'Total Investment': [500.00, 300.00, 3850.00, 50.00, 500.00],
                    'Equity': [470.65, 295.62, 3524.51, 98.90, 249.75],
                    'Return': [-29.35, -4.38, -325.49, 48.90, 49.75]
        }

        template = pd.DataFrame(template)
        st.write(template)

        st.download_button(label='Download', data=template.to_csv(), file_name='Stock Prediction Template.csv')

    uploaded_file = st.file_uploader("Upload the template:")

    if uploaded_file is not None:
        # Convert csv to dataframe
        template = pd.read_csv(uploaded_file, index_col=False)
        st.write("Convert csv to DataFrame")
        template.drop(columns=template.columns[0], axis=1, inplace=True)
        view = st.checkbox("View Uploaded Data")

        if view:
            st.write(template)
            
        st.write("")
        st.write("")
        
        # Download relevant stock detail
        presentable_data = pd.DataFrame()
        day=1
        while presentable_data.empty:
            ytd = date.today() - timedelta(days = day)
            presentable_data = yf.download(stock_symbol, ytd, date.today(), auto_adjust=True)
            day = day+1
        
        st.write("Today's Stock Value: ")
        st.write(presentable_data)

        # Check if stock appear in DataFrame
        if((template['Tickers'] == stock_symbol).any()):
            st.write("Stock Appear in Table")

            filtered_df =[]
            filtered_df = template.loc[template['Tickers'] == stock_symbol]

            # st.write("Old Stock Detail")
            # st.write(filtered_df)

            # st.write("Latest Stock Detail")
            # st.write(presentable_data)

            # Update old record
            filtered_df['Date'] = datetime.today().strftime("%d/%m/%Y")
            filtered_df['Open'] = float(presentable_data['Open'])
            filtered_df['Close'] = float(presentable_data['Close'])
            filtered_df['High'] = float(presentable_data['High'])
            filtered_df['Low'] = float(presentable_data['Low'])
            filtered_df['Volume'] = int(presentable_data['Volume'])

            filtered_df['Equity'] = filtered_df["Open"]*filtered_df["Share Bought"] # How many equity we have in that company
            filtered_df['Return'] = filtered_df["Equity"]-filtered_df["Total Investment"] # Earn/Loss from today market
            filtered_df['Sell/Hold'] = np.where((filtered_df['Open'] >= filtered_df['Current Share Price (Sell Price)']), "Sell", "Hold")

            st.write(filtered_df)

            strategy = filtered_df.iloc[0]['Sell/Hold']

            if strategy == "Sell":
                st.subheader("Strategy (Sell or Hold): ")
                msg = '<p style="font-family:sans-serif; color:Green; font-size: 18px;"><strong>Sell</strong></p>'
                st.markdown(msg, unsafe_allow_html=True)
            else:
                st.subheader("Strategy (Sell or Hold): ")
                msg = '<p style="font-family:sans-serif; color:Red; font-size: 18px;"><strong>Hold</strong></p>'
                st.markdown(msg, unsafe_allow_html=True)

            # Update original DataFrame
            template.loc[template['Tickers'] == stock_symbol] = filtered_df
            
        else:
            err_msg = '<p style="font-family:sans-serif; color:Red; font-size: 18px;">!!Error (Stock Not Appear): Please Proceed to "Update Stock" to Buy Stock!!</p>'
            st.markdown(err_msg, unsafe_allow_html=True)

    else:
        err_msg = '<p style="font-family:sans-serif; color:Red; font-size: 18px;">!!Error (File Type): Empty File!!</p>'
        st.markdown(err_msg, unsafe_allow_html=True)

elif sb =='Update Stock':
    st.header("Update Stock (Buy / Sell)")

    stock_bought = float(sideb.text_input(
        label = 'Enter Amount of Stock: ', 
        value=0.00
    ))

    choice = sideb.radio("",
        ["Buy Stock", "Sell Stock"],
        label_visibility="collapsed"
    )

    if choice=="Buy Stock":
        buy_Prc = float(sideb.text_input(
        label = "Enter Current Stock Price (Buy Price): ",
        value=90.00,
        format = "%.2f"
        ))
    else:
        sell_Prc = float(sideb.text_input(
        label = "Enter Current Stock Price (Sell Price): ",
        value=90.00,
        format = "%.2f"
        ))

    dl_template = st.checkbox("Download Sample Template")

    if dl_template:
        st.subheader("Sample Template")
        st.write("You may download the template and make amendments. Uplaod the file to see its updates")
        # Download sample csv
        template = {'Tickers': ['AMZN', 'AAPL', 'NFLX', 'GOOGL', 'MSFT'],
                    'Date': ['04/12/2022', '04/12/2022','04/12/2022', '04/12/2022', '04/12/2022'],
                    'Open': [94.47, 145.96, 310.49, 99.05, 249.82],
                    'High': [95.36, 148.00, 321.98, 100.77, 256.05],
                    'Low': [93.78, 145.65, 310.00, 98.90, 249.75],
                    'Close': [94.13, 147.81, 320.41, 98.90, 249.75],
                    'Volume': [72096414, 62231328, 12690602, 21480703, 21332290],
                    'Share Bought': [5.00, 2.00, 11.00, 1.00, 1.00],
                    'Current Share Price (Buy Price)': [100.00, 150.00, 350.00, 50.00, 200.00],
                    'Current Share Price (Sell Price)': [80.00, 100.00, 250.00, 200.00, 300.00],
                    'Total Investment': [500.00, 300.00, 3850.00, 50.00, 500.00],
                    'Equity': [470.65, 295.62, 3524.51, 98.90, 249.75],
                    'Return': [-29.35, -4.38, -325.49, 48.90, 49.75]
        }

        template = pd.DataFrame(template)
        st.write(template)

        st.download_button(label='Download', data=template.to_csv(), file_name='Stock Prediction Template.csv')

    uploaded_file = st.file_uploader("Upload the template:")

    if uploaded_file is not None:
        # Convert csv to dataframe
        template = pd.read_csv(uploaded_file)
        st.write("Upload Successful")
        template.drop(columns=template.columns[0], axis=1, inplace=True)

        view = st.checkbox("View Uploaded Data")

        if view:
            st.write(template)

        presentable_data = pd.DataFrame()
        day=1
        while presentable_data.empty:
            ytd = date.today() - timedelta(days = day)
            presentable_data = yf.download(stock_symbol, ytd, date.today(), auto_adjust=True)
            day = day+1
        
        st.write("Today's Stock Value: ")
        st.write(presentable_data)

        # Check if stock appear in DataFrame
        if((template['Tickers'] == stock_symbol).any()):
            st.write("Stock Appear in Table")
            st.write("Stock Bought: "+str(stock_bought))

            filtered_df =[]
            filtered_df = template.loc[template['Tickers'] == stock_symbol]
            filtered_df['Date'] = datetime.today().strftime("%d/%m/%Y")
            filtered_df['Open'] = float(presentable_data['Open'])
            filtered_df['Close'] = float(presentable_data['Close'])
            filtered_df['High'] = float(presentable_data['High'])
            filtered_df['Low'] = float(presentable_data['Low'])
            filtered_df['Volume'] = int(presentable_data['Volume'])

            # If user select Buy Stock
            if choice=="Buy Stock":
                st.write("Stock Buy Price: "+str(buy_Prc))
                filtered_df['Share Bought'] = float(filtered_df['Share Bought']) + stock_bought
                filtered_df['Current Share Price (Buy Price)'] = buy_Prc

                investment = stock_bought*buy_Prc
                filtered_df['Total Investment'] = filtered_df['Total Investment'] + investment
                filtered_df['Equity'] = filtered_df["Open"]*filtered_df["Share Bought"] # How many equity we have in that company
                filtered_df['Return'] = filtered_df["Equity"]-filtered_df["Total Investment"] # Earn/Loss from today market
                filtered_df['Sell/Hold'] = np.where((filtered_df['Open'] >= filtered_df['Current Share Price (Sell Price)']), "Sell", "Hold")

                st.write("Latest Data")
                st.write(filtered_df)

            # if user wants to sell
            else:
                st.write("Stock Sell Price: "+str(sell_Prc))

                obtained_stock = filtered_df.iloc[0]['Share Bought']

                if (obtained_stock>stock_bought):
                    filtered_df['Share Bought'] = obtained_stock - stock_bought
                    filtered_df['Current Share Price (Sell Price)'] = sell_Prc

                    filtered_df['Date'] = datetime.today().strftime("%d/%m/%Y")
                    investment = stock_bought*sell_Prc
                    filtered_df['Total Investment'] = filtered_df['Total Investment'] - investment
                    filtered_df['Equity'] = filtered_df["Open"]*filtered_df["Share Bought"] # How many equity we have in that company
                    filtered_df['Return'] = filtered_df["Equity"]-filtered_df["Total Investment"] # Earn/Loss from today market
                    filtered_df['Sell/Hold'] = np.where((filtered_df['Open'] >= filtered_df['Current Share Price (Sell Price)']), "Sell", "Hold")

                    st.write("Latest Data")
                    st.write(filtered_df)

                else:
                    err_msg = '<p style="font-family:sans-serif; color:Red; font-size: 18px;">!!Error (Amount of Stock): Cannot Sell Stock more than Original Amount!!</p>'
                    st.markdown(err_msg, unsafe_allow_html=True)

            # Update into original table
            template.loc[template['Tickers'] == stock_symbol] = filtered_df

            confirm = st.checkbox("View Template before download")

            if confirm:
                st.write(template)
            
            st.download_button(label='Download as CSV', data=template.to_csv(), file_name='Stock Prediction Template.csv')

        # if not appear in dataframe            
        else:
            # Create new row
            st.write("Stock not appear in Table, creating a new record")               

            # Downlaod new record
            presentable_data = pd.DataFrame()
            day=1
            while presentable_data.empty:
                ytd = date.today() - timedelta(days = day)
                presentable_data = yf.download(stock_symbol, ytd, date.today(), auto_adjust=True)
                day = day+1
            
            presentable_data.insert(0, "Date", datetime.today().strftime("%d/%m/%Y"), True)

            # If user select Buy Stock
            if choice=="Buy Stock":
                st.write("Stock Bought: "+str(stock_bought))
                st.write("Stock Buy Price: "+str(buy_Prc))

                presentable_data.insert(0, "Tickers", stock_symbol, True)
                presentable_data = presentable_data.rename(index={presentable_data.index[0]:len(template)})
                investment = stock_bought*buy_Prc
                presentable_data['Share Bought'] = stock_bought
                presentable_data['Current Share Price (Buy Price)'] = buy_Prc
                presentable_data['Current Share Price (Sell Price)'] = 0
                presentable_data['Total Investment'] = investment 
                presentable_data['Equity'] = presentable_data["Open"]*presentable_data["Share Bought"] # How many equity we have in that company
                presentable_data['Return'] = presentable_data["Equity"]-presentable_data["Total Investment"] # Earn/Loss from today market
                presentable_data['Sell/Hold'] = np.where((presentable_data['Open'] >= presentable_data['Current Share Price (Sell Price)']), "Sell", "Hold")

                st.write("New Record")
                st.write(presentable_data)

                template = template.append(presentable_data)

                confirm = st.checkbox("View Template before download")

                if confirm:
                    st.write(template)
                
                st.download_button(label='Download as CSV', data=template.to_csv(), file_name='Stock Prediction Template.csv')

            else:
                err_msg = '<p style="font-family:sans-serif; color:Red; font-size: 18px;">!!Error (Amount of Stock): Empty Stock Amount!!</p>'
                st.markdown(err_msg, unsafe_allow_html=True)
    
        
    else:
        err_msg = '<p style="font-family:sans-serif; color:Red; font-size: 18px;">!!Error (File Type): Empty File!!</p>'
        st.markdown(err_msg, unsafe_allow_html=True)

    







    


    










