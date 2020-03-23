# Stock Prediction using Supervised learning Techniques
The program is written in Pythion 3.7
The purpose of this program is to apply supervised learning techniques on real data gained from finance.yahoo.com.
Independent features in this program as Open, Hign and Prices of the stock and the targeted value is the Close price of the stock.

# Input/Output:
The program asks the user to input an abbreviation of a company. Example, Amazon = AMZN
The program later ask the user to input a start date. Example, 2015-01-01
The end date is set to be the current date.
The program outputs a several graphs showing high and low of the stock from the start date until now.
The Program also shows scores of all the models used in the program for the user to get a sense of the best model for this purpose.
The program will then ask the user if he/she would like to use a deep learning model which may take a while to complete.
If the input is yes the program predicts the closing price from the testing set and displays its scores

# Running the program
1) Install Libraries:
pip3 install numpy
pip3 install pandas
pip3 install pandas-datareader
pip3 install sklearn
pip3 install keras
pip3 install datatime
pip3 install seaborn

2) python3 app.py
