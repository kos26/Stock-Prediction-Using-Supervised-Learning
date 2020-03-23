#!/usr/bin/env python
# coding: utf-8

# In[9]:


from StockPrediction import StockPrediction

answer=True
print("This program will gain real time data from Yahoo Finance API,")
print("You may use this program to analyze the historical behaviour of the stock\nand to make predictions using popular regression models\n")
print("\n\nThe purpose of making this program is to get a sense of which regression model would perform better in predicting the closing balance of a stock")
print('\n\n\n')
user = input("Would You to begin (Yes/No): ")

while (answer==True):    
    if (user.lower()=='yes'):
        print("\n\nPlease input the abbreviation of the company first\nExample: \nAmazon = AMZN\nMicrosoft = MSFT\nApple = AAPL\nGoogle = GOOG\n")
        company = input("Abbreviation of Company: ")
        company = company.upper()
        print("\n\nThank you, \n\nBy default the end date is the current date, please input a starting date")
        print("\nExample: 2015-01-01, if you like to use approximately 5 years old data for this program.")
        date = input("Starting date (YYYY-MM-DD): ")
        print('Thank you.\n\n')
        test = StockPrediction(company, date)
        print(test.stock_analysis())
        print(test.predict_stock())
        lstm = input("The LSTM model may take a while would you like to continue (yes/no): ")
        if (lstm.lower() == 'yes'):
            print(test.LSTM_predict_stock())
            user = input("\n\nWould you to like to use another stock (Yes/No): ")
        elif(lstm.lower() == 'no'):
            print('Thank You.\n')
            user = input("\n\nWould you to like to use another stock (Yes/No): ")
        else:
            print("Invalid input, try again\n")
            lstm = input("The LSTM model may take a while would you like to continue (yes/no): ")
    elif(user=='no'):
        answer=False
        print('Thank you and stay healthy.')
        break
    else:
        print("Invalid input, try again")
        user = input("\nWould You to begin (Yes/No): ")
        
        
        


# In[ ]:





# In[ ]:




