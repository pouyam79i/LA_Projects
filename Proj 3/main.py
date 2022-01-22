# Coded by Pouya Mohammadi - 9829039

# Importing libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# flagger
def flag(flag_val):
    # If true it prints debug flags
    debug_state = False
    
    if debug_state:
        print("*** Debug Flag: " + flag_val)


# Linear algorithm
def linear():
    # Reading dataset
    chart_csv = pd.read_csv('covid_cases.csv', usecols=[0, 1])
    flag("1 - csv file readed")

    chart_csv.head()
    dates = chart_csv[['date']]
    covid_cases = chart_csv[['World']]
    dates.tail()
    covid_cases.tail()
    flag("2 - Date of csv categorized!")

    covid_matrix = np.array(covid_cases.values, 'float')
    X_date = np.array(dates)[:, 0]
    Y = covid_matrix[:, 0]
    X_num = np.arange(len(Y))
    limit = int(len(Y) * 0.95)
    A = np.vstack((np.ones(limit), X_num[0:limit])).T
    flag("3 - Basic matrixes are built")

    A_tanspose = A.transpose()
    params = np.linalg.inv(A_tanspose.dot(A)).dot(A_tanspose).dot(Y[0:limit])
    flag("4 - Parameters calculater")

    Y_reg = params[1] * X_num + params[0]
    plt.plot(X_date, Y_reg)
    plt.plot(X_date, Y)
    plt.title("Covid Cases - Linear Regression")
    plt.show()
    flag("5 - Flag is drawn")
    
    X_var = X_num[limit:]
    Error = np.zeros(len(Y) - limit)
    for i in X_var:
        Error[i - limit] = Y[i] - Y_reg[i]
    flag("6 - Error Calculated")

    print(" *** Report with regards of linear regression")
    print()
    for i in X_var[:5]:
        print("Date: " + X_date[i])
        print("Real Val: " + str(Y[i]))
        print("Estimated Val: " + str(Y_reg[i]))
        print("Error: " + str(Error[i - limit]))
        print()
    flag("7 - Report has been printed")


    # Square algorithm
def square():
    # Reading dataset
    chart_csv = pd.read_csv('covid_cases.csv', usecols=[0, 1])
    flag("1 - csv file readed")

    chart_csv.head()
    dates = chart_csv[['date']]
    covid_cases = chart_csv[['World']]
    dates.tail()
    covid_cases.tail()
    flag("2 - Date of csv categorized!")

    covid_matrix = np.array(covid_cases.values, 'float')
    X_date = np.array(dates)[:, 0]
    Y = covid_matrix[:, 0]
    X_num = np.arange(len(Y))
    limit = int(len(Y) * 0.95)
    A = np.vstack((np.ones(limit), X_num[0:limit], X_num[0:limit] * X_num[0:limit])).T
    flag("3 - Basic matrixes are built")

    A_tanspose = A.transpose()
    params = np.linalg.inv(A_tanspose.dot(A)).dot(A_tanspose).dot(Y[0:limit])
    flag("4 - Parameters calculater")

    Y_reg = params[2] * X_num * X_num + params[1] * X_num + params[0]
    plt.plot(X_date, Y_reg)
    plt.plot(X_date, Y)
    plt.title("Covid Cases - Square Regression")
    plt.show()
    flag("5 - Flag is drawn")
    
    X_var = X_num[limit:]
    Error = np.zeros(len(Y) - limit)
    for i in X_var:
        Error[i - limit] = Y[i] - Y_reg[i]
    flag("6 - Error Calculated")

    print(" *** Report with regards of square regression")
    print()
    for i in X_var[:5]:
        print("Date: " + X_date[i])
        print("Real Val: " + str(Y[i]))
        print("Estimated Val: " + str(Y_reg[i]))
        print("Error: " + str(Error[i - limit]))
        print()
    flag("7 - Report has been printed")

# Run each part 
linear()
square()
