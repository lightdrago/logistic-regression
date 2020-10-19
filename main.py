import modules
import pandas as pd

if __name__ == '__main__':
    data = modules.delete_nan_from_data(pd.read_csv('logistic-regression\data\data_new.csv'))
    while True:
        print('Enter what you want to do:\n\t1 - Show dataset info\n\t2 - Start learning and classification\n\t3 - Exit')
        choice = input('-> ')
        if choice == '1':
            modules.print_data_info(data)
        elif choice == '2':
            modules.logistic_regression(data)
        elif choice == '3':
            break
        else:
            print('Error. Please try again!')
