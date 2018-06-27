'''
Created on 2018年6月25日

@author: root
'''
import pandas
def get_data(file_name):
    return pandas.read_csv(file_name)

def print_limit_data(data):
    print("获取前4行数据 ：")
    print(data.head(4))
    print("获取后3行数据 ：")
    print(data.tail(3))
    print("获取1-3行数据 ：")
    print(data.loc[1:3])
    
def get_columns_data(data,columns,start=0,count=5):
    print("获取如下列数据 ：%s"%columns)
    print(data[columns][start:start+count])
    
def get_all_columns(data):
    return data.columns.tolist()

if __name__ == '__main__':
    food_data = get_data("food_info.csv")
    print(food_data.dtypes)
    print_limit_data(food_data)
    get_columns_data(food_data,["NDB_No","Cholestrl_(mg)"])
    print(get_all_columns(food_data))