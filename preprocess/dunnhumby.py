import pandas as pd

user_order = pd.read_csv('../DataSource/dunnhumby_50k/transactions_200607.csv', usecols=['CUST_CODE', 'SHOP_DATE', 'BASKET_ID', 'PROD_CODE'])
file_list = ['200608.csv', '200609.csv', '200610.csv', '200611.csv', '200612.csv', '200613.csv', '200614.csv']
for date in file_list:
    order_file = '../DataSource/dunnhumby_50k/transactions_' + date
    temp = pd.read_csv(order_file, usecols=['CUST_CODE', 'SHOP_DATE', 'BASKET_ID', 'PROD_CODE'])
    user_order = pd.concat([user_order, temp], ignore_index=True)
user_order = user_order.dropna(how='any')
print(user_order)

print('Construct the basic basket sequence... and filter via user sequence length..')
baskets = None
for user, user_data in user_order.groupby('CUST_CODE'):
    date_list = list(set(user_data['SHOP_DATE'].tolist()))
    date_list = sorted(date_list)
    if len(date_list)>=3 and len(date_list)<=50:

        date_num = 1
        for date in date_list:
            date_data = user_data[user_data['SHOP_DATE'].isin([date])]
            date_item = list(set(date_data['PROD_CODE'].tolist()))
            item_num = len(date_item)
            if baskets is None:
                baskets = pd.DataFrame({'user_id': pd.Series([user for i in range(item_num)]),
                                        'order_number': pd.Series([date_num for i in range(item_num)]),
                                        'product_id': pd.Series(date_item),
                                        'eval_set': pd.Series(['prior' for i in range(item_num)])})
                date_num += 1
            else:
                if date == date_list[-1]:#if date is the last. then add a tag here
                    temp = pd.DataFrame({'user_id': pd.Series([user for i in range(item_num)]),
                                            'order_number': pd.Series([date_num for i in range(item_num)]),
                                            'product_id': pd.Series(date_item),
                                            'eval_set': pd.Series(['train' for i in range(item_num)])})
                    date_num += 1
                    baskets = pd.concat([baskets, temp], ignore_index=True)
                else:
                    temp = pd.DataFrame({'user_id': pd.Series([user for i in range(item_num)]),
                                            'order_number': pd.Series([date_num for i in range(item_num)]),
                                            'product_id': pd.Series(date_item),
                                            'eval_set': pd.Series(['prior' for i in range(item_num)])})
                    date_num += 1
                    baskets = pd.concat([baskets, temp], ignore_index=True)

print('total transcations:', len(baskets))
baskets.to_csv('dataset/dunnhumby_temp.csv', index=False)
##### Filter items, keep items which appreas k times in the training set.
print('Filter data use the training items...')
item_set_all = set()
item_filter_dict = dict()
history_baskets = baskets[baskets['eval_set'].isin(['prior'])].reset_index()

for ind in range(len(history_baskets)):
    product_id = history_baskets['product_id'].iloc[ind]
    if product_id not in item_filter_dict:
        item_filter_dict[product_id] = 1
    else:
        item_filter_dict[product_id] += 1

for key in item_filter_dict.keys():
    if item_filter_dict[key]>=17:
        item_set_all.add(key)

baskets = baskets[baskets['product_id'].isin(item_set_all)].reset_index()
print('After transcations:', len(baskets))

print('Reset the user_id and product_id....')
item_dict = dict()
item_ind = 1
user_dict = dict()
user_ind = 1
for ind in range(len(baskets)):
    product_id = baskets.at[ind, 'product_id']
    if product_id not in item_dict:
        item_dict[product_id] = item_ind
        item_ind += 1
    baskets.at[ind, 'product_id'] = item_dict[product_id]

    user_id = baskets.at[ind, 'user_id']
    if user_id not in user_dict:
        user_dict[user_id] = user_ind
        user_ind += 1
    baskets.at[ind, 'user_id'] = user_dict[user_id]

baskets.to_csv('dataset/dunnhumby.csv', index=False)
print('Done...')