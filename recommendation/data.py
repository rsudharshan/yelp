import pandas as pd
stars = pd.read_csv('review_stars.csv')

def transform_ids(col_field):
    ueid = {}
    ueid_col = []
    ueid_count = 0
    for user_id in stars[col_field]:
        if user_id not in ueid:
            ueid[user_id] = ueid_count
            ueid_col.append(ueid_count)
            ueid_count = ueid_count+1
        else:
            ueid_col.append(ueid[user_id])
    ueid_series =pd.Series(ueid_col)
    return ueid_series

stars["beid"]=transform_ids('business_id')
stars["ueid"]=transform_ids('user_id')

stars.to_csv("stars.csv")