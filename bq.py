from google.cloud import bigquery
client = bigquery.Client()

sql = """
SELECT business_id,review_id,text,stars,useful,RestaurantsPriceRange2 FROM `yelp-243412.yelp.reviews_price` 
WHERE RAND() < 1000000/4142751
  """

filename="bq-1M-reviews.csv"
df = client.query(sql).to_dataframe()
df.to_csv(filename)
