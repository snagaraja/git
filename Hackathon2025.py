# Databricks notebook source
# MAGIC %pip install -U -qqq langchain_core langchain_databricks langchain_community
# MAGIC %restart_python

# COMMAND ----------

location = "Chicago"
special_interests = ["shopping", "museum", "beach"]

# COMMAND ----------

# MAGIC %md
# MAGIC AirBnB

# COMMAND ----------

import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_databricks import ChatDatabricks
from databricks.sdk import WorkspaceClient
import os

# configure workspace tokens
w = WorkspaceClient()
os.environ["DATABRICKS_HOST"] = w.config.host
os.environ["DATABRICKS_TOKEN"] = w.tokens.create(comment="for model serving", lifetime_seconds=1200).token_value

llm = ChatDatabricks(endpoint="databricks-llama-4-maverick")

special_interest_text = " OR ".join(special_interests)
print(special_interest_text)
inputs = {"location": location, "special_need": special_interests}

def format_context(df: pd.DataFrame) -> str:
    """
    Converts the DataFrame into a JSON string to ensure all data is passed
    to the model without truncation. JSON is also a great format for structured data
    like you have in 'description_by_sections'.
    """
    return df.to_json(orient='records', indent=2)

def find_airbnb_properties(kwargs) -> pd.DataFrame:
  """
  Queries the Bright Initiative Airbnb dataset for properties in a specific location
  that have reviews mentioning "wheelchair".
  """

  location = kwargs['location']
  special_need = kwargs['special_need']
  
  where_clause= "EXISTS(reviews, review -> review ILIKE '%{need}%')"
  final_list = []
  for need in special_need:
    wherelist = where_clause.format(need=need)
    final_list.append(wherelist)
    final_where_clause = " OR ".join(final_list)


  query = f"""
    SELECT
      listing_name,
      location_details,
      location,
      details,
      description_by_sections,
      reviews,
      final_url,
      total_price
    FROM `dais-hackathon-2025`.bright_initiative.airbnb_properties_information
    WHERE
      location ILIKE '%{location}%'
      AND host_number_of_reviews > 100
      AND ({final_where_clause})
    LIMIT 5
  """
  
  return format_context(spark.sql(query).toPandas())

# Define the prompt template for the LLM
prompt_template = PromptTemplate(
  template = """
  You are a helpful assistant for {special_needs} travel. Your goal is to summarize potential Airbnb listings for a user.

  The following listing *mention* {special_needs} needs. Closely review the descriptions and review,
  and then summarize the features and also provide a link to the listing along with the price of the listing

  Here is the JSON data:
  {context}
  """,
  input_variables = ["special_need", "context"]
)
prompt_template = prompt_template.partial(special_needs=special_interest_text)
llm = ChatDatabricks(endpoint="databricks-llama-4-maverick")

# This is our simple "agentic" chain
chain = (
    find_airbnb_properties
    | prompt_template
    | llm
    | StrOutputParser()
)

# Let's run the chain for Chicago!
result = chain.invoke(inputs)

print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC Booking.com

# COMMAND ----------

import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_databricks import ChatDatabricks
from databricks.sdk import WorkspaceClient
import os

# configure workspace tokens
w = WorkspaceClient()
os.environ["DATABRICKS_HOST"] = w.config.host
os.environ["DATABRICKS_TOKEN"] = w.tokens.create(comment="for model serving", lifetime_seconds=1200).token_value

llm = ChatDatabricks(endpoint="databricks-llama-4-maverick")

special_interest_text = " OR ".join(special_interests)
print(special_interest_text)
inputs = {"location": location, "special_need": special_interests}

def format_context(df: pd.DataFrame) -> str:
    """
    Converts the DataFrame into a JSON string to ensure all data is passed
    to the model without truncation. JSON is also a great format for structured data
    like you have in 'description_by_sections'.
    """
    return df.to_json(orient='records', indent=2)

def find_airbnb_properties(kwargs) -> pd.DataFrame:
  """
  Queries the Bright Initiative Airbnb dataset for properties in a specific location
  that have reviews mentioning "wheelchair".
  """

  location = kwargs['location']
  special_need = kwargs['special_need']
  
  where_clause= "description ILIKE '%{need}%'"
  final_list = []
  for need in special_need:
    wherelist = where_clause.format(need=need)
    final_list.append(wherelist)
    final_where_clause = " OR ".join(final_list)


  query = f"""
      SELECT 
      title,
      description,
      fine_print,
      most_popular_facilities,
      property_highlights,
      property_information,
      property_surroundings,
      house_rules,
      top_reviews, 
      url
      FROM `dais-hackathon-2025`.bright_initiative.booking_hotel_listings
    WHERE
      city ILIKE '%{location}%'
      AND ({final_where_clause})
    LIMIT 5
  """
  
  return format_context(spark.sql(query).toPandas())

# Define the prompt template for the LLM
prompt_template = PromptTemplate(
  template = """
  You are a helpful assistant for {special_needs} travel. Your goal is to summarize potential Booking.com listings for a user.

  The following listing *mention* {special_needs} needs. Closely review the descriptions and review,
  and then summarize the features and also provide a link to the listing along with the price of the listing

  Here is the JSON data:
  {context}
  """,
  input_variables = ["special_need", "context"]
)
prompt_template = prompt_template.partial(special_needs=special_interest_text)
llm = ChatDatabricks(endpoint="databricks-llama-4-maverick")

# This is our simple "agentic" chain
chain = (
    find_airbnb_properties
    | prompt_template
    | llm
    | StrOutputParser()
)

# Let's run the chain for Chicago!
result = chain.invoke(inputs)

print(result)

# COMMAND ----------

special_need = ["shopping", "museum"]
where_clause= "EXISTS(reviews, review -> review ILIKE '%{need}%')"
final_list = []
for need in special_need:
  print(need)
  wherelist = where_clause.format(need=need)
  final_list.append(wherelist)
print(final_list)

final_where_clause = " OR ".join(final_list)

print(final_where_clause)


# COMMAND ----------

special_need = ["shopping", "museum"]

special_needs = " OR ".join(special_need)
print(special_needs)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT reviews, * FROM `dais-hackathon-2025`.bright_initiative.airbnb_properties_information
# MAGIC WHERE
# MAGIC   location = 'Chicago, Illinois, United States'
# MAGIC   AND host_number_of_reviews > 1000
# MAGIC   AND EXISTS(reviews, review -> review ILIKE '%beach%')
# MAGIC LIMIT 5;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM `dais-hackathon-2025`.bright_initiative.booking_hotel_listings

# COMMAND ----------

# MAGIC %sql
<<<<<<< Updated upstream
# MAGIC SELECT * FROM `dais-hackathon-2025`.bright_initiative.google_maps_businesses
=======
# MAGIC SELECT * FROM `dais-hackathon-2025`.bright_initiative.google_maps_businesses
>>>>>>> Stashed changes
