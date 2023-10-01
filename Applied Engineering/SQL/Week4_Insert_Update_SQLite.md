<center>
    <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DB0201EN-SkillsNetwork/labs/FinalModule_Coursera_V5/images/SN_web_lightmode.png" width="300" alt="cognitiveclass.ai logo">
</center>

# Create & Access SQLite database using Python

Estimated time needed: **15** minutes

## Objectives

After completing this lab you will be able to:

*   Create a database
*   Create a table
*   Insert data into the table
*   Query data from the table
*   Retrieve the result set into a pandas dataframe
*   Close the database connection


SQLite is a software library that implements a self-contained, serverless, zero-configuration, transactional SQL database engine. SQLite is the most widely deployed SQL database engine in the world.

## Task 1: Create database using SQLite



```python
#Install & load sqlite3
#!pip install sqlite3  ##Uncomment the code to install sqlite3
import sqlite3
```


```python
# Connecting to sqlite
# connection object
conn = sqlite3.connect('INSTRUCTOR.db')
```

Cursor class is an instance using which you can invoke methods that execute SQLite statements, fetch data from the result sets of the queries. You can create Cursor object using the cursor() method of the Connection object/class.



```python
# cursor object
cursor_obj = conn.cursor()
```

## Task 2: Create a table in the database

In this step we will create a table in the database with following details:

<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DB0201EN-SkillsNetwork/labs/FinalModule_Coursera_V5/images/table.png" align="center">


Before creating a table, let's first if the table already exist or not. To drop the table from a database use **DROP** query. A cursor is an object which helps to execute the query and fetch the records from the database.



```python
# Drop the table if already exists.
cursor_obj.execute("DROP TABLE IF EXISTS INSTRUCTOR")
```

## Dont worry if you get this error:

If you see an exception/error similar to the following, indicating that INSTRUCTOR is an undefined name, that's okay. It just implies that the INSTRUCTOR table does not exist in the table - which would be the case if you had not created it previously.

<strong>Exception: \[IBM]\[CLI Driver]\[DB2/LINUXX8664] SQL0204N  "ABC12345.INSTRUCTOR" is an undefined name.  SQLSTATE=42704 SQLCODE=-204</strong>



```python
# Creating table
table = """ create table IF NOT EXISTS INSTRUCTOR(ID INTEGER PRIMARY KEY NOT NULL, FNAME VARCHAR(20), LNAME VARCHAR(20), CITY VARCHAR(20), CCODE CHAR(2));"""
 
cursor_obj.execute(table)
 
print("Table is Ready")
```

## Task 3: Insert data into the table

In this step we will insert some rows of data into the table.

The INSTRUCTOR table we created in the previous step contains 3 rows of data:

<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DB0201EN-SkillsNetwork/labs/FinalModule_Coursera_V5/images/table1.png" align="center">

We will start by inserting just the first row of data, i.e. for instructor Rav Ahuja



```python
cursor_obj.execute('''insert into INSTRUCTOR values (1, 'Rav', 'Ahuja', 'TORONTO', 'CA')''')
```

The output you will get something as: <strong>sqlite3.Cursor at 0x27a1a491260</strong> which means mySql database has sqlite3.Cursor object at 0x27a1a49126 as output in table. But you may get the different number.

Now use a single query to insert the remaining two rows of data



```python
cursor_obj.execute('''insert into INSTRUCTOR values (2, 'Raul', 'Chong', 'Markham', 'CA'), (3, 'Hima', 'Vasudevan', 'Chicago', 'US')''')
```

## Task 4: Query data in the table

In this step we will retrieve data we inserted into the INSTRUCTOR table.



```python
statement = '''SELECT * FROM INSTRUCTOR'''
cursor_obj.execute(statement)

print("All the data")
output_all = cursor_obj.fetchall()
for row_all in output_all:
  print(row_all)
```


```python
## Fetch few rows from the table
statement = '''SELECT * FROM INSTRUCTOR'''
cursor_obj.execute(statement)
  
print("All the data")
# If you want to fetch few rows from the table we use fetchmany(numberofrows) and mention the number how many rows you want to fetch
output_many = cursor_obj.fetchmany(2) 
for row_many in output_many:
  print(row_many)
```


```python
# Fetch only FNAME from the table
statement = '''SELECT FNAME FROM INSTRUCTOR'''
cursor_obj.execute(statement)
  
print("All the data")
output_column = cursor_obj.fetchall()
for fetch in output_column:
  print(fetch)
```

<Strong>Bonus: now write and execute an update statement that changes the Rav's CITY to MOOSETOWN</strong>



```python
query_update='''update INSTRUCTOR set CITY='MOOSETOWN' where FNAME="Rav"'''
cursor_obj.execute(query_update)
```


```python
statement = '''SELECT * FROM INSTRUCTOR'''
cursor_obj.execute(statement)
  
print("All the data")
output1 = cursor_obj.fetchmany(2)
for row in output1:
  print(row)
```

## Task 5: Retrieve data into Pandas

In this step we will retrieve the contents of the INSTRUCTOR table into a Pandas dataframe



```python
import pandas as pd
#retrieve the query results into a pandas dataframe
df = pd.read_sql_query("select * from instructor;", conn)

#print the dataframe
df
```


```python
#print just the LNAME for first row in the pandas data frame
df.LNAME[0]
```

Once the data is in a Pandas dataframe, you can do the typical pandas operations on it.

For example you can use the shape method to see how many rows and columns are in the dataframe



```python
df.shape
```

## Task 6: Close the Connection

We free all resources by closing the connection. Remember that it is always important to close connections so that we can avoid unused connections taking up resources.



```python
# Close the connection
conn.close()
```

## Summary

In this tutorial you created a database & table in Python notebook using SQLite3. Then created a table and insert a few rows of data into it. Then queried the data. You also retrieved the data into a pandas dataframe.


## Author

<a href="https://www.linkedin.com/in/ravahuja/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDB0201ENSkillsNetwork20127838-2021-01-01" target="_blank">Rav Ahuja</a>

<a href="https://www.linkedin.com/in/malika-goyal-04798622/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDB0201ENSkillsNetwork20127838-2021-01-01" target="_blank">Malika</a>

## Other Contributor(s)

<a href="https://www.linkedin.com/in/lakshmi-holla-b39062149/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDB0201ENSkillsNetwork20127838-2021-01-01" target="_blank">Lakshmi Holla</a>

## Change Log

| Date (YYYY-MM-DD) | Version | Changed By | Change Description        |
| ----------------- | ------- | ---------- | ------------------------- |
| 2022-03-03        | 1.0     | Malika     | Created lab using sqlite3 |

<hr>

## <h3 align="center"> © IBM Corporation 2022. All rights reserved. <h3/>

