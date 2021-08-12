CREATE TABLE Exercise3_SQL_raw (
    user_id INTEGER,
    transaction_date DATETIME,
    amount REAL
);
---Imported through SQLite the CSV File
--Create a Table which seperates the subscribers and defines the times they were subscribed.
CREATE TABLE Subscribers_TimeFrame AS
--Here we test the start of the subscription and the end of the subscription whether it is one month after the last payment or the last transaction date available.
SELECT min(Date(transaction_date)) AS Start, min(max(Date(transaction_date, '+1 month')),'2019-02-01') AS End,
       user_id, round((julianday(min(max(Date(transaction_date, '+1 month')),'2019-02-01'))- julianday(min(Date(transaction_date))))/30,2) AS Num_Months
FROM Exercise3_SQL_raw
WHERE amount == 8.99
GROUP BY user_id
ORDER BY user_id;

--Create a more in depth table that summarizes the monthly orders for subscribers which can be joined with the table just created to find monthly orders and purchases.
CREATE TABLE Subscribers_Monthly_Stats AS
SELECT  Subscribers_TimeFrame.user_id, Num_Months, sum(amount != 8.99) AS Total_Orders, round(sum(amount),2) AS Total_Purchases, round(sum(amount)/Num_Months,2) AS Monthly_Purchases, round(sum(amount != 8.99)/Num_Months,2) AS Monthly_Orders
FROM Exercise3_SQL_raw
--These Joins are for the where clause which needs to take into account subscriptions and subscription timeframe
LEFT JOIN Subscribers_TimeFrame
ON Exercise3_SQL_raw.user_id = Subscribers_TimeFrame.user_id
--For someone to be a subscriber they need to be in the TimeFrame that was ditated in the first table.
WHERE Exercise3_SQL_raw.user_id == Subscribers_TimeFrame.user_id and transaction_date >= Start and transaction_date <= End
GROUP BY Exercise3_SQL_raw.user_id
ORDER BY Subscribers_TimeFrame.user_id;

--find when the customer started ordering the first time because if they are new in 2019 it will skew the data, also add the number of orders and sums to the users.
SELECT min(Date(transaction_date)) AS Start_R, Exercise3_SQL_raw.user_id, sum(amount), count(amount)
FROM Exercise3_SQL_raw
--These Joins are for the where clause which needs to take into account subscriptions and subscription timeframe
LEFT JOIN Subscribers_TimeFrame
ON Exercise3_SQL_raw.user_id = Subscribers_TimeFrame.user_id
--For someone to be a subscriber they need to be in the TimeFrame that was ditated in the first table
WHERE amount != 8.99 or transaction_date < Subscribers_TimeFrame.Start or transaction_date > Subscribers_TimeFrame.End
GROUP BY Exercise3_SQL_raw.user_id
ORDER BY Exercise3_SQL_raw.user_id;

--This is for the dummy table I will have to make later on to avoid null values on the unsubscribed
CREATE TABLE Subscriptions AS
SELECT user_id, sum(amount == 8.99) AS subscriptions
FROM Exercise3_SQL_raw
GROUP BY user_id
ORDER BY user_id;

--Create a dummy table with 0s for the start and ends since unsubscribed orders never have a timeframe. This is simply putting all values back into a table to dictate subscriber times.
CREATE TABLE Dummy_Table AS
SELECT user_id, 0*user_id AS Start, 0*user_id AS End
FROM Subscriptions
--only allows unsubscribed users in or else the rest of the data would get mangled.
WHERE subscriptions == 0
GROUP BY user_id;

--Insert the subscribed values in now with the correct numbers.
INSERT INTO Dummy_Table (user_id,Start,End)
SELECT user_id, Start , End
FROM Subscribers_TimeFrame;

--Now that the dummy table is complete we can make a unsubscribed timeframe that wont be full of null values
CREATE TABLE Unsubscribed_TimeFrame AS
SELECT min(Date(transaction_date)) AS Start_R, round((julianday('2019-02-01')+ julianday(Dummy_Table.Start)-julianday(Dummy_Table.End)-julianday(min(Date(transaction_date))))/30,2) AS Months_Unsubscibed, Exercise3_SQL_raw.user_id, round(sum(amount),2) AS Total_Sum_R, round(count(amount),2) AS Total_Count_R
FROM Exercise3_SQL_raw
--These Joins are for the where clause which needs to take into account subscriptions and subscription timeframe
LEFT JOIN Subscribers_TimeFrame
ON Exercise3_SQL_raw.user_id = Subscribers_TimeFrame.user_id
LEFT JOIN Dummy_Table_Nonsubs
ON Exercise3_SQL_raw.user_id = Dummy_Table.user_id
LEFT JOIN Subscriptions
ON Exercise3_SQL_raw.user_id = Subscriptions.user_id
--since these are the unsubscribed values subscriptions can equal 0 or it can be outside of the users specific subscription timeframe.
WHERE subscriptions == 0 or transaction_date < Subscribers_TimeFrame.Start or transaction_date > Subscribers_TimeFrame.End
GROUP BY Exercise3_SQL_raw.user_id
ORDER BY Exercise3_SQL_raw.user_id;

--Get every transaction from unsubbed people
CREATE TABLE Unsubscribed_Raw AS
SELECT Exercise3_SQL_raw.user_id, strftime('%Y-%m',transaction_date) AS Month, amount
FROM Exercise3_SQL_raw
--These Joins are for the where clause which needs to take into account subscriptions and subscription timeframe
LEFT JOIN Subscribers_TimeFrame
ON Exercise3_SQL_raw.user_id = Subscribers_TimeFrame.user_id
LEFT JOIN Subscriptions
ON Exercise3_SQL_raw.user_id = Subscriptions.user_id
WHERE subscriptions == 0 or transaction_date < Subscribers_TimeFrame.Start or transaction_date > Subscribers_TimeFrame.End
ORDER BY transaction_date;

--Get every transaction from subbed people
CREATE TABLE Subscribed_Raw AS
SELECT Exercise3_SQL_raw.user_id, strftime('%Y-%m',transaction_date) AS Month, amount
FROM Exercise3_SQL_raw
--These Joins are for the where clause which needs to take into account subscriptions and subscription timeframe
LEFT JOIN Subscribers_TimeFrame
ON Exercise3_SQL_raw.user_id = Subscribers_TimeFrame.user_id
LEFT JOIN Subscriptions
ON Exercise3_SQL_raw.user_id = Subscriptions.user_id
WHERE subscriptions < 0 and transaction_date > Subscribers_TimeFrame.Start and transaction_date < Subscribers_TimeFrame.End
ORDER BY transaction_date;

--get avg subscriber stats total
CREATE TABLE Subscriber_Avg_Stats AS
SELECT round(avg(Monthly_Purchases),2) AS avg_Spent_Sub, round(avg(Monthly_Orders),2) AS avg_Monthly_Orders_Sub
FROM Subscribers_Monthly_Stats;

--get avg unsubscriber stats total
CREATE TABLE Unsubscribed_Avg_Stats AS
SELECT round(avg(Total_Sum_R/Months_Unsubscibed),2) AS avg_Spent_Unsub, round(avg(Total_Count_R/Months_Unsubscibed),2) AS avg_Monthly_Orders_Unsub
FROM Unsubscribed_TimeFrame;

--get avg subscriber monthly total
CREATE TABLE Avg_Sub_Monthly AS
SELECT Month, round(sum(amount)/count(DISTINCT user_id),2) AS avg_Sub_Over_Time
FROM Subscribed_Raw
GROUP BY Month;

--get avg unsubscriber monthly total
CREATE TABLE Avg_Unsub_Monthly AS
SELECT Month, round(sum(amount)/count(DISTINCT user_id),2) AS avg_Unsub_Over_Time
FROM Unsubscribed_Raw
GROUP BY Month;
--Join the tables together for easier to show on minitab later
CREATE TABLE Avg_Monthly_Spend AS
SELECT Avg_Sub_Monthly.Month, avg_Sub_Over_Time AS Avg_Sub_Monthly_Spend, avg_Unsub_Over_Time AS Avg_Unsub_Monthly_Spend
FROM Avg_Sub_Monthly
JOIN Avg_Unsub_Monthly
ON Avg_Sub_Monthly.Month = Avg_Unsub_Monthly.Month
GROUP BY Avg_Sub_Monthly.Month;

--get avg revenue subscriber monthly
CREATE TABLE Rev_Sub_Monthly AS
SELECT Month, round(sum(Subscribed_Raw.amount),2) AS Sum_Rev_Sub
FROM Subscribed_Raw
GROUP BY Subscribed_Raw.Month;

--get avg revenue unsubscriber monthly
CREATE TABLE Rev_Unsub_Monthly AS
SELECT Month, round(sum(Unsubscribed_Raw.amount),2) AS Sum_Rev_Unsub
FROM Unsubscribed_Raw
GROUP BY Unsubscribed_Raw.Month;

--CREATE TABLE Proportion_Sub_Revenue AS
SELECT Rev_Sub_Monthly.Month, round(Sum_Rev_Sub/(Sum_Rev_Sub + Sum_Rev_Unsub),4)*100 AS Proportion_Sub_Rev
FROM Rev_Sub_Monthly
JOIN Rev_Unsub_Monthly
ON Rev_Sub_Monthly.Month = Rev_Unsub_Monthly.Month
GROUP BY Rev_Sub_Monthly.Month;
