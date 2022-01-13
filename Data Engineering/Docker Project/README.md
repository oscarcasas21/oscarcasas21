# Docker: Tracking User Activity

We created a service that delivers assessments, 
and now lots of different customers (e.g., Pearson) want
to publish their assessments on it. We need to get ready for data scientists
who work for these customers to run queries on the data. 

# Tasks

Prepare the infrastructure to land the data in the form and structure it needs
to be to be queried.  We will need to:

- Publish and consume messages with Kafka
- Use Spark to transform the messages. 
- Use Spark to transform the messages so that you can land them in HDFS

# Methedology is in Notebook
