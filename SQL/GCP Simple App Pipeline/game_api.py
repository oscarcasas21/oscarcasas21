#!/usr/bin/env python
import json
from kafka import KafkaProducer
from flask import Flask, request

app = Flask(__name__)
producer = KafkaProducer(bootstrap_servers='kafka:29092')


def log_to_kafka(topic, event):
    event.update(request.headers)
    producer.send(topic, json.dumps(event).encode())


@app.route("/")
def default_response():
    default_event = {'event_type': 'default'}
    log_to_kafka('events', default_event)
    return "This is the default response!\n"

@app.route("/purchase_sword/<type>")
def purchase_sword(type):
    purchase_sword_event = {'event_type': 'purchase_sword','sword_type':type}
    log_to_kafka('events', purchase_sword_event)
    return "Sword Purchased! "+ type +"\n"

@app.route("/join_guild/<type>")
def join_guild(type):
    join_guild_event = {'event_type': 'join_guild','guild_name':type}
    log_to_kafka('events', join_guild_event)
    return "Guild Joined! "+ type +"\n"

@app.route("/login", methods=['POST'])
def login():
    id = request.args.get('id',default=0,type=int)
    login_event = {'event_type': 'login','id':id}
    log_to_kafka('events', login_event)
    return "User logged in = "+str(id)+"\n"

@app.route("/logout", methods=['POST'])
def logout():
    id = request.args.get('id',default=0,type=int)
    logout_event = {'event_type': 'logout','id':id}
    log_to_kafka('events', logout_event)
    return "User logged out = "+str(id)+"\n"