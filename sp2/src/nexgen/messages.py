"""
This module contains the message definitions for the NexGen system.
It'll be used to wrap fake and real data passed along the network.

"""

"""
class Message:
    data: pointer to the actual data being passed
    source: the device that sent the message
    destination: the device that will receive the message
    lifetime: how long the message is valid for, in seconds 
    # Using perplexity its said that for safety critical systems the
    # lifetime of a message should be:
    # "maximum permitted time between observing something and using that observation."
    # d_acc = tolerable_error / max(acceleration)
    timestamp: when the message was created, in seconds since epoch
    simulation_creation_time
    status

    constructor
    expire
    __str__() - To make message history array life easier


class MessageHistory:
    buffer of "infinite size" to log all of the messages as a string
    string format will be a timestamp - delivery time (all Xs if not delivered maybe) - Message Type - Message if applicable (if applicable because I cant put entire subset of a video here)    

"""