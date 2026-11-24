"""
This module contains the message definitions for the NexGen system.
It'll be used to send fake and real data along the network.

With my current understanding it'll be just housing all the tools for managing message lifetimes
and pointer to the actual data being passed, synthetic or real based on the training data.
"""

"""
class Message:
    data: pointer to the actual data being passed
    lifetime: how long the message is valid for, in seconds 
    # Using perplexity its said that for safety critical systems the
    # lifetime of a message should be:
    # "maximum permitted time between observing something and using that observation."
    # d_acc = tolerable_error / max(acceleration)
    timestamp: when the message was created, in seconds since epoch

    constructor
    expire

"""