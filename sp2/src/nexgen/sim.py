"""
This module runs the discrete event simulation for NexGen. 
It manages the simulation clock, device queues, packet transmissions, arrivals, and deadlines. 
It uses the current topology to check which links are available 
and records packet events for later analysis.

message logs are handled using tools written in the messages.py module
"""