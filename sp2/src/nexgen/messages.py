"""
This module contains the message definitions for the NexGen system.
It'll be used to wrap synthetic and real data passed along the network.

"""
from dataclasses import dataclass


@dataclass
class Message:
    messageId: int
    payload: bytearray
    # source: Device
    source: str
    destination: str # for both source and destination, expecting the device ids
    timestamp: str # Time of creation
    status: str = "PENDING" # PENDING | DELIVERED | DROPPED | EXPIRED
    # Pending absorbs state of a message traveling or waiting to be sent
    lifetime: float = 0

    def set_pending(self) -> None:
        self.status = "PENDING"

    def set_delivered(self) -> None:
        self.status = "DELIVERED"

    def set_dropped(self) -> None:
        self.status = "DROPPED"

    def expire(self) -> None:
        self.status = "EXPIRED"

    def __str__(self) -> str:
        deadline = self.lifetime + self.timestamp
        # Lua has something like this didnt know python could do it
        pay_size = len(self.payload) if self.payload is not None else 0         
        return (
            f"Message ID: {self.messageId} | Source: {self.source} --- Destination: {self.destination} | Status: {self.status}"
            f"Designated lifetime: {self.lifetime} | Time created (Simulated): {self.timestamp} | Deadline: {deadline}"
            f"Message payload size: {pay_size}"
        )


class MessageHistory:
    # a list of all of the messages ever generated.
    # Depending on how python handles lists I can see this becoming a big issue over time
    # Memory management, a linked list kind of structure would be better for this I'd assume.
    log_buffer: list

    def __init__(self):
        self.log_buffer = []