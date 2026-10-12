"""Decode static keyword configuration for GPU buffer callbacks."""

import pickle


def load_kwargs(kwargs_dump, /):
    """Decode the host-resident static keywords passed to a GPU callback."""
    return pickle.loads(bytes(kwargs_dump))
