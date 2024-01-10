
import numpy as np
from tkinter import *
from PIL import ImageTk, Image

from scissors.feature_extraction import Scissors


class Model:
    def __init__(self, canvas):
        self.canvas = canvas
        self.views = []

    def add_view(self, view):
        self.views.append(view)

    def update(self):
        for view in self.views:
            view.update()


class View:
    def __init__(self, model):
        self.model = model

    def update(self):