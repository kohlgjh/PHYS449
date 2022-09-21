import numpy as np
from shapes import ShapesBase

class Rectangle(ShapesBase):

    def __init__(self, a, b):
        super.__init__()

        self.a = a
        self.b = b

    def perimeter(self):
        '''Return perimeter of rectangle with sidelengths a, b'''
        return 2 * (self.a + self.b)

    def area(self):
        '''Return area of rectangle with sidelengths a, b'''
        return self.a * self.b