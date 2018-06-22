from __future__ import division

from math import atan2, cos, pi, sin, sqrt, tan

import numpy as np

from constants import Config, Globals


def normalize(v):
    norm = sqrt(np.dot(v, v))
    if norm == 0:
        return v
    return v / norm


def findStartAngle(x0, y0, x1, y1):
    return atan2(y1 - y0, x1 - x0) * 180 / pi


class Line:
    def __init__(self, p1=None, p2=None, m=None, b=None, normal_vector=None):
        self.p1 = p1
        self.p2 = p2
        self.m = m
        self.b = b
        self.normal_vector = normal_vector

    def astuple(self):
        return self.p1 + self.p2


class Bacterium:
    def __init__(self):
        # Dimensions
        self.length = Config.init_length
        self.width = Config.init_width
        self.radius = int(self.width/2)
        self.theta = 0  # for rotation
        self.bend =0

        # Structures
        self.line_1 = Line()
        self.line_2 = Line()
        self.line_3 = Line()
        self.line_4 = Line()

        # Coordinates
        self.pos = np.array([Globals.image_width/2, Globals.image_height/2, 0])

        self.head_pos = np.zeros(3)
        self.tail_pos = np.zeros(3)
        
        # Characteristics
        self.name = ''
        self.collided = False
        self.w = np.zeros(3)
        self.v = np.zeros(3)
        self.k = 10
        self.m = 1

    def update(self):
        length = self.length
        pos = self.pos
        theta = self.theta
        radius = self.radius
        m = tan(theta)
        width = self.width
        #there are 2 scenarios for bend: |>  vs <|
        #if we just had the bend angle be less than 180, then the two above would be considered the same
        if self.bend>pi:
            bend = 2*pi - self.bend
            # update head                                                                                     
            x = pos[0] - ((length-width)/2)*cos(pi - (pi/2+bend/2-theta))
            y = pos[1] - ((length-width)/2)*sin(pi - (pi/2+bend/2-theta))
            self.head_pos = np.array([x, y, 0])

            self.end_point_1 = np.array([x + radius*cos(pi - (pi/2+bend/2-theta) - pi/2),
                                         y + radius*sin(pi - (pi/2+bend/2-theta) - pi/2), 0])
            self.end_point_2 = np.array([x - radius*cos(pi - (pi/2+bend/2-theta) - pi/2),
                                         y - radius*sin(pi - (pi/2+bend/2-theta) - pi/2), 0])

            #update middle circle
            self.end_point_5 = np.array([pos[0] + radius*cos(pi - (pi/2+bend/2-theta) - pi/2), #end pt 1 connects to 5
                                         pos[1] + radius*sin(pi - (pi/2+bend/2-theta) - pi/2), 0])
            self.end_point_6 = np.array([pos[0] - radius*cos(pi - (pi/2+bend/2-theta) - pi/2), #end pt 2 connects to 6
                                         pos[1] - radius*sin(pi - (pi/2+bend/2-theta) - pi/2), 0])
            self.end_point_7 = np.array([pos[0] - radius*cos(pi - (3*pi/2 - bend/2 - theta) - pi/2), #end pt 7 connects to 3
                                         pos[1] - radius*sin(pi - (3*pi/2 - bend/2 - theta) - pi/2), 0])
            self.end_point_8 = np.array([pos[0] + radius*cos(pi - (3*pi/2 - bend/2 - theta) - pi/2), #end pt 8 connects to 4
                                         pos[1] + radius*sin(pi - (3*pi/2 - bend/2 - theta) - pi/2), 0])
            

            # update tail
            x = pos[0] + ((length-width)/2)*cos(pi - (3*pi/2 - bend/2 - theta))
            y = pos[1] + ((length-width)/2)*sin(pi - (3*pi/2 - bend/2 - theta))
            self.tail_pos = np.array([x, y, 0])

            self.end_point_3 = np.array([x - radius*cos(pi - (3*pi/2 - bend/2 - theta) - pi/2),
                                         y - radius*sin(pi - (3*pi/2 - bend/2 - theta) - pi/2), 0])
            self.end_point_4 = np.array([x + radius*cos(pi - (3*pi/2 - bend/2 - theta) - pi/2),
                                         y + radius*sin(pi - (3*pi/2 - bend/2 - theta) - pi/2), 0])

            # update top half
            normal = normalize(self.end_point_1 - self.end_point_2)

            
            
            # body line 1
            m = tan(pi - (pi/2+bend/2-theta))
            b = self.end_point_1[1] - m*self.end_point_1[0]
            self.line_1 = Line(self.end_point_1, self.end_point_5, m, b, normal)
                
            # body line 2
            b = self.end_point_2[1] - m*self.end_point_2[0]
            self.line_2 = Line(self.end_point_2, self.end_point_6, m, b, -normal)

            #update bottom half
            normal = normalize(self.end_point_3 - self.end_point_4)

            #body line 3
            m= tan(pi - (3*pi/2 - bend/2 - theta))
            b = self.end_point_3[1] - m*self.end_point_3[0]
            self.line_1 = Line(self.end_point_3, self.end_point_7, m, b, normal)

            # body line 2
            b = self.end_point_4[1] - m*self.end_point_4[0]
            self.line_2 = Line(self.end_point_4, self.end_point_8, m, b, -normal)

        else:
            bend = self.bend
            theta = pi - self.theta
        
            # update head                                                                                     
            x = pos[0] - ((length-width)/2)*cos(pi/2+bend/2-theta)
            y = pos[1] - ((length-width)/2)*sin(pi/2+bend/2-theta)
            self.head_pos = np.array([x, y, 0])

            self.end_point_1 = np.array([x + radius*cos(pi/2+bend/2-theta - pi/2),
                                         y + radius*sin(pi/2+bend/2-theta - pi/2), 0])
            self.end_point_2 = np.array([x - radius*cos(pi/2+bend/2-theta - pi/2),
                                         y - radius*sin(pi/2+bend/2-theta - pi/2), 0])

            #update middle circle
            self.end_point_5 = np.array([pos[0] + radius*cos(pi/2+bend/2-theta - pi/2), #end pt 1 connects to 5
                                         pos[1] + radius*sin(pi/2+bend/2-theta - pi/2), 0])
            self.end_point_6 = np.array([pos[0] - radius*cos(pi/2+bend/2-theta - pi/2), #end pt 2 connects to 6
                                         pos[1] - radius*sin(pi/2+bend/2-theta - pi/2), 0])
            self.end_point_7 = np.array([pos[0] - radius*cos(3*pi/2 - bend/2 - theta - pi/2), #end pt 7 connects to 3
                                         pos[1] - radius*sin(3*pi/2 - bend/2 - theta - pi/2), 0])
            self.end_point_8 = np.array([pos[0] + radius*cos(3*pi/2 - bend/2 - theta - pi/2), #end pt 8 connects to 4
                                         pos[1] + radius*sin(3*pi/2 - bend/2 - theta - pi/2), 0])
            

            # update tail
            x = pos[0] + ((length-width)/2)*cos(3*pi/2 - bend/2 - theta)
            y = pos[1] + ((length-width)/2)*sin(3*pi/2 - bend/2 - theta)
            self.tail_pos = np.array([x, y, 0])

            self.end_point_3 = np.array([x - radius*cos(3*pi/2 - bend/2 - theta - pi/2),
                                         y - radius*sin(3*pi/2 - bend/2 - theta - pi/2), 0])
            self.end_point_4 = np.array([x + radius*cos(3*pi/2 - bend/2 - theta - pi/2),
                                         y + radius*sin(3*pi/2 - bend/2 - theta - pi/2), 0])

            # update top half
            normal = normalize(self.end_point_1 - self.end_point_2)

            
            
            # body line 1
            m = tan(pi/2+bend/2-theta)
            b = self.end_point_1[1] - m*self.end_point_1[0]
            self.line_1 = Line(self.end_point_1, self.end_point_5, m, b, normal)

            # body line 2
            b = self.end_point_2[1] - m*self.end_point_2[0]
            self.line_2 = Line(self.end_point_2, self.end_point_6, m, b, -normal)

            #update bottom half
            normal = normalize(self.end_point_3 - self.end_point_4)

            #body line 3
            m= tan(3*pi/2 - bend/2 - theta)
            b = self.end_point_3[1] - m*self.end_point_3[0]
            self.line_3 = Line(self.end_point_3, self.end_point_7, m, b, normal)

            # body line 2
            b = self.end_point_4[1] - m*self.end_point_4[0]
            self.line_4 = Line(self.end_point_4, self.end_point_8, m, b, -normal)
