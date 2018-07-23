import pygame
import operator
import numpy as np
import math
import random
import matplotlib.pyplot as plt


class Vec2d(object):
    """2d vector class, supports vector and scalar operators,
       and also provides a bunch of high level functions
       """
    __slots__ = ['x', 'y']

    def __init__(self, x_or_pair, y = None):
        if y == None:
            self.x = x_or_pair[0]
            self.y = x_or_pair[1]
        else:
            self.x = x_or_pair
            self.y = y

    def __len__(self):
        return 2

    def __getitem__(self, key):
        if key == 0:
            return self.x
        elif key == 1:
            return self.y
        else:
            raise IndexError("Invalid subscript "+str(key)+" to Vec2d")

    def __setitem__(self, key, value):
        if key == 0:
            self.x = value
        elif key == 1:
            self.y = value
        else:
            raise IndexError("Invalid subscript "+str(key)+" to Vec2d")

    # String representaion (for debugging)
    def __repr__(self):
        return 'Vec2d(%s, %s)' % (self.x, self.y)

    # Comparison
    def __eq__(self, other):
        if hasattr(other, "__getitem__") and len(other) == 2:
            return self.x == other[0] and self.y == other[1]
        else:
            return False

    def __ne__(self, other):
        if hasattr(other, "__getitem__") and len(other) == 2:
            return self.x != other[0] or self.y != other[1]
        else:
            return True

    def __nonzero__(self):
        return bool(self.x or self.y)

    # Generic operator handlers
    def _o2(self, other, f):
        "Any two-operator operation where the left operand is a Vec2d"
        if isinstance(other, Vec2d):
            return Vec2d(f(self.x, other.x),
                         f(self.y, other.y))
        elif (hasattr(other, "__getitem__")):
            return Vec2d(f(self.x, other[0]),
                         f(self.y, other[1]))
        else:
            return Vec2d(f(self.x, other),
                         f(self.y, other))

    def _r_o2(self, other, f):
        "Any two-operator operation where the right operand is a Vec2d"
        if (hasattr(other, "__getitem__")):
            return Vec2d(f(other[0], self.x),
                         f(other[1], self.y))
        else:
            return Vec2d(f(other, self.x),
                         f(other, self.y))

    def _io(self, other, f):
        "inplace operator"
        if (hasattr(other, "__getitem__")):
            self.x = f(self.x, other[0])
            self.y = f(self.y, other[1])
        else:
            self.x = f(self.x, other)
            self.y = f(self.y, other)
        return self

    # Addition
    def __add__(self, other):
        if isinstance(other, Vec2d):
            return Vec2d(self.x + other.x, self.y + other.y)
        elif hasattr(other, "__getitem__"):
            return Vec2d(self.x + other[0], self.y + other[1])
        else:
            return Vec2d(self.x + other, self.y + other)
    __radd__ = __add__

    def __iadd__(self, other):
        if isinstance(other, Vec2d):
            self.x += other.x
            self.y += other.y
        elif hasattr(other, "__getitem__"):
            self.x += other[0]
            self.y += other[1]
        else:
            self.x += other
            self.y += other
        return self

    # Subtraction
    def __sub__(self, other):
        if isinstance(other, Vec2d):
            return Vec2d(self.x - other.x, self.y - other.y)
        elif (hasattr(other, "__getitem__")):
            return Vec2d(self.x - other[0], self.y - other[1])
        else:
            return Vec2d(self.x - other, self.y - other)
    def __rsub__(self, other):
        if isinstance(other, Vec2d):
            return Vec2d(other.x - self.x, other.y - self.y)
        if (hasattr(other, "__getitem__")):
            return Vec2d(other[0] - self.x, other[1] - self.y)
        else:
            return Vec2d(other - self.x, other - self.y)
    def __isub__(self, other):
        if isinstance(other, Vec2d):
            self.x -= other.x
            self.y -= other.y
        elif (hasattr(other, "__getitem__")):
            self.x -= other[0]
            self.y -= other[1]
        else:
            self.x -= other
            self.y -= other
        return self

    # Multiplication
    def __mul__(self, other):
        if isinstance(other, Vec2d):
            return Vec2d(self.x*other.x, self.y*other.y)
        if (hasattr(other, "__getitem__")):
            return Vec2d(self.x*other[0], self.y*other[1])
        else:
            return Vec2d(self.x*other, self.y*other)
    __rmul__ = __mul__

    def __imul__(self, other):
        if isinstance(other, Vec2d):
            self.x *= other.x
            self.y *= other.y
        elif (hasattr(other, "__getitem__")):
            self.x *= other[0]
            self.y *= other[1]
        else:
            self.x *= other
            self.y *= other
        return self

    # Division
    def __div__(self, other):
        return self._o2(other, operator.div)
    def __rdiv__(self, other):
        return self._r_o2(other, operator.div)
    def __idiv__(self, other):
        return self._io(other, operator.div)

    def __floordiv__(self, other):
        return self._o2(other, operator.floordiv)
    def __rfloordiv__(self, other):
        return self._r_o2(other, operator.floordiv)
    def __ifloordiv__(self, other):
        return self._io(other, operator.floordiv)

    def __truediv__(self, other):
        return self._o2(other, operator.truediv)
    def __rtruediv__(self, other):
        return self._r_o2(other, operator.truediv)
    def __itruediv__(self, other):
        return self._io(other, operator.floordiv)

    # Modulo
    def __mod__(self, other):
        return self._o2(other, operator.mod)
    def __rmod__(self, other):
        return self._r_o2(other, operator.mod)

    def __divmod__(self, other):
        return self._o2(other, operator.divmod)
    def __rdivmod__(self, other):
        return self._r_o2(other, operator.divmod)

    # Exponentation
    def __pow__(self, other):
        return self._o2(other, operator.pow)
    def __rpow__(self, other):
        return self._r_o2(other, operator.pow)

    # Bitwise operators
    def __lshift__(self, other):
        return self._o2(other, operator.lshift)
    def __rlshift__(self, other):
        return self._r_o2(other, operator.lshift)

    def __rshift__(self, other):
        return self._o2(other, operator.rshift)
    def __rrshift__(self, other):
        return self._r_o2(other, operator.rshift)

    def __and__(self, other):
        return self._o2(other, operator.and_)
    __rand__ = __and__

    def __or__(self, other):
        return self._o2(other, operator.or_)
    __ror__ = __or__

    def __xor__(self, other):
        return self._o2(other, operator.xor)
    __rxor__ = __xor__

    # Unary operations
    def __neg__(self):
        return Vec2d(operator.neg(self.x), operator.neg(self.y))

    def __pos__(self):
        return Vec2d(operator.pos(self.x), operator.pos(self.y))

    def __abs__(self):
        return Vec2d(abs(self.x), abs(self.y))

    def __invert__(self):
        return Vec2d(-self.x, -self.y)

    # vectory functions
    def get_length_sqrd(self):
        return self.x**2 + self.y**2

    def get_length(self):
        return math.sqrt(self.x**2 + self.y**2)
    def __setlength(self, value):
        length = self.get_length()
        self.x *= value/length
        self.y *= value/length
    length = property(get_length, __setlength, None, "gets or sets the magnitude of the vector")

    def rotate(self, angle_degrees):
        radians = math.radians(angle_degrees)
        cos = math.cos(radians)
        sin = math.sin(radians)
        x = self.x*cos - self.y*sin
        y = self.x*sin + self.y*cos
        self.x = x
        self.y = y

    def rotated(self, angle_degrees):
        radians = math.radians(angle_degrees)
        cos = math.cos(radians)
        sin = math.sin(radians)
        x = self.x*cos - self.y*sin
        y = self.x*sin + self.y*cos
        return Vec2d(x, y)

    def get_angle(self):
        if (self.get_length_sqrd() == 0):
            return 0
        return math.degrees(math.atan2(self.y, self.x))
    def __setangle(self, angle_degrees):
        self.x = self.length
        self.y = 0
        self.rotate(angle_degrees)
    angle = property(get_angle, __setangle, None, "gets or sets the angle of a vector")

    def get_angle_between(self, other):
        cross = self.x*other[1] - self.y*other[0]
        dot = self.x*other[0] + self.y*other[1]
        return math.degrees(math.atan2(cross, dot))

    def normalized(self):
        length = self.length
        if length != 0:
            return self/length
        return Vec2d(self)

    def normalize_return_length(self):
        length = self.length
        if length != 0:
            self.x /= length
            self.y /= length
        return length

    def perpendicular(self):
        return Vec2d(-self.y, self.x)

    def perpendicular_normal(self):
        length = self.length
        if length != 0:
            return Vec2d(-self.y/length, self.x/length)
        return Vec2d(self)

    def dot(self, other):
        return float(self.x*other[0] + self.y*other[1])

    def get_distance(self, other):
        return math.sqrt((self.x - other[0])**2 + (self.y - other[1])**2)

    def get_dist_sqrd(self, other):
        return (self.x - other[0])**2 + (self.y - other[1])**2

    def projection(self, other):
        other_length_sqrd = other[0]*other[0] + other[1]*other[1]
        projected_length_times_other_length = self.dot(other)
        return other*(projected_length_times_other_length/other_length_sqrd)

    def cross(self, other):
        return self.x*other[1] - self.y*other[0]

    def interpolate_to(self, other, range):
        return Vec2d(self.x + (other[0] - self.x)*range, self.y + (other[1] - self.y)*range)

    def convert_to_basis(self, x_vector, y_vector):
        return Vec2d(self.dot(x_vector)/x_vector.get_length_sqrd(), self.dot(y_vector)/y_vector.get_length_sqrd())

    def __getstate__(self):
        return [self.x, self.y]

    def __setstate__(self, dict):
        self.x, self.y = dict

SCOREBOARD_HEIGHT = 50
SCOREBOARD_BOUNDARY_WIDTH = 6

WINDOW_WIDTH = 400
WINDOW_HEIGHT = 400 + SCOREBOARD_HEIGHT + SCOREBOARD_BOUNDARY_WIDTH
PADDLE_THICKNESS = 10
PADDLE_LENGTH = 50

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

PUCK_SPEED = 50
PADDLE_SPEED = 40

# user paddle control status
MOVE_DOWN = False
MOVE_UP = False

mainloop = True
episode_done = False
paddle_2_reward = 0

RESET_PUCK = False

pygame.font.init()
myfont = pygame.font.SysFont(None, 60)

class puck:
    def __init__(self, pos_, speed_, width_ = 10, height_ = 10):
        self.pos = pos_
        self.width = width_
        self.height = height_
        self.speed = speed_
        self.force = 0

    def show(self, screen_):
        puck = pygame.Rect(self.pos.x, self.pos.y, self.height, self.width)
        pygame.draw.rect(screen_, WHITE, puck)

    def addForce(self, force):
        self.pos += force * self.speed

    def update(self):
        self.addForce(self.force)

    def bouncing_edge(self, paddle_1_, paddle_2_):
        global RESET_PUCK
        global episode_done
        global paddle_2_reward
        if self.pos.x > WINDOW_WIDTH - self.width:
            del self
            paddle_1_.score += 1
            paddle_2_reward = -1
            if paddle_1_.score == 21:
                paddle_1_.score = 0
                paddle_2_.score = 0
                episode_done = True
            RESET_PUCK = True
            return
            # self.dir.x *= -1
            # self.pos.x = WINDOW_WIDTH - self.width
        if self.pos.x < 0:
            del self
            paddle_2_.score += 1
            paddle_2_reward = 1
            if paddle_2_.score == 21:
                paddle_1_.score = 0
                paddle_2_.score = 0
                episode_done = True
            RESET_PUCK = True
            return
            # self.dir.x *= -1
            # self.pos.x = 0



        if self.pos.y > WINDOW_HEIGHT - self.height:
            self.force.y *= -1
            self.pos.y = WINDOW_HEIGHT - self.height
        if self.pos.y < 0 + SCOREBOARD_HEIGHT:
            self.force.y *= -1
            self.pos.y = 0 + SCOREBOARD_HEIGHT

    def get_pos(self):
        return self.pos

    def bouncing_paddle(self, paddle_):
        if self.pos.y + self.height > paddle_.posY and self.pos.y < paddle_.posY + paddle_.get_length():
            # left paddle
            if paddle_.get_ID() == 1:
                 if self.pos.x < paddle_.get_posX() + paddle_.get_thickness():
                    bouncing_angle = (((self.pos.y + self.height - paddle_.posY) / (paddle_.length + self.height)) * math.pi / 2) - math.pi / 4
                    # print(bouncing_angle)
                    self.force = Vec2d(math.cos(bouncing_angle), math.sin(bouncing_angle))
                    self.pos.x = 0 + paddle_.get_posX() + paddle_.get_thickness()
            # right paddle
            if paddle_.get_ID() == 2:
                if self.pos.x + self.width > paddle_.get_posX():
                    bouncing_angle = (((self.pos.y + self.height - paddle_.posY) / (paddle_.length + self.height)) * math.pi / 2) - math.pi / 4
                    # print(bouncing_angle)
                    self.force = Vec2d(math.cos(bouncing_angle), math.sin(bouncing_angle))
                    self.force.x *= -1
                    self.pos.x = paddle_.get_posX() - self.width

class paddle:
    def __init__(self, ID_, posY_ = WINDOW_HEIGHT/2, length_ = PADDLE_LENGTH, thickness_ = PADDLE_THICKNESS):
        self.posY = posY_
        self.length = length_
        self.thickness = thickness_
        self.ID = ID_
        self.score = 0
        if self.ID is 1:
            self.posX = 0
        if self.ID is 2:
            self.posX = WINDOW_WIDTH - thickness_

    def show(self, screen_):
        global myfont
        self.score_text = myfont.render(str(self.score), False, WHITE)
        paddle = pygame.Rect(self.posX, self.posY, self.thickness, self.length)
        pygame.draw.rect(screen_, WHITE, paddle)
        if self.ID == 1:
            screen_.blit(self.score_text, (WINDOW_WIDTH / 4 - 15, SCOREBOARD_HEIGHT / 4))
        if self.ID == 2:
            screen_.blit(self.score_text, (WINDOW_WIDTH*3 / 4, SCOREBOARD_HEIGHT / 4))


    def input_action(self, action_):
        self.posY += action_ * PADDLE_SPEED

    def edge(self):
        if self.posY < 0 + SCOREBOARD_HEIGHT:
            self.posY = 0 + SCOREBOARD_HEIGHT
        if self.posY > WINDOW_HEIGHT - self.length:
            self.posY = WINDOW_HEIGHT - self.length

    def get_length(self):
        return self.length

    def get_ID(self):
        return self.ID

    def get_posX(self):
        return self.posX

    def get_thickness(self):
        return self.thickness

class pong:
    def __init__(self):
        # RESET_PUCK = False

        self.screen = pygame.display.set_mode([WINDOW_WIDTH, WINDOW_HEIGHT])

        self.my_puck = self.reset_puck()

        self.paddle_1 = paddle(ID_=1)
        self.paddle_2 = paddle(ID_=2)

    def paddle_2_control(self, action_):
        global episode_done
        global paddle_2_reward
        self.paddle_2.input_action(action_)
        return episode_done, paddle_2_reward


    def naive_AI_control(self, paddle, target):
        # offset = random.uniform(0, paddle.length)
        direction = target - (paddle.posY + paddle.length / 2)
        if direction > 25:
            paddle.input_action(1)
        if direction < -25:
            paddle.input_action(-1)
        else:
            paddle.input_action(0)

    def reset_puck(self):
        angle = random.uniform(-math.pi / 8, math.pi / 8)

        puck_init_force_dir = Vec2d(math.cos(angle), math.sin(angle))

        random_num = random.uniform(-1, 1)

        if random_num > 0:
            direction = 1
        else:
            direction = -1

        puck_force = puck_init_force_dir.normalized() * direction

        my_puck_init_pos = Vec2d(WINDOW_WIDTH / 2, (WINDOW_HEIGHT - (SCOREBOARD_HEIGHT + SCOREBOARD_BOUNDARY_WIDTH)) / 2 + (SCOREBOARD_HEIGHT + SCOREBOARD_BOUNDARY_WIDTH))
        my_puck = puck(my_puck_init_pos, PUCK_SPEED)
        my_puck.force = puck_force

        return my_puck

    def get_next_frame(self):
        global RESET_PUCK
        global episode_done
        global paddle_2_reward

        if episode_done == True:
            episode_done = False

        if RESET_PUCK:
            paddle_2_reward = 0
            self.my_puck = self.reset_puck()
            RESET_PUCK = False

        self.screen.fill(BLACK)

        # paddle_keyboard_control(self.paddle_2)


        pygame.draw.line(self.screen, WHITE, (0, SCOREBOARD_HEIGHT), (WINDOW_WIDTH, SCOREBOARD_HEIGHT), SCOREBOARD_BOUNDARY_WIDTH)

        self.my_puck.bouncing_edge(self.paddle_1, self.paddle_2)
        self.my_puck.bouncing_paddle(self.paddle_1)
        self.my_puck.bouncing_paddle(self.paddle_2)
        self.my_puck.update()
        self.my_puck.show(self.screen)

        self.naive_AI_control(self.paddle_1, self.my_puck.get_pos().y)

        self.paddle_1.edge()
        self.paddle_2.edge()

        self.paddle_1.show(self.screen)
        self.paddle_2.show(self.screen)

        frame_data = pygame.surfarray.array2d(pygame.display.get_surface())

        # plt.imshow(image_data[:, SCOREBOARD_HEIGHT + 3 :WINDOW_HEIGHT].T)
        # plt.show()

        pygame.display.update()

        return frame_data

def main():
    global mainloop
    global MOVE_UP
    global MOVE_DOWN
    global RESET_PUCK

    my_pong = pong()

    while mainloop:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                mainloop = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    mainloop = False  # user pressed ESC
                if event.key == pygame.K_UP:
                    MOVE_UP = True
                if event.key == pygame.K_DOWN:
                    MOVE_DOWN = True
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_UP:
                    MOVE_UP = False
                if event.key == pygame.K_DOWN:
                    MOVE_DOWN = False

        paddle_keyboard_control(my_pong)

        my_pong.get_next_frame()

        # clock = pygame.time.Clock()
        # clock.tick(100)
        # print(clock.get_fps())



def paddle_keyboard_control(pong_game_):
    if MOVE_UP:
        pong_game_.paddle_2_control(-1)
    if MOVE_DOWN:
        pong_game_.paddle_2_control(1)
    else:
        pong_game_.paddle_2_control(0)

if __name__ == "__main__":
    main()

