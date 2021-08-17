import math
from rl_car_simulator.network import Network
import numpy as np
import cv2
import copy

WALL = (0,0,0)
WINDSHIELD = (0,0,0)
SPAWN = (150,0,150)
GOAL = (0,255,0)
GOAL_APPROACH = (0,0,0)
LIDAR = (255,0,255)

KEYBOARD = (255,0,0)
KEYBOARD_GOAL = (0,215,255)

NETWORK = (0,0,255)
NETWORKEXPLORATION = (0,0,155)

FEEDBACK = (0,255,0)
FEEDBACKEXPLORATION = (0,155,0)

RANDOM = (100,100,155)

class Graphics:
    def __init__(self, settings, world, reporting):
        self.settings = settings
        self.world = world
        self.reporting

        w = int(self.settings.world.size_x * self.settings.graphics.pixels_per_m)
        h = int(self.settings.world.size_y * self.settings.graphics.pixels_per_m)
        base_world_frame = np.full((h,w,3), (255,255,255), dtype=np.uint8)
        self.base_world_frame = self.fill_base_frame(base_world_frame)

    def close(self):
        cv2.destroyAllWindows()
    
    def fill_base_frame(self, frame):
        for wall in self.world.walls:
            self.draw_wall(frame, wall)
        for goal in self.settings.world.goal_points:
            self.draw_goal(frame, goal)
        for pt in self.settings.world.spawn_points:
            self.draw_spawn_pt(frame, pt)
        return frame

    def draw_wall(self, frame, wall):
        p1 = (wall.x1 * self.settings.graphics.pixels_per_m).astype(np.int32)
        p1 = tuple(p1.tolist())
        p2 = (wall.x2 * self.settings.graphics.pixels_per_m).astype(np.int32)
        p2 = tuple(p2.tolist())
        cv2.line(frame, p1, p2, color=WALL, thickness=4)

    def draw_goal(self, frame, goal):
        goal = tuple([int(g* self.settings.graphics.pixels_per_m) for g in goal])
        cv2.circle(frame, goal, color=GOAL, radius=int(1 * self.settings.graphics.pixels_per_m), thickness=-1)

    def draw_spawn_pt(self, frame, pt):
        pt = tuple([int(g* self.settings.graphics.pixels_per_m) for g in pt])
        cv2.circle(frame, pt, color=SPAWN, radius=int(0.5 * self.settings.graphics.pixels_per_m), thickness=-1)


    def draw_autonomous_car(self, base_frame, car, color):
        self.draw_car(base_frame, car, color)

        if self.settings.graphics.draw_goals:
            curr = tuple([int(c* self.settings.graphics.pixels_per_m) for c in [car.state.x, car.state.y]])
            goal = tuple([int(g* self.settings.graphics.pixels_per_m) for g in car.goal])
            cv2.line(base_frame, curr, goal, color=GOAL_APPROACH, thickness=1)

        if car.sensed_state is not None and self.settings.graphics.draw_lidar:
            for ang, dist in zip(self.settings.car_properties.lidar_angles, car.lidar_state):
                ang = ang + car.state.h
                x = curr[0] + int(dist * math.cos(ang) * self.settings.graphics.pixels_per_m)
                y = curr[1] + int(dist * math.sin(ang) * self.settings.graphics.pixels_per_m)
                cv2.line(base_frame, curr, (x,y), color=LIDAR, thickness=1)
                cv2.circle(base_frame, (x,y), color=LIDAR, radius=5, thickness=-1)

    def create_current_frame(self, base_frame):

        for car in self.world.keyboard_cars:
            self.draw_car(base_frame, car, KEYBOARD)

            goal = tuple([int(g* self.settings.graphics.pixels_per_m) for g in car.goal])
            cv2.circle(base_frame, goal, color=KEYBOARD_GOAL, radius=10, thickness=-1)

        for car in self.world.network_cars:
            self.draw_autonomous_car(base_frame, car, NETWORK)
        for car in self.world.network_exploration_cars:
            self.draw_autonomous_car(base_frame, car, NETWORKEXPLORATION)
        for car in self.world.random_cars:
            self.draw_autonomous_car(base_frame, car, RANDOM)
        for car in self.world.feedback_cars:
            self.draw_autonomous_car(base_frame, car, FEEDBACK)
        for car in self.world.feedback_exploration_cars:
            self.draw_autonomous_car(base_frame, car, FEEDBACKEXPLORATION)

        return base_frame

    def draw_car(self, frame, car, color):
        # Car Body
        corners = car.get_corners()
        #corners = np.concatenate(corners, axis=1).T
        #corners = (corners * self.settings.graphics.pixels_per_m).astype(np.int32)
        use_corners = []
        for corner in corners:
            #print(corner)
            use_corners.append((corner * self.settings.graphics.pixels_per_m).astype(np.int32))
        cv2.fillPoly(frame, [np.array(use_corners)], color)

        # Windshield
        corners = car.get_windshield_corners()
        corners = np.concatenate(corners, axis=1).T
        corners = (corners * self.settings.graphics.pixels_per_m).astype(np.int32)
        cv2.fillPoly(frame, [corners], WINDSHIELD)

    def show_current_world(self):
        frame = self.create_current_frame(copy.copy(self.base_world_frame))
        cv2.imshow('test', frame)

    def sleep(self):
        cv2.waitKey(self.settings.graphics.show_ms)
