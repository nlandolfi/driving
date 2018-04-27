import json
import math

import numpy as np

import car
import dynamics
import feature
import neural
import world

theta_normal     = [1., -50., 10., 100., 10., -50.]

# dynamics {{{

# DYNAMICS_NORMAL indicates the regular car dynamics
# used in the journal paper.
DYNAMICS_NORMAL = "normal"

# dynamics_from constructs a dynamics object from a
# declarative definition. e.g.,
#
#   def = { "kind": DYNAMICS_NORMAL, "params": { "dt": 0.1, }, }
#   dyn = dynamics_from(def)
def dynamics_from(definition):
    if "kind" not in definition:
        raise Exception("dynamics_from: dynamics definition must include 'kind'")

    if definition["kind"] == DYNAMICS_NORMAL:
        dt = 0.1

        if "params" in definition and "dt" in definition["params"]:
            dt = definition["params"]["dt"]

        return dynamics.CarDynamics(dt=definition["params"]["dt"])

    raise Exception("dynamics_from: unknown dynamics kind: " + definition["kind"])

# }}}

# world {{{

# WORLD_CUSTOM indicates that the world definition
# will be explicitly constructed. Other world
# definitions are pre-made worlds.
WORLD_CUSTOM = "custom"

# WORLD_HIGHWAY indicates the classic three-lane highway
# from the journal paper
WORLD_HIGHWAY = "highway"
WORLD_HIGHWAY_EXIT = "highway-exit"

# world_from creates a world object from a declarative
# definition. e.g.,
#
#  def = { "kind": WORLD_HIGHWAY }
#  world = world_from(def)
def world_from(definition):
    if "kind" not in definition:
        raise Exception("world_from: world definition must include 'kind'")

    if definition["kind"] == WORLD_HIGHWAY:
        return world.highway()

    if definition["kind"] == WORLD_HIGHWAY_EXIT:
        return world.highwayexit()

    if definition["kind"] == WORLD_CUSOM:
        # TODO(nlandolfi): WORLD_CUSTOM
        raise NotImplementedError("world_from: kind 'custom' not implemented")

    raise Exception("world_from: unknown world kind: " + definition["kind"])

# }}}

# COLOR_* are different car colors
COLOR_RED    = "red"
COLOR_BLUE    = "blue"
COLOR_YELLOW = "yellow"
COLOR_ORANGE = "orange"
COLOR_GRAY   = "gray"
COLOR_LIGHT_GRAY   = "lightgray"
COLOR_WHITE  = "white"
COLOR_FIRE   = "fire"

# CAR_* are different car kinds
CAR_SIMPLE = "simple"
CAR_USER   = "user"
CAR_NESTED = "nested"
CAR_BELIEF = "belief"
CAR_CANNED = "canned"
CAR_NEURAL = "neural"
CAR_COPY   = "copy"
CAR_COHESIVE = "cohesive"

# Rewards are various custom reward functions
REWARD_MIDDLE_LANE = "r_middle_lane"
REWARD_LEFT_LANE   = "r_left_lane"
REWARD_RIGHT_LANE  = "r_right_lane"
REWARD_RIGHT_HALF  = "r_right_half"
REWARD_LEFT_HALF   = "r_left_half"

REWARD_FUNCTIONS = {
    REWARD_MIDDLE_LANE: feature.Feature(lambda t, x, u: -(x[0])**2),
    REWARD_LEFT_LANE:   feature.Feature(lambda t, x, u: np.exp((-0.5*(x[0]+0.13)**2)/.04)),
    REWARD_RIGHT_LANE:  feature.Feature(lambda t, x, u: -(x[0]-0.13)**2),
    REWARD_RIGHT_HALF:  feature.Feature(lambda t, x, u: x[0]),
    REWARD_LEFT_HALF:   feature.Feature(lambda t, x, u: -x[0]),
}

# car_from {{{

# car_from constructs a car from a car declarative
# definition. e.g.,
#
#   def = {
#       "kind": CAR_SIMPLE,
#       "x0": [-.13, 0.0, math.pi/2., 0.5],
#       "color": COLOR_GRAY
#       "reward": theta_normal,
#       "T": 5,
#   }
#   car = car_from(def)
def car_from(dyn, definition):
    if "kind" not in definition:
        raise Exception("car definition must include 'kind'")

    if "x0" not in definition:
        raise Exception("car definition must include 'x0'")

    if "color" not in definition:
        raise Exception("car definition must include 'color'")

    if "T" not in definition:
        raise Exception("car definition must include 'T'")

    if definition["kind"] == CAR_SIMPLE:
        return car.SimpleOptimizerCar(
                dyn, definition["x0"],
                color=definition["color"], T=definition["T"])

    if definition["kind"] == CAR_USER:
        return car.UserControlledCar(
                dyn, definition["x0"],
                color=definition["color"], T=definition["T"])

    if definition["kind"] == CAR_NESTED:
        return car.NestedOptimizerCar(
                dyn, definition["x0"],
                color=definition["color"], T=definition["T"])

    if definition["kind"] == CAR_BELIEF:
        return car.BeliefOptimizerCar(
                dyn, definition["x0"],
                color=definition["color"], T=definition["T"])

    if definition["kind"] == CAR_CANNED:
        c = car.CannedCar(
                dyn, definition["x0"],
                color=definition["color"], T=definition["T"])

        if "controls" not in definition:
            raise Exception("definition doesn't contain 'controls' key")

        c.follow(definition["controls"])

        return c

    if definition["kind"] == CAR_NEURAL:
        c = car.NeuralCar(
                dyn, definition["x0"],
                color=definition["color"], T=definition["T"])

        if "model" not in definition:
            raise Exception("definition doesn't contain 'model' key")

        mu = None
        if "mu" in definition:
            mu = definition["mu"]

        c.use(neural.load(definition["model"]), mu=mu)

        return c

    if definition["kind"] == CAR_COPY:
        c = car.CopyCar(
                dyn, definition["x0"],
                color=definition["color"], T=definition["T"])
        return c

    if definition["kind"] == CAR_COHESIVE:
        c = car.CohesiveCar(
                dyn, definition["x0"],
                color=definition["color"], T=definition["T"])
        return c

    raise Exception("car kind not recognized " + definition["kind"])

# }}}

# cars_from {{{

# attach_cars_from constructs a list of cars from
# declarative definitions, and adds them to the world.
#
# In particular, this function also handles setting
# the reward functions for the particular cars.
#
# The dyn dependency is for constructing the cars,
# the world is for the reward features.
def attach_cars_from(dyn, world, deflist):
    cars = []

    for definition in deflist:
        cars.append(car_from(dyn, definition))

    # set world cars here, BEFORE consructing reward functions
    world.cars = cars

    # can assume "kind" is in definition because
    # car_from worked for each of these definitions
    for (definition, car)  in zip(deflist, cars):
        if definition["kind"] == CAR_USER:
            # this car kind requires no initialization, it will
            # be controlled by a user.
            continue

        if (definition["kind"] == CAR_SIMPLE
                or definition["kind"] == CAR_NESTED
                or definition["kind"] == CAR_COPY
                or definition["kind"] == CAR_COHESIVE
                or (definition["kind"] == CAR_NEURAL
                    and "mu" in definition
                    and definition["mu"] < 1.0)):

            if "theta" not in definition:
                raise Exception(definition["kind"] + " car definition must include 'theta'")

            exclude = []
            if "exclude" in definition:
                exclude = [cars[i] for i in definition["exclude"]]
            exclude.append(car)

            r = world.features(definition["theta"], exclude, 'linear')

            if "extra" in definition:
                for extra in definition["extra"]:
                    # kind specifies one of several preset reward functions
                    # listed above, as constants in REWARD_FUNCTIONS dict
                    if "kind" not in extra:
                        raise Exception("extra reward must have 'kind'")

                    # gain specifies the coefficent multiplying the reward
                    if "gain" not in extra:
                        raise Exception("extra reward must have 'gain'")

                    # kind should be recognized in the known reward functions
                    # dictionary
                    if extra["kind"] not in REWARD_FUNCTIONS:
                        raise Exception("extra reward function kind not recognized" + k)

                    r = r + extra["gain"]*REWARD_FUNCTIONS[extra["kind"]]

            car.reward = r

            if definition["kind"] == CAR_SIMPLE:
                continue

            if definition["kind"] == CAR_COHESIVE:
                continue

        if definition["kind"] == CAR_NESTED:
            if "reward_h" not in definition:
                raise Exception("nested car definition must include 'reward_h'")

            reward_h = definition["reward_h"]

            if "index" not in reward_h:
                raise Exception("nested car 'reward_h' definition must include 'index'")

            car_h = cars[reward_h["index"]]

            if "theta" not in reward_h:
                raise Exception("nested car 'reward_h' must include 'theta'")

            r_h = world.features(reward_h["theta"], car_h, 'linear')

            if "extra" in reward_h:
                for extra in definition["extra"]:
                    if "kind" not in extra:
                        raise Exception("extra reward must have 'kind'")

                    if "gain" not in extra:
                        raise Exception("extra reward must have 'gain'")

                    if extra["kind"] not in REWARD_FUNCTIONS:
                        raise Exception("extra reward function kind not recognized" + k)

                    r_h = r_h + extra["gain"]*REWARD_FUNCTIONS[extra["kind"]]

            car.rewards = (r_h, car.reward)
            car.human = car_h
            continue


        if definition["kind"] == CAR_CANNED:
            continue

        if definition["kind"] == CAR_NEURAL:
            continue

        if definition["kind"] == CAR_COPY:
            continue

        raise Exception("car reward not implemented for kind " + definition["kind"])

    return cars

# }}}

# definition extensions {{{

# merge merges the contents of def2 into
# def1, creating a copy to do so.
#
# use this when you want to create a definition
# that extends def1 but includes aspects of,
# for example cars, from def2
def merge(def1, def2):
    copy = def1.copy()
    if "dynamics" in def2:
        copy["dynamics"] = def2["dynamics"]
    if "world" in def2:
        copy["world"] = def2["world"]
    if "cars" in def2:
        if "cars" in copy:
            copy["cars"] = copy["cars"] + def2["cars"]
        else:
            copy["cars"] = def2["cars"]
    if "main_car" in def2:
        copy["main_car"] = def2["main_car"]
    return copy

# expand checks for the "extends" key, and merges
# the definitions recursively.
#
# use on a simple definition to ensure that it
# inherits the aspects of the definitions it
# "extends"
def expand(definition):
    if "extends" not in definition:
        return definition

    extends = definition["extends"]

    if type(extends) is str:
        other = load(extends)
    if type(extends) is dict:
        other = extends

    return merge(expand(other), definition)

# }}}

# use env_from to construct a world from a
# declarative definition.
def env_from(definition):
    definition = expand(definition)

    if "dynamics" not in definition:
        raise Exception("definition must include 'dynamics'")

    if "world" not in definition:
        raise Exception("definition must include 'world'")

    if "cars" not in definition or len(definition["cars"]) == 0:
        raise Exception("definition must include 'cars'")

    dyn = dynamics_from(definition["dynamics"])
    world = world_from(definition["world"])
    attach_cars_from(dyn, world, definition["cars"])

    if "main_car" in definition:
        world.main_car = definition["main_car"]

    return world

# Marshaling {{{

def save(declaration, filename):
    with open(filename, 'w') as f:
        json.dump(declaration, f, indent=4)

def load(filename):
    with open(filename, 'r') as f:
        return json.load(f)

# }}}

# base worlds {{{

highway_base = {
    "dynamics": {
        "kind": DYNAMICS_NORMAL,
        "params": {
            "dt": 0.1,
        },
    },
    "world": {
        "kind": WORLD_HIGHWAY,
    },
}

highway_exit_base = {
    "dynamics": {
        "kind": DYNAMICS_NORMAL,
        "params": {
            "dt": 0.1,
        },
    },
    "world": {
        "kind": WORLD_HIGHWAY_EXIT,
    },
}

# }}}

# social cohesion {{{

base_cohesive = {
    "extends": highway_base,
    "cars": [
        {
            "kind": CAR_COHESIVE,
            "x0": [0., 0., math.pi/2, 0.6],
            "color": COLOR_ORANGE,
            "theta": [1., -50., 1., 100., 10. , -70.],
            "T": 5,
        },
    ],
}

## speed {{{

speed_base = {
    "extends": highway_base,
    "cars": [

        # left lane {{{
        {
            "kind": CAR_SIMPLE,
            "x0": [-0.13, -0.2, math.pi/2, 0.72],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 30. , -50.],
            "T": 5,
        },
        {
            "kind": CAR_SIMPLE,
            "x0": [-.13, -0.55, math.pi/2, 0.72],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 30. , -50.],
            "T": 5,
        },
        {
            "kind": CAR_SIMPLE,
            "x0": [-.13, -1.05, math.pi/2, 0.72],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 30. , -50.],
            "T": 5,
        },
        {
            "kind": CAR_SIMPLE,
            "x0": [-.13, -1.45, math.pi/2, 0.72],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 30. , -50.],
            "T": 5,
        },
        {
            "kind": CAR_SIMPLE,
            "x0": [-.13, -2., math.pi/2, 0.72],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 30. , -50.],
            "T": 5,
        },

        # }}}

        # center lane {{{

        {
            "kind": CAR_SIMPLE,
            "x0": [0.0, -0.05, math.pi/2, 0.72],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 30. , -50.],
            "T": 5,
        },
        {
            "kind": CAR_SIMPLE,
            "x0": [0.0, -0.4, math.pi/2, 0.72],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 30. , -50.],
            "T": 5,
        },

        # }}}

        # right lane {{{

        {
            "kind": CAR_SIMPLE,
            "x0": [.13, 0.5, math.pi/2, 0.72],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 30. , -50.],
            "T": 5,
        },
        {
            "kind": CAR_SIMPLE,
            "x0": [.13, 0.00, math.pi/2, 0.72],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 30. , -50.],
            "T": 5,
        },
        {
            "kind": CAR_SIMPLE,
            "x0": [.13, -0.8, math.pi/2, 0.72],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 30. , -50.],
            "T": 5,
        },
        {
            "kind": CAR_SIMPLE,
            "x0": [.13, -1.3, math.pi/2, 0.72],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 30. , -50.],
            "T": 5,
        },

        # }}}

    ],
    "main_car": 11,
}

speed_naive = {
    "extends": speed_base,
    "cars": [
        {
            "kind": CAR_SIMPLE,
            "x0": [0.0, -0.7, math.pi/2, 0.5],
            "color": COLOR_BLUE,
            "theta": [1., -50., 1., 100., 5. , -50.],
            "T": 5,
        },
    ],
}

speed_lane_features = {
    "extends": speed_base,
    "cars": [
        {
            "kind": CAR_SIMPLE,
            "x0": [0.0, -0.7, math.pi/2, 0.5],
            "color": COLOR_ORANGE,
            "theta": [1., 0., 0., 0., 0. , 0.],
            "T": 5,
        },
    ],
}

speed_fence_features = {
    "extends": speed_base,
    "cars": [
        {
            "kind": CAR_SIMPLE,
            "x0": [0.0, -0.7, math.pi/2, 0.5],
            "color": COLOR_ORANGE,
            "theta": [0., -1., 0., 0., 0. , 0.],
            "T": 5,
        },
    ],
}

speed_car_features = {
    "extends": speed_base,
    "cars": [
        {
            "kind": CAR_SIMPLE,
            "x0": [0.0, -0.7, math.pi/2, 0.5],
            "color": COLOR_ORANGE,
            "theta": [0., 0., 0., 0., 0. , -50.],
            "T": 5,
        },
    ],
}

speed_cohesive = {
    "extends": speed_base,
    "cars": [
        {
            "kind": CAR_COHESIVE,
            "x0": [0.0, -0.7, math.pi/2, 0.5],
            "color": COLOR_ORANGE,
            "theta": [1., -50., 1., 100., 5. , -50.],
            "T": 5,
        },
    ],
}

# }}}

## swerve {{{

swerve_small_base = {
    "extends": highway_base,
    "cars": [
        # block car
        {
            "kind": CAR_SIMPLE,
            "x0": [-.13 -.02, 0.00, math.pi/2, 0.0],
            "color": COLOR_GRAY,
            "T": 5,
            "theta": [0., 0., 0., 0., 0. , 0.],
        },

        # left lane cars {{{

        # first car
        {
            "kind": CAR_SIMPLE,
            "x0": [-.13, -0.6, math.pi/2, 0.6],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 10. , -70.],
            "T": 5,
        },

        # second car
        {
            "kind": CAR_SIMPLE,
            "x0": [-.13, -0.9, math.pi/2, 0.6],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 10. , -70.],
            "T": 5,
        },

        # third car
        {
            "kind": CAR_SIMPLE,
            "x0": [-.13, -1.2, math.pi/2, 0.6],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 10. , -70.],
            "T": 5,
        },

        # }}}

        # center lane cars {{{

        # first car

        {
            "kind": CAR_SIMPLE,
            "x0": [.0, 0.4, math.pi/2, 0.6],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 10. , -70.],
            "T": 5,
        },

        {
            "kind": CAR_SIMPLE,
            "x0": [.0, 0.10, math.pi/2, 0.6],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 10. , -70.],
            "T": 5,
        },

        # }}}

        # right lane cars {{{

        # first car
        {
            "kind": CAR_SIMPLE,
            "x0": [.13, 0.8, math.pi/2, 0.6],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 10. , -70.],
            "T": 5,
        },

        # second car
        {
            "kind": CAR_SIMPLE,
            "x0": [.13, 0.3, math.pi/2, 0.6],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 10. , -70.],
            "T": 5,
        },

        # third car
        {
            "kind": CAR_SIMPLE,
            "x0": [.13, -0.4, math.pi/2, 0.6],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 10. , -70.],
            "T": 5,
        },

        # fourth car
        {
            "kind": CAR_SIMPLE,
            "x0": [.13, -0.85, math.pi/2, 0.6],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 10. , -70.],
            "T": 5,
        },

        # }}}

    ],
}

swerve_small_cohesive = {
    "extends": swerve_small_base,
    "cars": [
        {
            "kind": CAR_COHESIVE,
            "x0": [-.13, -1.5, math.pi/2, 0.6],
            "color": COLOR_ORANGE,
            "theta": [1., -50., 1., 100., 10. , -70.],
            "T": 5,
            "exclude": [0],
        },
    ],
}

swerve_small_naive = {
    "extends": swerve_small_base,
    "cars": [
        {
            "kind": CAR_SIMPLE,
            "x0": [-.13, -1.5, math.pi/2, 0.6],
            "color": COLOR_BLUE,
            "theta": [1., -50., 1., 100., 10. , -70.],
            "T": 5,
            "exclude": [0],
        },
    ],
}

swerve_base = {
    "extends": highway_base,
    "cars": [
        # block car
        {
            "kind": CAR_SIMPLE,
            "x0": [-.13 -.02, 0.00, math.pi/2, 0.0],
            "color": COLOR_GRAY,
            "T": 5,
            "theta": [0., 0., 0., 0., 0. , 0.],
        },

        # left lane cars {{{

        # first car
        {
            "kind": CAR_SIMPLE,
            "x0": [-.13, -3.3, math.pi/2, 0.6],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 10. , -70.],
            "T": 5,
        },

        # second car
        {
            "kind": CAR_SIMPLE,
            "x0": [-.13, -3.6, math.pi/2, 0.6],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 10. , -70.],
            "T": 5,
        },

        # third car
        {
            "kind": CAR_SIMPLE,
            "x0": [-.13, -3.9, math.pi/2, 0.6],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 10. , -70.],
            "T": 5,
        },

        {
            "kind": CAR_SIMPLE,
            "x0": [-.13, -4.2, math.pi/2, 0.6],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 10. , -70.],
            "T": 5,
        },

        {
            "kind": CAR_SIMPLE,
            "x0": [-.13, -4.5, math.pi/2, 0.6],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 10. , -70.],
            "T": 5,
        },

        {
            "kind": CAR_SIMPLE,
            "x0": [-.13, -4.5, math.pi/2, 0.6],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 10. , -70.],
            "T": 5,
        },

        {
            "kind": CAR_SIMPLE,
            "x0": [-.13, -4.8, math.pi/2, 0.6],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 10. , -50.],
            "T": 5,
        },

        {
            "kind": CAR_SIMPLE,
            "x0": [-.13, -5.1, math.pi/2, 0.6],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 10. , -70.],
            "T": 5,
        },

        {
            "kind": CAR_SIMPLE,
            "x0": [-.13, -5.4, math.pi/2, 0.6],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 10. , -70.],
            "T": 5,
        },

        {
            "kind": CAR_SIMPLE,
            "x0": [-.13, -5.7, math.pi/2, 0.6],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 10. , -70.],
            "T": 5,
        },

        {
            "kind": CAR_SIMPLE,
            "x0": [-.13, -6.0, math.pi/2, 0.6],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 10. , -70.],
            "T": 5,
        },

        # }}}

        # middle lane cars {{{
        {
            "kind": CAR_SIMPLE,
            "x0": [0., -2.3, math.pi/2, 0.55],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 7. , -70.],
            "T": 5,
        },
        {
            "kind": CAR_SIMPLE,
            "x0": [0., -2.9, math.pi/2, 0.55],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 7. , -70.],
            "T": 5,
        },
        {
            "kind": CAR_SIMPLE,
            "x0": [0., -3.5, math.pi/2, 0.55],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 7. , -70.],
            "T": 5,
        },
        {
            "kind": CAR_SIMPLE,
            "x0": [0., -4.1, math.pi/2, 0.55],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 7. , -70.],
            "T": 5,
        },
        {
            "kind": CAR_SIMPLE,
            "x0": [0., -4.7, math.pi/2, 0.55],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 7. , -70.],
            "T": 5,
        },
        {
            "kind": CAR_SIMPLE,
            "x0": [0., -5.3, math.pi/2, 0.55],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 7. , -70.],
            "T": 5,
        },
        {
            "kind": CAR_SIMPLE,
            "x0": [0., -5.9, math.pi/2, 0.55],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 7. , -70.],
            "T": 5,
        },
        {
            "kind": CAR_SIMPLE,
            "x0": [0., -6.5, math.pi/2, 0.55],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 7. , -70.],
            "T": 5,
        },

        # }}}
    ],
}

swerve_naive = {
    "extends": swerve_base,
    "cars": [
        {
            "kind": CAR_SIMPLE,
            "x0": [-.13, -6.3, math.pi/2, 0.6],
            "color": COLOR_ORANGE,
            "theta": [1., -50., 1., 100., 10. , -70.],
            "T": 5,
            "exclude": [0],
        },
    ],
}

swerve_variance = {
    "extends": swerve_base,
    "cars": [
        {
            "kind": CAR_COHESIVE,
            "x0": [-.13, -6.3, math.pi/2, 0.6],
            "color": COLOR_ORANGE,
            "theta": [1., -50., 1., 100., 10. , -70.],
            "T": 5,
            "exclude": [0],
        },
    ],
}

swerve_copy = {
    "extends": swerve_base,
    "cars": [
        {
            "kind": CAR_COPY,
            "x0": [0.0, -0.6, math.pi/2, 0.6],
            "color": COLOR_ORANGE,
            "theta": [1., -50., 1., 100., 10. , -50.],
            "T": 5,
            "exclude": [3],
        },
    ],
}

## }}}

## right_lane {{{

right_lane_base = {
    "extends": highway_base,
    "cars": [
        # right lane already {{{
        {
            "kind": CAR_SIMPLE,
            "x0": [.13, 0.9, math.pi/2, 0.6],
            "color": COLOR_WHITE,
            "theta": [1., -50., 10., 100., 10. , -50.],
            "extra": [
                {
                    "kind": REWARD_RIGHT_LANE,
                    "gain": 200.0,
                },
            ],
            "T": 5,
        },
        # }}}

        # left lane {{{
        {
            "kind": CAR_SIMPLE,
            "x0": [-.13, -0.25, math.pi/2, 0.6],
            "color": COLOR_WHITE,
            "theta": [1., -100., 10., 100., 10. , -50.],
            "extra": [
                {
                    "kind": REWARD_RIGHT_LANE,
                    "gain": 200.0,
                },
            ],
            "T": 5,
        },
        {
            "kind": CAR_SIMPLE,
            "x0": [-.13, -0.75, math.pi/2, 1.0],
            "color": COLOR_FIRE,
            "theta": [1., -50., 10., 100., 100. , -50.],
            "T": 5,
        },
        # }}}

        # center lane {{{
        {
            "kind": CAR_SIMPLE,
            "x0": [-0., -0.6, math.pi/2, 0.6],
            "color": COLOR_WHITE,
            "theta": [1., -50., 10., 100., 10. , -50.],
            "extra": [
                {
                    "kind": REWARD_RIGHT_LANE,
                    "gain": 200.0,
                },
            ],
            "T": 5,
        },
        {
            "kind": CAR_SIMPLE,
            "x0": [0., 0.50, math.pi/2, 0.6],
            "color": COLOR_WHITE,
            "theta": [1., -50., 10., 100., 10. , -50.],
            "extra": [
                {
                    "kind": REWARD_RIGHT_LANE,
                    "gain": 200.0,
                },
            ],
            "T": 5,
        },
        # }}}
    ],
}

right_lane_naive = {
    "extends": right_lane_base,
    "main_car": 4,
    "cars": [
        {
            "kind": CAR_SIMPLE,
            "x0": [0.0, 0.0, math.pi/2, 0.6],
            "color": COLOR_BLUE,
            "theta": [1., -50., 1., 100., 10. , -50.],
            "T": 5,
        },
    ],
}

right_lane_cohesive = {
    "extends": right_lane_base,
    "main_car": 5,
    "cars": [
        {
            "kind": CAR_COHESIVE,
            "x0": [0.0, 0.1, math.pi/2, 0.6],
            "color": COLOR_ORANGE,
            "theta": [1., -50., 1., 100., 10. , -50.],
            "T": 5,
        },
    ],
}

right_lane_copy = {
    "extends": right_lane_base,
    "main_car": 3,
    "cars": [
        {
            "kind": CAR_COPY,
            "x0": [0.0, -0.6, math.pi/2, 0.6],
            "color": COLOR_ORANGE,
            "theta": [1., -50., 1., 100., 10. , -50.],
            "T": 5,
        },
    ],
}

# }}}

## two_merge_out {{{

two_merge_out_base = {
    "extends": highway_base,
    "cars": [

        # center lane {{{
        {
            "kind": CAR_SIMPLE,
            "x0": [0., .8, math.pi/2, 0.6],
            "color": COLOR_WHITE,
            "theta": [1., -50., 10., 100., 10. , -50.],
            "extra": [
                {
                    "kind": REWARD_RIGHT_LANE,
                    "gain": 200.0,
                },
            ],
            "T": 5,
        },
        {
            "kind": CAR_SIMPLE,
            "x0": [0., .4, math.pi/2, 0.6],
            "color": COLOR_WHITE,
            "theta": [1., -50., 10., 100., 10. , -50.],
            "T": 5,
        },
        {
            "kind": CAR_SIMPLE,
            "x0": [0., -0.4, math.pi/2, 0.6],
            "color": COLOR_WHITE,
            "theta": [1., -.5, 10., 100., 10. , -50.],
            "extra": [
                {
                    "kind": REWARD_RIGHT_LANE,
                    "gain": 200.0,
                },
            ],
            "T": 5,
        },
        {
            "kind": CAR_SIMPLE,
            "x0": [0., -.8, math.pi/2, 0.6],
            "color": COLOR_WHITE,
            "theta": [1., -1., 10., 100., 10. , -50.],
            "T": 5,
        },

        # }}}

    ],
}

two_merge_out_naive = {
    "extends": two_merge_out_base,
    "main_car": 4,
    "cars": [
        {
            "kind": CAR_SIMPLE,
            "x0": [0.0, 0.0, math.pi/2, 0.6],
            "color": COLOR_BLUE,
            "theta": [1., -1., 1., 100., 10. , -50.],
            "T": 5,
        },
    ],
}

two_merge_out_cohesive = {
    "extends": two_merge_out_base,
    "main_car": 4,
    "cars": [
        {
            "kind": CAR_COHESIVE,
            "x0": [0.0, 0.0, math.pi/2, 0.6],
            "color": COLOR_BLUE,
            "theta": [1., -1., 1., 100., 10. , -50.],
            "T": 5,
        },
    ],
}


# }}}

## highway_exit_all {{{

highway_exit_all_base = {
    "extends": highway_exit_base,
    "cars": [
        {
            "kind": CAR_SIMPLE,
            "x0": [.13, .0, math.pi/2, 0.6],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 10. , -50.],
            "T": 5,
            "extra": [
                {
                    "kind": REWARD_RIGHT_HALF,
                    "gain": 2.0,
                },
            ],
        },
        {
            "kind": CAR_SIMPLE,
            "x0": [.13, -.3, math.pi/2, 0.6],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 10. , -50.],
            "T": 5,
            "extra": [
                {
                    "kind": REWARD_RIGHT_HALF,
                    "gain": 2.0,
                },
            ],
        },
        {
            "kind": CAR_SIMPLE,
            "x0": [.13, -.6, math.pi/2, 0.6],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 10. , -50.],
            "T": 5,
            "extra": [
                {
                    "kind": REWARD_RIGHT_HALF,
                    "gain": 2.0,
                },
            ],
        },
    ],
}

highway_exit_all_naive = {
    "extends": highway_exit_all_base,
    "cars": [
        {
            "kind": CAR_SIMPLE,
            "x0": [0.13, -.9, math.pi/2, 0.6],
            "color": COLOR_BLUE,
            "theta": [1., -50., 100., 100., 10. , -50.],
            "T": 5,
        },
    ],
}

highway_exit_all_cohesive = {
    "extends": highway_exit_all_base,
    "cars": [
        {
            "kind": CAR_COHESIVE,
            "x0": [.13, -.9, math.pi/2, 0.6],
            "color": COLOR_ORANGE,
            "theta": [1., -50., 100., 100., 10. , -50.],
            "T": 5,
        },
    ],
}

# }}}

# }}}

swerve_neural = {
    "extends": highway_base,
    "main_car": 0,
    "cars": [
        {
            "kind": CAR_NEURAL,
            "x0": [0.0, 0.00, math.pi/2, 0.6],
            "color": COLOR_ORANGE,
            "T": 5,
            "excludes": [1],
            "model": "swerve-3layers-1000reps",
            "mu": .5,
            "theta": [1., -50., 1., 100., 10. , -50.],
        },
        # block car
        {
            "kind": CAR_SIMPLE,
            "x0": [-0.02, 0.50, math.pi/2, 0.0],
            "color": COLOR_GRAY,
            "T": 5,
            "theta": [0., 0., 0., 0., 0. , 0.],
        },
    ],
}

# bounded rationality {{{

crash_base = {
    "extends": highway_base,
    "cars": [
        # merge car
        {
            "kind": CAR_CANNED,
            "x0": [+.13, 0.22, math.pi/2., 0.7],
            "color": COLOR_GRAY,
            "T": 20,
            "controls": [([ 0.71598   ,  0.69208262]),  ([ 0.70819742,  0.69534356]),  ([ 0.70017753,  0.69714366]),  ([ 0.69050416,  0.69705493]),  ([ 0.6712254 ,  0.69178569]),  ([ 0.62889238,  0.68024151]),  ([ 0.51708621,  0.67136581]),  ([-0.62256188,  0.66215711]),  ([-0.65346835,  0.6578402 ]),  ([-0.66612079,  0.65872826]),  ([-0.67128989,  0.6626553 ]),  ([-0.67200172,  0.66740692]),  ([-0.66871722,  0.67156702]),  ([-0.6598298 ,  0.67414042]),  ([-0.64034248,  0.67416649]),  ([-0.60268203,  0.67212932]),  ([-0.54502419,  0.67059391]),  ([-0.05829414,  0.66988283]),  ([ 0.50956669,  0.66956543]),  ([ 0.50169908,  0.66957544]),  ([ 0.47060134,  0.66964394]),  ([ 0.40836934,  0.66973165]),  ([ 0.12701486,  0.66975035]),  ([-0.24805782,  0.66983425]),  ([-0.02020778,  0.66987636]),  ([-0.00756135,  0.6699145 ]),  ([ 0.00162548,  0.66994832]),  ([ 0.00108227,  0.6699784 ]),  ([  5.64470289e-04,   6.70005115e-01]),  ([  3.27592305e-04,   6.70028839e-01]),  ([  1.67712590e-04,   6.70049905e-01]),  ([  9.83664140e-05,   6.70068607e-01]),  ([  4.94744798e-05,   6.70085215e-01]),  ([  2.94775318e-05,   6.70099964e-01]),  ([  1.45184268e-05,   6.70113065e-01]),  ([  8.83608525e-06,   6.70124706e-01]),  ([  4.24058945e-06,   6.70135050e-01]),  ([  2.65421351e-06,   6.70144245e-01]),  ([  1.23276143e-06,   6.70152418e-01]),  ([  8.00050747e-07,   6.70159685e-01]),  ([  3.56399357e-07,   6.70166145e-01]),  ([  2.42264770e-07,   6.70171890e-01]),  ([  1.02310742e-07,   6.70176999e-01]),  ([  7.37732657e-08,   6.70181542e-01]),  ([  2.90909843e-08,   6.70185582e-01]),  ([  2.26149905e-08,   6.70189174e-01]),  ([  8.16265164e-09,   6.70192369e-01]),  ([  6.98656819e-09,   6.70195209e-01]),  ([  2.24731457e-09,   6.70197735e-01]),  ([  2.17777979e-09,   6.70199982e-01])],
        },
        # crash car
        {
            "kind": CAR_CANNED,
            "x0": [-.13, 0.10, math.pi/2, 0.7],
            "color": COLOR_GRAY,
            "T": 20,
            "controls": [ ([-0.40803463,  0.66688008]),  ([-0.01178236,  0.66722892]),  ([ 0.28196057,  0.66759778]),  ([ 0.1616365 ,  0.66788353]),  ([-0.00689171,  0.66815119]),  ([-0.00260256,  0.66837202]),  ([-0.0552785 ,  0.66815789]),  ([ 0.42801833,  0.66569158]),  ([ 0.18917435,  0.66556055]),  ([ 0.04545126,  0.6662067 ]),  ([-0.03686632,  0.66750549]),  ([-0.12959546,  0.66897452]),  ([-0.21441317,  0.67029078]),  ([-0.28530649,  0.67104277]),  ([-0.29633317,  0.67114489]),  ([-0.25775584,  0.67072296]),  ([-0.10110433,  0.67019774]),  ([ 0.08691868,  0.66992593]),  ([ 0.24736296,  0.66988372]),  ([ 0.24481931,  0.66991008]),  ([ 0.14176646,  0.66995658]),  ([-0.00682281,  0.66999236]),  ([ 0.04726598,  0.67004999]),  ([-0.00619012,  0.67010317]),  ([-0.04707631,  0.6701064 ]),  ([-0.02813267,  0.6701243 ]),  ([ 0.0038863 ,  0.67014239]),  ([ 0.00512286,  0.67015908]),  ([ 0.00527079,  0.67017385]),  ([ 0.00301855,  0.67018688]),  ([ 0.00160195,  0.67019841]),  ([ 0.00080015,  0.67020862]),  ([  3.94747629e-04,   6.70217684e-01]),  ([  1.92430858e-04,   6.70225732e-01]),  ([  9.24368282e-05,   6.70232883e-01]),  ([  4.31419995e-05,   6.70239238e-01]),  ([  1.91438031e-05,   6.70244887e-01])],
        },
        {
            "kind": CAR_CANNED,
            "x0": [-.13, 0.30, math.pi/2, 0.7],
            "color": COLOR_GRAY,
            "T": 20,
            "controls": [ ([-0.40803463,  0.66688008]),  ([-0.01178236,  0.66722892]),  ([ 0.28196057,  0.66759778]),  ([ 0.1616365 ,  0.66788353]),  ([-0.00689171,  0.66815119]),  ([-0.00260256,  0.66837202]),  ([-0.0552785 ,  0.66815789]),  ([ 0.42801833,  0.66569158]),  ([ 0.18917435,  0.66556055]),  ([ 0.04545126,  0.6662067 ]),  ([-0.03686632,  0.66750549]),  ([-0.12959546,  0.66897452]),  ([-0.21441317,  0.67029078]),  ([-0.28530649,  0.67104277]),  ([-0.29633317,  0.67114489]),  ([-0.25775584,  0.67072296]),  ([-0.10110433,  0.67019774]),  ([ 0.08691868,  0.66992593]),  ([ 0.24736296,  0.66988372]),  ([ 0.24481931,  0.66991008]),  ([ 0.14176646,  0.66995658]),  ([-0.00682281,  0.66999236]),  ([ 0.04726598,  0.67004999]),  ([-0.00619012,  0.67010317]),  ([-0.04707631,  0.6701064 ]),  ([-0.02813267,  0.6701243 ]),  ([ 0.0038863 ,  0.67014239]),  ([ 0.00512286,  0.67015908]),  ([ 0.00527079,  0.67017385]),  ([ 0.00301855,  0.67018688]),  ([ 0.00160195,  0.67019841]),  ([ 0.00080015,  0.67020862]),  ([  3.94747629e-04,   6.70217684e-01]),  ([  1.92430858e-04,   6.70225732e-01]),  ([  9.24368282e-05,   6.70232883e-01]),  ([  4.31419995e-05,   6.70239238e-01]),  ([  1.91438031e-05,   6.70244887e-01])],
        },
    ],
}


crash_3 = {
    "extends": crash_base,
    "main_car": -1,
    "cars": [
        {
            "kind": CAR_SIMPLE,
            "x0": [0.0, 0.20, math.pi/2, 0.7],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 1. , -50.],
            "T": 3,
        },
    ],
}

crash_10 = {
    "extends": crash_base,
    "main_car": -1,
    "cars": [
        {
            "kind": CAR_SIMPLE,
            "x0": [0.0, 0.20, math.pi/2, 0.7],
            "color": COLOR_WHITE,
            "theta": [1., -50., 1., 100., 1. , -50.],
            "T": 10,
        },
    ],
}

# }}}
