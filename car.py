import numpy as np
import utils
import theano as th
import theano.tensor as tt
import theano.tensor.slinalg as ts
from trajectory import Trajectory
import feature
import cohesion

class Car(object):
    def __init__(self, dyn, x0, color='yellow', T=5):
        self.data0 = {'x0': x0}
        self.T = T
        self.dyn = dyn

        self.traj = Trajectory(T, dyn)
        self.traj.x0.set_value(x0)

        self.traj2 = Trajectory(T, dyn)
        self.traj2.x0.set_value(x0)

        self.x_hist = [x0]*T
        self.past = Trajectory(T, dyn)
        self.past.x0.set_value(x0)
        self.linear = Trajectory(T, dyn)
        self.linear.x0.set_value(x0)
        self.color = color
        self.default_u = np.zeros(self.dyn.nu)
    def reset(self):
        self.x_hist = [self.data0['x0']]*self.T

        self.traj.x0.set_value(self.data0['x0'])
        self.traj2.x0.set_value(self.data0['x0'])

        self.past.x0.set_value(self.data0['x0'])
        self.linear.x0.set_value(self.data0['x0'])
        for t in range(self.T):

            self.traj.u[t].set_value(np.zeros(self.dyn.nu))
            self.traj2.u[t].set_value(np.zeros(self.dyn.nu))

            self.past.u[t].set_value(np.zeros(self.dyn.nu))
            self.linear.u[t].set_value(self.default_u)
    def past_tick(self):
        self.x_hist = self.x_hist[1:]+[self.x]
        self.past.tick()
        self.past.x0.set_value(self.x_hist[0])
        self.past.u[self.T-1].set_value(self.u)
    def move(self):
        self.past_tick()
        self.traj.tick()
        self.traj2.tick()
        self.linear.x0.set_value(self.traj.x0.get_value())
    @property
    def x(self):
        return self.traj.x0.get_value()
    @property
    def u(self):
        return self.traj.u[0].get_value()
    @u.setter
    def u(self, value):
        self.traj.u[0].set_value(value)
    def control(self, steer, gas):
        pass

class UserControlledCar(Car):
    def __init__(self, *args, **vargs):
        Car.__init__(self, *args, **vargs)
    def control(self, steer, gas):
        self.u = [steer, gas]


class CannedCar(Car):
    def __init__(self, *args, **vargs):
        Car.__init__(self, *args, **vargs)
        self.controls = None
        self.step = 0

    def follow(self, controls):
        self.controls =  controls
        self.step = 0

    def reset(self):
        Car.reset(self)
        self.step = 0

    def control(self, _steer, _gas):
        if self.controls == None:
            raise 'CannedCar.control: self.controls are None'

        if self.step >= len(self.controls):
            raise 'CannedCar.control: off the end of pre-canned controls!'

        steer, gas = self.controls[self.step]

        self.u = [steer, gas]

        self.step += 1

class SimpleOptimizerCar(Car):
    def __init__(self, *args, **vargs):
        Car.__init__(self, *args, **vargs)
    @property
    def reward(self):
        return self._reward
    @reward.setter
    def reward(self, reward):
        self._reward = reward
        self.optimizer = None
    def control(self, steer, gas):
        if self.optimizer is None:
            r = self.traj.total(self.reward)
            self.optimizer = utils.Maximizer(r, self.traj.u)
        self.optimizer.maximize()

# Cohesive Car {{{
class CohesiveCar(SimpleOptimizerCar):
    def __init__(self, *args, **vargs):
        SimpleOptimizerCar.__init__(self, *args, **vargs)
        self.beta = utils.scalar()
        self.beta.set_value(1.)
        self.hist = []
        self.stats = cohesion.CohesiveStats()
        self.inspected = None

        # cars in my position {{{

        self.lambdaX  = utils.scalar()
        self.lambdaY  = utils.scalar()
        self.lambdaO  = utils.scalar()
        self.lambdaXN = utils.scalar()
        self.lambdaYN = utils.scalar()
        self.lambdaON = utils.scalar()

        self.muX  = utils.scalar()
        self.muY  = utils.scalar()
        self.muO  = utils.scalar()
        self.muXN = utils.scalar()
        self.muYN = utils.scalar()
        self.muON = utils.scalar()

        # }}}

        # cars in my Lane {{{

        self.lambdaXL  = utils.scalar()
        self.lambdaYL  = utils.scalar()
        self.lambdaOL  = utils.scalar()
        self.lambdaXNL = utils.scalar()
        self.lambdaYNL = utils.scalar()
        self.lambdaONL = utils.scalar()

        self.muXL  = utils.scalar()
        self.muYL  = utils.scalar()
        self.muOL  = utils.scalar()
        self.muXNL = utils.scalar()
        self.muYNL = utils.scalar()
        self.muONL = utils.scalar()

        # }}}

        # cars in my Road {{{

        self.lambdaXR  = utils.scalar()
        self.lambdaYR  = utils.scalar()
        self.lambdaOR  = utils.scalar()
        self.lambdaXNR = utils.scalar()
        self.lambdaYNR = utils.scalar()
        self.lambdaONR = utils.scalar()

        self.muXR  = utils.scalar()
        self.muYR  = utils.scalar()
        self.muOR  = utils.scalar()
        self.muXNR = utils.scalar()
        self.muYNR = utils.scalar()
        self.muONR = utils.scalar()

        # }}}

    def reset(self):
        Car.reset(self)
        self.stats.reset()

    def ingest_hist(self, hist):
        others = hist[1:len(hist)-1]
        for i, other in enumerate(others):
            self.stats.processNewState(i, other)

    def set_hist(self, hist):
        self.hist = hist

    def update_stats(self):
        stats = cohesion.variances(self.traj.x0.get_value(), self.hist)

        # My position {{{
        posStats = stats[0]
        x = posStats[:, 0]
        y = posStats[:, 1]
        o = posStats[:, 2]
        xn = posStats[:, 3]
        yn = posStats[:, 4]
        on = posStats[:, 5]

        self.lambdaX.set_value(x[0])
        self.muX.set_value(x[1])

        self.lambdaY.set_value(y[0])
        self.muY.set_value(y[1])

        self.lambdaO.set_value(o[0])
        self.muO.set_value(o[1])

        self.lambdaXN.set_value(xn[0])
        self.muXN.set_value(xn[1])

        self.lambdaYN.set_value(yn[0])
        self.muYN.set_value(yn[1])

        self.lambdaON.set_value(on[0])
        self.muON.set_value(on[1])

        # }}}

        # My lane {{{
        laneStats = stats[1]
        x = laneStats[:, 0]
        y = laneStats[:, 1]
        o = laneStats[:, 2]
        xn = laneStats[:, 3]
        yn = laneStats[:, 4]
        on = laneStats[:, 5]

        self.lambdaXL.set_value(x[0])
        self.muXL.set_value(x[1])

        self.lambdaYL.set_value(y[0])
        self.muYL.set_value(y[1])

        self.lambdaOL.set_value(o[0])
        self.muOL.set_value(o[1])

        self.lambdaXNL.set_value(xn[0])
        self.muXNL.set_value(xn[1])

        self.lambdaYNL.set_value(yn[0])
        self.muYNL.set_value(yn[1])

        self.lambdaONL.set_value(on[0])
        self.muONL.set_value(on[1])

        # }}}

        # Whole Road {{{
        roadStats = stats[2]
        x = roadStats[:, 0]
        y = roadStats[:, 1]
        o = roadStats[:, 2]
        xn = roadStats[:, 3]
        yn = roadStats[:, 4]
        on = roadStats[:, 5]

        self.lambdaXR.set_value(x[0])
        self.muXR.set_value(x[1])

        self.lambdaYR.set_value(y[0])
        self.muYR.set_value(y[1])

        self.lambdaOR.set_value(o[0])
        self.muOR.set_value(o[1])

        self.lambdaXNR.set_value(xn[0])
        self.muXNR.set_value(xn[1])

        self.lambdaYNR.set_value(yn[0])
        self.muYNR.set_value(yn[1])

        self.lambdaONR.set_value(on[0])
        self.muONR.set_value(on[1])

        # }}}

    def set_beta(self, beta):
        self.beta.set_value(beta)

    # returns numpy array for later inspection
    @property
    def inspect(self):
        if self.inspected is None:
            raise Exception("inspected is none")
        return self.inspected

    def control(self, _steer, _gas):
        dyn = self.traj.dyn

        # Cohesion Reward
        r = self.traj.total(self.reward)
        xReward = (self.beta * self.lambdaX * tt.sqr(self.muX - feature.deltaX(dyn)(0, self.traj.x[0], self.traj.u[0])))
        yReward = (self.beta * self.lambdaY * tt.sqr(self.muY - feature.deltaY(dyn)(0, self.traj.x[0], self.traj.u[0])))
        oReward = (self.beta * self.lambdaO * tt.sqr(self.muO - feature.deltaO(dyn)(0, self.traj.x[0], self.traj.u[0])))
        posRewards = [xReward, yReward, oReward]
        r = r + sum(posRewards)

        xLReward = (self.beta * self.lambdaXL * tt.sqr(self.muXL - feature.deltaX(dyn)(0, self.traj.x[0], self.traj.u[0])))
        yLReward = (self.beta * self.lambdaYL * tt.sqr(self.muYL - feature.deltaY(dyn)(0, self.traj.x[0], self.traj.u[0])))
        oLReward = (self.beta * self.lambdaOL * tt.sqr(self.muOL - feature.deltaO(dyn)(0, self.traj.x[0], self.traj.u[0])))
        laneRewards = [xLReward, yLReward, oLReward]
        r = r + sum(laneRewards)

        xRReward = (self.beta * self.lambdaXR * tt.sqr(self.muXR - feature.deltaX(dyn)(0, self.traj.x[0], self.traj.u[0])))
        yRReward = (self.beta * self.lambdaYR * tt.sqr(self.muYR - feature.deltaY(dyn)(0, self.traj.x[0], self.traj.u[0])))
        oRReward = (self.beta * self.lambdaOR * tt.sqr(self.muOR - feature.deltaO(dyn)(0, self.traj.x[0], self.traj.u[0])))
        roadRewards = [xRReward, yRReward, oRReward]
        r = r + sum(roadRewards)

        # Original Reward
        originalR = self.traj2.total(self.reward)

        if self.optimizer is None:
            self.optimizer  = utils.Maximizer(r, self.traj.u)
            self.optimizer2 = utils.Maximizer(originalR, self.traj2.u)

        # We should have recieved a new history array. So recompute the stats.
        self.update_stats()
        self.optimizer.maximize()
        self.optimizer2.maximize()

        originalReward = (self.traj.total(self.reward)).eval()
        print("Original Reward:  %f" % (originalReward))
        #print("Reward of Normal: %f" % ((self.traj2.total(self.reward) +
        #    (self.beta * self.lambdaX * tt.sqr(self.muX - feature.deltaX(dyn)(0, self.traj2.x[0], self.traj2.u[0]))) +
        #    (self.beta * self.lambdaY * tt.sqr(self.muY - feature.deltaY(dyn)(0, self.traj2.x[0], self.traj2.u[0]))) +
        #    (self.beta * self.lambdaO * tt.sqr(self.muO - feature.deltaO(dyn)(0, self.traj2.x[0], self.traj2.u[0])))).eval()))
        print("Reward of Cohesive: %f" % (r.eval()))

        print("Features (POSITION)")
        print("(mx, my, mo) = [ %f, %f, %f ]" % (self.muX.get_value(), self.muY.get_value(), self.muO.get_value()))
        print("(lx, ly, lo) = [ %f, %f, %f ]" % (self.lambdaX.get_value(), self.lambdaY.get_value(), self.lambdaO.get_value()))
        print("(rx, ry, ro) = [ %f, %f, %f ]" % (xReward.eval(), yReward.eval(), oReward.eval()))

        print("Features (LANE)")
        print("(mx, my, mo) = [ %f, %f, %f ]" % (self.muXL.get_value(), self.muYL.get_value(), self.muOL.get_value()))
        print("(lx, ly, lo) = [ %f, %f, %f ]" % (self.lambdaXL.get_value(), self.lambdaYL.get_value(), self.lambdaOL.get_value()))
        print("(rx, ry, ro) = [ %f, %f, %f ]" % (xLReward.eval(), yLReward.eval(), oLReward.eval()))

        print("Features (ROAD)")
        print("(mx, my, mo) = [ %f, %f, %f ]" % (self.muXR.get_value(), self.muYR.get_value(), self.muOR.get_value()))
        print("(lx, ly, lo) = [ %f, %f, %f ]" % (self.lambdaXR.get_value(), self.lambdaYR.get_value(), self.lambdaOR.get_value()))
        print("(rx, ry, ro) = [ %f, %f, %f ]" % (xRReward.eval(), yRReward.eval(), oRReward.eval()))

        print("Actual")
        features = np.array([
            feature.deltaX(dyn)(0, self.traj.x[0], self.traj.u[0]).eval(),
            feature.deltaY(dyn)(0, self.traj.x[0], self.traj.u[0]).eval(),
            feature.deltaO(dyn)(0, self.traj.x[0], self.traj.u[0]).eval(),
        ])
        print("(dx, dy, do) = [ %f, %f, %f ]" % (features[0], features[1], features[2]))

        print("Controls")
        originalControl = self.traj2.u[0].get_value()
        cohesiveControl = self.traj.u[0].get_value()
        controlDeviations = originalControl - cohesiveControl
        print("Original Control:   [ %f, %f ]" % (originalControl[0],   originalControl[1]))
        print("Cohesive Control:   [ %f, %f ]" % (cohesiveControl[0],   cohesiveControl[1]))
        print("Control Deviations: [ %f, %f ]" % (controlDeviations[0], controlDeviations[1]))
        print("")

        self.inspected = (
            originalReward, # original reward R 0
            r.eval(), # R_c # 1
            # mus for pos
            (self.muX.get_value(), self.muY.get_value(), self.muO.get_value()),
            # lambdas for pos
            (self.lambdaX.get_value(), self.lambdaY.get_value(), self.lambdaO.get_value()),
            # rewards for pos
            (xReward.eval(), yReward.eval(), oReward.eval()),
            (self.muXL.get_value(), self.muYL.get_value(), self.muOL.get_value()),
            (self.lambdaXL.get_value(), self.lambdaYL.get_value(), self.lambdaOL.get_value()),
            (xLReward.eval(), yLReward.eval(), oLReward.eval()),
            (self.muXR.get_value(), self.muYR.get_value(), self.muOR.get_value()),
            (self.lambdaXR.get_value(), self.lambdaYR.get_value(), self.lambdaOR.get_value()),
            (xRReward.eval(), yRReward.eval(), oRReward.eval()),
            features,
            originalControl,
            cohesiveControl,
            controlDeviations,
        )

# }}}

# Neural Car {{{
class NeuralCar(SimpleOptimizerCar):
    def __init__(self, *args, **vargs):
        SimpleOptimizerCar.__init__(self, *args, **vargs)
        self.mu = 1.0

    """
        Expects a keras model object
    """
    def use(self, model, mu=None):
        self.model = model

        if mu is not None:
            self.mu = mu

    def control(self, _steer, _gas):
        if self.model == None:
            raise Exception("NeuralCar.model is None")

        if self.mu == 1.0:
            self.u = self.model.predict(np.array([self.x]))[0]
            return

        if self.optimizer is None:
            r = self.traj.total(self.reward)
            self.optimizer = utils.Maximizer(r, self.traj.u)

        self.optimizer.maximize()

        self.u = (1-self.mu)*self.u + self.mu*self.model.predict(np.array([self.x]))[0]

# }}}

# CopyCar {{{
class CopyCar(SimpleOptimizerCar):
    def __init__(self, *args, **vargs):
        SimpleOptimizerCar.__init__(self, *args, **vargs)

        self.social_u = utils.vector(2)
        self.l = utils.scalar()
        self.watching = []
        self.copyx = None
        self.copyu = None
        self.l_default = 3

    def watch(self, cars):
        self.watching = cars

    def use(self, x, u, l = 3):
        self.copyx = x
        self.copyu = u

    # fix and make this automatic
    def use_also(self, x, u):
        if self.copyx == None:
            self.copyx = x
        else:
            self.copyx = np.vstack((self.copyx, x))

        if self.copyu == None:
            self.copyu = u
        else:
            self.copyu = np.vstack((self.copyu, u))

    def control(self, _steer, _gas):
        if self.copyx != None:
            print(self.copyx.shape[0], " examples")
            dists = np.array([np.linalg.norm(self.x - x) for x in self.copyx])
            i = np.argmin(dists)
            self.social_u.set_value(self.copyu[i])

            print("SOCIAL U_i", i)
            self.l.set_value(self.l_default)
            if dists[i] > 1.0:
                print("ignoring social")
                self.l.set_value(0)
            else:
                self.l.set_value(self.l_default)
        else:
            self.l.set_value(0)

        if self.optimizer is None:
            r = self.traj.total(self.reward) - self.l * (self.traj.u[0] - self.social_u).norm(2)
            self.optimizer = utils.Maximizer(r, self.traj.u)

        self.optimizer.maximize()

# }}}

# NestedOptimizerCar {{{
class NestedOptimizerCar(Car):
    def __init__(self, *args, **vargs):
        Car.__init__(self, *args, **vargs)
        self.bounds = [(-3., 3.), (-2., 2.)]
    @property
    def human(self):
        return self._human
    @human.setter
    def human(self, value):
        self._human = value
        self.traj_h = Trajectory(self.T, self.human.dyn)
    def move(self):
        Car.move(self)
        self.traj_h.tick()
    @property
    def rewards(self):
        return self._rewards
    @rewards.setter
    def rewards(self, vals):
        self._rewards = vals
        self.optimizer = None
    def control(self, steer, gas):
        #import ipdb; ipdb.set_trace()
        if self.optimizer is None:
            reward_h, reward_r = self.rewards
            reward_h = self.traj_h.total(reward_h)
            reward_r = self.traj.total(reward_r)
            self.optimizer = utils.NestedMaximizer(reward_h, self.traj_h.u, reward_r, self.traj.u)
        self.traj_h.x0.set_value(self.human.x)
        self.optimizer.maximize(bounds = self.bounds)

# }}}

# BeliefOptimizerCar {{{
class BeliefOptimizerCar(Car):
    def __init__(self, *args, **vargs):
        Car.__init__(self, *args, **vargs)
        self.bounds = [(-3., 3.), (-2., 2.)]
        self.dumb = False
        self.collab = False
        self.eta = 1.
        self.t = 0 ## ADDED BY NICK FEB 6
    @property
    def human(self):
        return self._human
    @human.setter
    def human(self, value):
        self._human = value
        self.traj_hs = []
        self.log_ps = []
        self.rewards = []
        self.optimizer = None
    def add_model(self, reward, log_p=0.):
        self.traj_hs.append(Trajectory(self.T, self.human.dyn))
        weight = utils.scalar()
        weight.set_value(log_p)
        self.log_ps.append(weight)
        self.rewards.append(reward)
        self.data0['log_ps'] = [log_p.get_value() for log_p in self.log_ps]
        self.optimizer = None
    @property
    def objective(self):
        return self._objective
    @objective.setter
    def objective(self, value):
        self._objective = value
        self.optimizer = None
    def reset(self):
        Car.reset(self)
        for log_p, val in zip(self.log_ps, self.data0['log_ps']):
            log_p.set_value(val)
        if hasattr(self, 'normalize'):
            self.normalize()
        self.t = 0
        if self.dumb:
            self.useq = self.objective
    def move(self):
        Car.move(self)
        self.t += 1
    def entropy(self, traj_h):
        new_log_ps = [traj_h.log_p(reward('traj'))+log_p for log_p, reward in zip(self.log_ps, self.rewards)]
        mean_log_p = sum(new_log_ps)/len(new_log_ps)
        new_log_ps = [x-mean_log_p for x in new_log_ps]
        s = tt.log(sum(tt.exp(x) for x in new_log_ps))
        new_log_ps = [x-s for x in new_log_ps]
        return sum(x*tt.exp(x) for x in new_log_ps)
    def control(self, steer, gas):
        if self.optimizer is None:
            u = sum(log_p for log_p in self.log_ps)/len(self.log_ps)
            self.prenormalize = th.function([], None, updates=[(log_p, log_p-u) for log_p in self.log_ps])
            s = tt.log(sum(tt.exp(log_p) for log_p in self.log_ps))
            self.normalize = th.function([], None, updates=[(log_p, log_p-s) for log_p in self.log_ps])
            self.update_belief = th.function([], None, updates=[(log_p, log_p + self.human.past.log_p(reward('past'))) for reward, log_p in zip(self.rewards, self.log_ps)])
            self.normalize()
            self.t = 0
            if self.dumb:
                self.useq = self.objective
                self.optimizer = True
            else:
                if hasattr(self.objective, '__call__'):
                    obj_h = sum([traj_h.total(reward('traj')) for traj_h, reward in zip(self.traj_hs, self.rewards)])
                    var_h = sum([traj_h.u for traj_h in self.traj_hs], [])
                    obj_r = sum(tt.exp(log_p)*self.objective(traj_h) for traj_h, log_p in zip(self.traj_hs, self.log_ps))
                    if self.collab:
                        obj_r = obj_r + self.eta*sum([tt.exp(10*(1-tt.exp(log_p)))*traj_h.total(reward('traj')) for traj_h, reward, log_p in zip(self.traj_hs, self.rewards, self.log_ps)])
                    self.optimizer = utils.NestedMaximizer(obj_h, var_h, obj_r, self.traj.u)
                else:
                    obj_r = self.objective
                    if self.collab:
                        obj_r = obj_r + self.eta*sum([tt.exp(10*(1-tt.exp(log_p)))*traj_h.total(reward('traj')) for traj_h, reward, log_p in zip(self.traj_hs, self.rewards, self.log_ps)])
                    self.optimizer = utils.Maximizer(obj_r, self.traj.u)
        if self.t == self.T:
            self.update_belief()
            self.t = 0
        if self.dumb:
            self.u = self.useq[0]
            self.useq = self.useq[1:]
        if self.t == 0:
            self.prenormalize()
            self.normalize()
            for traj_h in self.traj_hs:
                traj_h.x0.set_value(self.human.x)
            if not self.dumb:
                self.optimizer.maximize(bounds = self.bounds)
        for log_p in self.log_ps:
            print '%.2f'%np.exp(log_p.get_value()),
        print
        #for traj in self.traj_hs:
        #    traj.x0.set_value(self.human.x)
        #self.optimizer.maximize(bounds = self.bounds)

# }}}
