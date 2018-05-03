
# coding: utf-8

# In[123]:

from pyomo.environ import *
import pyomo.opt
import time
from time import strftime
import os
import pickle
import sys
import random
import numpy as np

M = 99999
LOSS_AVERSION_FACTOR = 0.1
ATTACKER_LOSS_FACTOR = 0.3
REWARDS_FILE = "20180429-rewards.pickle"
BONMIN_EXEC = "/home/ashton/Bonmin-1.8.6/bin/bonmin"
random.seed(2018)



# In[140]:
class Experiment:

    def __init__(self, attacker_prob, service_prob, rewards, pickle_path=None):
        self.attacker_prob = attacker_prob
        # changing x to self.c
        self.c = service_prob
        self.rewards = rewards
        self.pickle_path = pickle_path

        self.modified_rewards = []
        for i in range(len(rewards)):
            for j in range(len(rewards[i])):
                for k in range(len(rewards[i][j])):
                    self.modified_rewards.append((i,j,k))

        self.m = ConcreteModel()
        self.prop_m = ConcreteModel()
        self.prop_h = [(1 - sum(self.c)) * (c_s / sum(self.c)) for c_s in self.c]

    def init_vars(self):
        self.m.attackers = Set(initialize=list(range(len(self.attacker_prob))))
        self.prop_m.attackers = Set(initialize=list(range(len(self.attacker_prob))))
        self.m.service_set = Set(initialize=list(range(len(self.c))))
        self.prop_m.service_set = Set(initialize=list(range(len(self.c))))
        self.m.attacks = Set(initialize=self.modified_rewards, dimen=3)
        self.prop_m.attacks = Set(initialize=self.modified_rewards, dimen=3)

        self.m.h = Var(self.m.service_set, domain=PositiveReals)
        self.m.n = Var(self.m.attacks, domain=Binary)
        self.prop_m.n = Var(self.m.attacks, domain=Binary)
        self.m.v = Var(self.m.attackers, domain=Reals)
        self.prop_m.v = Var(self.m.attackers, domain=Reals)
        self.m.q = Var(self.m.service_set, domain=PositiveReals)
        self.prop_m.q = Var(self.m.service_set, domain=PositiveReals)

    def set_obj(self):

        def obj_rule(model):
            return sum(self.attacker_prob[theta] * model.n[(s, theta, a)] * exp(model.q[s]) * \
                    (self.c[s] * -1 * self.rewards[s][theta][a]['is'] + \
                    LOSS_AVERSION_FACTOR * model.h[s] * self.rewards[s][theta][a]['is']) \
                    for s, theta, a in model.attacks)

        self.m.OBJ = Objective(rule=obj_rule, sense=maximize)
    
    def set_prop_obj(self):
        def obj_rule(model):
            return sum(self.attacker_prob[theta] * model.n[(s, theta, a)] * exp(model.q[s]) * \
                    (self.c[s] * -1 * self.rewards[s][theta][a]['is'] + \
                    LOSS_AVERSION_FACTOR * self.prop_h[s] * self.rewards[s][theta][a]['is']) \
                    for s, theta, a in model.attacks)
        
        self.prop_m.OBJ = Objective(rule=obj_rule, sense=maximize)       
    
    def set_constraints(self):
        def h_positive(model, s):
            return model.h[s] >= 0
        self.m.h_pos_constr = Constraint(self.m.service_set, rule=h_positive)

        def x_sum(model):
            return sum(model.h[s] + self.c[s] for s in model.service_set) == 1
        self.m.x_sum = Constraint(rule=x_sum)

        def best_attacker_u1(model, s, theta, a):
            #s, theta, a = attack
            return model.v[theta] - exp(model.q[s]) * (self.rewards[s][theta][a]['bs'] * self.c[s] + \
                    -1 * ATTACKER_LOSS_FACTOR * self.rewards[s][theta][a]['es']  * model.h[s]) \
                    >= 0.001

        self.m.best_attacker_constr1 = Constraint(self.m.attacks, rule=best_attacker_u1)

        def best_attacker_u2(model, s, theta, a):
        #s, theta, a = attack
            return model.v[theta] - exp(model.q[s]) * (self.rewards[s][theta][a]['bs'] * self.c[s] \
                    + -1 * ATTACKER_LOSS_FACTOR * self.rewards[s][theta][a]['es']  * model.h[s]) \
                    <= (1 - model.n[(s, theta, a)]) * M + 0.001
        self.m.best_attacker_constr2 = Constraint(self.m.attacks, rule=best_attacker_u2)

        def only_one_action(model, attacker):
            return sum(model.n[(s, theta, a)] for s, theta, a in model.attacks if theta == attacker) == 1

        self.m.only_one_action_constr = Constraint(self.m.attackers, rule=only_one_action)

        def q_rule(model, s):
            return model.q[s] == -1 * log(self.c[s] + model.h[s])
        self.m.q_constr = Constraint(self.m.service_set, rule=q_rule)

    def set_prop_constraints(self):

        def best_attacker_u1(model, s, theta, a):
            #s, theta, a = attack
            return model.v[theta] - exp(model.q[s]) * (self.rewards[s][theta][a]['bs'] * self.c[s] + \
                    -1 * ATTACKER_LOSS_FACTOR * self.rewards[s][theta][a]['es']  * self.prop_h[s]) \
                    >= 0.001

        self.prop_m.best_attacker_constr1 = Constraint(self.prop_m.attacks, rule=best_attacker_u1)

        def best_attacker_u2(model, s, theta, a):
        #s, theta, a = attack
            return model.v[theta] - exp(model.q[s]) * (self.rewards[s][theta][a]['bs'] * self.c[s] \
                    + -1 * ATTACKER_LOSS_FACTOR * self.rewards[s][theta][a]['es']  * self.prop_h[s]) \
                    <= (1 - model.n[(s, theta, a)]) * M + 0.001
        self.prop_m.best_attacker_constr2 = Constraint(self.prop_m.attacks, rule=best_attacker_u2)

        def only_one_action(model, attacker):
            return sum(model.n[(s, theta, a)] for s, theta, a in model.attacks if theta == attacker) == 1

        self.prop_m.only_one_action_constr = Constraint(self.prop_m.attackers, rule=only_one_action)

        def q_rule(model, s):
            return model.q[s] == -1 * log(self.c[s] + self.prop_h[s])
        self.prop_m.q_constr = Constraint(self.prop_m.service_set, rule=q_rule)       
        
    # used to simulate difference between believed attacker probabilities and "real" attacker probabilities
    def real_attacker_obj(self, model, real_attacker_probs):
        return sum(real_attacker_probs[theta] * model.n[(s, theta, a)].value * exp(model.q[s].value) * \
                (self.c[s] * -1 * self.rewards[s][theta][a]['is'] + \
                LOSS_AVERSION_FACTOR * model.h[s].value * self.rewards[s][theta][a]['is']) \
                for s, theta, a in model.attacks)

    def solve(self, tee_p=False):
        self.solver = pyomo.opt.SolverFactory('bonmin', executable=BONMIN_EXEC, validate=False)
        start_time = time.time()
        self.solver.solve(self.m, tee=tee_p)
        end_time = time.time()
        self.duration = end_time - start_time
        
    def solve_prop(self, tee_p=False):
        self.prop_solver = pyomo.opt.SolverFactory('bonmin', executable=BONMIN_EXEC, validate=False)
        self.prop_solver.solve(self.prop_m, tee=tee_p)

    # gets the csv log itself
    def get_log(self):
        return ",".join([
            str(len(self.c)),
            str(self.c),
            # honeypots
            str([self.m.h[i].value for i in range(len(self.m.h))]),
            str(self.duration),
            self.pickle_path if self.pickle_path else "",
            # num vars
            str(sum(sum(len(a) for a in b) for b in self.rewards) + 2 * len(self.rewards) + 3),
            # num constraints
            str(sum(sum(len(a) for a in b) for b in self.rewards) * 2 + 2 * len(self.rewards) + 3),
            str(self.m.OBJ()),
            str(self.prop_m.OBJ())
        ])

    def print_results(self):
        print("Attacks")
        print("(service, attacker, attack), was_selected, CVE Info")
        for n_i in self.m.n:
            if self.m.n[n_i].value > 0:
                s, theta, a = n_i
                print (n_i, self.m.n[n_i].value, self.rewards[s][theta][a])

        print()
        print("Obj: ", self.m.OBJ())

        print()
        print("Service Honeypot Distribution:")
        
# gets the header for the csv log
def get_log_header():
    return ",".join([
        "num_services",
        "c",
        "h",
        "duration",
        "experiment_pickle_path",
        "num_vars",
        "num_constraints",
        "obj",
        "prop_obj"
    ]) + "\n"
       
# experiment running
def run_experiments(experiments, experiments_dir):
    # create dir
    new_dir_name = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime(time.time()))
    os.mkdir(experiments_dir + new_dir_name)
    new_dir_path = os.path.join(experiments_dir, new_dir_name)
    
    # create file
    with open(os.path.join(new_dir_path, "results.csv"), 'w') as f:
        # write header
        f.write(get_log_header())
        for i, e in enumerate(experiments):
            # run experiment
            e.init_vars()
            e.set_obj()
            e.set_prop_obj()
            e.set_constraints()
            e.set_prop_constraints()
            e.solve()
            e.solve_prop()
            # write pickle
            e.pickle_path = os.path.join(new_dir_path, str(i) + ".pickle")
            pickle.dump(e, open(e.pickle_path, 'w'))
            # write line to file
            f.write(e.get_log() + '\n')
            e.print_results()
            sys.stdout.flush()
            f.flush()



rewards = pickle.load(open(REWARDS_FILE, 'rb'))

#c = [0.5 / 2] * 2
# add null option
for i in range(len(rewards)):
    for j in range(len(rewards[i])):
        rewards[i][j].append({'bs': 0, 'cve_id': 'NULL', 'es': 0, 'is': 0})


# Experiment 1
#PROP_LEGIT = 0.5
#experiments = []
#attacker_prob=[0.5, 0.4, 0.1]
#num_services = 3
#for _ in range(30):
    #cuts = [0] + sorted([random.uniform(0, PROP_LEGIT) for _ in range(num_services - 1)]) + [PROP_LEGIT]
    #c = [cuts[i+1] - cuts[i] for i in range(len(cuts) - 1)]
    #experiments.append(Experiment(attacker_prob=attacker_prob, service_prob=c, \
#                                  rewards=random.sample(rewards, num_services)))

# Experiment 2
#experiments = []
#attacker_prob = [0.5, 0.4, 0.1]
#for service_count in [1, 2, 4, 8, 16]:
#    experiments.append(Experiment(attacker_prob=[0.5, 0.4, 0.1], service_prob=[0.5 / service_count] * service_count, rewards=rewards[:service_count]))

experiments = []
cur_service_prob = [0.3, 0.1, 0.1]
attacker_prob = [0.5, 0.4, 0.1]
for modified_attacker_ind in range(len(attacker_prob)):
    for prop_change in np.arange(-1, 1.01, 0.1):
        new_attacker_prob = []
        remaining = sum(attacker_prob) - attacker_prob[modified_attacker_ind]
        for i in range(len(attacker_prob)):
            if i == modified_attacker_ind:
                new_attacker_prob.append((1 + prop_change)  * attacker_prob[i])
            else:
                gain_or_loss = -1 if prop_change > 0 else 1
                new_attacker_prob.append(attacker_prob[i] * (1 - prop_change * attacker_prob[modified_attacker_ind] / remaining))
        print(new_attacker_prob)
        experiments.append(Experiment(attacker_prob=new_attacker_prob, service_prob=cur_service_prob, rewards=rewards[:3]))



run_experiments(experiments, "/home/ashton/828m/game-theory-honeypot/results/")

