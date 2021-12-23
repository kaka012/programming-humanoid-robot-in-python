'''In this exercise you need to implement inverse kinematics for NAO's legs

* Tasks:
    1. solve inverse kinematics for NAO's legs by using analytical or numerical method.
       You may need documentation of NAO's leg:
       http://doc.aldebaran.com/2-1/family/nao_h21/joints_h21.html
       http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
    2. use the results of inverse kinematics to control NAO's legs (in InverseKinematicsAgent.set_transforms)
       and test your inverse kinematics implementation.
'''


import time
from forward_kinematics import ForwardKinematicsAgent
import numpy as np
import scipy.optimize
import sympy
import pickle

from forward_kinematics import draw

def calc_sym_exprs(chains, err_fun):
    print("Calculating symbolic derivatives...")
    sym_errs = {}
    sym_diffs = {}
    for chain_name in chains.keys():
        print(f"Current chain: {chain_name}")
        angles = sympy.symbols(
            [f"{j}_angle" for j in chains[chain_name]], real=True
        )
        mat = np.array(sympy.symbols("m:4:4")).reshape(4,4)
        print("  Symbolic error function: ", end="", flush=True)
        sym_err = err_fun(angles, chain_name, mat, True)
        sym_errs[chain_name] = sym_err
        print("Done\n  Symbolic derivatives: ", end="", flush=True)
        sym_diffs[chain_name] = [sympy.diff(sym_err, a) for a in angles]
        print("Done")
    return sym_errs, sym_diffs

def compile_sym_exprs(chains, sym_exprs, desc=None):
    print("Compiling symbolic expressions ", end="")
    if desc is not None:
        print(f"({desc}) ...")
    else:
        print("...")
    funs = {}
    for chain_name in chains.keys():
        print(f"Current chain: {chain_name} ... ", end="", flush=True)
        angles = sympy.symbols(
            [f"{j}_angle" for j in chains[chain_name]], real=True
        )
        mat = np.array(sympy.symbols("m:4:4")).reshape(4,4)

        expr = sym_exprs[chain_name]
        funs[chain_name] = sympy.lambdify([angles, mat], expr, modules=["numpy"], cse=True)
        print("Done")
    return funs

def get_fk_funs(chains, err_fun):
    # Note: The compilation of the symbolic expressions also takes quite a bit
    # of time, but caching them isn't possible, as lambdas can't be pickled
    # (without other picklers like "dill").
    try:
        with open("kinematics/fk_err_exprs.pkl", "rb") as f:
            err_exprs, diff_exprs = pickle.load(f)
        print("Loaded cached expressions.")
    except FileNotFoundError:
        err_exprs, diff_exprs = calc_sym_exprs(chains, err_fun)
        print("Caching calculation ...")
        with open("kinematics/fk_err_exprs.pkl", "wb") as f:
            pickle.dump((err_exprs, diff_exprs), f)

    ik_err_funs = compile_sym_exprs(chains, err_exprs, "error functions")
    ik_diff_funs = compile_sym_exprs(chains, diff_exprs, "derivative functions")

    print("Combining functions ...")
    def ik_err_fun(angles, chain, target):
        return ik_err_funs[chain](angles, target)
    def ik_diff_fun(angles, chain, target):
        return ik_diff_funs[chain](angles, target)
    return ik_err_fun, ik_diff_fun

class InverseKinematicsAgent(ForwardKinematicsAgent):
    def __init__(self, *args, **kwargs):
        super(InverseKinematicsAgent, self).__init__(*args, **kwargs)

        # Symbolically calculate the FK function and its derivatives for each
        # chain, and generate code for them, based on self.ik_err_fun.
        # This is not only way faster (10x faster, ~15ms per IK trial), but
        # somehow also seems to be more numerically stable, requiring less
        # trials to find the global minimum (even if the derivative is
        # left unused).
        t0 = time.time()
        self.ik_err_fun, self.ik_diff_fun = get_fk_funs(self.chains, self.ik_err_fun_base)
        print(f"IK preparation done in {time.time() - t0:.3f}s.")

    def ik_err_fun_base(self, angles, chain_name, target, symbolic=False):
        weights = np.array([
            [1, 1, 1, 100],
            [1, 1, 1, 100],
            [1, 1, 1, 100],
            [0, 0, 0, 0],
        ])

        d = {j: 0.0 for j in self.joint_names}
        for i, j in enumerate(self.chains[chain_name]):
            d[j] = angles[i]
        transforms = self.forward_kinematics(d, symbolic)

        d = transforms[self.chains[chain_name][-1]] - target
        d = np.multiply(d, weights)
        return sum([x ** 2 for x in d.flatten()])

    def inverse_kinematics(self, effector_name, transform):
        '''solve the inverse kinematics

        :param str effector_name: name of end effector, e.g. LLeg, RLeg
        :param transform: 4x4 transform matrix
        :return: list of joint angles
        '''
        angles_curr = np.array([self.target_joints.get(j, 0.0) for j in self.chains[effector_name]], np.float32)
        angles_bound = [self.sensor_limits.get(j, (-np.pi, np.pi)) for j in self.chains[effector_name]]

        # minimize() may not convert to the global minimum, but only to a local
        # minimum. In that case it's best to restart the IK search from a
        # slightly different state.
        best_err = float('inf')
        best_x = None
        ik_trials = 10
        for ik_i in range(ik_trials):
            # A tiny bit of randomness is required to escape bad initial states
            angles_init = angles_curr + (np.random.rand(*angles_curr.shape) - 0.5) * 0.01

            m = scipy.optimize.minimize(
                self.ik_err_fun,
                angles_init,
                args = (effector_name, transform),
                jac = self.ik_diff_fun,
                tol=10**-9,
                bounds = angles_bound,
            )
            err = self.ik_err_fun(m.x, effector_name, transform)
            if err < best_err:
                best_err = err
                best_x = m.x
            if best_err < 10**-9:
                return best_x
            print(f"IK stuck (err={err}, best={best_err}). Retrying... ({ik_i + 1}/{ik_trials})")
        print("Trial count for IK exceeded. Returning best solution so far.")
        return m.x

    def set_transforms(self, effector_name, transform, interpolate=False):
        '''solve the inverse kinematics and control joints use the results
        '''
        angles = self.inverse_kinematics(effector_name, transform)
        d = self.sensor_joints
        d.update({j: angles[i] for i, j in enumerate(self.chains[effector_name])})
        self.transforms = self.forward_kinematics(d)

        self.animation_running = False # Force animation restart
        names = []
        times = []
        keys = []
        for j in self.joint_names:
            if j in self.locked_joints:
                continue
            names.append(j)
            key = []
            if interpolate:
                times.append([0.0, 1.0])
                key += [[self.sensor_joints[j], [3, -0.1, 0.0], [3, 0.1, 0.0]]]
            else:
                times.append([0.0])
            if j in self.chains[effector_name]:
                self.target_joints[j] = angles[self.chains[effector_name].index(j)]
            key += [[self.target_joints[j], [3, -0.1, 0.0], [3, 0.1, 0.0]]]
            keys.append(key)
        self.keyframes = (names, times, keys)

if __name__ == '__main__':
    agent = InverseKinematicsAgent()
    # test inverse kinematics
    T = np.identity(4)
    pos = [0, 0.05, -0.33309 + 0.1] # x, y, z
    T[0, -1] = pos[0]
    T[1, -1] = pos[1]
    T[2, -1] = pos[2]
    agent.set_transforms('LLeg', T, True)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1,1,1])
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    draw(0, ax, agent)
    ax.plot(*pos, 'cx')
    plt.show(block=True)

    agent.run()
