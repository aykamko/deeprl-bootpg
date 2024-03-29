#!/usr/bin/env python3.5
"""
Reference implementation: https://github.com/wojzaremba/trpo
"""

import atexit
import click
import gym
import inspect
import logging
import numpy as np
import os
import pickle
import scipy.signal
import tensorflow as tf
import logz
import threading
import multiprocessing
from os import path as osp
from tqdm import tqdm
from numpy_ringbuffer import RingBuffer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa
import mpld3_custom.mpld3 as mpld3  # noqa
import seaborn as sns  # noqa

logger = logging.getLogger('trpo')
logger.setLevel(logging.DEBUG)


MACHINE_EPS = np.finfo(np.float32).eps
CG_DAMP = 0.1
ROLL_MEM_SIZE = 1000
CLEAR_LOGS = True
EP_AVG_LEN = 5

REFRESH_RATE = -1
SERVER_STARTED = False


class RollingMem:
    def __init__(self, maxlen=ROLL_MEM_SIZE):
        self.maxlen = maxlen
        self.mem = []

    def append(self, val):
        if len(self.mem) < self.maxlen:
            self.mem += [val]
        else:
            self.mem[:-1] = self.mem[1:]
            self.mem[-1] = val

    def mean(self):
        return np.mean(self.mem)

    def std(self):
        return np.std(self.mem)

    def standardize(self, val):
        return (val - self.mean()) / (self.std() + MACHINE_EPS)


def discount(x, gamma):
    """
    Compute discounted sum of future values
    out[i] = in[i] + gamma * in[i+1] + gamma^2 * in[i+2] + ...
    """
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def explained_variance_1d(ypred, y):
    """
    Var[ypred - y] / var[y].
    https://www.quora.com/What-is-the-meaning-proportion-of-variance-explained-in-linear-regression
    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary == 0 else 1 - np.var(y-ypred)/vary


def normc_initializer(std=1.0):
    """
    Initialize array with normalized columns
    """
    def _initializer(shape, dtype=None, partition_info=None):  # pylint: disable=W0613
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


def dense(x, size, name, weight_init=None):
    """
    Dense (fully connected) layer
    """
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init)
    b = tf.get_variable(name + "/b", [size], initializer=tf.zeros_initializer())
    return tf.matmul(x, w) + b


def pathlength(path):
    return len(path["reward"])


def lrelu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)


def sample_bootstrap_heads(rewards):
    heads = len(rewards)
    min_reward, min_head = min((rewards[i], i) for i in range(heads))
    head = np.random.randint(heads+1)
    if head == heads:
        head = min_head
    return head


def flatgrad(loss, tfvars):
    grads = tf.gradients(loss, tfvars)
    flat_grads = [tf.reshape(grad, [np.prod(var.get_shape().as_list())])
                  for (var, grad) in zip(tfvars, grads)]
    return tf.concat(flat_grads, axis=0)


def construct_set_vars_from_flat_op(var_list):
    shapes = [var.get_shape().as_list() for var in var_list]
    total_size = sum(np.prod(shape) for shape in shapes)
    sy_theta = tf.placeholder(shape=[total_size], dtype=tf.float32)

    start = 0
    assign_ops = []
    for shape, var in zip(shapes, var_list):
        size = np.prod(shape)
        assign_ops += [tf.assign(var, tf.reshape(sy_theta[start:start+size], shape))]
        start += size

    return sy_theta, tf.group(*assign_ops)


def linesearch(f, x, fullstep, expected_improve_rate, head_i, smallest=False):
    accept_ratio = 0.1
    max_backtracks = 30
    fval = f(x)

    search_space = enumerate(0.5 ** np.arange(max_backtracks))
    if smallest:
        search_space = reversed(list(search_space))

    for step_i, stepfrac in search_space:
        step = stepfrac * fullstep
        xnew = x + step
        try:
            newfval = f(xnew)
        except Exception:
            print('linesearch stepfrac {} caused except!'.format(stepfrac))
            continue
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        if ratio > accept_ratio and actual_improve > 0:
            print(head_i, 'i/a/e/r/f', step_i, actual_improve, expected_improve, ratio, newfval)
            return xnew, step
    return x, 0


def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)
    for i in range(cg_iters):
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr
        if rdotr < residual_tol:
            break
    return x


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


class LinearValueFunction:
    coef = None

    def __init__(self, ob_dim, **kwargs):
        pass  # noop

    def fit(self, X, y):
        Xp = self.preproc(X)
        A = Xp.T.dot(Xp)
        nfeats = Xp.shape[1]
        A[np.arange(nfeats), np.arange(nfeats)] += 1e-3  # a little ridge regression
        b = Xp.T.dot(y)
        self.coef = np.linalg.solve(A, b)

    def predict(self, X):
        if self.coef is None:
            return np.zeros(X.shape[0])
        else:
            return self.preproc(X).dot(self.coef)

    def preproc(self, X):
        return np.concatenate([np.ones([X.shape[0], 1]), X, np.square(X)/2.0], axis=1)


class NnValueFunction:
    def __init__(self, ob_dim, n_epochs=20, stepsize=1e-3):
        self.ob_dim = ob_dim
        self.n_epochs = n_epochs
        self.stepsize = stepsize
        self.init_tf()

    def init_tf(self):
        self.sy_ob = tf.placeholder(shape=[None, self.ob_dim], name='v_ob', dtype=tf.float32)

        sy_h1 = tf.nn.elu(dense(self.sy_ob, 64, "v_h1", weight_init=normc_initializer(1.0)))
        sy_h2 = tf.nn.elu(dense(sy_h1, 64, "v_h2", weight_init=normc_initializer(1.0)))
        self.sy_value = tf.reshape(dense(sy_h2, 1, 'value', weight_init=normc_initializer(0.1)), [-1])

        self.sy_targ_v = tf.placeholder(shape=[None], name='targ_v', dtype=tf.float32)

        loss = tf.reduce_mean(tf.square(self.sy_value - self.sy_targ_v))

        self.update_op = tf.train.AdamOptimizer(self.stepsize).minimize(loss)

    def fit(self, X, y):
        logger.info('Fitting NNValueFunction')
        for _ in tqdm(range(self.n_epochs)):
            self.update_op.run(feed_dict={
                self.sy_ob: X,
                self.sy_targ_v: y,
            })

    def predict(self, X):
        return np.squeeze(self.sy_value.eval(feed_dict={
            self.sy_ob: X
        }))


def dump_stats(logdir, k, stats):
    os.makedirs(logdir, exist_ok=True)
    statfile_path = osp.join(logdir, 'k{}_{}_stats.pkl'.format(k, str(os.getpid())))
    with open(statfile_path, 'wb') as f:
        pickle.dump(stats, f)

def _main(gym_env, logdir, seed, n_iter, gamma, bootstrap_heads, min_timesteps_per_batch,  # noqa
          initial_stepsize, desired_kl, vf_type, vf_params, animate=False,
          mpld3_start_port=-1, softmax_temp=250):

    tf.set_random_seed(seed)
    np.random.seed(seed)
    env = gym.make(gym_env)
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]
    print("ob_dim is:", ob_dim)
    print("ac_dim is:", ac_dim)

    vfs = []
    for i in range(bootstrap_heads):
        with tf.variable_scope('value_head%d' % i):
            vf = {
                'linear': LinearValueFunction,
                'nn': NnValueFunction,
            }[vf_type](ob_dim, **vf_params)
            vfs += [vf]

    with tf.variable_scope('shared'):
        sy_ob = tf.placeholder(shape=[None, ob_dim], name='ob', dtype=tf.float32)  # batch of observations
        sy_ac = tf.placeholder(shape=[None, ac_dim], name='ac', dtype=tf.float32)  # batch of actions taken by the policy, used for policy gradient computation
        sy_adv = tf.placeholder(shape=[None], name='adv', dtype=tf.float32)  # advantage function estimate

        sy_h1 = lrelu(dense(sy_ob, 64, 'h1', weight_init=normc_initializer(1.0)))
        sy_h2 = lrelu(dense(sy_h1, 64, 'h2', weight_init=normc_initializer(1.0)))

    shared_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='shared')

    sy_sampled_acs = []
    sy_means = []
    sy_logstds = []
    sy_old_means = []
    sy_old_logstds = []
    sy_kls = []
    sy_ents = []
    sy_policy_grads = []
    sy_surrs = []
    sy_flat_tans = []
    sy_fisher_vec_prods = []
    ops_get_vars_flat = []
    ops_set_vars_from_flat = []
    sy_thetas = []

    for i in range(bootstrap_heads):
        with tf.variable_scope('head%d' % i):
            sy_mean = dense(sy_h2, ac_dim, 'mean%d' % i, weight_init=normc_initializer(0.1))  # Mean control output
            sy_logstd = tf.get_variable('logstd%d' % i, [ac_dim], initializer=tf.zeros_initializer())  # Variance
            sy_std = tf.exp(sy_logstd)
            sy_dist = tf.contrib.distributions.Normal(
                name='dist%d' % i, loc=sy_mean, scale=sy_std, validate_args=True)

            sy_old_mean = tf.placeholder(shape=[None, ac_dim], name='oldmean', dtype=tf.float32)
            sy_old_logstd = tf.placeholder(shape=[ac_dim], name='oldlogstd', dtype=tf.float32)
            sy_old_std = tf.exp(sy_old_logstd)
            # sy_old_dist = tf.contrib.distributions.Normal(
            #     name='olddist%d' % i, loc=sy_old_mean, scale=sy_old_std, validate_args=True)
            sy_old_dist = tf.contrib.distributions.Normal(
                name='olddist%d' % i, loc=sy_old_mean, scale=sy_old_std, validate_args=True)

            sy_sampled_ac = sy_dist.sample()

            sy_ac_prob = tf.squeeze(sy_dist.prob(sy_ac))
            sy_old_ac_prob = tf.squeeze(sy_old_dist.prob(sy_ac))
            sy_ac_prob_ratio = sy_ac_prob / (sy_old_ac_prob + MACHINE_EPS)
            if ac_dim > 1:
                sy_ac_prob_ratio = tf.reduce_prod(sy_ac_prob_ratio, axis=1)
            sy_surr = -tf.reduce_mean(sy_ac_prob_ratio * sy_adv)

            # sy_ac_prob = tf.squeeze(sy_dist.prob(sy_ac))
            # sy_ac_safe_prob = tf.maximum(sy_ac_prob, MACHINE_EPS)
            # sy_ac_logprob = tf.log(sy_ac_safe_prob)
            # sy_surr = -tf.reduce_mean(sy_ac_logprob * sy_adv)

            var_list = shared_vars + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=('head%d' % i))

            sy_kl = tf.reduce_mean(tf.contrib.distributions.kl(sy_old_dist, sy_dist))
            sy_ent = tf.reduce_mean(sy_dist.entropy())

            sy_policy_grad = flatgrad(sy_surr, var_list)

            # sy_fixed_dist = tf.contrib.distributions.Normal(
            #     loc=tf.stop_gradient(sy_mean), scale=tf.stop_gradient(sy_std))
            sy_fixed_dist = tf.contrib.distributions.Normal(
                loc=tf.stop_gradient(sy_mean), scale=tf.stop_gradient(sy_std))
            sy_kl_firstfixed = tf.reduce_mean(tf.contrib.distributions.kl(sy_fixed_dist, sy_dist))
            sy_grads = tf.gradients(sy_kl_firstfixed, var_list)

            sy_flat_tan = tf.placeholder(shape=[None], dtype=tf.float32)
            var_shapes = [var.get_shape().as_list() for var in var_list]
            start = 0
            sy_tans = []
            for var_shape in var_shapes:
                size = np.prod(var_shape)
                sy_tans += [tf.reshape(sy_flat_tan[start:start+size], var_shape)]
                start += size

            sy_gvp = [tf.reduce_sum(g * t) for (g, t) in zip(sy_grads, sy_tans)]  # gradient vector product
            sy_fisher_vec_prod = flatgrad(sy_gvp, var_list)

            op_get_vars_flat = tf.concat(
                [tf.reshape(var, [np.prod(var.get_shape().as_list())]) for var in var_list],
                axis=0)
            sy_theta, op_set_vars_from_flat = construct_set_vars_from_flat_op(var_list)

        sy_sampled_acs += [sy_sampled_ac]
        sy_means += [sy_mean]
        sy_logstds += [sy_logstd]
        sy_old_means += [sy_old_mean]
        sy_old_logstds += [sy_old_logstd]
        sy_kls += [sy_kl]
        sy_ents += [sy_ent]
        sy_policy_grads += [sy_policy_grad]
        sy_surrs += [sy_surr]
        sy_flat_tans += [sy_flat_tan]
        sy_fisher_vec_prods += [sy_fisher_vec_prod]
        ops_get_vars_flat += [op_get_vars_flat]
        ops_set_vars_from_flat += [op_set_vars_from_flat]
        sy_thetas += [sy_theta]

    sess = tf.Session()
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.__enter__()  # equivalent to `with sess:`
    tf.global_variables_initializer().run()  # pylint: disable=E1101

    total_timesteps = 0
    total_episodes = 0

    # ----- PLOTTING
    dynamic_html = mpld3.DynamicHTML(None)
    fig, axes = plt.subplots(4, 1, figsize=(7, 8))
    rew_ax, loss_ax, kl_ax, ev_ax = axes
    plt.tight_layout(pad=2.0, h_pad=1.5)

    x_start = 0
    x_end = 50

    def set_xbounds():
        for ax in axes:
            ax.set_xlim(x_start, x_end)
    set_xbounds()

    rew_ax.set_title('Average Reward (Rollout Head)')
    rew_y_by_head = []
    rew_x_by_head = []
    rew_points_by_head = []
    for i in range(bootstrap_heads):
        rew_y, rew_x = [], []
        rew_y_by_head += [rew_y]
        rew_x_by_head += [rew_x]
        rew_points, = rew_ax.plot(rew_x, rew_y, marker='.')
        rew_points_by_head += [rew_points]

    iter_x = []

    loss_ax.set_title('Loss (After train iteration, by head)')
    loss_y_by_head, loss_points_by_head = [], []
    for i in range(bootstrap_heads):
        loss_y = []
        loss_y_by_head += [loss_y]
        loss_points, = loss_ax.plot(iter_x, loss_y)
        loss_points_by_head += [loss_points]

    kl_ax.set_title('KL')
    kl_ax.set_ylim(0, 2*desired_kl)
    kl_y_by_head, kl_points_by_head = [], []
    for i in range(bootstrap_heads):
        kl_y = []
        kl_y_by_head += [kl_y]
        kl_points, = kl_ax.plot(iter_x, kl_y)
        kl_points_by_head += [kl_points]

    ev_ax.set_title('Explained Variance (After update, by head)')
    ev_ax.set_ylim(-1, 1)
    ev_y_by_head, ev_points_by_head = [], []
    for i in range(bootstrap_heads):
        ev_y = []
        ev_y_by_head += [ev_y]
        ev_points, = ev_ax.plot(iter_x, ev_y)
        ev_points_by_head += [ev_points]

    plt.ion()
    # ----- PLOTTING

    args, _, _, values = inspect.getargvalues(inspect.currentframe())
    program_args = {arg: values[arg] for arg in args}

    atexit.register(dump_stats, logdir, bootstrap_heads, {
        'program_args': program_args,
        'rew_x_by_head': rew_x_by_head,
        'rew_y_by_head': rew_y_by_head,
        'iter_x': iter_x,
        'loss_y_by_head': loss_y_by_head,
        'kl_y_by_head': kl_y_by_head,
        'ev_y_by_head': ev_y_by_head,
    })

    ep_avg_history = np.empty(bootstrap_heads, dtype=np.object)
    for i in range(bootstrap_heads):
        ep_avg_history[i] = RingBuffer(capacity=EP_AVG_LEN, dtype=np.int)
    past_ep_avg = np.empty(bootstrap_heads)
    past_ep_avg.fill(np.nan)

    for iter_i in range(n_iter):
        print("********** Iteration %i ************" % iter_i)
        iter_x += [iter_i]

        # Randomly sample a bootstrap head
        for k in range(bootstrap_heads):
            past_ep_avg[k] = np.mean(np.array(ep_avg_history[k]))

        if any(np.isnan(past_ep_avg)):
            bootstrap_i = np.random.randint(bootstrap_heads)
        else:
            print('past_ep_avg:', past_ep_avg)
            past_ep_avg_softmax = softmax(past_ep_avg / softmax_temp)
            print('softmax:', past_ep_avg_softmax)
            bootstrap_i = np.random.choice(bootstrap_heads, p=past_ep_avg_softmax)

        # bootstrap_i = sample_bootstrap_heads(rewards_saved)
        logger.info('Sampled head %d', bootstrap_i)

        sy_sampled_ac = sy_sampled_acs[bootstrap_i]
        sy_mean = sy_means[bootstrap_i]
        sy_logstd = sy_logstds[bootstrap_i]
        rollout_vf = vfs[bootstrap_i]

        # Collect paths until we have enough timesteps
        old_logstd = sy_logstd.eval()
        timesteps_this_batch = 0
        episodes_this_batch = 0
        paths = []
        with tqdm(total=min_timesteps_per_batch) as pbar:
                while True:
                    ob = env.reset().reshape((ob_dim,)); pbar.update(1)  # noqa
                    terminated = False
                    obs, acs, rewards, means = [], [], [], []
                    animate_this_episode = (len(paths) == 0 and (i % 10 == 0) and animate)
                    while True:
                        if animate_this_episode:
                            env.render()

                        obs.append(ob)
                        mean, ac = sess.run([sy_mean, sy_sampled_ac],
                                            feed_dict={sy_ob: ob[None]})

                        means.append(np.squeeze(mean, axis=0))
                        ac = np.squeeze(ac, axis=0)
                        acs.append(ac)
                        ob, rew, done, _ = env.step(ac); pbar.update(1)  # noqa
                        ob = np.squeeze(ob)

                        rewards.append(rew)
                        if done:
                            break

                    path = {"observation": np.array(obs),
                            "terminated": terminated,
                            "reward": np.array(rewards),
                            "action": np.array(acs),
                            "dist_means": np.array(means),
                            "mask": (np.random.rand(bootstrap_heads) > 0.5) if bootstrap_heads > 1 else np.array([True])}
                    paths.append(path)
                    episodes_this_batch += 1
                    timesteps_this_batch += pathlength(path)
                    if timesteps_this_batch > min_timesteps_per_batch:
                        break

        total_episodes += episodes_this_batch
        total_timesteps += timesteps_this_batch

        max_kl = -np.inf
        avg_ent = 0
        for i in range(bootstrap_heads):
            sy_old_mean = sy_old_means[i]
            sy_old_logstd = sy_old_logstds[i]
            sy_kl = sy_kls[i]
            sy_ent = sy_ents[i]
            sy_policy_grad = sy_policy_grads[i]
            sy_flat_tan = sy_flat_tans[i]
            sy_fisher_vec_prod = sy_fisher_vec_prods[i]
            sy_surr = sy_surrs[i]
            op_get_vars_flat = ops_get_vars_flat[i]
            op_set_vars_from_flat = ops_set_vars_from_flat[i]
            sy_theta = sy_thetas[i]
            vf = vfs[i]

            # Estimate advantage function
            vtargs, baselines, advs = [], [], []
            for path in paths:
                if not path["mask"][i]:
                    continue
                vtargs.append(discount(path["reward"], gamma))
                rew_t = path["reward"]
                return_t = discount(rew_t, gamma)
                # TODO: use rollout vf or this head's vf? former seems to do a bit better?
                baseline_t = rollout_vf.predict(path["observation"])
                baselines.append(baseline_t)
                adv_t = return_t - baseline_t
                advs.append(adv_t)

            # Build arrays for policy update
            ob = np.concatenate([path["observation"] for path in paths if path["mask"][i]])
            ac = np.concatenate([path["action"] for path in paths if path["mask"][i]])
            old_means = np.concatenate([path["dist_means"] for path in paths if path["mask"][i]])
            adv = np.concatenate(advs)
            standardized_adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # Fit value function
            vtarg = np.concatenate(vtargs)
            vf.fit(ob, vtarg)  # fit!
            ev_after = explained_variance_1d(np.squeeze(vf.predict(ob)), vtarg)

            feed = {
                sy_ob: ob,
                sy_ac: ac,
                sy_adv: standardized_adv,
                sy_old_mean: old_means,
                sy_old_logstd: old_logstd,
            }

            this_head_old_means = sy_means[i].eval(feed_dict={sy_ob: ob})
            this_head_old_logstd = sy_logstds[i].eval()

            def fisher_vector_product(p):
                feed[sy_flat_tan] = p
                return sy_fisher_vec_prod.eval(feed_dict=feed) + CG_DAMP * p

            theta_old = op_get_vars_flat.eval()
            policy_grad = sy_policy_grad.eval(feed_dict=feed)
            stepdir = conjugate_gradient(fisher_vector_product, -policy_grad)

            fvp = fisher_vector_product(stepdir)
            shs = 0.5 * stepdir.dot(fvp)
            lamb = np.sqrt(shs / desired_kl)
            print(i, 'lamb:', lamb, 'pg_norm:', np.linalg.norm(policy_grad))
            fullstep = stepdir / lamb
            neggdotstepdir = -policy_grad.dot(stepdir)

            def surr_loss(theta):
                op_set_vars_from_flat.run(feed_dict={sy_theta: theta})
                return sess.run(sy_surr, feed_dict=feed)

            theta_new, step_taken = linesearch(
                surr_loss, theta_old, fullstep, neggdotstepdir / lamb, i, smallest=False)

            # measure KL old-new with this heads parameters (not with rollout head params)
            feed[sy_old_mean] = this_head_old_means
            feed[sy_old_logstd] = this_head_old_logstd
            op_set_vars_from_flat.run(feed_dict={sy_theta: theta_new})
            surr_after, kl, ent = sess.run([sy_surr, sy_kl, sy_ent], feed_dict=feed)
            if kl > 2 * desired_kl:
                op_set_vars_from_flat.run(feed_dict={sy_theta: theta_old})
                print(i, 'no update')
            else:
                print(i, 'kl, step norm, step max', kl, np.linalg.norm(step_taken), max(step_taken))

            # ----- PLOTTING
            loss_y = loss_y_by_head[i]
            loss_y += [surr_after]
            loss_points_by_head[i]._y = loss_y  # XXX: bug in matplotlib
            loss_points_by_head[i].set_data(iter_x, loss_y)
            # recompute loss axis
            loss_ax.relim()
            loss_ax.autoscale()
            kl_y = kl_y_by_head[i]
            kl_y += [kl]
            kl_points_by_head[i]._y = kl_y  # XXX: bug in matplotlib
            kl_points_by_head[i].set_data(iter_x, kl_y)
            ev_y = ev_y_by_head[i]
            ev_y += [ev_after]
            ev_points_by_head[i]._y = ev_y  # XXX: bug in matplotlib
            ev_points_by_head[i].set_data(iter_x, ev_y)
            # ----- PLOTTING

            max_kl = max(max_kl, kl)
            avg_ent += ent

        avg_ent /= bootstrap_heads

        # Log diagnostics
        logz.log_tabular("Head", bootstrap_i)
        ep_rew_mean = np.mean([path["reward"].sum() for path in paths])
        ep_avg_history[bootstrap_i].append(ep_rew_mean)
        logz.log_tabular("EpRewMean", ep_rew_mean)
        logz.log_tabular("EpLenMean", np.mean([pathlength(path) for path in paths]))
        logz.log_tabular("KLOldNew", max_kl)
        logz.log_tabular("AvgEntropy", avg_ent)
        # logz.log_tabular("EVBefore", ev_before)
        # logz.log_tabular("EVAfter", ev_after)
        logz.log_tabular("SurrAfter", surr_after)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        logz.log_tabular("EpisodesSoFar", total_episodes)
        logz.log_tabular("BatchTimesteps", timesteps_this_batch)
        logz.log_tabular("BatchEpisodes", episodes_this_batch)
        # If you're overfitting, EVAfter will be way larger than EVBefore.
        # Note that we fit value function AFTER using it to compute the advantage function to avoid introducing bias
        logz.dump_tabular()

        # ------- PLOTTING
        rew_x = rew_x_by_head[bootstrap_i]
        rew_y = rew_y_by_head[bootstrap_i]
        rew_x += [iter_i]
        rew_y += [ep_rew_mean]
        rew_points_by_head[bootstrap_i]._y = rew_y  # XXX: bug in matplotlib
        rew_points_by_head[bootstrap_i].set_data(rew_x, rew_y)
        # recompute loss axis
        rew_ax.relim()
        rew_ax.autoscale()

        if iter_i > 45:
            x_start += 1
            x_end += 1
        set_xbounds()

        if mpld3_start_port < 0:
            plt.pause(0.05)
        else:
            dynamic_html.content = mpld3.fig_to_html(fig)
            global SERVER_STARTED
            if not SERVER_STARTED:
                def d3show():
                    mpld3.show(dynamic=dynamic_html, refresh_rate=REFRESH_RATE, ip='0.0.0.0',
                               port=mpld3_start_port + (bootstrap_heads - 1))
                threading.Thread(target=d3show, args=()).start()
                SERVER_STARTED = True
        # ------- PLOTTING


def _main1(d):
    try:
        _main(**d)
    finally:
        atexit._run_exitfuncs()


@click.command()
@click.option('--env', default='Pendulum-v0')
@click.option('--gamma', '-g', default=0.97)
@click.option('--bootstrap-heads', '-k', default='1')
@click.option('--batch-timesteps', '-t', default=10000)
@click.option('--desired-kl', default=2e-3)
@click.option('--n-iter', default=400)
@click.option('--mpld3-start-port', default=-1)
@click.option('--vf-epochs', default=25)
@click.option('--softmax-temp', default=250)
def main(bootstrap_heads, env, gamma, batch_timesteps, desired_kl, n_iter, mpld3_start_port,
         vf_epochs, softmax_temp):
    bootstrap_heads = [int(k) for k in bootstrap_heads.split(',')]

    general_params = dict(
        animate=False,
        n_iter=n_iter,
        gamma=gamma,
        min_timesteps_per_batch=batch_timesteps,
        vf_params={'n_epochs': vf_epochs},
        vf_type='nn',
        seed=0,
        logdir=osp.join(osp.dirname(osp.abspath(__file__)), 'log'),
        gym_env=env,
        initial_stepsize=1e-3,
        desired_kl=desired_kl,
        mpld3_start_port=mpld3_start_port,
        softmax_temp=softmax_temp,
    )

    params = [dict(bootstrap_heads=k, **general_params) for k in bootstrap_heads]
    p = multiprocessing.Pool(len(bootstrap_heads))
    p.map(_main1, params)


if __name__ == '__main__':
    main()
