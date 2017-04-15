#!/usr/bin/env python3.5
import numpy as np
import tensorflow as tf
import gym
import logz
import scipy.signal
from os import path as osp
import click

MACHINE_EPS = np.finfo(np.float32).eps
CG_DAMP = 0.1


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
    assigns = []
    shapes = [var.get_shape().as_list() for var in var_list]
    total_size = sum(np.prod(shape) for shape in shapes)
    sy_theta = tf.placeholder(shape=[total_size], dtype=tf.float32)

    start = 0
    assigns = []
    for shape, v in zip(shapes, var_list):
        size = np.prod(shape)
        assigns.append(
            tf.assign(v, tf.reshape(sy_theta[start:(start + size)], shape)))
        start += size
        return sy_theta, tf.group(*assigns)


def linesearch(f, x, fullstep, expected_improve_rate):
    accept_ratio = .1
    max_backtracks = 10
    fval = f(x)
    for _n_backtracks, stepfrac in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        newfval = f(xnew)
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        if ratio > accept_ratio and actual_improve > 0:
            return xnew
    return x


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
    def __init__(self, ob_dim, n_epochs=50, stepsize=1e-3):
        self.ob_dim = ob_dim
        self.n_epochs = n_epochs
        self.stepsize = stepsize
        self.init_tf()

    def init_tf(self):
        self.sy_ob = tf.placeholder(shape=[None, self.ob_dim], name='v_ob', dtype=tf.float32)

        sy_h1 = tf.nn.elu(dense(self.sy_ob, 128, "v_h1", weight_init=normc_initializer(1.0)))
        sy_h2 = tf.nn.elu(dense(sy_h1, 128, "v_h2", weight_init=normc_initializer(1.0)))
        self.sy_value = dense(sy_h2, 1, 'value', weight_init=normc_initializer(0.1))

        self.sy_targ_v = tf.placeholder(shape=[None], name='targ_v', dtype=tf.float32)

        loss = tf.reduce_mean(tf.square(self.sy_value - self.sy_targ_v))

        self.update_op = tf.train.AdamOptimizer(self.stepsize).minimize(loss)

    def fit(self, X, y):
        for _ in range(self.n_epochs):
            self.update_op.run(feed_dict={
                self.sy_ob: X,
                self.sy_targ_v: y,
            })

    def predict(self, X):
        return np.squeeze(self.sy_value.eval(feed_dict={
            self.sy_ob: X
        }))


def pendulum(logdir, seed, n_iter, gamma, bootstrap_heads, min_timesteps_per_batch, initial_stepsize, initial_reward,
             desired_kl, vf_type, vf_params, animate=False):
    tf.set_random_seed(seed)
    np.random.seed(seed)
    env = gym.make("HalfCheetah-v1")
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]
    print("ob_dim is ", ob_dim)
    print("ac_dim is ", ac_dim)
    vf = {
        'linear': LinearValueFunction,
        'nn': NnValueFunction,
    }(ob_dim, **vf_params)

    with tf.variable_scope('shared'):
        sy_ob = tf.placeholder(shape=[None, ob_dim], name='ob', dtype=tf.float32)  # batch of observations
        sy_ac = tf.placeholder(shape=[None, ac_dim], name='ac', dtype=tf.float32)  # batch of actions taken by the policy, used for policy gradient computation
        sy_adv = tf.placeholder(shape=[None], name='adv', dtype=tf.float32)  # advantage function estimate

        sy_h1 = lrelu(dense(sy_ob, 128, 'h1', weight_init=normc_initializer(1.0)))
        sy_h2 = lrelu(dense(sy_h1, 128, 'h2', weight_init=normc_initializer(1.0)))

    shared_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='shared')

    sy_sampled_acs = []
    sy_means = []
    sy_logstds = []
    sy_old_means = []
    sy_old_logstds = []
    sy_kls = []
    sy_ents = []
    sy_surrs = []
    sy_flat_tans = []
    sy_flat_gvps = []
    ops_get_vars_flat = []
    ops_set_vars_from_flat = []
    sy_thetas = []

    for i in range(bootstrap_heads):
        with tf.variable_scope('head%d' % i):
            sy_mean = dense(sy_h2, ac_dim, 'mean%d' % i, weight_init=normc_initializer(0.1))  # Mean control output
            sy_logstd = tf.get_variable('logstd%d' % i, [ac_dim], initializer=tf.zeros_initializer())  # Variance
            sy_std = tf.exp(sy_logstd)
            sy_dist = tf.contrib.distributions.Normal(mu=sy_mean, sigma=sy_std, validate_args=True)

            sy_old_mean = tf.placeholder(shape=[None, ac_dim], name='old_mean%d' % i, dtype=tf.float32)
            sy_old_logstd = tf.placeholder(shape=[ac_dim], name='old_logstd%d' % i, dtype=tf.float32)
            sy_old_std = tf.exp(sy_old_logstd)
            sy_old_dist = tf.contrib.distributions.Normal(mu=sy_old_mean, sigma=sy_old_std, validate_args=True)

            sy_sampled_ac = sy_dist.sample()

            sy_ac_prob = tf.reduce_prod(sy_dist.prob(sy_ac), axis=1)
            sy_safe_ac_prob = tf.maximum(sy_ac_prob, MACHINE_EPS)  # to avoid log(0) -> nan issues
            sy_ac_logprob = tf.squeeze(tf.log(sy_safe_ac_prob))  # log-prob of actions taken -- used for policy gradient calculation

        var_list = shared_vars + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=('head%d' % i))

        sy_kl = tf.reduce_mean(tf.contrib.distributions.kl(sy_old_dist, sy_dist, allow_nan=False))
        sy_ent = tf.reduce_mean(sy_dist.entropy())

        sy_surr = -tf.reduce_mean(sy_ac_logprob * sy_adv)

        sy_pg = flatgrad(sy_surr, var_list)

        sy_fixed_dist = tf.contrib.distributions.Normal(
            mu=tf.stop_gradient(sy_mean),
            sigma=tf.stop_gradient(sy_std),
            validate_args=True)
        sy_kl_firstfixed = tf.reduce_mean(tf.contrib.distributions.kl(sy_fixed_dist, sy_dist, allow_nan=False))
        sy_grads = tf.gradients(sy_kl_firstfixed, var_list)

        sy_flat_tan = tf.placeholder(shape=[None], dtype=tf.float32)
        var_shapes = [var.get_shape().as_list() for var in var_list]
        start = 0
        sy_tans = []
        for var_shape in var_shapes:
            size = np.prod(var_shape)
            sy_tan = tf.reshape(sy_flat_tan[start:(start + size)], var_shape)
            sy_tans.append(sy_tan)
            start += size

        sy_gvp = [tf.reduce_sum(g * t) for (g, t) in zip(sy_grads, sy_tans)]  # gradient vector product
        sy_flat_gvp = flatgrad(sy_gvp, var_list)

        op_get_vars_flat = tf.concat(
            [tf.reshape(v, [np.prod(v.get_shape().as_list())]) for v in var_list],
            axis=0)
        sy_theta, op_set_vars_from_flat = construct_set_vars_from_flat_op(var_list)

        # sy_stepsize = tf.placeholder(shape=[], dtype=tf.float32)
        # update_op = tf.train.AdamOptimizer(sy_stepsize).minimize(sy_surr)

        sy_sampled_acs.append(sy_sampled_ac)
        sy_means.append(sy_mean)
        sy_logstds.append(sy_logstd)
        sy_old_means.append(sy_old_mean)
        sy_old_logstds.append(sy_old_logstd)
        sy_kls.append(sy_kl)
        sy_ents.append(sy_ent)
        sy_surrs += [sy_surr]
        sy_flat_tans += [sy_flat_tan]
        sy_flat_gvps += [sy_flat_gvp]
        ops_get_vars_flat += [op_get_vars_flat]
        ops_set_vars_from_flat += [op_set_vars_from_flat]
        sy_thetas += [sy_theta]

    sess = tf.Session()
    sess.__enter__()  # equivalent to `with sess:`
    tf.global_variables_initializer().run()  # pylint: disable=E1101

    total_timesteps = 0
    total_episodes = 0
    stepsizes = [initial_stepsize for _ in range(bootstrap_heads)]

    rewards_saved = [initial_reward for _ in range(bootstrap_heads)]

    for i in range(n_iter):
        print("********** Iteration %i ************" % i)

        # Randomly sample a bootstrap head
        # bootstrap_i = np.random.randint(bootstrap_heads)
        bootstrap_i = sample_bootstrap_heads(rewards_saved)
        sy_sampled_ac = sy_sampled_acs[bootstrap_i]
        sy_mean = sy_means[bootstrap_i]
        sy_logstd = sy_logstds[bootstrap_i]

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        episodes_this_batch = 0
        paths = []
        while True:
            ob = env.reset().reshape((ob_dim,))
            terminated = False
            logstd = None
            obs, acs, rewards, means = [], [], [], []
            animate_this_episode = (len(paths) == 0 and (i % 10 == 0) and animate)
            while True:
                if animate_this_episode:
                    env.render()
                obs.append(ob)
                mean, logstd, ac = sess.run([sy_mean, sy_logstd, sy_sampled_ac],
                                            feed_dict={sy_ob: ob[None]})
                means.append(np.squeeze(mean))
                ac = np.squeeze(ac)
                acs.append(ac)
                ob, rew, done, _ = env.step(ac)
                ob = np.squeeze(ob)
                rewards.append(rew)
                if done:
                    break

            path = {"observation": np.array(obs),
                    "terminated": terminated,
                    "reward": np.array(rewards),
                    "action": np.array(acs),
                    "dist_means": np.array(means),
                    "dist_logstd": logstd,
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
            sy_mean = sy_means[i]
            sy_old_mean = sy_old_means[i]
            sy_logstd = sy_logstds[i]
            sy_old_logstd = sy_old_logstds[i]
            sy_kl = sy_kls[i]
            sy_ent = sy_ents[i]
            sy_flat_tan = sy_flat_tans[i]
            sy_flat_gvp = sy_flat_gvps[i]
            sy_surr = sy_surrs[i]
            op_get_vars_flat = ops_get_vars_flat[i]
            op_set_vars_from_flat = ops_set_vars_from_flat[i]
            sy_theta = sy_thetas[i]

            # Estimate advantage function
            advs = []
            for path in paths:
                if not path["mask"][i]:
                    continue
                rew_t = path["reward"]
                return_t = discount(rew_t, gamma)
                baseline_t = vf.predict(path["observation"])
                adv_t = return_t - baseline_t
                advs.append(adv_t)

            # Build arrays for policy update
            ob = np.concatenate([path["observation"] for path in paths if path["mask"][i]])
            ac = np.concatenate([path["action"] for path in paths if path["mask"][i]])
            old_means = np.concatenate([path["dist_means"] for path in paths if path["mask"][i]])
            old_logstd = paths[0]["dist_logstd"]  # TODO: ugly
            adv = np.concatenate(advs)
            standardized_adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            feed = {
                sy_ob: ob,
                sy_ac: ac,
                sy_old_mean: old_means,
                sy_old_logstd: old_logstd,
                sy_adv: standardized_adv,
            }

            def fisher_vector_product(p):
                feed[sy_flat_tan] = p
                return sess.run(sy_flat_gvp, feed) + CG_DAMP * p

            theta_old = op_get_vars_flat.eval()
            pg = sy_pg.eval(feed_dict=feed)
            stepdir = conjugate_gradient(fisher_vector_product, -pg)
            shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
            lm = np.sqrt(shs / desired_kl)
            fullstep = stepdir / lm
            neggdotstepdir = -pg.dot(stepdir)

            def loss(theta):
                op_set_vars_from_flat.run(feed_dict={sy_theta: theta})
                return sess.run(sy_surr, feed_dict=feed)

            theta = linesearch(loss, theta_old, fullstep, neggdotstepdir / lm)
            op_set_vars_from_flat.run(feed_dict={sy_theta: theta})

            lossafter, kl, ent = sess.run([sy_surr, sy_kl, sy_ent], feed_dict=feed)

            if kl > 2 * desired_kl:  # TODO: ???
                op_set_vars_from_flat.run(feed_dict={sy_theta: theta_old})

            max_kl = max(max_kl, kl)
            avg_ent += ent

        avg_ent /= bootstrap_heads
        rewards_saved[bootstrap_i] = np.mean([path["reward"].sum() for path in paths])

        # Fit value function
        vtargs, vpreds = [], []
        for path in paths:
            vtargs.append(discount(path["reward"], gamma))
            vpreds.append(vf.predict(path["observation"]))
        ob = np.concatenate([path["observation"] for path in paths])
        vtarg = np.concatenate(vtargs)
        vpred = np.concatenate(vpreds)
        vf.fit(ob, vtarg)  # fit!
        ev_before = explained_variance_1d(vpred, vtarg)
        ob = np.concatenate([path["observation"] for path in paths])
        ev_after = explained_variance_1d(np.squeeze(vf.predict(ob)), vtarg)

        # Log diagnostics
        logz.log_tabular("Head", bootstrap_i)
        logz.log_tabular("EpRewMean", np.mean([path["reward"].sum() for path in paths]))
        logz.log_tabular("EpLenMean", np.mean([pathlength(path) for path in paths]))
        logz.log_tabular("MaxKLOldNew", max_kl)
        logz.log_tabular("AvgEntropy", avg_ent)
        logz.log_tabular("EVBefore", ev_before)
        logz.log_tabular("EVAfter", ev_after)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        logz.log_tabular("EpisodesSoFar", total_episodes)
        logz.log_tabular("BatchTimesteps", timesteps_this_batch)
        logz.log_tabular("BatchEpisodes", episodes_this_batch)
        # If you're overfitting, EVAfter will be way larger than EVBefore.
        # Note that we fit value function AFTER using it to compute the advantage function to avoid introducing bias
        logz.dump_tabular()


@click.command()
@click.option('-k', default=1)
def main(k):
    pendulum(
        logdir=osp.join(osp.abspath(__file__), '/log'),
        animate=False,
        gamma=0.99,
        bootstrap_heads=k,
        # min_timesteps_per_batch=20000,
        min_timesteps_per_batch=25000,
        n_iter=1500,
        initial_stepsize=1e-3,
        initial_reward=-650,
        seed=0,
        # desired_kl=2e-3,
        desired_kl=0.01,
        vf_type='nn',
        vf_params={}
    )


if __name__ == '__main__':
    main()
