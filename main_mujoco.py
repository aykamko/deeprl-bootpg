#!/usr/bin/env python3.5
import numpy as np
import tensorflow as tf
import gym
import logz
import scipy.signal
from os import path as osp
import click


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


class LinearValueFunction:
    coef = None

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


def pendulum(logdir, seed, n_iter, gamma, bootstrap_heads, min_timesteps_per_batch, initial_stepsize, initial_reward,
             desired_kl, vf_type, vf_params, animate=False):
    tf.set_random_seed(seed)
    np.random.seed(seed)
    env = gym.make("HalfCheetah-v1")
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]
    print("ob_dim is ", ob_dim)
    print("ac_dim is ", ac_dim)
    vf = LinearValueFunction()

    sy_ob = tf.placeholder(shape=[None, ob_dim], name='ob', dtype=tf.float32)  # batch of observations
    sy_ac = tf.placeholder(shape=[None, ac_dim], name='ac', dtype=tf.float32)  # batch of actions taken by the policy, used for policy gradient computation
    sy_adv = tf.placeholder(shape=[None], name='adv', dtype=tf.float32)  # advantage function estimate

    sy_h1 = lrelu(dense(sy_ob, 128, 'h1', weight_init=normc_initializer(1.0)))
    sy_h2 = lrelu(dense(sy_h1, 128, 'h2', weight_init=normc_initializer(1.0)))

    sy_sampled_acs = []
    sy_means = []
    sy_logstds = []
    sy_old_means = []
    sy_old_logstds = []
    sy_kls = []
    sy_ents = []
    sy_stepsizes = []
    update_ops = []

    for i in range(bootstrap_heads):
        sy_mean = dense(sy_h2, ac_dim, 'mean_out%d' % i, weight_init=normc_initializer(0.1))  # Mean control output
        sy_logstd = tf.get_variable('logstd%d' % i, [ac_dim], initializer=tf.zeros_initializer())  # Variance
        sy_std = tf.exp(sy_logstd)

        sy_dist = tf.contrib.distributions.Normal(mu=sy_mean, sigma=sy_std, validate_args=True)
        sy_sampled_ac = sy_dist.sample()

        sy_ac_prob = tf.reduce_prod(sy_dist.prob(sy_ac), axis=1)
        sy_safe_ac_prob = tf.maximum(sy_ac_prob, np.finfo(np.float32).eps)  # to avoid log(0) -> nan issues
        sy_ac_logprob = tf.squeeze(tf.log(sy_safe_ac_prob))  # log-prob of actions taken -- used for policy gradient calculation

        sy_old_mean = tf.placeholder(shape=[None, ac_dim], name='old_mean%d' % i, dtype=tf.float32)
        sy_old_logstd = tf.placeholder(shape=[ac_dim], name='old_logstd%d' % i, dtype=tf.float32)
        sy_old_std = tf.exp(sy_old_logstd)

        sy_old_dist = tf.contrib.distributions.Normal(mu=sy_old_mean, sigma=sy_old_std, validate_args=True)

        sy_kl = tf.reduce_mean(tf.contrib.distributions.kl(sy_old_dist, sy_dist, allow_nan=False))
        sy_ent = tf.reduce_mean(sy_dist.entropy())

        sy_surr = -tf.reduce_mean(sy_adv * sy_ac_logprob)  # Loss function that we'll differentiate to get the policy gradient ("surr" is for "surrogate loss")

        sy_stepsize = tf.placeholder(shape=[], dtype=tf.float32)
        update_op = tf.train.AdamOptimizer(sy_stepsize).minimize(sy_surr)

        sy_sampled_acs.append(sy_sampled_ac)
        sy_means.append(sy_mean)
        sy_logstds.append(sy_logstd)
        sy_old_means.append(sy_old_mean)
        sy_old_logstds.append(sy_old_logstd)
        sy_kls.append(sy_kl)
        sy_ents.append(sy_ent)
        sy_stepsizes.append(sy_stepsize)
        update_ops.append(update_op)

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

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        episodes_this_batch = 0
        paths = []
        while True:
            ob = env.reset().reshape((ob_dim,))
            terminated = False
            obs, acs, rewards = [], [], []
            animate_this_episode = (len(paths) == 0 and (i % 10 == 0) and animate)
            while True:
                if animate_this_episode:
                    env.render()
                obs.append(ob)
                ac = sess.run(sy_sampled_ac, feed_dict={sy_ob: ob[None]})
                ac = ac.reshape((ac.shape[1]))
                acs.append(ac)
                ob, rew, done, _ = env.step(ac)
                ob = ob.reshape((ob_dim,))
                rewards.append(rew)
                if done:
                    break

            path = {"observation": np.array(obs),
                    "terminated": terminated,
                    "reward": np.array(rewards),
                    "action": np.array(acs),
                    "mask": (np.random.rand(bootstrap_heads) > 0.5) if bootstrap_heads > 1 else np.array([True])}
            paths.append(path)
            episodes_this_batch += 1
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > min_timesteps_per_batch:
                break

        total_episodes += episodes_this_batch
        total_timesteps += timesteps_this_batch

        # Fit value function
        vtargs, vpreds = [], []
        for path in paths:
            vtargs.append(discount(path["reward"], gamma))
            vpreds.append(vf.predict(path["observation"]))
        ob = np.concatenate([path["observation"] for path in paths])
        vtarg = np.concatenate(vtargs)
        vpred = np.concatenate(vpreds)
        vf.fit(ob, vtarg)

        vtarg = vtarg.reshape((vtarg.shape[0]))
        # print("vpred shape is ",vpred.shape)
        # print("vtarg shape is",vtarg.shape)
        # print("vpred dim is ", vpred.ndim)
        # print("vtarg dim is ", vtarg.ndim)

        ev_before = explained_variance_1d(vpred, vtarg)
        ob = np.concatenate([path["observation"] for path in paths])
        ev_after = explained_variance_1d(vf.predict(ob).reshape((vtarg.shape[0])), vtarg)

        max_kl = -np.inf
        avg_ent = 0
        for i in range(bootstrap_heads):
            update_op = update_ops[i]
            sy_mean = sy_means[i]
            sy_old_mean = sy_old_means[i]
            sy_logstd = sy_logstds[i]
            sy_old_logstd = sy_old_logstds[i]
            sy_kl = sy_kls[i]
            sy_ent = sy_ents[i]
            sy_stepsize = sy_stepsizes[i]

            # Estimate advantage function
            advs = []
            for path in paths:
                if not path["mask"][i]:
                    continue
                rew_t = path["reward"]
                return_t = discount(rew_t, gamma)
                vpred_t = vf.predict(path["observation"])
                adv_t = return_t - vpred_t
                advs.append(adv_t)

            # Build arrays for policy update
            ob = np.concatenate([path["observation"] for path in paths if path["mask"][i]])
            ac = np.concatenate([path["action"] for path in paths if path["mask"][i]])
            adv = np.concatenate(advs)
            standardized_adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # Policy update
            _, old_mean, old_logstd = sess.run([update_op, sy_mean, sy_logstd], feed_dict={
                sy_ob: ob,
                sy_ac: ac,
                sy_adv: standardized_adv,
                sy_stepsize: stepsizes[i]})

            print("head {}, stddev: {}".format(i, old_logstd))
            kl, ent = sess.run([sy_kl, sy_ent], feed_dict={
                sy_ob: ob,
                sy_old_mean: old_mean,
                sy_old_logstd: old_logstd})

            if kl > desired_kl * 2:
                stepsizes[i] /= 1.5
                print('stepsizes[%d] -> %s' % (i, stepsizes[i]))
            elif max_kl < desired_kl / 2:
                stepsizes[i] *= 1.5
                print('stepsizes[%d] -> %s' % (i, stepsizes[i]))
            else:
                print('stepsize OK')

            max_kl = max(max_kl, kl)
            avg_ent += ent

        avg_ent /= bootstrap_heads
        rewards_saved[bootstrap_i] = np.mean([path["reward"].sum() for path in paths])

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
        min_timesteps_per_batch=5000,
        n_iter=1500,
        initial_stepsize=1e-3,
        initial_reward=-650,
        seed=0,
        desired_kl=2e-3,
        vf_type='linear',
        vf_params={}
    )


if __name__ == '__main__':
    main()
