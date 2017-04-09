import numpy as np
import tensorflow as tf
import gym
import logz
import scipy.signal
import sklearn.utils
from os import path as osp
# import click


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
    b = tf.get_variable(name + "/b", [size], initializer=tf.zeros_initializer)
    return tf.matmul(x, w) + b


def pathlength(path):
    return len(path["reward"])


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


def main(logdir, seed, n_iter, gamma, bootstrap_heads, min_timesteps_per_batch, initial_stepsize,
         desired_kl, vf_type, vf_params, animate=False):
    tf.set_random_seed(seed)
    np.random.seed(seed)
    env = gym.make("Pendulum-v0")
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]
    # logz.configure_output_dir(logdir, CLEAR_LOGS)
    vf = LinearValueFunction()

    sy_ob = tf.placeholder(shape=[None, ob_dim], name='ob', dtype=tf.float32)  # batch of observations
    sy_ac = tf.placeholder(shape=[None, ac_dim], name='ac', dtype=tf.float32)  # batch of actions taken by the policy, used for policy gradient computation
    sy_adv = tf.placeholder(shape=[None], name='adv', dtype=tf.float32)  # advantage function estimate

    sy_h1 = tf.nn.elu(dense(sy_ob, 128, 'h1', weight_init=normc_initializer(1.0)))
    sy_h2 = tf.nn.elu(dense(sy_h1, 128, 'h2', weight_init=normc_initializer(1.0)))

    sy_stepsize = tf.placeholder(shape=[], dtype=tf.float32)

    sy_sampled_acs = []
    sy_means = []
    sy_logstds = []
    sy_old_means = []
    sy_old_logstds = []
    sy_kls = []
    sy_ents = []
    update_ops = []
    for i in range(bootstrap_heads):
        sy_mean = dense(sy_h2, ac_dim, 'mean_out%d' % i, weight_init=normc_initializer(0.1))  # Mean control output
        sy_logstd = tf.get_variable('logstd%d' % i, [ac_dim], initializer=tf.zeros_initializer)  # Variance
        sy_std = tf.exp(sy_logstd)

        sy_dist = tf.contrib.distributions.Normal(mu=sy_mean, sigma=sy_std, validate_args=True)
        sy_sampled_ac = sy_dist.sample(ac_dim)[0, :, 0]
        sy_ac_logprob = tf.squeeze(tf.log(sy_dist.prob(sy_ac)))  # log-prob of actions taken -- used for policy gradient calculation

        sy_old_mean = tf.placeholder(shape=[None, ac_dim], name='old_mean%d' % i, dtype=tf.float32)
        sy_old_logstd = tf.placeholder(shape=[ac_dim], name='old_logstd%d' % i, dtype=tf.float32)
        sy_old_std = tf.exp(sy_old_logstd)
        sy_old_dist = tf.contrib.distributions.Normal(mu=sy_old_mean, sigma=sy_old_std, validate_args=True)

        sy_kl = tf.reduce_mean(tf.contrib.distributions.kl(sy_old_dist, sy_dist, allow_nan=False))
        sy_ent = tf.reduce_mean(sy_dist.entropy())

        sy_surr = -tf.reduce_mean(sy_adv * sy_ac_logprob)  # Loss function that we'll differentiate to get the policy gradient ("surr" is for "surrogate loss")

        update_op = tf.train.AdamOptimizer(sy_stepsize).minimize(sy_surr)

        sy_sampled_acs.append(sy_sampled_ac)
        sy_means.append(sy_mean)
        sy_logstds.append(sy_logstd)
        sy_old_means.append(sy_old_mean)
        sy_old_logstds.append(sy_old_logstd)
        sy_kls.append(sy_kl)
        sy_ents.append(sy_ent)
        update_ops.append(update_op)

    sess = tf.Session()
    sess.__enter__()  # equivalent to `with sess:`
    tf.global_variables_initializer().run()  # pylint: disable=E1101

    total_timesteps = 0
    total_episodes = 0
    stepsize = initial_stepsize

    for i in range(n_iter):
        print("********** Iteration %i ************" % i)

        # Randomly sample a bootstrap head
        bootstrap_i = np.random.randint(bootstrap_heads)
        sy_sampled_ac = sy_sampled_acs[bootstrap_i]

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        episodes_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            terminated = False
            obs, acs, rewards = [], [], []
            animate_this_episode = (len(paths) == 0 and (i % 10 == 0) and animate)
            while True:
                if animate_this_episode:
                    env.render()
                obs.append(ob)
                ac = sess.run(sy_sampled_ac, feed_dict={sy_ob: ob[None]})
                acs.append(ac)
                ob, rew, done, _ = env.step(ac)
                rewards.append(rew)
                if done:
                    break

            path = {"observation": np.array(obs),
                    "terminated": terminated,
                    "reward": np.array(rewards),
                    "action": np.array(acs)}
            paths.append(path)
            episodes_this_batch += 1
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > min_timesteps_per_batch:
                break

        total_episodes += episodes_this_batch
        total_timesteps += timesteps_this_batch

        avg_kl = 0
        avg_ent = 0
        avg_ev_before = 0
        avg_ev_after = 0
        adjusted_stepsize = stepsize / bootstrap_heads
        for i in range(bootstrap_heads):
            bootstrapped_paths = sklearn.utils.resample(paths)
            update_op = update_ops[i]
            sy_mean = sy_means[i]
            sy_old_mean = sy_old_means[i]
            sy_logstd = sy_logstds[i]
            sy_old_logstd = sy_old_logstds[i]
            sy_kl = sy_kls[i]
            sy_ent = sy_ents[i]

            # Estimate advantage function
            vtargs, vpreds, advs = [], [], []
            for path in bootstrapped_paths:
                rew_t = path["reward"]
                return_t = discount(rew_t, gamma)
                vpred_t = vf.predict(path["observation"])
                adv_t = return_t - vpred_t
                advs.append(adv_t)
                vtargs.append(return_t)
                vpreds.append(vpred_t)

            # Build arrays for policy update
            ob = np.concatenate([path["observation"] for path in bootstrapped_paths])
            ac = np.concatenate([path["action"] for path in bootstrapped_paths])
            adv = np.concatenate(advs)
            standardized_adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            vtarg = np.concatenate(vtargs)
            vpred = np.concatenate(vpreds)
            vf.fit(ob, vtarg)

            # Policy update
            _, old_mean, old_logstd = sess.run([update_op, sy_mean, sy_logstd], feed_dict={
                sy_ob: ob,
                sy_ac: ac,
                sy_adv: standardized_adv,
                sy_stepsize: adjusted_stepsize})
            kl, ent = sess.run([sy_kl, sy_ent], feed_dict={
                sy_ob: ob,
                sy_old_mean: old_mean,
                sy_old_logstd: old_logstd})

            avg_kl += kl
            avg_ent += ent
            avg_ev_before += explained_variance_1d(vpred, vtarg)
            avg_ev_after += explained_variance_1d(vf.predict(ob), vtarg)

        avg_kl /= bootstrap_heads
        avg_ent /= bootstrap_heads
        avg_ev_before /= bootstrap_heads
        avg_ev_after /= bootstrap_heads

        if avg_kl > desired_kl * 2:
            stepsize /= 1.5
            print('stepsize -> %s' % stepsize)
        elif avg_kl < desired_kl / 2:
            stepsize *= 1.5
            print('stepsize -> %s' % stepsize)
        else:
            print('stepsize OK')

        # Log diagnostics
        logz.log_tabular("EpRewMean", np.mean([path["reward"].sum() for path in paths]))
        logz.log_tabular("EpLenMean", np.mean([pathlength(path) for path in paths]))
        logz.log_tabular("AvgKLOldNew", avg_kl)
        logz.log_tabular("AvgEntropy", avg_ent)
        logz.log_tabular("AvgEVBefore", avg_ev_before)
        logz.log_tabular("AvgEVAfter", avg_ev_after)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        logz.log_tabular("EpisodesSoFar", total_episodes)
        logz.log_tabular("BatchTimesteps", timesteps_this_batch)
        logz.log_tabular("BatchEpisodes", episodes_this_batch)
        # If you're overfitting, EVAfter will be way larger than EVBefore.
        # Note that we fit value function AFTER using it to compute the advantage function to avoid introducing bias
        logz.dump_tabular()


if __name__ == '__main__':
    main(
        logdir=osp.join(osp.abspath(__file__), '/log'),
        animate=True,
        gamma=0.97,
        bootstrap_heads=5,
        min_timesteps_per_batch=2500,
        n_iter=300,
        initial_stepsize=1e-3,
        seed=0,
        desired_kl=2e-3,
        vf_type='linear',
        vf_params={}
    )
