import logging
import timeit

from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
import jax
import gpjax as gpx

import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp
import tqdm

log = logging.getLogger(__name__)

def sll(q, Dtest, Dtrain):
    m, S = q.mean(), q.covariance().diagonal()
    loss_model = (0.5 * jnp.log(2 * jnp.pi * S) + (Dtest.y.flatten() - m)**2 / (2 * S)).mean()
    res = loss_model
    data_mean = Dtrain.y.mean()
    data_var = Dtrain.y.var()
    loss_trivial_model = (
        0.5 * jnp.log(2 * jnp.pi * data_var) + (Dtest.y.flatten() - data_mean)**2 / (2 * data_var)
    ).mean()
    res = res - loss_trivial_model
    return res

def nlpd(q, Dtest, likelihood):
    return jnp.mean(-likelihood.link_function(q.mean()).log_prob(Dtest.y.flatten()))

def kl(q1, q2, likelihood):
    m1, S1 = q1.mean(), q1.covariance()
    m2, S2 = q2.mean(), q2.covariance()
    k = m1.shape[0]
    R1,_ = jsp.linalg.cho_factor(S1 + jnp.identity(k)*likelihood.obs_stddev**2, lower=True)
    R2,_ = jsp.linalg.cho_factor(S2 + jnp.identity(k)*likelihood.obs_stddev**2, lower=True)
    tr_term = jnp.trace(jsp.linalg.cho_solve((R2, True), R1))
    log_det_term = 2 * jnp.sum(jnp.log(R2.diagonal())) - 2 * jnp.sum(jnp.log(R1.diagonal()))
    diff = m1 - m2
    quad_term = diff @ jsp.linalg.cho_solve((R2, True), diff)
    return 1/2 * (tr_term - k + quad_term + log_det_term)

def rmse(y1, y2):
    return jnp.sqrt(jnp.mean((y1.mean() - y2.mean())**2))

def eval_data(alg, data, cfg):
    log.parent.disabled = True
    train_data, test_data = data
    # Vanilla GP
    log.info("Running standard GP")
    gp = alg.prior * alg.likelihood
    gp_yhat = gp.predict(test_data.X, train_data)
    # Update w/ data
    alg = alg.update_with_batch(train_data)
    # Standard prediction
    log.info("Running standard HGP")
    yhat, q = alg.predict(test_data.X)
    t = timeit.repeat('alg.predict(test_data.X)',
                      repeat=5,
                      number=5,
                      globals=locals())
    name = type(alg).__name__
    result = {
        name : dict(
        kl = kl(gp_yhat, yhat, alg.likelihood),
        relative_kl = 0.,
        nlpd = nlpd(yhat, test_data, alg.likelihood),
        rmse = rmse(gp_yhat, yhat),
        relative_rmse = 0.,
        time_min = np.min(t),
        time_max = np.max(t),
        m = q.mean().shape[0],
    )}
    log.info("Running approximate HGP")
    aname = 'A' + name
    # Approximate timings
    for fraction in tqdm.tqdm(cfg.fractions, desc="Fraction: "):
        alg = alg.replace(approximate_selector=alg.approximate_selector.replace(fraction=fraction))
        approx_yhat, approx_q = alg.predict(test_data.X, approx=True)
        approx_t = timeit.repeat('alg.predict(test_data.X, approx=True)',
                                repeat=5,
                                number=5,
                                globals=locals())
        if aname not in result.keys():
            result[aname] = dict(
                kl = [kl(gp_yhat, approx_yhat, alg.likelihood)], # KL towards standard GP
                relative_kl = [kl(yhat, approx_yhat, alg.likelihood)], # KL towards HGP
                nlpd = [nlpd(approx_yhat, test_data, alg.likelihood)],
                rmse = [rmse(gp_yhat, approx_yhat)], # RMSE towards standard GP
                relative_rmse = [rmse(yhat, approx_yhat)], # RMSE towards HGP
                time_min = [np.min(approx_t)],
                time_max = [np.max(approx_t)],
                m = [approx_q.mean().shape[0]],
            )
        else:
            result[aname]["kl"].append(kl(gp_yhat, approx_yhat, alg.likelihood))
            result[aname]["relative_kl"].append(kl(yhat, approx_yhat, alg.likelihood))
            result[aname]["nlpd"].append(nlpd(approx_yhat, test_data, alg.likelihood))
            result[aname]["rmse"].append(rmse(gp_yhat, approx_yhat))
            result[aname]["relative_rmse"].append(rmse(yhat, approx_yhat))
            result[aname]["time_min"].append(np.min(approx_t))
            result[aname]["time_max"].append(np.max(approx_t))
            result[aname]["m"].append(approx_q.mean().shape[0])
    result[aname]["fraction"] = list(cfg.fractions)
    return result

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    jax.config.update("jax_enable_x64", True)
    log.info("Instantiating objects")
    data_generator = instantiate(cfg.data_generator)
    alg = instantiate(cfg.alg)
    log.info("Generating data")
    D = data_generator()
    log.info("Data generated!")
    res = eval_data(alg, D, cfg)
    log.info("Evaluation complete")
    log.info("Saving data and quitting")
    name = type(alg).__name__
    aname = 'A' + name
    np.savez('result.npz',
             **res[name])
    np.savez('aresult.npz',
             **res[aname])

if __name__ == "__main__":
    main()
