import numpy as np
from bearing_dt.qrei.endpoint import endpoint_history


def test_current_threshold_and_sensor_extrema():
    t = np.array([0., .01, .025, .05])
    features = endpoint_history(np.array([20., 60., 90., 112.]), np.array([22., 55., 91., 108.]),
                                np.ones(4)*21, np.array([1., 5., 7., 9.]), np.array([2., 3., 5., 11.]), t)
    assert np.allclose(features['ep_thermal_margin_C'], [88, 50, 19, -2])
    assert np.array_equal(features['ep_vibration_crossings_8'], [0, 0, 0, 1])
    assert features['ep_temperature_running_max_C'][-1] == 112


def test_future_temperature_and_vibration_do_not_change_prefix():
    rng = np.random.default_rng(23)
    hours = np.cumsum(rng.uniform(.002, .03, 90))
    args = [rng.uniform(20, 115, 90), rng.uniform(20, 115, 90), rng.uniform(18, 30, 90),
            rng.uniform(0, 12, 90), rng.uniform(0, 12, 90), hours]
    full = endpoint_history(*args)
    prefix = endpoint_history(*(v[:37] for v in args))
    for key in full:
        assert np.allclose(full[key][:37], prefix[key], atol=1e-12)
    args[0][37:] = 10000
    args[3][37:] = 10000
    changed = endpoint_history(*args)
    for key in full:
        assert np.allclose(full[key][:37], changed[key][:37], atol=1e-12)
