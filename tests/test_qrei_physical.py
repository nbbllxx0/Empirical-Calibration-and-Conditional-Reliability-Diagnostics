import numpy as np
import pytest

from bearing_dt.qrei.signal import harmonize_record, physical_features, causal_history, defect_orders


def make_mat(fs, amplitude=0.1, frequency=100.0):
    t=np.arange(int(fs*1.6))/fs
    x=amplitude*np.sin(2*np.pi*frequency*t)
    return {"measTime":t,"accHorizRear_A":x,"accHorizFrontal_C":x*2}


def test_volts_to_g_and_two_channels():
    x,fs=harmonize_record(make_mat(64000))
    assert fs==pytest.approx(64000)
    assert np.sqrt(np.mean(x[0]**2))==pytest.approx(1/np.sqrt(2),rel=1e-4)
    assert np.sqrt(np.mean(x[1]**2))==pytest.approx(np.sqrt(2),rel=1e-4)


def test_resampling_preserves_duration_and_rejects_alias():
    a,_=harmonize_record(make_mat(128000,frequency=1000))
    b,_=harmonize_record(make_mat(128000,frequency=50000))
    assert a.shape==(2,102400)
    assert np.sqrt(np.mean(b**2))<.02*np.sqrt(np.mean(a**2))


def test_no_per_window_amplitude_loss():
    x,_=harmonize_record(make_mat(64000))
    f,w=physical_features(x,1800)
    f2,w2=physical_features(x*3,1800)
    assert f2["A_rms_g"]==pytest.approx(3*f["A_rms_g"])
    assert f2["A_power_0.5_200_g2"]==pytest.approx(9*f["A_power_0.5_200_g2"])
    assert np.allclose(w2,3*w,rtol=1e-5,atol=1e-5)
    assert w.shape==(2,2048)


def test_causal_history_prefix_invariant():
    t=np.arange(12)/120
    values=np.arange(12,dtype=float)+1
    short=causal_history(values[:6],t[:6])
    values[6:]=1e6
    assert np.array_equal(short,causal_history(values,t)[:6])


def test_defect_orders_geometry():
    orders=defect_orders()
    assert orders["BPFO"]+orders["BPFI"]==pytest.approx(19)
    assert orders["BPFO"]==pytest.approx(8.649014,rel=1e-6)


def test_nonuniform_measurement_time_rejected():
    mat=make_mat(64000)
    mat["measTime"][100]+=.01
    with pytest.raises(ValueError):
        harmonize_record(mat)
