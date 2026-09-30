"""Publication figures with observed targets, empirical intervals and controls."""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .signal import harmonize_record, defect_orders
from .summarize import NAMES


def main():
    root=Path("QREI submission/results/endpoint_v3")
    out=root/"figures"
    p=pd.read_csv(root/"joined_predictions.csv")
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    def save(fig,name):
        fig.savefig(out/(name+".pdf"),bbox_inches="tight")
        fig.savefig(out/(name+".png"),bbox_inches="tight",dpi=300)
        plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for ax,b in zip(axs.flat,("B02","B03","B10","B17")):
        g=p[(p.bearing_id==b)&(p.model=="representation")].sort_values("elapsed_hours")
        ax.fill_between(g.elapsed_hours,g.lower_hours,g.upper_hours,color="#2962a3",alpha=.15,label="Empirical 90% interval")
        ax.plot(g.elapsed_hours,g.pred_hours,color="#2962a3",lw=1,label="Latent-state forecast")
        ax.plot(g.elapsed_hours,g.rul_hours,color="black",lw=1.7,label="Observed residual time")
        h=p[(p.bearing_id==b)&(p.model=="competing_stop")].sort_values("elapsed_hours")
        ax.plot(h.elapsed_hours,h.pred_hours,color="#c56f20",lw=1,label="Competing-stop forecast")
        h=p[(p.bearing_id==b)&(p.model=="time_only")].sort_values("elapsed_hours")
        ax.plot(h.elapsed_hours,h.pred_hours,color="#8a465d",lw=1,ls="--",label="Median-life clock")
        ax.set_title(b+ (" (thermal stop)" if b=="B03" else " (vibration stop)"))
        ax.set_xlabel("Elapsed hours");ax.set_ylabel("Remaining hours")
    axs[0,0].legend(fontsize=7)
    save(fig,"absolute_forecasts")
    d=pd.read_csv(root/"maintenance_decisions.csv")
    fig,axs=plt.subplots(1,2,figsize=(11,4.7),constrained_layout=True)
    names=("representation","attention","competing_stop","competing_threshold","time_only","always_action","never_action")
    palette=plt.get_cmap("tab10")
    for ax,policy in zip(axs,("point","lower_bound")):
        for j,name in enumerate(names):
            g=d[(d.model==name)&(d.required_lead_hours==1)&d.policy.isin([policy,"control"])]
            curve=g.groupby("failure_cost_ratio").loss.mean()
            ax.plot(curve.index,curve.values,marker="o",ms=4,color=palette(j),label=NAMES.get(name,name.replace("_"," ")))
        ax.set_title("Point trigger" if policy=="point" else "Lower interval trigger")
        ax.set_xlabel("Missed-action penalty");ax.set_ylabel("Mean retrospective loss")
    axs[1].legend(fontsize=7,loc="upper left")
    save(fig,"policy_comparison")
    features=pd.read_csv("data/processed/phme_tvoc_10b_v2/features.csv")
    fig,axs=plt.subplots(2,2,figsize=(6.6,4.7),constrained_layout=True)
    choices=[]
    for i,b in enumerate(("B02","B10")):
        g=features[features.bearing_id==b]
        # Use the existing 1,800 rpm regime cut point and require the strongest 3-80 Hz vibration line at the set
        # shaft frequency; this excludes idle records, including two B10 records taken while the shaft stood still.
        shaft_hz=g.speed_rpm/60
        running=g[(g.speed_rpm>=1800)&(np.abs(g.spectral_peak_3_80_Hz-shaft_hz)<=np.maximum(1.25,.03*shaft_hz))]
        assert len(running),b
        early=running.iloc[np.argmin(np.abs(running.elapsed_hours.to_numpy()-.25))]
        late=g.iloc[-2]
        for row,label,color in ((early,"Early running acquisition","#2962a3"),(late,"Penultimate acquisition","#c56f20")):
            record=int(row.record_id)
            folder=Path("data/raw/phme_tvoc")/b/b/"vibrationData"
            raw=next(f for f in folder.glob("*.mat") if int(re.search(r"_M(\d+)",f.stem)[1])==record)
            x,fs=harmonize_record(loadmat(raw))
            sos=signal.butter(4,[6000,10000],fs=64000,btype="bandpass",output="sos")
            env=np.abs(signal.hilbert(signal.sosfiltfilt(sos,x,axis=1),axis=1))
            f,psd=signal.periodogram(env-env.mean(axis=1,keepdims=True),fs=64000,window="hann",axis=1,scaling="density")
            rotation=row.speed_rpm/60
            assert rotation>0,(b,record)
            orders=f/rotation
            for j,ax in enumerate(axs[i]):
                keep=(orders>=.2)&(orders<=34)
                assert keep.sum()>1,(b,record,'No spectrum in displayed order range')
                ax.semilogy(orders[keep],np.maximum(psd[j,keep]*rotation,1e-9),color=color,lw=.8,label=label)
            choices.append({"bearing_id":b,"record_id":record,"elapsed_hours":row.elapsed_hours,"speed_rpm":row.speed_rpm,"source_fs_Hz":fs,"selection":label})
        for j,ax in enumerate(axs[i]):
            for label in ("BPFO","BPFI"):
                order=defect_orders()[label]
                ax.axvline(order,ls=":" if label=="BPFO" else "--",color="#65717d",lw=.8,label=label+" ideal first order")
            ax.set_title(b+" / channel "+("A" if j==0 else "C"))
            ax.set_xlabel(r"Frequency order ($f/f_r$)");ax.set_ylabel(r"Power density ($g^2$/order)")
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside upper center",ncol=2,fontsize=8,frameon=False)
    save(fig,"envelope_spectra")
    pd.DataFrame(choices).to_csv(root/"comparison/spectrum_record_selection.csv",index=False)
    fig,axs=plt.subplots(4,2,figsize=(11,11),constrained_layout=True)
    for ax,(b,g) in zip(axs.flat,p[p.model=="representation"].groupby("bearing_id")):
        ax.fill_between(g.elapsed_hours,g.lower_hours,g.upper_hours,color="#2962a3",alpha=.2)
        ax.plot(g.elapsed_hours,g.pred_hours,color="#2962a3",lw=1)
        ax.plot(g.elapsed_hours,g.rul_hours,color="black",lw=1.5)
        ax.set_title(b);ax.set_xlabel("Elapsed hours");ax.set_ylabel("Remaining hours")
    save(fig,"all_bearing_forecasts")
    print("Built four publication figures, with record-selection ledger.",flush=True)


if __name__=="__main__":
    main()
