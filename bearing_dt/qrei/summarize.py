from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .evaluate import LEARNED


NAMES={"representation":"Latent-state network","random_forest":"Random forest","standard_boosting":"Gradient boosting",
       "custom_boosting":"Boosted stumps","tcn":"Temporal convolutional network","attention":"Attention network",
       "subspace_ridge":"Random-subspace ridge","time_only":"Median-life clock","elapsed_clock":"Elapsed-time clock",
       "degradation":"Exponential vibration extrapolation","context_only":"Operating-context control","random_forest_no_elapsed":"Forest without explicit age",
       "temperature_only":"Temperature-context control","competing_stop":"Competing-stop Weibull model","competing_threshold":"Competing-threshold extrapolation"}


def metrics(group):
    true=group.rul_hours.to_numpy()
    pred=group.pred_hours.to_numpy()
    lo=group.lower_hours.to_numpy()
    up=group.upper_hours.to_numpy()
    duration=float(group.observed_duration_hours.iloc[0])
    elapsed=group.elapsed_hours.to_numpy()
    error=pred-true
    late=elapsed>=.5*duration
    score=up-lo+20*np.maximum(lo-true,0)+20*np.maximum(true-up,0)
    slope=float(np.polyfit(elapsed,pred,1)[0]) if len(elapsed)>2 else np.nan
    # Standard relative alpha bounds; report PH=0 when no sustained accuracy occurs.
    eligible=true>0
    accurate=np.abs(error[eligible])<=.2*true[eligible]
    sustained=np.logical_and.accumulate(accurate[::-1])[::-1]
    first=np.flatnonzero(sustained)
    ph=float((duration-elapsed[eligible][first[0]])/duration) if len(first) else 0.0
    al=[]
    for lam in (.5,.75,.9):
        idx=np.flatnonzero(elapsed<=lam*duration)
        if len(idx):
            i=idx[-1]
            al.append(float(abs(error[i])<=.2*true[i]))
    return {"nMAE":float(np.mean(np.abs(error))/duration),"MAE_hours":float(np.mean(np.abs(error))),
            "late_MAE_hours":float(np.mean(np.abs(error[late]))),"coverage":float(np.mean((true>=lo)&(true<=up))),
            "interval_score_hours":float(score.mean()),"normalized_interval_score":float(score.mean()/duration),
            "mean_width_hours":float((up-lo).mean()),"slope":slope,
            "monotonic_violation_rate":float(np.mean(np.diff(pred)>1e-6)),
            "asymmetric_5":float(np.mean(np.maximum(-error,0)+5*np.maximum(error,0))/duration),
            "asymmetric_10":float(np.mean(np.maximum(-error,0)+10*np.maximum(error,0))/duration),
            "alpha_lambda_accuracy":float(np.mean(al)),"prognostic_horizon_fraction":ph,
            "coverage_error":float(abs(np.mean((true>=lo)&(true<=up))-.9)),"records":len(group),"duration_hours":duration}


def decision_rows(predictions):
    rows=[]
    for (name,b),g in predictions.groupby(["model","bearing_id"]):
        g=g.sort_values("elapsed_hours")
        for lead in (.5,1.0,2.0):
            for policy,col in (("point","pred_hours"),("lower_bound","lower_hours")):
                triggers=np.flatnonzero(g[col].to_numpy()<=lead)
                remaining=float(g.rul_hours.iloc[triggers[0]]) if len(triggers) else 0.0
                has_action=bool(len(triggers))
                for ratio in (5,10,20):
                    failure=(not has_action) or remaining<lead
                    burden=max(remaining-lead,0)/float(g.observed_duration_hours.iloc[0])
                    rows.append({"model":name,"bearing_id":b,"policy":policy,"required_lead_hours":lead,"failure_cost_ratio":ratio,
                                 "action":has_action,"actual_lead_hours":remaining,"too_late":failure,"unused_life_fraction":burden,
                                 "loss":ratio*float(failure)+burden})
    for b,g in predictions[predictions.model=="time_only"].groupby("bearing_id"):
        duration=float(g.observed_duration_hours.iloc[0])
        for lead in (.5,1.0,2.0):
            for ratio in (5,10,20):
                rows.append({"model":"always_action","bearing_id":b,"policy":"control","required_lead_hours":lead,"failure_cost_ratio":ratio,
                             "action":True,"actual_lead_hours":duration,"too_late":duration<lead,"unused_life_fraction":max(duration-lead,0)/duration,
                             "loss":ratio*float(duration<lead)+max(duration-lead,0)/duration})
                rows.append({"model":"never_action","bearing_id":b,"policy":"control","required_lead_hours":lead,"failure_cost_ratio":ratio,
                             "action":False,"actual_lead_hours":0,"too_late":True,"unused_life_fraction":0,"loss":ratio})
    return pd.DataFrame(rows)


def summarize(inputs:list[Path],out:Path,processed:Path,protocol:Path,allow_partial=False):
    cfg=json.loads(protocol.read_text(encoding="utf-8"))
    parts=[]
    for folder in inputs:
        parts.extend(pd.read_csv(p) for p in sorted(folder.rglob("predictions.csv")))
    predictions=pd.concat(parts,ignore_index=True)
    if predictions.duplicated(["model","bearing_id","record_id"]).any():
        raise ValueError("Duplicate forecast rows")
    expected=pd.read_csv(processed/"features.csv")
    count_expected=int(expected.event_observed.sum())
    counts=predictions.groupby("model").size()
    learned=cfg.get("learned_models",list(LEARNED))
    if not allow_partial and (not set(learned).issubset(counts.index) or any(counts!=count_expected)):
        raise ValueError(f"Incomplete prediction matrix: {counts.to_dict()}; expected {count_expected} per model")
    out.mkdir(parents=True,exist_ok=True)
    figures=out/"figures"; figures.mkdir(exist_ok=True)
    predictions.to_csv(out/"joined_predictions.csv",index=False)
    rows=[]
    for (model,b),g in predictions.groupby(["model","bearing_id"]):
        rows.append({"model":model,"bearing_id":b,**metrics(g.sort_values("elapsed_hours"))})
    per=pd.DataFrame(rows)
    per.to_csv(out/"per_bearing_metrics.csv",index=False)
    columns=[c for c in per if c not in ("model","bearing_id","records","duration_hours")]
    summary=per.groupby("model")[columns].mean()
    summary["median_slope"]=per.groupby("model").slope.median()
    summary["bearings_nMAE_le_0.20"]=per.groupby("model").nMAE.apply(lambda x:int((x<=.2).sum()))
    summary["worst_bearing_nMAE"]=per.groupby("model").nMAE.max()
    summary["worst_bearing_coverage"]=per.groupby("model").coverage.min()
    summary.to_csv(out/"model_summary.csv")
    pooled=[]
    for name,g in predictions.groupby("model"):
        error=np.abs(g.pred_hours-g.rul_hours)
        norm=error/g.observed_duration_hours
        pooled.append({"model":name,"nMAE":float(norm.mean()),"MAE_hours":float(error.mean()),
                       "coverage":float(((g.rul_hours>=g.lower_hours)&(g.rul_hours<=g.upper_hours)).mean())})
    pd.DataFrame(pooled).to_csv(out/"window_pooled_metrics.csv",index=False)

    # Shared bearing draws give paired conditional-on-fit comparisons.
    names=sorted(summary.index)
    bearings=cfg["event_bearings"]
    rng=np.random.default_rng(cfg["seeds"][0])
    draws=rng.integers(0,len(bearings),size=(cfg["bootstrap_draws"],len(bearings)))
    bootstrap=[]
    probability=[]
    criteria={"nMAE":False,"MAE_hours":False,"normalized_interval_score":False,"interval_score_hours":False,
              "asymmetric_5":False,"asymmetric_10":False,"coverage_error":False,"alpha_lambda_accuracy":True,"prognostic_horizon_fraction":True}
    modelset=[n for n in names if n in learned]
    for col,higher in criteria.items():
        matrix=per.pivot(index="bearing_id",columns="model",values=col).reindex(index=bearings,columns=names).to_numpy()
        boot=matrix[draws].mean(axis=1)
        for j,name in enumerate(names):
            ci=np.quantile(boot[:,j],[.025,.975])
            bootstrap.append({"model":name,"metric":col,"estimate":float(matrix[:,j].mean()),"CI_low":float(ci[0]),"CI_high":float(ci[1]),"draws":len(draws),"independent_units":len(bearings)})
        indices=[names.index(n) for n in modelset]
        scores=boot[:,indices]
        optimum=scores.max(axis=1,keepdims=True) if higher else scores.min(axis=1,keepdims=True)
        ties=np.isclose(scores,optimum,rtol=1e-10,atol=1e-12)
        probs=(ties/ties.sum(axis=1,keepdims=True)).mean(axis=0)
        for name,p in zip(modelset,probs):
            probability.append({"model":name,"criterion":col,"P_rank_1":float(p),"higher_is_better":higher})
    pd.DataFrame(bootstrap).to_csv(out/"cluster_bootstrap_intervals.csv",index=False)
    rank=pd.DataFrame(probability)
    rank.to_csv(out/"rank_probabilities.csv",index=False)
    time_late=float(summary.loc["time_only","late_MAE_hours"])
    gates=[]
    for name in modelset:
        row=summary.loc[name]
        gates.append({"model":name,"G1":bool(row.nMAE<=.15 and row["bearings_nMAE_le_0.20"]>=6),
                      "G1_nMAE":float(row.nMAE),"G1_bearings":int(row["bearings_nMAE_le_0.20"]),
                      "G2":bool(row.late_MAE_hours<time_late),"G2_late_MAE_hours":float(row.late_MAE_hours),
                      "G2_time_only_hours":time_late,"G3":bool(row.median_slope<=-.7),"G3_median_slope":float(row.median_slope)})
    coherent=[r["model"] for r in gates if r["G1"] and r["G2"] and r["G3"]]
    winners=rank.loc[rank.groupby("criterion").P_rank_1.idxmax()]
    stable=winners[winners.P_rank_1>=.7]
    g4=len(stable.model.unique())>=2
    verdict={"G1_G3_coherent_models":coherent,"G4":bool(g4),"G4_stable_criterion_winners":stable.to_dict("records"),
             "per_model_gates":gates,"outcome":"ranking paper" if coherent and g4 else "instability framing" if coherent else "STOP: physical prediction gate failed",
             "thresholds_changed":False,"claim":"Results are conditional on the fixed protocol and fitted models; they do not prove prediction is impossible."}
    (out/"gate.json").write_text(json.dumps(verdict,indent=2),encoding="utf-8")
    decisions=decision_rows(predictions)
    decisions.to_csv(out/"maintenance_decisions.csv",index=False)
    decisions.groupby(["model","policy","required_lead_hours","failure_cost_ratio"])[["loss","too_late","unused_life_fraction"]].mean().to_csv(out/"maintenance_summary.csv")
    print(summary[["nMAE","late_MAE_hours","median_slope","bearings_nMAE_le_0.20","coverage"]].to_string(),flush=True)
    print(json.dumps(verdict,indent=2),flush=True)
    plot_figures(expected,predictions,per,summary,rank,decisions,figures,learned)
    return verdict


def plot_figures(features,predictions,per,summary,rank,decisions,out,learned=LEARNED):
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":300,"pdf.fonttype":42})
    palette=plt.get_cmap("tab10")
    def save(fig,name):
        fig.savefig(out/(name+".pdf"),bbox_inches="tight")
        fig.savefig(out/(name+".png"),bbox_inches="tight",dpi=300)
        plt.close(fig)
    fig,axs=plt.subplots(5,2,figsize=(11,12),constrained_layout=True)
    for ax,(b,g) in zip(axs.flat,features.groupby("bearing_id")):
        g=g.sort_values("elapsed_hours")
        ax.plot(g.elapsed_hours,g.A_rms_g,color="#2962a3",lw=.7,alpha=.45,label="Channel A")
        ax.plot(g.elapsed_hours,g.C_rms_g,color="#c56f20",lw=.7,alpha=.45,label="Channel C")
        ax.plot(g.elapsed_hours,g.hi_rms_g,color="#242d35",lw=1.2,label="Causal indicator")
        ax.axhspan(6,10,color="#ca5252",alpha=.12,label="Documented stop-threshold range")
        ax.set_title(b+(" — diagnostics only" if b in ("B01","B05") else ""))
        ax.set_xlabel("Elapsed hours"); ax.set_ylabel("Acceleration RMS (g)")
    axs[0,0].legend(fontsize=7,loc="upper left")
    save(fig,"health_trajectories")
    chosen=("B02","B03","B10","B17")
    displayed=("representation","random_forest","custom_boosting","time_only")
    fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for ax,b in zip(axs.flat,chosen):
        g=predictions[predictions.bearing_id==b]
        truth=g[g.model=="time_only"].sort_values("elapsed_hours")
        ax.plot(truth.elapsed_hours,truth.rul_hours,color="black",lw=1.8,label="Observed residual time")
        for j,name in enumerate(displayed):
            h=g[g.model==name].sort_values("elapsed_hours")
            ax.plot(h.elapsed_hours,h.pred_hours,color=palette(j),lw=1.1,label=NAMES[name])
        ax.set_title(b); ax.set_xlabel("Elapsed hours"); ax.set_ylabel("Predicted remaining hours")
    axs[0,0].legend(fontsize=7)
    save(fig,"independent_bearing_predictions")
    selected=summary.loc[[n for n in learned if n in summary.index]]
    fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
    for j,(name,row) in enumerate(selected.iterrows()):
        ax.scatter(row.nMAE,row.coverage,s=55,color=palette(j),label=f"{j+1}. "+NAMES.get(name,name.replace("_"," ")))
        ax.annotate(str(j+1),(row.nMAE,row.coverage),xytext=(5,5),textcoords="offset points",fontsize=8)
    ax.axhline(.9,color="#626b75",ls="--",lw=1)
    ax.set_xlabel("Mean normalized absolute error (equal bearings)"); ax.set_ylabel("Empirical coverage (equal bearings)")
    ax.set_ylim(0,1.08)
    ax.legend(fontsize=8,loc="lower right")
    save(fig,"error_coverage")
    p=rank.pivot(index="model",columns="criterion",values="P_rank_1").reindex(index=[n for n in learned if n in rank.model.unique()])
    fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True)
    im=ax.imshow(p.to_numpy(),vmin=0,vmax=1,cmap="Blues",aspect="auto")
    ax.set_yticks(np.arange(len(p)),[NAMES.get(n,n.replace("_"," ")) for n in p.index])
    ax.set_xticks(np.arange(len(p.columns)),[c.replace("_"," ") for c in p.columns],rotation=35,ha="right")
    for i in range(len(p)):
        for j in range(len(p.columns)):
            v=p.iloc[i,j]; ax.text(j,i,f"{v:.3f}",ha="center",va="center",color="white" if v>.5 else "black",fontsize=8)
    fig.colorbar(im,ax=ax,label="Fraction of bearing bootstrap draws ranked first")
    save(fig,"rank_probabilities")
    fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
    for j,name in enumerate(displayed+("always_action","never_action")):
        d=decisions[(decisions.model==name)&(decisions.required_lead_hours==1)&decisions.policy.isin(["point","control"])]
        curve=d.groupby("failure_cost_ratio").loss.mean()
        ax.plot(curve.index,curve.values,marker="o",label=NAMES.get(name,name.replace("_"," ")),color=palette(j))
    ax.set_xlabel("Cost of a missed timely action relative to unused life"); ax.set_ylabel("Retrospective mean policy loss")
    ax.legend(fontsize=8)
    save(fig,"maintenance_loss")
    growth=[]
    for b,g in features.groupby("bearing_id"):
        n=max(10,int(len(g)*.1))
        early=float(g.hi_rms_g.iloc[:n].median()); late=float(g.hi_rms_g.iloc[-n:].median())
        growth.append({"bearing_id":b,"early_HI_g":early,"late_HI_g":late,"late_to_early_ratio":late/max(early,1e-12),
                       "primary_event":b not in ("B01","B05")})
    pd.DataFrame(growth).to_csv(out.parent/"health_indicator_growth.csv",index=False)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--inputs",nargs="+",type=Path,default=[Path("QREI submission/results/lobo_tabular"),Path("QREI submission/results/lobo_neural")])
    p.add_argument("--out",type=Path,default=Path("QREI submission/results/primary"))
    p.add_argument("--processed",type=Path,default=Path("data/processed/phme_tvoc_10b_v2"))
    p.add_argument("--protocol",type=Path,default=Path("QREI submission/protocol.json"))
    p.add_argument("--allow-partial",action="store_true")
    a=p.parse_args()
    summarize(a.inputs,a.out,a.processed,a.protocol,a.allow_partial)
