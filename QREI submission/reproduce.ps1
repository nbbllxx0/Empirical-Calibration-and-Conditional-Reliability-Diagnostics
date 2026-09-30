param(
    [Parameter(Mandatory=$true)][string]$AnalysisPython,
    [string]$NeuralPython = 'py',
    [ValidateSet('prepare','fit','summarize','sensitivity','verify','paper')][string]$Stage = 'summarize'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
function Invoke-Analysis {
    param([string[]]$Arguments)
    & $AnalysisPython @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Analysis command failed: $Arguments" }
}
if ($Stage -eq 'prepare') {
    Invoke-Analysis -Arguments @('-u','-m','bearing_dt.qrei.prepare','--workers','6')
    Invoke-Analysis -Arguments @('-u','-m','bearing_dt.qrei.endpoint')
}
if ($Stage -eq 'fit') {
    Invoke-Analysis -Arguments @('-u','-m','bearing_dt.qrei.evaluate','--out','QREI submission/results/lobo_neural','--models','representation','tcn','attention','--gpu-python',$NeuralPython)
    Invoke-Analysis -Arguments @('-u','-m','bearing_dt.qrei.batches')
    Invoke-Analysis -Arguments @('-u','-m','bearing_dt.qrei.evaluate','--processed','data/processed/phme_tvoc_10b_endpoint_v3','--protocol','QREI submission/protocol_endpoint_v3.json','--out','QREI submission/results/endpoint_neural','--models','representation','tcn','attention','--gpu-python',$NeuralPython)
    Invoke-Analysis -Arguments @('-u','-m','bearing_dt.qrei.batches','--endpoint')
    foreach ($Seed in @(20260930,20260931)) {
        Invoke-Analysis -Arguments @('-u','-m','bearing_dt.qrei.evaluate','--processed','data/processed/phme_tvoc_10b_endpoint_v3','--protocol','QREI submission/protocol_endpoint_v3.json','--out',"QREI submission/results/endpoint_seed_$Seed",'--models','representation','--seed',"$Seed",'--gpu-python',$NeuralPython)
    }
}
if ($Stage -eq 'summarize') {
    Invoke-Analysis -Arguments @('-u','-m','bearing_dt.qrei.summarize','--inputs','QREI submission/results/lobo_tabular','QREI submission/results/lobo_tabular_groups','QREI submission/results/lobo_neural')
    Invoke-Analysis -Arguments @('-u','-m','bearing_dt.qrei.summarize','--inputs','QREI submission/results/endpoint_tabular_groups','QREI submission/results/endpoint_neural','--out','QREI submission/results/endpoint_v3','--processed','data/processed/phme_tvoc_10b_endpoint_v3','--protocol','QREI submission/protocol_endpoint_v3.json')
    Invoke-Analysis -Arguments @('-u','-m','bearing_dt.qrei.compare')
    Invoke-Analysis -Arguments @('-u','-m','bearing_dt.qrei.seed_summary')
}
if ($Stage -eq 'sensitivity') {
    # Registered secondary refits and diagnostics (protocol_sensitivity_addendum.json); run after 'summarize', before 'paper'.
    Invoke-Analysis -Arguments @('-u','-m','bearing_dt.qrei.sensitivity_runs','--gpu-python',$NeuralPython)
    Invoke-Analysis -Arguments @('-u','-m','bearing_dt.qrei.sensitivity')
}
if ($Stage -eq 'verify') {
    Invoke-Analysis -Arguments @('-u','-m','bearing_dt.qrei.verify')
    Invoke-Analysis -Arguments @('-u','-m','bearing_dt.qrei.verify','--endpoint')
}
if ($Stage -eq 'paper') {
    # Spectrum record selection ledger (read by the figure script) and result-folder figures.
    Invoke-Analysis -Arguments @('-u','-m','bearing_dt.qrei.publication_plots')
    Invoke-Analysis -Arguments @('-u','-m','bearing_dt.qrei.build_revision')
    Invoke-Analysis -Arguments @('-u','-m','bearing_dt.qrei.build_supplement')
    Invoke-Analysis -Arguments @('-u','-m','bearing_dt.qrei.audit_tables')
    Invoke-Analysis -Arguments @('-u','QREI submission/manuscript/figure_sources/data_figures.py')
}
