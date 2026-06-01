'''
helper functions for creating summary tables and plots
'''

from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
from pathlib import Path
import ast
import mlflow
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
ROOT = Path(__file__).resolve().parent.parent

#mlflow db path 
MLFLOW_DB = ROOT / "data" / "mlflow.db"
VISUALS = ROOT / "visuals"
CLIENT = MlflowClient(tracking_uri=f"sqlite:///{MLFLOW_DB}")
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")

from mlflow.tracking import MlflowClient



def create_table(experiment_id, run_id,dataname):
    root_run_id = run_id

    root_run = mlflow.get_run(root_run_id)
    exp_id = root_run.info.experiment_id
    if experiment_id is not None and str(experiment_id) != str(exp_id):
        print(f"Warning: provided experiment_id={experiment_id} but root run belongs to experiment_id={exp_id}.")

    exp = mlflow.get_experiment(exp_id)
    experiment_name = exp.name if exp is not None else None

    seen = {root_run_id}
    frontier = [root_run_id]
    all_runs = []

    while frontier:
        pid = frontier.pop()
        kids = mlflow.search_runs(
            experiment_ids=[exp_id],
            filter_string=f"tags.mlflow.parentRunId = '{pid}'",
        )
        if kids.empty:
            continue
        all_runs.append(kids)
        for rid in kids["run_id"].tolist():
            if rid not in seen:
                seen.add(rid)
                frontier.append(rid)

    if not all_runs:
        return pd.DataFrame(), experiment_name

    runs = pd.concat(all_runs, ignore_index=True)

    parents = set(runs["tags.mlflow.parentRunId"].dropna())
    leaf_ids = sorted(set(runs["run_id"]) - parents)

    leaf_runs = runs[runs["run_id"].isin(leaf_ids)].copy()
    leaf_runs["experiment_name"] = experiment_name

    # Add parent run name + parent experiment name (direct parent of each leaf)
    client = MlflowClient()
    parent_ids = leaf_runs["tags.mlflow.parentRunId"].dropna().unique().tolist()

    parent_run_name = {}
    parent_experiment_name = {}
    for pid in parent_ids:
        prun = client.get_run(pid)
        parent_run_name[pid] = prun.data.tags.get("mlflow.runName")
        pexp = client.get_experiment(prun.info.experiment_id)
        parent_experiment_name[pid] = pexp.name if pexp is not None else None

    leaf_runs["parent_run_name"] = leaf_runs["tags.mlflow.parentRunId"].map(parent_run_name)
    leaf_runs["parent_experiment_name"] = leaf_runs["tags.mlflow.parentRunId"].map(parent_experiment_name)
    
    metrics= { 
    "metrics.RMSE": "RMSE",
    "metrics.Mean_Winkler_Score": "WinklerScore",
    "metrics.Negative_log_likelihood": "NLL",
    "metrics.Winkler_Coverage": "WinklerCoverage",
    "runtime": "Runtime (s)",}
    
    df = leaf_runs[["parent_run_name","metrics.RMSE",'metrics.Mean_Winkler_Score', 'metrics.Negative_log_likelihood','metrics.Winkler_Coverage','start_time',
       'end_time']]    
    #df = df.copy()
    #df = df.fillna(value=0)
    
    
    df["runtime"] = (pd.to_datetime(df["end_time"])-pd.to_datetime(df["start_time"])).dt.total_seconds()
    df["parent_run_name"] = df["parent_run_name"].apply(lambda x: "BDE" if x == "ModelBDE_robust" else "XGBoostLSS" if x == "ModelXGboostLSS_robust" else "TabIcL" if x == "ModelTabICL_robust" else "LR" if x =="ModelLG_robust" else "RF")
    grouped_data =df.groupby("parent_run_name")[["metrics.RMSE",'metrics.Mean_Winkler_Score', 'metrics.Negative_log_likelihood','metrics.Winkler_Coverage',"runtime"]].agg(["mean","std"])

    def pm(cell):
        m = cell[("mean")]
        s = cell[("std")]
        if pd.isna(m):  # optional
            return "-"
        return f"{m:.4f} \\pm {s:.4f}"
    
    final_df = pd.DataFrame({"Model": grouped_data.index})
    for col, name in metrics.items():
        final_df[name] = grouped_data[col].apply(pm, axis=1).values

    latex = final_df.to_latex(index=False, escape=False)
    
    with open(VISUALS/ "ablation" / f"metrics_{dataname}.tex","w") as f:
        f.write(latex)
    
    return df, experiment_name



def create_bde_hpo_parallel_plot(
    experiment_id: str,
    hpo_parent_run_id: str,
    dataname: str,
    param_cols: list[str],
    target_metric: str,
    visuals_dir: Path = VISUALS,
    lower_is_better: bool = True,
    color_metric: str | None = None,  # optional: e.g. "metrics.Winkler_Coverage"
    fixed_param_filters: dict[str, object] | None = None,
):
    def with_prefix(name: str, prefix: str) -> str:
        return name if name.startswith(prefix) else f"{prefix}{name}"

    def pretty(name: str) -> str:
        return name.replace("params.", "").replace("metrics.", "")

    def compact_category_label(value):
        text = str(value)
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)) and parsed:
                if all(isinstance(item, (int, float)) for item in parsed):
                    if len(set(parsed)) == 1:
                        base = int(parsed[0]) if float(parsed[0]).is_integer() else parsed[0]
                        text = f"{base}x{len(parsed)}"
                    else:
                        shown = [str(int(x) if float(x).is_integer() else x) for x in parsed[:3]]
                        suffix = "+" if len(parsed) > 3 else ""
                        text = "-".join(shown) + suffix
        except Exception:
            pass

        max_len = 14
        if len(text) > max_len:
            text = text[: max_len - 1] + "."
        return text

    root_run = mlflow.get_run(hpo_parent_run_id)
    exp_id = root_run.info.experiment_id
    if experiment_id is not None and str(experiment_id) != str(exp_id):
        print(f"Warning: provided experiment_id={experiment_id} but root run belongs to experiment_id={exp_id}.")

    trials = mlflow.search_runs(
        experiment_ids=[exp_id],
        filter_string=f"tags.mlflow.parentRunId = '{hpo_parent_run_id}'",
        output_format="pandas",
    )
    if trials.empty:
        raise ValueError("No HPO child runs found for the given parent run id.")

    if fixed_param_filters:
        for param_name, param_value in fixed_param_filters.items():
            filter_col = with_prefix(param_name, "params.")
            if filter_col not in trials.columns:
                raise ValueError(f"Filter column not found in MLflow runs: {filter_col}")
            trials = trials[trials[filter_col].astype(str) == str(param_value)]
        if trials.empty:
            raise ValueError(f"No HPO runs left after applying filters: {fixed_param_filters}")

    params = [with_prefix(c, "params.") for c in param_cols]
    target_col = with_prefix(target_metric, "metrics.")
    color_col_raw = with_prefix(color_metric, "metrics.") if color_metric else target_col

    required = list(dict.fromkeys(params + [target_col, color_col_raw]))
    missing = [c for c in required if c not in trials.columns]
    if missing:
        raise ValueError(f"Missing columns in MLflow runs: {missing}")

    plot_df = trials[required].copy()

    encoded_maps = {}
    dim_cols = []
    dimensions = []

    # Build numeric dimensions (encode categorical params if needed)
    for col in list(dict.fromkeys(params + [target_col])):
        dim_col = f"{col}__plot"
        series = plot_df[col]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        numeric = pd.to_numeric(series, errors="coerce")

        if numeric.notna().mean() >= 0.9:
            plot_df[dim_col] = numeric
            dim_def = dict(
                label=pretty(col),
                values=plot_df[dim_col],
                range=[float(plot_df[dim_col].min()), float(plot_df[dim_col].max())],
            )
        else:
            codes, uniques = pd.factorize(series.astype(str).fillna("NA"), sort=True)
            plot_df[dim_col] = codes.astype(float)
            encoded_maps[col] = dict(enumerate(uniques.tolist()))
            tickvals = list(encoded_maps[col].keys())
            if "hidden_layers" in col:
                ticktext = [str(v) for v in encoded_maps[col].values()]
            else:
                ticktext = [compact_category_label(v) for v in encoded_maps[col].values()]

            # Keep categorical axes readable when many categories exist.
            if len(tickvals) > 8:
                step = max(1, len(tickvals) // 8)
                keep_idx = list(range(0, len(tickvals), step))
                if keep_idx[-1] != len(tickvals) - 1:
                    keep_idx.append(len(tickvals) - 1)
                tickvals = [tickvals[i] for i in keep_idx]
                ticktext = [ticktext[i] for i in keep_idx]

            dim_def = dict(
                label=pretty(col),
                values=plot_df[dim_col],
                range=[float(plot_df[dim_col].min()), float(plot_df[dim_col].max())],
                tickvals=tickvals,
                ticktext=ticktext,
            )

        dim_cols.append(dim_col)
        dimensions.append(dim_def)

    plot_df[color_col_raw] = pd.to_numeric(plot_df[color_col_raw], errors="coerce")
    plot_df = plot_df.dropna(subset=[dim_cols[-1], color_col_raw])  # ensure target+color available
    if plot_df.empty:
        raise ValueError("No valid rows left after numeric conversion/dropna.")

    color_values = plot_df[color_col_raw].copy()

    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=color_values,
                colorscale="Jet",   # rainbow style like your screenshot
                reversescale=lower_is_better,
                cmin=float(color_values.min()),
                cmax=float(color_values.max()),
                showscale=True,
                colorbar=dict(title=pretty(color_col_raw)),
            ),
            dimensions=dimensions,
        )
    )

    fig.update_layout(
        title=dict(text=f"BDE HPO Parallel Plot - {dataname}", font=dict(size=16)),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(size=22, color="#444444"),
        width=1700,
        height=900,
        margin=dict(l=70, r=120, t=90, b=40),
    )

    visuals_dir.mkdir(parents=True, exist_ok=True)
    safe_target = target_col.replace("metrics.", "").replace(".", "_")
    pdf_path = visuals_dir / f"parallel_bde_hpo_{dataname}_{safe_target}.pdf"
    fig.write_image(str(pdf_path), format="pdf", scale=2)

    corr = plot_df[dim_cols + [target_col]].corr(method="spearman")[target_col].drop(target_col)
    corr.index = [c.replace("params.", "").replace("__plot", "") for c in corr.index]
    corr = corr.sort_values(key=lambda s: s.abs(), ascending=False)

    return fig, corr, encoded_maps, {"pdf_path": pdf_path, "n_trials": len(plot_df)}


def create_bde_hpo_parallel_plots_by_group(
    experiment_id: str,
    hpo_parent_run_id: str,
    dataname: str,
    group_by_param: str,
    param_cols: list[str],
    target_metric: str,
    visuals_dir: Path = VISUALS,
    lower_is_better: bool = True,
    color_metric: str | None = None,
):
    def with_prefix(name: str, prefix: str) -> str:
        return name if name.startswith(prefix) else f"{prefix}{name}"

    root_run = mlflow.get_run(hpo_parent_run_id)
    exp_id = root_run.info.experiment_id
    if experiment_id is not None and str(experiment_id) != str(exp_id):
        print(f"Warning: provided experiment_id={experiment_id} but root run belongs to experiment_id={exp_id}.")

    trials = mlflow.search_runs(
        experiment_ids=[exp_id],
        filter_string=f"tags.mlflow.parentRunId = '{hpo_parent_run_id}'",
        output_format="pandas",
    )
    if trials.empty:
        raise ValueError("No HPO child runs found for the given parent run id.")

    group_col = with_prefix(group_by_param, "params.")
    if group_col not in trials.columns:
        raise ValueError(f"group_by_param not found in MLflow runs: {group_col}")

    group_values = sorted(trials[group_col].dropna().astype(str).unique().tolist())
    if not group_values:
        raise ValueError(f"No non-null group values found for {group_col}.")

    plot_param_cols = [p for p in param_cols if with_prefix(p, "params.") != group_col]
    if not plot_param_cols:
        plot_param_cols = param_cols[:]

    outputs = {}
    for group_value in group_values:
        safe_group = (
            str(group_value)
            .replace(" ", "")
            .replace("[", "")
            .replace("]", "")
            .replace(",", "-")
            .replace("/", "-")
        )
        group_dataname = f"{dataname}_{group_by_param}_{safe_group}"
        _, corr, encoded_maps, info = create_bde_hpo_parallel_plot(
            experiment_id=experiment_id,
            hpo_parent_run_id=hpo_parent_run_id,
            dataname=group_dataname,
            param_cols=plot_param_cols,
            target_metric=target_metric,
            visuals_dir=visuals_dir,
            lower_is_better=lower_is_better,
            color_metric=color_metric,
            fixed_param_filters={group_by_param: group_value},
        )
        outputs[group_value] = {
            "corr": corr,
            "encoded_maps": encoded_maps,
            "info": info,
        }

    return outputs




def ablation_plot():
    
    pass

if __name__ =="__main__":
    df, exp_name = create_table("5","021de29ea6b34023a3c62202d4a0060a","fiat_1200")
    print(df)
    '''outputs = create_bde_hpo_parallel_plots_by_group(
    experiment_id="2",
    hpo_parent_run_id="7e99f7fdc8a44f1b9461c3c1bdc6ec46",
    dataname="miami_housing",
    group_by_param="hidden_layers",
    param_cols=["desired_energy_var_start", "desired_energy_var_end","hidden_layers"],
    target_metric="Mean_Winkler_Score",
    lower_is_better=True,
    color_metric=None,
)'''
