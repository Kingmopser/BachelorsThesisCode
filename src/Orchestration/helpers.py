'''
helper functions for creating summary tables and plots
'''

from mlflow.tracking import MlflowClient
from pathlib import Path
import ast
import mlflow
import numpy as np
import pandas as pd
import plotly.graph_objects as go
ROOT = Path(__file__).resolve().parent.parent

#mlflow db path 
MLFLOW_DB = ROOT / "data" / "mlflow.db"
VISUALS = ROOT / "visuals"
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")


def fetch_direct_child_runs(experiment_id, run_id, include_parent_metadata=True):
    """
    Fetch all direct child runs for one MLflow parent run.

    This is intentionally non-recursive and only returns the runs whose
    ``tags.mlflow.parentRunId`` matches ``run_id``.
    """
    root_run = mlflow.get_run(run_id)
    exp_id = root_run.info.experiment_id
    if experiment_id is not None and str(experiment_id) != str(exp_id):
        print(
            f"Warning: provided experiment_id={experiment_id} but root run belongs to experiment_id={exp_id}."
        )

    exp = mlflow.get_experiment(exp_id)
    experiment_name = exp.name if exp is not None else None

    child_runs = mlflow.search_runs(
        experiment_ids=[exp_id],
        filter_string=f"tags.mlflow.parentRunId = '{run_id}'",
        output_format="pandas",
    )

    if child_runs.empty:
        return pd.DataFrame(), experiment_name

    child_runs = child_runs.copy()
    child_runs["experiment_name"] = experiment_name

    if include_parent_metadata:
        child_runs["parent_run_id"] = run_id
        child_runs["parent_run_name"] = root_run.data.tags.get("mlflow.runName")
        child_runs["parent_experiment_id"] = exp_id
        child_runs["parent_experiment_name"] = experiment_name

    return child_runs, experiment_name


def extract_hpo_model_parameters(
    experiment_id,
    hpo_parent_run_id,
    include_metrics=False,
    output_path=None,
):
    """
    Extract one row per HPO child run with the logged model parameters.

    This is useful for runs like ``HPO_Study_miami_housing`` where each nested
    child run corresponds to one trial configuration.
    """
    root_run = mlflow.get_run(hpo_parent_run_id)
    exp_id = root_run.info.experiment_id
    if experiment_id is not None and str(experiment_id) != str(exp_id):
        print(
            f"Warning: provided experiment_id={experiment_id} but root run belongs to experiment_id={exp_id}."
        )

    trials = mlflow.search_runs(
        experiment_ids=[exp_id],
        filter_string=f"tags.mlflow.parentRunId = '{hpo_parent_run_id}'",
        output_format="pandas",
    )
    if trials.empty:
        raise ValueError("No HPO child runs found for the given parent run id.")

    trials = trials.copy()
    param_cols = sorted(col for col in trials.columns if col.startswith("params."))
    metric_cols = sorted(col for col in trials.columns if col.startswith("metrics."))

    preferred_param_order = [
        "params.hidden_layers",
        "params.desired_energy_var_start",
        "params.desired_energy_var_end",
        "params.warmup_steps",
        "params.n_samples",
        "params.epochs",
        "params.n_members",
    ]
    ordered_param_cols = [col for col in preferred_param_order if col in param_cols]
    ordered_param_cols.extend(col for col in param_cols if col not in ordered_param_cols)

    base_cols = [
        "run_id",
        "tags.mlflow.runName",
        "status",
    ]
    selected_cols = base_cols + ordered_param_cols
    if include_metrics:
        selected_cols.extend(metric_cols)

    result = trials[selected_cols].copy()
    rename_map = {
        "tags.mlflow.runName": "run_name",
        **{col: col.removeprefix("params.") for col in ordered_param_cols},
        **{col: col.removeprefix("metrics.") for col in metric_cols},
    }
    result = result.rename(columns=rename_map)
    result = result.sort_values(by="run_name").reset_index(drop=True)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)

    return result


def create_table(experiment_id, run_id,dataname):
    root_run_ids = [run_id] if isinstance(run_id, str) else list(run_id)
    leaf_frames = []
    experiment_names = []

    for root_run_id in root_run_ids:
        root_run = mlflow.get_run(root_run_id)
        exp_id = root_run.info.experiment_id
        if experiment_id is not None and str(experiment_id) != str(exp_id):
            print(f"Warning: provided experiment_id={experiment_id} but root run belongs to experiment_id={exp_id}.")

        exp = mlflow.get_experiment(exp_id)
        experiment_name = exp.name if exp is not None else None
        experiment_names.append(experiment_name)

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
            continue

        runs = pd.concat(all_runs, ignore_index=True)

        parents = set(runs["tags.mlflow.parentRunId"].dropna())
        leaf_ids = sorted(set(runs["run_id"]) - parents)

        root_leaf_runs = runs[runs["run_id"].isin(leaf_ids)].copy()
        root_leaf_runs["experiment_name"] = experiment_name
        root_leaf_runs["root_run_id"] = root_run_id
        root_leaf_runs["root_run_name"] = root_run.data.tags.get("mlflow.runName")
        leaf_frames.append(root_leaf_runs)

    experiment_name = ", ".join([name for name in dict.fromkeys(experiment_names) if name])
    if not leaf_frames:
        return pd.DataFrame(), experiment_name

    leaf_runs = pd.concat(leaf_frames, ignore_index=True).drop_duplicates("run_id")

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
    
    df = leaf_runs[["root_run_id","root_run_name","parent_run_name","metrics.RMSE",'metrics.Mean_Winkler_Score', 'metrics.Negative_log_likelihood','metrics.Winkler_Coverage','start_time',
       'end_time']].copy()
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
    
    output_dir = VISUALS / "ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"metrics_{dataname}.tex","w") as f:
        f.write(latex)
    
    return df, experiment_name



def create_bde_hpo_parallel_plot(
    experiment_id: str,
    hpo_parent_run_id: str,
    dataname: str,
    param_cols: list[str],
    target_metric: str | list[str],
    visuals_dir: Path = VISUALS,
    lower_is_better: bool = True,
    color_metric: str | None = None,  # optional: e.g. "metrics.Winkler_Coverage"
    fixed_param_filters: dict[str, object] | None = None,
    color_group_param: str | None = None,
):
    okabe_ito = [
        "#0072B2",  # blue
        "#D55E00",  # vermillion
        "#009E73",  # bluish green
        "#CC79A7",  # reddish purple
        "#E69F00",  # orange
        "#56B4E9",  # sky blue
        "#F0E442",  # yellow
        "#000000",  # black
    ]
    grouped_color_map = {
        "Lower variance": "#5D3A9B",
        "Higher variance": "#E66100",
    }
    pretty_label_map = {
        "metrics.Mean_Winkler_Score": "Winkler Score",
        "Mean_Winkler_Score": "Winkler Score",
        "metrics.Negative_log_likelihood": "NLL",
        "Negative_log_likelihood": "NLL",
    }

    def with_prefix(name: str, prefix: str) -> str:
        return name if name.startswith(prefix) else f"{prefix}{name}"

    def pretty(name: str) -> str:
        if name in pretty_label_map:
            return pretty_label_map[name]
        return name.replace("params.", "").replace("metrics.", "")

    def is_nll_column(name: str) -> bool:
        return "Negative_log_likelihood" in name or pretty(name) == "NLL"

    def format_log_tick(value: float) -> str:
        if value >= 1000:
            return f"{value:,.0f}"
        if value >= 1:
            return f"{value:.0f}" if float(value).is_integer() else f"{value:g}"
        return f"{value:g}"

    def build_log_tick_spec(values: pd.Series, n_ticks: int = 5) -> dict:
        positive = pd.to_numeric(values, errors="coerce")
        positive = positive[positive > 0]
        if positive.empty:
            return {}

        min_val = float(positive.min())
        max_val = float(positive.max())

        if np.isclose(min_val, max_val):
            return {
                "range": [float(np.floor(np.log10(min_val))), float(np.log10(max_val))],
                "tickvals": [float(np.log10(min_val))],
                "ticktext": [format_log_tick(min_val)],
            }

        min_power = int(np.floor(np.log10(min_val)))
        max_power = int(np.floor(np.log10(max_val)))
        tick_values = np.array(
            [10.0**power for power in range(min_power, max_power + 1)],
            dtype=float,
        )
        tick_values = tick_values[tick_values <= max_val]

        if len(tick_values) < 2:
            tick_values = np.geomspace(min_val, max_val, num=n_ticks)
            tick_values = np.unique(np.round(tick_values, 12))

        if len(tick_values) == 0 or not np.isclose(tick_values[-1], max_val):
            tick_values = np.append(tick_values, max_val)

        tick_values = np.unique(np.round(tick_values, 12))

        return {
            "range": [float(min_power), float(np.log10(max_val))],
            "tickvals": np.log10(tick_values).tolist(),
            "ticktext": [format_log_tick(value) for value in tick_values],
        }

    def classify_energy_variance_level(value):
        parsed = value
        try:
            parsed = ast.literal_eval(str(value))
        except Exception:
            pass

        if isinstance(parsed, (list, tuple)) and parsed:
            numeric = parsed[-1]
        else:
            numeric = parsed

        try:
            numeric = float(numeric)
        except Exception:
            return compact_category_label(value)

        return "Higher variance" if numeric >= 0.01 else "Lower variance"

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
    target_metrics = [target_metric] if isinstance(target_metric, str) else list(target_metric)
    target_cols = [with_prefix(metric, "metrics.") for metric in target_metrics]
    if color_group_param:
        color_col_raw = with_prefix(color_group_param, "params.")
    else:
        color_col_raw = with_prefix(color_metric, "metrics.") if color_metric else target_cols[-1]

    required = list(dict.fromkeys(params + target_cols + [color_col_raw]))
    missing = [c for c in required if c not in trials.columns]
    if missing:
        raise ValueError(f"Missing columns in MLflow runs: {missing}")

    plot_df = trials[required].copy()

    encoded_maps = {}
    dim_cols = []
    dimension_specs = []

    # Build numeric dimensions (encode categorical params if needed)
    for col in list(dict.fromkeys(params + target_cols)):
        dim_col = f"{col}__plot"
        series = plot_df[col]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        numeric = pd.to_numeric(series, errors="coerce")

        if numeric.notna().mean() >= 0.9:
            use_log_nll = is_nll_column(col) and numeric.dropna().gt(0).all()
            plot_values = np.log10(numeric.astype(float)) if use_log_nll else numeric
            plot_df[dim_col] = plot_values
            log_tick_spec = build_log_tick_spec(numeric) if use_log_nll else {}
            dim_spec = dict(
                label="NLL(log-scale)" if use_log_nll and is_nll_column(col) else pretty(col),
                range=log_tick_spec.get(
                    "range",
                    [float(plot_df[dim_col].min()), float(plot_df[dim_col].max())],
                ),
            )
            if use_log_nll:
                dim_spec.update(
                    {key: value for key, value in log_tick_spec.items() if key != "range"}
                )
            elif is_nll_column(col):
                dim_spec["tickformat"] = ",.0f"
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

            dim_spec = dict(
                label=pretty(col),
                range=[float(plot_df[dim_col].min()), float(plot_df[dim_col].max())],
                tickvals=tickvals,
                ticktext=ticktext,
            )

        dim_cols.append(dim_col)
        dimension_specs.append((dim_col, dim_spec))

    def dimensions_for(df_subset):
        dims = []
        for dim_col, dim_spec in dimension_specs:
            dims.append(
                {
                    **dim_spec,
                    "values": df_subset[dim_col],
                }
            )
        return dims

    if color_group_param:
        plot_df[color_col_raw] = plot_df[color_col_raw].astype(str).fillna("NA")
        plot_df = plot_df.dropna(subset=dim_cols)
    else:
        plot_df[color_col_raw] = pd.to_numeric(plot_df[color_col_raw], errors="coerce")
        plot_df = plot_df.dropna(subset=list(dict.fromkeys(dim_cols + [color_col_raw])))
    if plot_df.empty:
        raise ValueError("No valid rows left after numeric conversion/dropna.")

    legend_shapes = []
    legend_annotations = []
    if color_group_param:
        category_series = plot_df[color_col_raw].copy()
        if "energy_var" in color_col_raw:
            category_series = category_series.map(classify_energy_variance_level)
        else:
            category_series = category_series.map(compact_category_label)

        color_codes, color_labels = pd.factorize(category_series, sort=True)
        discrete_colors = [
            grouped_color_map.get(label, okabe_ito[idx % len(okabe_ito)])
            for idx, label in enumerate(color_labels.tolist())
        ]
        if len(discrete_colors) == 1:
            colorscale = [[0.0, discrete_colors[0]], [1.0, discrete_colors[0]]]
        else:
            colorscale = []
            step = 1.0 / len(discrete_colors)
            for idx, color in enumerate(discrete_colors):
                start = idx * step
                end = (idx + 1) * step
                colorscale.append([start, color])
                colorscale.append([end, color])

        fig = go.Figure(
            data=go.Parcoords(
                line=dict(
                    color=color_codes.astype(float),
                    colorscale=colorscale,
                    cmin=-0.5,
                    cmax=len(color_labels) - 0.5,
                    showscale=False,
                ),
                dimensions=dimensions_for(plot_df),
            )
        )
        for idx, (color_label, color) in enumerate(zip(color_labels.tolist(), discrete_colors)):
            x_start = 0.34 + idx * 0.18
            legend_shapes.append(
                dict(
                    type="rect",
                    xref="paper",
                    yref="paper",
                    x0=x_start,
                    x1=x_start + 0.02,
                    y0=-0.11,
                    y1=-0.09,
                    line=dict(color=color, width=1),
                    fillcolor=color,
                )
            )
            legend_annotations.append(
                dict(
                    x=x_start + 0.025,
                    y=-0.10,
                    xref="paper",
                    yref="paper",
                    text=color_label,
                    showarrow=False,
                    xanchor="left",
                    yanchor="middle",
                    font=dict(size=18, color="#444444"),
                )
            )
    else:
        color_values = plot_df[color_col_raw].copy()
        use_log_nll_color = is_nll_column(color_col_raw) and color_values.dropna().gt(0).all()
        colorbar_config = dict(title=pretty(color_col_raw))
        if use_log_nll_color:
            log_tick_spec = build_log_tick_spec(plot_df[color_col_raw])
            color_values = np.log10(color_values.astype(float))
            colorbar_config["title"] = "NLL(log-scale)"
            colorbar_config.update(
                {key: value for key, value in log_tick_spec.items() if key != "range"}
            )
        elif is_nll_column(color_col_raw):
            colorbar_config["tickformat"] = ",.0f"

        line_config = dict(
            color=color_values,
            colorscale="Cividis",
            reversescale=lower_is_better,
            cmin=float(log_tick_spec["range"][0]) if use_log_nll_color else float(color_values.min()),
            cmax=float(color_values.max()),
            showscale=True,
            colorbar=colorbar_config,
        )

        fig = go.Figure(
            data=go.Parcoords(
                line=line_config,
                dimensions=dimensions_for(plot_df),
            )
        )

    fig.update_layout(
        title=dict(
            text=f"BDE HPO Parallel Plot - {dataname}",
            font=dict(size=30),
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
        ),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(size=22, color="#444444"),
        annotations=legend_annotations,
        shapes=legend_shapes,
        width=1900,
        height=980,
        margin=dict(l=190, r=160, t=160, b=150),
    )

    visuals_dir.mkdir(parents=True, exist_ok=True)
    safe_target = "_".join(
        col.replace("metrics.", "").replace(".", "_") for col in target_cols
    )
    pdf_path = visuals_dir / f"parallel_bde_hpo_{dataname}_{safe_target}.pdf"
    html_path = visuals_dir / f"parallel_bde_hpo_{dataname}_{safe_target}.html"
    export_path = pdf_path
    export_error = None
    try:
        fig.write_image(str(pdf_path), format="pdf", scale=2)
    except Exception as exc:
        export_error = str(exc)
        fig.write_html(str(html_path), include_plotlyjs="cdn")
        export_path = html_path

    target_dim_cols = [f"{col}__plot" for col in target_cols]
    param_dim_cols = [col for col in dim_cols if col not in target_dim_cols]
    corr_matrix = plot_df[list(dict.fromkeys(param_dim_cols + target_dim_cols))].corr(method="spearman")
    param_labels = [c.replace("params.", "").replace("__plot", "") for c in param_dim_cols]

    if len(target_dim_cols) == 1:
        corr = corr_matrix[target_dim_cols[0]].loc[param_dim_cols]
        corr.index = param_labels
        corr = corr.sort_values(key=lambda s: s.abs(), ascending=False)
    else:
        corr = corr_matrix.loc[param_dim_cols, target_dim_cols]
        corr.index = param_labels
        corr.columns = [col.replace("metrics.", "") for col in target_cols]
        corr = corr.reindex(corr.abs().max(axis=1).sort_values(ascending=False).index)

    return fig, corr, encoded_maps, {
        "pdf_path": pdf_path,
        "html_path": html_path,
        "export_path": export_path,
        "export_error": export_error,
        "n_trials": len(plot_df),
        "color_mode": "grouped" if color_group_param else "continuous",
    }


def create_bde_hpo_parallel_plots_by_group(
    experiment_id: str,
    hpo_parent_run_id: str,
    dataname: str,
    group_by_param: str | None = None,
    param_cols: list[str] | None = None,
    target_metric: str | list[str] = "Mean_Winkler_Score",
    visuals_dir: Path = VISUALS,
    lower_is_better: bool = True,
    color_metric: str | None = None,
    color_group_param: str | None = None,
):
    def with_prefix(name: str, prefix: str) -> str:
        return name if name.startswith(prefix) else f"{prefix}{name}"

    if param_cols is None:
        param_cols = []

    if group_by_param in (None, "", "None", "none"):
        _, corr, encoded_maps, info = create_bde_hpo_parallel_plot(
            experiment_id=experiment_id,
            hpo_parent_run_id=hpo_parent_run_id,
            dataname=dataname,
            param_cols=param_cols,
            target_metric=target_metric,
            visuals_dir=visuals_dir,
            lower_is_better=lower_is_better,
            color_metric=color_metric,
            color_group_param=color_group_param,
        )
        return {
            "all": {
                "corr": corr,
                "encoded_maps": encoded_maps,
                "info": info,
            }
        }

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
            color_group_param=color_group_param,
        )
        outputs[group_value] = {
            "corr": corr,
            "encoded_maps": encoded_maps,
            "info": info,
        }

    return outputs



    
if __name__ =="__main__":
    # Example usage for manually exporting one HPO run's trial parameters.
    '''df, exp_name = create_table("5",
                                ["49038efc38c7463dbbb0a1dbe6da7f3f"],
                                "fiat500_ablation")
             print(df)
    fig, corr, encoded_maps, info = create_bde_hpo_parallel_plot(
        experiment_id="4",
        hpo_parent_run_id="c963c22d0c7b4f13a6ee0dc353b3ba1c",
        dataname="Wine Quality",
        param_cols=["desired_energy_var_end", "hidden_layers"],
        target_metric=["Mean_Winkler_Score", "Negative_log_likelihood"],
        lower_is_better=True,
        color_group_param="desired_energy_var_end",
        )
    '''
    df = extract_hpo_model_parameters(
    experiment_id="2",
    hpo_parent_run_id="383b387cb6aa46b08171ac5f45f466d8",
    include_metrics=True,
    output_path=VISUALS / "raw" / "miami_housing_hpo_params.csv",
    )
    print(df)
