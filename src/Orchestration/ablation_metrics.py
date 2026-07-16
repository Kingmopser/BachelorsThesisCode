"""
Utilities for turning exported LaTeX metric tables into percentage-delta series.
"""

from pathlib import Path
import re
import statistics


METRIC_EXPORT_NAMES = {
    "RMSE": "rmse",
    "NLL": "nll",
    "WinklerScore": "winkler",
    "WinklerCoverage": "winklercoverage",
    "Runtime (s)": "runtime",
}

ROOT = Path(__file__).resolve().parent.parent
MLFLOW_DB = ROOT / "data" / "mlflow.db"


def _get_mlflow_client():
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    return MlflowClient(tracking_uri=f"sqlite:///{MLFLOW_DB}")


def _extract_mean_from_latex_cell(cell):
    cell = cell.strip()
    if cell == "-":
        return None

    match = re.match(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", cell)
    if match is None:
        raise ValueError(f"Could not parse numeric value from LaTeX cell: {cell}")

    return float(match.group(1))


def parse_latex_metric_table(table_path, model_name="BDE"):
    """
    Parse one LaTeX metrics table and return the mean values for the chosen model.
    """
    table_path = Path(table_path)
    lines = [line.strip() for line in table_path.read_text().splitlines() if "&" in line and "\\\\" in line]

    if not lines:
        raise ValueError(f"No table rows found in {table_path}")

    header = [cell.strip() for cell in lines[0].removesuffix("\\\\").split("&")]
    metric_names = header[1:]

    for row in lines[1:]:
        cells = [cell.strip() for cell in row.removesuffix("\\\\").split("&")]
        if not cells or cells[0] != model_name:
            continue

        return {
            metric_name: _extract_mean_from_latex_cell(cell)
            for metric_name, cell in zip(metric_names, cells[1:])
        }

    raise ValueError(f"Model '{model_name}' not found in {table_path}")


def _resolve_base_table(dataset_slug, summary_tables, baseline_table_map=None):
    baseline_table_map = baseline_table_map or {}
    baseline_slug = baseline_table_map.get(dataset_slug, dataset_slug)

    direct_match = summary_tables.get(baseline_slug)
    if direct_match is not None:
        return direct_match

    raise ValueError(
        f"Could not find the baseline summary table '{baseline_slug}' for dataset '{dataset_slug}'. "
        f"Available summary tables: {sorted(summary_tables)}"
    )


def _percent_delta(new_value, base_value):
    if base_value == 0:
        raise ZeroDivisionError("Base value is zero; percentage delta is undefined.")
    return ((new_value - base_value) / base_value) * 100


def _format_number(value, digits=1):
    rounded = round(value, digits)
    if float(rounded).is_integer():
        return str(int(rounded))
    return f"{rounded:.{digits}f}".rstrip("0").rstrip(".")


def build_ablation_delta_series(
    visuals_root,
    metrics=("RMSE", "NLL", "WinklerScore","WinklerCoverage"),
    compared_epochs=(800, 1200),
    baseline_table_map=None,
    dataset_name_map=None,
    model_name="BDE",
    round_digits=1,
):
    """
    Build per-metric percentage delta lists like
    [0, delta_800_vs_baseline, delta_1200_vs_baseline].
    """
    visuals_root = Path(visuals_root)
    baseline_table_map = baseline_table_map or {}
    dataset_name_map = dataset_name_map or {}

    summary_tables = {
        path.stem.removeprefix("metrics_"): path
        for path in sorted((visuals_root / "summary_tables").glob("metrics_*.tex"))
        if path.stem != "metrics"
    }

    ablation_tables = {}
    for path in sorted((visuals_root / "ablation").glob("metrics_*.tex")):
        slug = path.stem.removeprefix("metrics_")
        dataset_slug, epoch_text = slug.rsplit("_", 1)
        if not epoch_text.isdigit():
            continue
        ablation_tables.setdefault(dataset_slug, {})[int(epoch_text)] = path

    delta_series = {
        METRIC_EXPORT_NAMES.get(metric, metric.lower()): {}
        for metric in metrics
    }

    for dataset_slug in sorted(ablation_tables):
        base_table = _resolve_base_table(
            dataset_slug,
            summary_tables,
            baseline_table_map=baseline_table_map,
        )
        base_metrics = parse_latex_metric_table(base_table, model_name=model_name)
        dataset_key = dataset_name_map.get(dataset_slug, dataset_slug)
        epoch_tables = ablation_tables[dataset_slug]

        for metric in metrics:
            export_name = METRIC_EXPORT_NAMES.get(metric, metric.lower())
            base_value = base_metrics[metric]
            values = [0]

            for epoch in compared_epochs:
                if epoch not in epoch_tables:
                    raise ValueError(f"Missing ablation table for dataset '{dataset_slug}' and epoch {epoch}.")
                epoch_metrics = parse_latex_metric_table(epoch_tables[epoch], model_name=model_name)
                delta = _percent_delta(epoch_metrics[metric], base_value)
                values.append(round(delta, round_digits))

            delta_series[export_name][dataset_key] = values

    return delta_series


def format_metric_delta_assignments(metric_delta_series, variable_prefix="data", digits=1):
    """
    Turn the computed delta series into paste-ready Python dict assignments.
    """
    blocks = []
    for metric_name, dataset_values in metric_delta_series.items():
        lines = [f"{variable_prefix}_{metric_name} = {{"]
        for dataset_name, values in dataset_values.items():
            formatted_values = ", ".join(_format_number(value, digits=digits) for value in values)
            lines.append(f"    {dataset_name!r}: [{formatted_values}],")
        lines.append("}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def build_epoch_metric_latex_table(
    metric_tables,
    epochs=(400, 800, 1200),
    caption="Results across epochs.",
    label="tab:epoch_results",
    digits=4,
):
    """
    Build a LaTeX table from metric->dataset->[epoch values] mappings.
    """
    latex_lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\begin{tabular}{ll" + "r" * len(epochs) + "}",
        r"\hline",
        "Metric & Dataset & " + " & ".join(str(epoch) for epoch in epochs) + r" \\",
        r"\hline",
    ]

    for metric_name, metric_data in metric_tables.items():
        first_row = True
        for dataset_name, values in metric_data.items():
            escaped_dataset = dataset_name.replace("_", r"\_")
            formatted_values = [f"{value:.{digits}f}" for value in values]
            row = [metric_name if first_row else "", escaped_dataset, *formatted_values]
            latex_lines.append(" & ".join(row) + r" \\")
            first_row = False
        latex_lines.append(r"\hline")

    latex_lines.extend(
        [
            r"\end{tabular}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table}",
        ]
    )

    return "\n".join(latex_lines)


def save_epoch_metric_latex_table(
    output_path,
    metric_tables,
    epochs=(400, 800, 1200),
    caption="Results across epochs.",
    label="tab:epoch_results",
    digits=4,
):
    """
    Build and save a LaTeX table to disk.
    """
    output_path = Path(output_path)
    latex = build_epoch_metric_latex_table(
        metric_tables=metric_tables,
        epochs=epochs,
        caption=caption,
        label=label,
        digits=digits,
    )
    output_path.write_text(latex)
    return output_path


def summarize_bde_early_stopping_epochs(experiment_id, robustness_run_id=None):
    """
    Extract BDE early-stopping epoch lengths from the seed runs of one robustness run.

    If ``robustness_run_id`` is omitted, the latest run whose name starts with
    ``Robustnesstest_Data_`` inside the given experiment is used.

    The function accepts either ``bde_history_length_model0`` or
    ``bde_epochs_ran_model0`` as the seed-run param storing the epoch count.
    """
    client = _get_mlflow_client()
    experiment_id = str(experiment_id)

    if robustness_run_id is None:
        robustness_runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string="attributes.run_name LIKE 'Robustnesstest_Data_%'",
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if not robustness_runs:
            raise ValueError(
                f"No robustness run found in experiment {experiment_id}."
            )
        robustness_run = robustness_runs[0]
    else:
        robustness_run = client.get_run(robustness_run_id)

    bde_model_runs = client.search_runs(
        experiment_ids=[robustness_run.info.experiment_id],
        filter_string=(
            f"tags.mlflow.parentRunId = '{robustness_run.info.run_id}' "
            "AND attributes.run_name = 'ModelBDE_robust'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not bde_model_runs:
        raise ValueError(
            f"No BDE robustness child run found under parent run {robustness_run.info.run_id}."
        )

    bde_run = bde_model_runs[0]
    seed_runs = client.search_runs(
        experiment_ids=[bde_run.info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{bde_run.info.run_id}'",
        order_by=["attributes.start_time ASC"],
        max_results=100,
    )
    if not seed_runs:
        raise ValueError(
            f"No seed runs found under BDE run {bde_run.info.run_id}."
        )

    seed_epochs = {}
    for run in seed_runs:
        epoch_text = (
            run.data.params.get("bde_history_length_model0")
            or run.data.params.get("bde_epochs_ran_model0")
        )
        if epoch_text is None:
            continue
        seed_name = run.data.tags.get("mlflow.runName", run.info.run_id)
        seed_epochs[seed_name] = int(epoch_text)

    if not seed_epochs:
        raise ValueError(
            f"No seed runs under BDE run {bde_run.info.run_id} contain "
            "'bde_history_length_model0' or 'bde_epochs_ran_model0'."
        )

    values = list(seed_epochs.values())
    mean_epoch = statistics.mean(values)
    std_epoch = statistics.stdev(values) if len(values) > 1 else 0.0

    return {
        "experiment_id": str(robustness_run.info.experiment_id),
        "robustness_run_id": robustness_run.info.run_id,
        "robustness_run_name": robustness_run.data.tags.get("mlflow.runName"),
        "bde_run_id": bde_run.info.run_id,
        "seed_epochs": seed_epochs,
        "mean_epoch": mean_epoch,
        "std_epoch": std_epoch,
        "n_seeds": len(values),
        "summary": f"{mean_epoch:.1f} \\pm {std_epoch:.1f}",
    }
