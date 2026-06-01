"""
Utilities for turning exported LaTeX metric tables into percentage-delta series.
"""

from pathlib import Path
import re


METRIC_EXPORT_NAMES = {
    "RMSE": "rmse",
    "NLL": "nll",
    "WinklerScore": "winkler",
    "WinklerCoverage": "winklercoverage",
    "Runtime (s)": "runtime",
}


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
