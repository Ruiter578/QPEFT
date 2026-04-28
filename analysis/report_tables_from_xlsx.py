from __future__ import annotations

import csv
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis/report_tables"
CSV_DIR = OUT_DIR / "csv"
MD_DIR = OUT_DIR / "markdown"

NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
REL_NS = {"pr": "http://schemas.openxmlformats.org/package/2006/relationships"}

NATURAL = ["Cifar100", "Caltech101", "DTD", "Flowers102", "Pets", "SVHN", "Sun397"]
SPECIALIZED = ["Camelyon", "EuroSAT", "Resisc45", "Retinopathy"]
STRUCTURED = ["Clevr-Count", "Clevr-Dist", "DMLAB", "KITTI-Dist", "dSpr-Loc", "dSpr-Ori", "sNORB-Azim", "sNORB-Ele"]
VTAB_TASKS = NATURAL + SPECIALIZED + STRUCTURED


def colrow(ref: str) -> tuple[int, int]:
    match = re.match(r"([A-Z]+)(\d+)", ref)
    if not match:
        raise ValueError(f"Bad cell reference: {ref}")
    col = 0
    for char in match.group(1):
        col = col * 26 + ord(char) - 64
    return col, int(match.group(2))


def cell_to_number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.startswith("="):
        return None
    if text.endswith("s"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def fmt_num(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.2f}"


def read_workbook(path: Path) -> dict[str, list[dict[int, object]]]:
    with zipfile.ZipFile(path) as zf:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", NS):
                shared.append("".join(t.text or "" for t in item.iter(f"{{{NS['a']}}}t")))

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rid_to_target = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels.findall("pr:Relationship", REL_NS)}

        sheets: dict[str, list[dict[int, object]]] = {}
        for sheet in workbook.find("a:sheets", NS):
            sheet_name = sheet.attrib["name"]
            target = rid_to_target[sheet.attrib[f"{{{NS['r']}}}id"]]
            if target.startswith("worksheets/"):
                sheet_path = "xl/" + target
            elif target.startswith("/xl/"):
                sheet_path = target.lstrip("/")
            else:
                sheet_path = "xl/worksheets/" + target

            root = ET.fromstring(zf.read(sheet_path))
            rows: list[dict[int, object]] = []
            for row in root.findall(".//a:row", NS):
                row_index = int(row.attrib.get("r", "0"))
                values: dict[int, object] = {0: row_index}
                for cell in row.findall("a:c", NS):
                    col, _ = colrow(cell.attrib["r"])
                    cell_type = cell.attrib.get("t")
                    value_node = cell.find("a:v", NS)
                    formula_node = cell.find("a:f", NS)
                    if cell_type == "s":
                        value = shared[int(value_node.text)] if value_node is not None and value_node.text else ""
                    elif cell_type == "inlineStr":
                        value = "".join(t.text or "" for t in cell.iter(f"{{{NS['a']}}}t"))
                    elif value_node is not None and value_node.text is not None:
                        value = value_node.text
                    elif formula_node is not None:
                        value = "=" + (formula_node.text or "")
                    else:
                        value = ""
                    values[col] = value
                rows.append(values)
            sheets[sheet_name] = rows
        return sheets


def row_value(row: dict[int, object], col: int) -> str:
    return str(row.get(col, "")).strip()


def build_header_map(header_row: dict[int, object]) -> dict[str, int]:
    return {str(v).strip(): k for k, v in header_row.items() if isinstance(k, int) and k > 0 and str(v).strip()}


def is_vtab_sheet(rows: list[dict[int, object]]) -> bool:
    if len(rows) < 2:
        return False
    header = build_header_map(rows[1])
    return all(task in header for task in VTAB_TASKS)


def extract_vtab_rows(workbook_name: str, sheet_name: str, rows: list[dict[int, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    top_header = build_header_map(rows[0])
    header = build_header_map(rows[1])
    params_col = top_header.get("Params") or header.get("Params")
    comments_col = top_header.get("comments") or top_header.get("Comments") or header.get("comments") or header.get("Comments")
    output: list[dict[str, object]] = []
    anomalies: list[dict[str, object]] = []

    last_method = ""
    for row in rows[2:]:
        method_cell = row_value(row, 1)
        wa_cell = row_value(row, 2)
        params_cell = row_value(row, params_col) if params_col else ""

        if method_cell:
            last_method = method_cell

        task_values = {task: cell_to_number(row.get(header[task])) for task in VTAB_TASKS}
        present_values = [v for v in task_values.values() if v is not None]

        if len(present_values) != len(VTAB_TASKS):
            continue
        if any(v > 100 for v in present_values):
            continue

        method = method_cell or last_method
        if not method and not wa_cell:
            continue

        natural_values = [task_values[t] for t in NATURAL]
        specialized_values = [task_values[t] for t in SPECIALIZED]
        structured_values = [task_values[t] for t in STRUCTURED]
        natural_avg = mean(natural_values)
        specialized_avg = mean(specialized_values)
        structured_avg = mean(structured_values)
        group_avg = mean([natural_avg, specialized_avg, structured_avg])
        tot_avg = mean(present_values)

        notes: list[str] = []
        for group_name, group_tasks, group_values in [
            ("Natural", NATURAL, natural_values),
            ("Specialized", SPECIALIZED, specialized_values),
            ("Structured", STRUCTURED, structured_values),
        ]:
            group_median = median(group_values)
            for task, value in zip(group_tasks, group_values):
                if 0 < value < 1 and group_median > 50:
                    note = (
                        f"Suspicious percentage scale: {task}={value}; "
                        f"{group_name} median={group_median:.2f}. Possibly {value * 100:.2f}."
                    )
                    notes.append(note)
                    anomalies.append(
                        {
                            "Workbook": workbook_name,
                            "Sheet": sheet_name,
                            "Row": row.get(0),
                            "Method": method,
                            "W/A": wa_cell,
                            "Task": task,
                            "Value": value,
                            "Group": group_name,
                            "Group Median": group_median,
                            "Suggested Check": f"Maybe {value * 100:.2f} if entered as a fraction.",
                        }
                    )

        output.append(
            {
                "Workbook": workbook_name,
                "Sheet": sheet_name,
                "Row": row.get(0),
                "Method": method,
                "W/A": wa_cell,
                "Params": params_cell,
                "Natural Avg": natural_avg,
                "Specialized Avg": specialized_avg,
                "Structured Avg": structured_avg,
                "Group Avg": group_avg,
                "Tot. Avg": tot_avg,
                "Comments": row_value(row, comments_col) if comments_col else "",
                "Anomaly Notes": " | ".join(notes),
            }
        )

    return output, anomalies


def extract_full_img_cls(workbook_name: str, sheet_name: str, rows: list[dict[int, object]]) -> list[dict[str, object]]:
    if not rows:
        return []
    headers = {str(v).strip().lower(): k for k, v in rows[0].items() if isinstance(k, int) and k > 0 and str(v).strip()}
    needed = ["cifar", "food101", "svhn"]
    if not all(k in headers for k in needed):
        return []
    output: list[dict[str, object]] = []
    for row in rows[1:]:
        method = row_value(row, 1)
        values = [cell_to_number(row.get(headers[k])) for k in needed]
        if not method or any(v is None for v in values):
            continue
        output.append(
            {
                "Workbook": workbook_name,
                "Sheet": sheet_name,
                "Row": row.get(0),
                "Method": method,
                "CIFAR": values[0],
                "Food101": values[1],
                "SVHN": values[2],
                "Avg": mean(values),
                "Comments": row_value(row, 5),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            formatted = {}
            for col in columns:
                value = row.get(col, "")
                if isinstance(value, float):
                    formatted[col] = f"{value:.4f}"
                else:
                    formatted[col] = value
            writer.writerow(formatted)


def markdown_table(rows: list[dict[str, object]], columns: list[str], numeric_cols: set[str] | None = None) -> str:
    numeric_cols = numeric_cols or set()
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---:" if col in numeric_cols else "---" for col in columns) + " |")
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                cells.append(fmt_num(value))
            else:
                text = str(value).replace("\n", " ").replace("|", "\\|")
                cells.append(text)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    MD_DIR.mkdir(parents=True, exist_ok=True)

    workbooks = [ROOT / "qpeft_exps.xlsx", ROOT / "QPEFT_PTQ对照实验结果.xlsx"]
    all_vtab: list[dict[str, object]] = []
    all_anomalies: list[dict[str, object]] = []
    all_full_img: list[dict[str, object]] = []
    markdown_sections: list[str] = ["# Clean Report Tables\n"]

    vtab_columns = [
        "Workbook",
        "Sheet",
        "Method",
        "W/A",
        "Params",
        "Natural Avg",
        "Specialized Avg",
        "Structured Avg",
        "Group Avg",
        "Tot. Avg",
        "Comments",
        "Anomaly Notes",
    ]
    vtab_md_columns = [
        "Method",
        "W/A",
        "Params",
        "Natural Avg",
        "Specialized Avg",
        "Structured Avg",
        "Group Avg",
        "Tot. Avg",
        "Comments",
        "Anomaly Notes",
    ]
    numeric_vtab = {"Natural Avg", "Specialized Avg", "Structured Avg", "Group Avg", "Tot. Avg"}

    for workbook_path in workbooks:
        workbook_name = workbook_path.name
        sheets = read_workbook(workbook_path)
        workbook_rows: list[dict[str, object]] = []
        markdown_sections.append(f"## {workbook_name}\n")
        for sheet_name, rows in sheets.items():
            if is_vtab_sheet(rows):
                table_rows, anomalies = extract_vtab_rows(workbook_name, sheet_name, rows)
                all_vtab.extend(table_rows)
                workbook_rows.extend(table_rows)
                all_anomalies.extend(anomalies)
                if table_rows:
                    write_csv(CSV_DIR / f"{workbook_path.stem}__{sheet_name}__vtab_summary.csv", table_rows, vtab_columns)
                    markdown_sections.append(f"### {sheet_name}\n")
                    markdown_sections.append(markdown_table(table_rows, vtab_md_columns, numeric_vtab))
                    markdown_sections.append("")
            else:
                full_rows = extract_full_img_cls(workbook_name, sheet_name, rows)
                if full_rows:
                    all_full_img.extend(full_rows)
                    cols = ["Workbook", "Sheet", "Method", "CIFAR", "Food101", "SVHN", "Avg", "Comments"]
                    write_csv(CSV_DIR / f"{workbook_path.stem}__{sheet_name}__summary.csv", full_rows, cols)
                    markdown_sections.append(f"### {sheet_name} (non-VTAB)\n")
                    markdown_sections.append(markdown_table(full_rows, ["Method", "CIFAR", "Food101", "SVHN", "Avg", "Comments"], {"CIFAR", "Food101", "SVHN", "Avg"}))
                    markdown_sections.append("")
        if workbook_rows:
            write_csv(CSV_DIR / f"{workbook_path.stem}__all_vtab_summary.csv", workbook_rows, vtab_columns)

    write_csv(CSV_DIR / "all_vtab_report_summary.csv", all_vtab, vtab_columns)
    if all_full_img:
        write_csv(CSV_DIR / "all_full_img_cls_summary.csv", all_full_img, ["Workbook", "Sheet", "Method", "CIFAR", "Food101", "SVHN", "Avg", "Comments"])

    anomaly_columns = ["Workbook", "Sheet", "Row", "Method", "W/A", "Task", "Value", "Group", "Group Median", "Suggested Check"]
    write_csv(CSV_DIR / "data_anomalies.csv", all_anomalies, anomaly_columns)

    markdown_sections.append("## Data Anomalies\n")
    if all_anomalies:
        markdown_sections.append(markdown_table(all_anomalies, anomaly_columns, {"Value", "Group Median"}))
    else:
        markdown_sections.append("No suspicious percentage-scale anomalies detected.")
    markdown_sections.append("")

    write_markdown(MD_DIR / "all_report_tables.md", "\n".join(markdown_sections))
    write_markdown(MD_DIR / "all_vtab_report_summary.md", markdown_table(all_vtab, vtab_columns, numeric_vtab))
    if all_full_img:
        write_markdown(MD_DIR / "all_full_img_cls_summary.md", markdown_table(all_full_img, ["Workbook", "Sheet", "Method", "CIFAR", "Food101", "SVHN", "Avg", "Comments"], {"CIFAR", "Food101", "SVHN", "Avg"}))
    write_markdown(MD_DIR / "data_anomalies.md", markdown_table(all_anomalies, anomaly_columns, {"Value", "Group Median"}) if all_anomalies else "No suspicious percentage-scale anomalies detected.\n")

    print(f"CSV directory: {CSV_DIR.relative_to(ROOT)}")
    print(f"Markdown directory: {MD_DIR.relative_to(ROOT)}")
    print(f"VTAB rows: {len(all_vtab)}")
    print(f"Full image classification rows: {len(all_full_img)}")
    print(f"Anomalies: {len(all_anomalies)}")


if __name__ == "__main__":
    main()
