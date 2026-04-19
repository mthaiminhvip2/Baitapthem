from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Iterable, Mapping

from pose_analysis import FrameAnalysis, sort_frame_ids


def write_html_report(
    output_path: str | Path,
    before: Mapping[int | str, FrameAnalysis],
    after: Mapping[int | str, FrameAnalysis],
    selected_frame_ids: Iterable[int | str],
    title: str = "Learnable IK Solver Report",
) -> Path:
    path = Path(output_path)
    frame_ids = sort_frame_ids(selected_frame_ids)
    rows = "\n".join(_frame_section(frame_id, before.get(frame_id), after.get(frame_id)) for frame_id in frame_ids)
    html = f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #202124; }}
    h1 {{ font-size: 26px; margin-bottom: 8px; }}
    h2 {{ margin-top: 28px; border-bottom: 1px solid #d8dce3; padding-bottom: 8px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 22px; }}
    th, td {{ border: 1px solid #d8dce3; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f6fa; }}
    code {{ background: #f3f6fa; padding: 2px 5px; border-radius: 4px; }}
    .muted {{ color: #5f6368; }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <p class="muted">Báo cáo lưu các tập M, D và thống kê Q1, Q3, mean trước và sau bước sửa pose.</p>
  {rows}
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
    return path


def _frame_section(
    frame_id: int | str,
    before: FrameAnalysis | None,
    after: FrameAnalysis | None,
) -> str:
    return f"""
  <h2>Frame {escape(str(frame_id))}</h2>
  <table>
    <thead>
      <tr>
        <th>Giai đoạn</th>
        <th>M: mâu thuẫn hướng</th>
        <th>D: sai số khoảng cách lớn</th>
        <th>Q1</th>
        <th>Q3</th>
        <th>Mean</th>
        <th>Ngưỡng D</th>
      </tr>
    </thead>
    <tbody>
      {_analysis_row("Trước sửa", before)}
      {_analysis_row("Sau sửa", after)}
    </tbody>
  </table>
"""


def _analysis_row(label: str, analysis: FrameAnalysis | None) -> str:
    if analysis is None:
        return f"<tr><td>{escape(label)}</td><td colspan=\"6\">Không có dữ liệu</td></tr>"
    return f"""
      <tr>
        <td>{escape(label)}</td>
        <td>{_list_text(analysis.m_conflicts)}</td>
        <td>{_list_text(analysis.d_distance_errors)}</td>
        <td>{_num(analysis.q1)}</td>
        <td>{_num(analysis.q3)}</td>
        <td>{_num(analysis.mean)}</td>
        <td>{analysis.threshold:.6f}</td>
      </tr>
"""


def _list_text(values: list[str]) -> str:
    if not values:
        return "<span class=\"muted\">Rỗng</span>"
    return ", ".join(f"<code>{escape(value)}</code>" for value in values)


def _num(value: float | None) -> str:
    if value is None:
        return "<span class=\"muted\">N/A</span>"
    return f"{value:.6f}"
