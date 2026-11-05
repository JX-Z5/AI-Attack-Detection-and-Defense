from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


TEXT = RGBColor(24, 24, 24)


def _set_font(run, size: float | None = None, bold: bool | None = None, italic: bool | None = None) -> None:
    run.font.name = "Times New Roman"
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), "Times New Roman")
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), "Times New Roman")
    run.font.color.rgb = TEXT
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if italic is not None:
        run.font.italic = italic


def _conference_border(cell, row_index: int, last_row: int) -> None:
    properties = cell._tc.get_or_add_tcPr()
    borders = properties.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        properties.append(borders)
    settings = {
        "top": ("single", "10") if row_index == 0 else ("nil", "0"),
        "bottom": ("single", "6") if row_index == 0 else (("single", "10") if row_index == last_row else ("nil", "0")),
        "left": ("nil", "0"),
        "right": ("nil", "0"),
        "insideH": ("nil", "0"),
        "insideV": ("nil", "0"),
    }
    for edge, values in settings.items():
        tag = "w:" + edge
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn("w:val"), values[0])
        element.set(qn("w:sz"), values[1])
        element.set(qn("w:color"), "000000")


def _set_cell_margin(cell, top: int = 70, start: int = 80, bottom: int = 70, end: int = 80) -> None:
    properties = cell._tc.get_or_add_tcPr()
    margins = properties.first_child_found_in("w:tcMar")
    if margins is None:
        margins = OxmlElement("w:tcMar")
        properties.append(margins)
    for name, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        element = margins.find(qn("w:" + name))
        if element is None:
            element = OxmlElement("w:" + name)
            margins.append(element)
        element.set(qn("w:w"), str(value))
        element.set(qn("w:type"), "dxa")


def _repeat_header(row) -> None:
    properties = row._tr.get_or_add_trPr()
    element = OxmlElement("w:tblHeader")
    element.set(qn("w:val"), "true")
    properties.append(element)


def _prevent_row_split(row) -> None:
    row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))


def _keep_with_next(paragraph) -> None:
    properties = paragraph._p.get_or_add_pPr()
    element = properties.find(qn("w:keepNext"))
    if element is None:
        element = OxmlElement("w:keepNext")
        properties.append(element)
    element.set(qn("w:val"), "true")


def _add_table(doc: Document, headers: list[str], rows: list[list[str]], widths: list[float] | None = None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.autofit = False
    for index, header in enumerate(headers):
        table.rows[0].cells[index].text = header
    for values in rows:
        cells = table.add_row().cells
        for index, value in enumerate(values):
            cells[index].text = str(value)
    _repeat_header(table.rows[0])
    last_row = len(table.rows) - 1
    for row_index, row in enumerate(table.rows):
        _prevent_row_split(row)
        for column_index, cell in enumerate(row.cells):
            _conference_border(cell, row_index, last_row)
            _set_cell_margin(cell, 45, 70, 45, 70)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if widths:
                cell.width = Inches(widths[column_index])
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_before = Pt(0)
                paragraph.paragraph_format.space_after = Pt(0)
                paragraph.paragraph_format.line_spacing = 1.0
                if row_index < last_row:
                    _keep_with_next(paragraph)
                if row_index == 0 or (row_index > 0 and len(headers) > 1 and column_index > 0 and len(str(rows[row_index - 1][column_index])) <= 18):
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in paragraph.runs:
                    _set_font(run, 8.5, row_index == 0)
    after = doc.add_paragraph()
    after.paragraph_format.space_after = Pt(1)
    return table


def _add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = "PAGE"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instruction, end])


def _configure(doc: Document) -> None:
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.7)
    section.bottom_margin = Inches(0.68)
    section.left_margin = Inches(0.78)
    section.right_margin = Inches(0.78)
    section.header_distance = Inches(0.28)
    section.footer_distance = Inches(0.28)
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10.2)
    normal.font.color.rgb = TEXT
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    normal.paragraph_format.space_after = Pt(4.5)
    normal.paragraph_format.line_spacing = 1.05
    settings = {"Title": (18, 0, 4), "Subtitle": (10.5, 0, 4), "Heading 1": (13, 9, 4), "Heading 2": (10.8, 6, 2.5), "Heading 3": (10.2, 4, 2)}
    for name, values in settings.items():
        style = doc.styles[name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(values[0])
        style.font.bold = name != "Subtitle"
        style.font.color.rgb = RGBColor(0, 0, 0)
        style._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
        style.paragraph_format.space_before = Pt(values[1])
        style.paragraph_format.space_after = Pt(values[2])
        style.paragraph_format.keep_with_next = True
        if name == "Title":
            borders = style._element.get_or_add_pPr().find(qn("w:pBdr"))
            if borders is not None:
                style._element.get_or_add_pPr().remove(borders)
    for style_name in ("List Number",):
        style = doc.styles[style_name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(10.1)
        style._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
        style.paragraph_format.space_after = Pt(2)
    _add_page_number(section.footer.paragraphs[0])
    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.CENTER
    header_run = header.add_run("AI Attack Detection and Defense")
    _set_font(header_run, 8, italic=True)


def _new_page(doc: Document, title: str) -> None:
    numbering = {
        "Research questions and hypotheses": "1. Introduction and research questions",
        "Laboratory design and safety": "2. Threat model and laboratory controls",
        "Attack generators and collected data": "3. Experimental design and data collection",
        "Features and evaluation protocol": "4. Detection method and evaluation protocol",
        "Six-model generalization study": "5. Cross-model generalization results",
        "Frozen detector development": "6. Frozen detector development",
        "Optimized browser-like detector": "6. Detector optimization and ablation",
        "Untouched Falcon3 blind test": "7. Blind evaluation",
        "Online defense results": "8. Online defense results",
        "Online defense evaluation": "8. Online defense results",
        "Discussion": "9. Discussion",
        "Appendix": "Appendix",
    }
    doc.add_heading(numbering.get(title, title), level=1)


def _add_caption(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(2)
    paragraph.paragraph_format.space_after = Pt(5)
    run = paragraph.add_run(text)
    _set_font(run, 8.6, italic=True)


def _add_figure(doc: Document, path: Path, caption: str, width: float = 6.55) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_with_next = True
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.add_run().add_picture(str(path), width=Inches(width))
    _add_caption(doc, caption)


def _add_code_line(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.left_indent = Inches(0.22)
    paragraph.paragraph_format.space_before = Pt(1)
    paragraph.paragraph_format.space_after = Pt(1)
    run = paragraph.add_run(text)
    run.font.name = "Consolas"
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), "Consolas")
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), "Consolas")
    run.font.size = Pt(8.4)


def _percent(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _make_figures(development: dict, generalization: dict, blind: dict, directory: Path) -> dict[str, Path]:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "grid.color": "#D9D9D9",
        "grid.linewidth": 0.7,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    palette = {
        "blue": "#0072B2",
        "sky": "#56B4E9",
        "green": "#009E73",
        "orange": "#E69F00",
        "vermillion": "#D55E00",
        "purple": "#CC79A7",
    }
    figures: dict[str, Path] = {}
    folds = generalization["model_family_holdout"]["folds"]
    order = ["llama3.2:3b", "gemma3:4b", "phi4-mini:3.8b", "mistral:7b", "granite3.3:2b", "smollm2:1.7b"]
    labels = ["Llama 3.2", "Gemma 3", "Phi-4 Mini", "Mistral", "Granite 3.3", "SmolLM2"]
    values = [folds[name]["macro_f1"] for name in order]
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    model_colors = [palette["orange"], palette["sky"], palette["green"], palette["purple"], palette["blue"], palette["vermillion"]]
    bars = ax.bar(labels, values, color=model_colors, edgecolor="#333333", linewidth=0.5)
    ax.axhline(np.mean(values), color="#333333", linewidth=1, linestyle="--", label=f"mean {np.mean(values):.3f}")
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("Macro F1")
    ax.set_title("Leave-one-model-out generalization before optimization")
    ax.grid(axis="y", alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path = directory / "model_generalization.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    figures["generalization"] = path
    importance = development["online_feature_importance"]
    top = sorted(importance.items(), key=lambda item: item[1], reverse=True)[:8][::-1]
    feature_colors = []
    for name, _ in top:
        if name == "repetition_rate":
            feature_colors.append(palette["vermillion"])
        elif name == "unique_path_ratio":
            feature_colors.append(palette["green"])
        elif "interval" in name:
            feature_colors.append(palette["orange"])
        elif "response" in name:
            feature_colors.append(palette["purple"])
        else:
            feature_colors.append(palette["blue"])
    fig, ax = plt.subplots(figsize=(7.2, 3.45))
    ax.barh([name.replace("_", " ") for name, _ in top], [value for _, value in top], color=feature_colors, edgecolor="#333333", linewidth=0.4)
    ax.set_xlabel("Random forest importance")
    ax.set_title("Most influential online session features")
    ax.grid(axis="x", alpha=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = directory / "feature_importance.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    figures["importance"] = path
    matrix = np.array(blind["offline_ai_vs_human"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.2, 3.65))
    image = ax.imshow(matrix, cmap="magma", vmin=0, vmax=100)
    for row in range(2):
        for column in range(2):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=15, color="#111111" if matrix[row, column] > 55 else "white")
    ax.set_xticks([0, 1], ["Human-like", "AI agent"])
    ax.set_yticks([0, 1], ["Human-like", "AI agent"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Falcon3 blind-test offline confusion matrix")
    fig.colorbar(image, ax=ax, fraction=0.047, pad=0.04)
    fig.tight_layout()
    path = directory / "blind_confusion_matrix.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    figures["confusion"] = path
    defense = blind["online_defense"]
    labels = ["AI agent\nblocked", "Browser-like script\nblocked", "Human-like\nfalsely blocked"]
    values = [defense["ai_block_rate"], defense["script_block_rate"], defense["human_false_block_rate"]]
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    bars = ax.bar(labels, values, color=[palette["green"], palette["orange"], palette["vermillion"]], edgecolor="#333333", linewidth=0.5, width=0.62)
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("Session rate")
    ax.set_title("Frozen online policy on the untouched blind set")
    ax.grid(axis="y", alpha=0.7)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, _percent(value, 0), ha="center", fontsize=9)
    fig.tight_layout()
    path = directory / "blind_defense.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    figures["defense"] = path
    values = [development["offline_group_oof"]["macro_f1"], development["without_timing_features"]["macro_f1"]]
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    bars = ax.bar(["All features", "Timing removed"], values, color=[palette["green"], palette["vermillion"]], edgecolor="#333333", linewidth=0.5, width=0.58)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Macro F1")
    ax.set_title("Development ablation")
    ax.grid(axis="y", alpha=0.7)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center")
    fig.tight_layout()
    path = directory / "timing_ablation.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    figures["ablation"] = path
    return figures


def build_report(artifacts_root: Path, output_path: Path) -> Path:
    generalization_path = artifacts_root / "generalization" / "results" / "generalization_metrics.json"
    generalization_metadata_path = artifacts_root / "generalization" / "processed" / "generalization_metadata.json"
    development_path = artifacts_root / "optimization" / "development" / "optimized_development_metrics.json"
    detector_metadata_path = artifacts_root / "optimization" / "development" / "optimized_detector_metadata.json"
    blind_path = artifacts_root / "optimization" / "blind" / "results" / "blind_metrics.json"
    blind_metadata_path = artifacts_root / "optimization" / "blind" / "processed" / "blind_metadata.json"
    pilot_path = artifacts_root / "results" / "metrics.json"
    required = [generalization_path, generalization_metadata_path, development_path, detector_metadata_path, blind_path, blind_metadata_path, pilot_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing report evidence: " + ", ".join(missing))
    generalization = _load_json(generalization_path)
    generalization_metadata = _load_json(generalization_metadata_path)
    development = _load_json(development_path)
    detector_metadata = _load_json(detector_metadata_path)
    blind = _load_json(blind_path)
    blind_metadata = _load_json(blind_metadata_path)
    pilot = _load_json(pilot_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as temporary:
        figures = _make_figures(development, generalization, blind, Path(temporary))
        doc = Document()
        _configure(doc)
        title = doc.add_paragraph(style="Title")
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        title.paragraph_format.space_before = Pt(4)
        title.add_run("AI Attack Detection and Defense")
        subtitle = doc.add_paragraph(style="Subtitle")
        subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
        subtitle.add_run("Behavior-Based Detection of Local AI Agents in an Isolated Web Security Laboratory")
        course = doc.add_paragraph()
        course.alignment = WD_ALIGN_PARAGRAPH.CENTER
        course.paragraph_format.space_before = Pt(2)
        course.paragraph_format.space_after = Pt(3)
        first = course.add_run("Student Name · Student ID\n")
        _set_font(first, 10.5, True)
        run = course.add_run("November 2026")
        _set_font(run, 9.5)

        _new_page(doc, "Abstract")
        abstract = doc.add_paragraph()
        abstract.paragraph_format.left_indent = Inches(0.35)
        abstract.paragraph_format.right_indent = Inches(0.35)
        abstract.paragraph_format.space_after = Pt(4)
        abstract.add_run("This paper investigates whether request-level behavior can support detection of AI-controlled web attacks without access to prompts, hidden reasoning, or model identity. An isolated Docker laboratory was built around DVWA, and local language models were restricted to an allowlisted SQL-injection action space. The formal data set comprises 2,100 sessions and 14,139 requests, balanced across AI agents, scripts, and human-like simulations. Six model families were used during development; Falcon3:3b was reserved for an untouched blind evaluation. An initial pilot achieved perfect scores but was rejected as evidence after generator-specific regularities were identified. The redesigned evaluation used grouped model holdouts, prompt and task variation, feature exclusion, confidence intervals, and frozen thresholds. On Falcon3, the offline classifier obtained macro F1 0.808 (95% bootstrap CI 0.757 to 0.857), with 63% AI recall and no false positives among 100 human-like sessions. The online policy blocked 85% of AI sessions after 4.09 requests on average, while producing no observed human-like false blocks (95% Wilson upper bound 3.7%). Together, the blind offline and online results show that observable session behavior can support useful detection of previously unseen AI-agent traffic in this laboratory. Performance differed across prompt strategies, timing patterns, and attack tasks.")
        keywords = doc.add_paragraph()
        keywords.paragraph_format.left_indent = Inches(0.35)
        keywords.paragraph_format.right_indent = Inches(0.35)
        run = keywords.add_run("Keywords: AI agents; intrusion detection; behavioral classification; random forest; blind evaluation; DVWA")
        _set_font(run, 9.2, italic=True)
        doc.add_heading("Experimental progression", level=2)
        _add_table(doc, ["Stage", "Purpose", "Result used in the report"], [["Pilot", "Check the complete pipeline", "Perfect scores exposed an overly simple data design"], ["Six-model study", "Test model-family transfer", "Mean leave-one-model-out macro F1 0.788; worst fold 0.506"], ["Optimization", "Freeze a browser-like AI detector", "Offline macro F1 0.851; online macro F1 0.872"], ["Blind test", "Evaluate an untouched model and new profiles", "Offline macro F1 0.808; online AI block rate 85%"]], [1.0, 2.35, 3.35])

        _new_page(doc, "Research questions and hypotheses")
        doc.add_paragraph("I wanted to study whether an AI-controlled attacker leaves request-level patterns that can be detected without reading its prompts or internal reasoning. I therefore built an isolated attack-and-defense laboratory in which local models interact with a known vulnerable web application through a restricted action interface. The study is organized around three research questions.")
        for text in ("RQ1. Can observable session behavior separate local AI-agent attacks from human-like simulated activity?", "RQ2. Does a detector trained on several model families transfer to a model that was not used during development?", "RQ3. Can an online policy block AI-agent sessions while keeping human-like false blocks below 10%?"):
            doc.add_paragraph(text, style="List Number")
        doc.add_heading("Hypotheses", level=2)
        _add_table(doc, ["ID", "Statement", "Measure"], [["H1", "Session features provide better than chance separation on grouped development data.", "Macro precision, recall, F1, and AI recall"], ["H2", "Performance remains useful when every session from one AI model is held out.", "Leave-one-model-out folds"], ["H3", "A frozen policy detects the untouched Falcon3 model with at most 10% human-like false blocks.", "Blind block rate and confidence intervals"], ["H4", "Timing contributes useful information beyond request order and repetition.", "Timing-feature ablation"]], [0.5, 4.15, 2.05])
        doc.add_heading("Scope of inference", level=2)
        doc.add_paragraph("The comparison class is a software-generated human-like simulation. The classifier estimates whether a session resembles the AI-agent behavior observed in this laboratory from request features; model identity, prompt text, and action rationale are not detector inputs.")
        doc.add_heading("Threat model", level=2)
        doc.add_paragraph("The attacker is a local language model that observes compact response markers and selects an approved next action. It cannot submit an arbitrary URL, SQL string, shell command, file, or network destination. The defender observes request timing, approved path and payload identifiers, status codes, response sizes, markers, and User-Agent categories. Hidden reasoning and prompt text are unavailable to the detector.")

        _new_page(doc, "Laboratory design and safety")
        doc.add_paragraph("DVWA is a deliberately vulnerable PHP and MariaDB application intended for security teaching in controlled environments [1]. Its documentation warns against Internet exposure. The DVWA Docker image was therefore used as the sole target, and the attack scope was limited to SQL injection and blind SQL injection.")
        _add_table(doc, ["Component", "Reachable services", "Role"], [["Runner", "Gateway and Ollama", "Runs generators and records structured actions"], ["Ollama", "Runner", "Serves local language models after they have been downloaded"], ["Gateway", "Runner and DVWA", "Checks allowlists, forwards approved requests, and applies defense"], ["DVWA", "Gateway and MariaDB", "Provides the known vulnerable teaching target"], ["MariaDB", "DVWA", "Stores disposable target data"]], [1.15, 1.75, 3.8])
        doc.add_heading("Controls used during every run", level=2)
        _add_table(doc, ["Control", "Implementation"], [["Network isolation", "agent_net, runner_net, and target_net are Docker internal networks"], ["Host exposure", "DVWA, Ollama, and the gateway bind only to 127.0.0.1"], ["Target allowlist", "The runner sends a path identifier, not an arbitrary URL"], ["Payload allowlist", "The gateway maps a small payload identifier to a fixed teaching probe"], ["Bounded tools", "The LangGraph agent receives no shell, file, persistence, or exfiltration tool"], ["Bounded sessions", "Every session contains 6 to 10 requests"], ["Local inference", "Model prompts and responses remain inside the laboratory"]], [1.65, 5.05])
        doc.add_paragraph("Docker defines an internal Compose network as a network without an external gateway [2]. Network isolation was treated as one defensive layer rather than a complete safety argument. Allowlist validation occurs before forwarding, and automated checks verify rejection of external hosts, arbitrary paths, unapproved payloads, and URL fields.")

        _new_page(doc, "Attack generators and collected data")
        doc.add_heading("AI agent", level=2)
        doc.add_paragraph("Each AI attack was represented as a short stateful LangGraph loop. The model received the previous approved action, status code, response marker, and response size, then selected one structured action from the allowlist. LangGraph supports workflows that combine deterministic transitions with model decisions [3], enabling model-directed action selection while preserving a fixed target and tool interface.")
        doc.add_paragraph("The development set used llama3.2:3b, gemma3:4b, phi4-mini:3.8b, mistral:7b, granite3.3:2b, and smollm2:1.7b. Each model completed 100 sessions distributed across methodical, cautious, exploratory, and adaptive prompt profiles. Falcon3:3b produced 100 later sessions and remained uninspected during feature and threshold development. Ollama served all seven local models [4].")
        doc.add_heading("Comparison generators", level=2)
        doc.add_paragraph("The script group followed program rules without a language model. Eight development profiles changed order, retry behavior, speed, and User-Agent style. The blind set added stealth_burst and adaptive_order profiles that used browser-like identifiers. The human-like generator introduced pauses, repeated requests, backtracking, and occasional changes of plan. Its development set used eight profiles; the blind set added bursty_recovery and careful_backtrack.")
        doc.add_heading("Formal data set", level=2)
        _add_table(doc, ["Partition", "AI agent", "Script", "Human-like", "Requests"], [["Development and generalization", "600", "600", "600", "11,726"], ["Untouched blind test", "100", "100", "100", "2,413"], ["Total", "700", "700", "700", "14,139"]], [2.55, 0.9, 0.9, 1.05, 1.15])
        doc.add_paragraph("Each raw JSONL event stores a session identifier, true generator label, virtual time, approved path and payload IDs, status code, response marker, response size, User-Agent profile, agent action, and gateway decision. The feature CSV has one row per complete session. Session identifiers and labels are not model features.")

        _new_page(doc, "Features and evaluation protocol")
        doc.add_heading("Session features", level=2)
        _add_table(doc, ["Group", "Examples", "Reason for inclusion"], [["Timing", "Mean, standard deviation, rate, duration, coefficient of variation, 80% range", "Measure pace and variation"], ["Navigation", "Unique path ratio and path sequence score", "Measure exploration and ordered movement"], ["Repetition", "Consecutive repetition rate and maximum payload share", "Measure loops and action collapse"], ["Payload choice", "Unique ratio, entropy, transition entropy, switch rate", "Measure diversity and changes of action"], ["Response use", "Response-size variation, marker entropy, marker-change switches", "Measure reaction to observations"], ["Request result", "Error rate, mean response bytes, request count", "Describe the completed interaction"], ["Client identity", "User-Agent anomaly rate", "Retain a simple rule signal without model identity"]], [1.25, 3.45, 2.0])
        doc.add_paragraph("The final representation contains 25 features. Prompt profile, generator profile, model identity, action rationale, and exact interval roundness were excluded. Interval roundness produced unusually high development scores because the generators used convenient delay values; it was removed before blind evaluation because it encoded an implementation habit with weak transfer validity.")
        doc.add_heading("Model and splitting", level=2)
        doc.add_paragraph("The detector is a random forest with 700 trees, minimum leaf size 5, 70% feature sampling, balanced-subsample class weights, and seed 20261102. Random forests aggregate decision trees trained with stochastic sampling of observations and features [5, 6]. The model was selected because it accommodates nonlinear feature interactions, exposes class probabilities for thresholding, and provides an interpretable feature-importance summary.")
        doc.add_paragraph("Development evaluation used group out-of-fold predictions to prevent model families or behavior profiles from appearing in both training and test partitions. Threshold selection used only development predictions. The permitted human-like false-positive rate was 10%, including the worst individual human-like profile. The offline model, online prefix model, feature list, and threshold 0.79 were frozen before Falcon3 data collection.")

        doc.add_heading("Pilot result and experimental redesign", level=1)
        doc.add_paragraph(f"The first complete run contained {sum(pilot['sample_counts'].values())} sessions and 1,190 requests. The locked-test random forest, rule baseline, and timing ablation each reached macro F1 1.000. Event-level inspection identified a design artifact: the Llama agent repeatedly selected the same quote-probe action, the script followed a fixed sequence, and the human-like generator visited multiple pages. Random session splitting therefore placed near-duplicate behavior in both training and test sets.")
        doc.add_heading("Diagnosis of the perfect pilot score", level=2)
        doc.add_paragraph("The pilot verified end-to-end operation of logging, feature extraction, training, and replay, but did not test transfer to a new model family. Perfect performance was attainable by learning generator implementations rather than transferable attack behavior. The unchanged 1.000 timing-ablation score further showed that several non-timing shortcuts separated the initial classes.")
        doc.add_heading("Corrective design changes", level=2)
        doc.add_paragraph(
            "(1) The AI group was expanded from one model to six development model families; "
            "(2) four prompt profiles, two attack tasks, eight script profiles, and eight human-like profiles were added; "
            "(3) evaluation was grouped by complete model family instead of mixing sessions randomly; "
            "(4) entropy, transition, response-variation, and observation-reaction features were added; "
            "(5) exact interval roundness was removed after it was identified as a generator fingerprint; and "
            "(6) Falcon3, two script profiles, and two human-like profiles were reserved for a later blind run."
        )
        doc.add_paragraph("The redesign also narrowed the classifier's inferential target. The pilot was a three-class systems demonstration; the final browser-like detector addresses AI-agent versus human-like sessions. Script traffic is retained as a separate online comparison because sensitivity to AI-style behavior should not be interpreted as general bot detection.")

        _new_page(doc, "Six-model generalization study")
        doc.add_paragraph("The formal development run collected 1,800 sessions and 11,726 requests. In the first generalization analysis, each fold removed every AI session from one model family and tested that model with held-out non-AI sessions. The mean macro F1 across six folds was 0.788. The range from 0.506 to 0.946 was more informative than the mean because it showed that model family strongly affected detectability.")
        _add_figure(doc, figures["generalization"], "Figure 1. Macro F1 when each development model family was held out in turn", 6.65)
        folds = generalization["model_family_holdout"]["folds"]
        _add_table(doc, ["Unseen model", "Macro F1", "AI recall", "95% CI for macro F1"], [[name, f"{folds[name]['macro_f1']:.3f}", _percent(folds[name]['classification_report']['ai_agent']['recall']), f"{folds[name]['macro_f1_95_ci']['lower_95']:.3f} to {folds[name]['macro_f1_95_ci']['upper_95']:.3f}"] for name in ("llama3.2:3b", "gemma3:4b", "phi4-mini:3.8b", "mistral:7b", "granite3.3:2b", "smollm2:1.7b")], [2.1, 1.1, 1.1, 2.4])
        doc.add_paragraph("SmolLM2 was the weakest fold. Its AI recall was 18% and macro F1 was 0.506. On that locked fold, the hand-written rule baseline reached macro F1 0.864, while the random forest without timing features fell to 0.321. The original online policy blocked none of the held-out SmolLM2 sessions. These results caused the second development step rather than being hidden behind the stronger folds.")

        _new_page(doc, "Frozen detector development")
        doc.add_paragraph("The primary detector was rebuilt around browser-like AI-agent and human-like sessions. Script sessions were excluded from fitting and retained for defense evaluation. The offline model consumes complete sessions, whereas the online model is trained on prefixes beginning at four requests, matching the information available at gateway decision time.")
        _add_table(doc, ["Development result", "Offline", "Online prefix"], [["Macro precision", f"{development['offline_group_oof']['macro_precision']:.3f}", f"{development['online_group_oof']['macro_precision']:.3f}"], ["Macro recall", f"{development['offline_group_oof']['macro_recall']:.3f}", f"{development['online_group_oof']['macro_recall']:.3f}"], ["Macro F1", f"{development['offline_group_oof']['macro_f1']:.3f}", f"{development['online_group_oof']['macro_f1']:.3f}"], ["95% interval", f"{development['offline_group_oof']['macro_f1_95_ci']['lower_95']:.3f} to {development['offline_group_oof']['macro_f1_95_ci']['upper_95']:.3f}", f"{development['online_group_oof']['macro_f1_95_ci']['lower_95']:.3f} to {development['online_group_oof']['macro_f1_95_ci']['upper_95']:.3f}"], ["AI recall", _percent(development['offline_group_oof']['ai_recall']), _percent(development['online_group_oof']['ai_recall'])], ["Human-like false-positive rate", _percent(development['offline_group_oof']['human_false_positive_rate']), _percent(development['online_group_oof']['human_false_positive_rate'])], ["Worst human-like profile", _percent(development['offline_policy']['oof_worst_human_profile_false_positive_rate']), _percent(development['online_policy']['oof_worst_human_profile_false_positive_rate'])]], [3.1, 1.8, 1.8])
        _add_figure(doc, figures["importance"], "Figure 2. Feature importance of the frozen online prefix model", 5.45)
        doc.add_paragraph("Repetition rate and unique path ratio provided most of the model's importance. Interval variation came next. User-Agent anomaly had zero importance in this browser-like model, which was expected because the compared sessions used normal browser-style identities.")

        doc.add_heading("6.1 Timing ablation and frozen configuration", level=2)
        _add_figure(doc, figures["ablation"], "Figure 3. Removing six timing features from the development classifier", 4.75)
        doc.add_paragraph("Removing timing features reduced development macro F1 from 0.851 to 0.444. AI recall fell from 73.0% to 11.2%, while the human-like false-positive rate remained low at 1.0%. The detector therefore depends heavily on timing and repeated navigation. That dependence is useful in the present lab, but an attacker that deliberately adds realistic delays may reduce detection.")
        doc.add_heading("Values frozen before the blind run", level=2)
        _add_table(doc, ["Item", "Frozen value"], [["Offline threshold", "0.79"], ["Online threshold", "0.79"], ["Online warm-up", "4 requests"], ["Feature count", "25 observable session features"], ["Classifier", "Random forest, 700 trees, minimum leaf size 5"], ["Development seed", str(detector_metadata['seed'])], ["Model bundle SHA-256", detector_metadata['bundle_sha256']]], [2.15, 4.55])
        doc.add_paragraph("No blind data were used to revise these values. The hash stored with the blind result matches the development bundle hash, and the metrics record confirms that thresholds were reused without blind-set tuning.")

        _new_page(doc, "Untouched Falcon3 blind test")
        doc.add_paragraph("After detector freezing, Falcon3:3b produced 100 AI-agent sessions. Two previously unused browser-like script profiles and two previously unused human-like profiles produced a further 100 sessions each. The blind run generated 2,413 requests in 18.3 minutes with no failed sessions.")
        _add_figure(doc, figures["confusion"], "Figure 4. Offline classification of 100 Falcon3 and 100 human-like blind sessions", 4.45)
        _add_table(doc, ["Offline blind metric", "Value"], [["Macro precision", f"{blind['offline_ai_vs_human']['macro_precision']:.3f}"], ["Macro recall", f"{blind['offline_ai_vs_human']['macro_recall']:.3f}"], ["Macro F1", f"{blind['offline_ai_vs_human']['macro_f1']:.3f}"], ["95% bootstrap interval", f"{blind['offline_ai_vs_human']['macro_f1_95_ci']['lower_95']:.3f} to {blind['offline_ai_vs_human']['macro_f1_95_ci']['upper_95']:.3f}"], ["Falcon3 recall", _percent(blind['offline_ai_vs_human']['ai_recall'])], ["Human-like false-positive rate", _percent(blind['offline_ai_vs_human']['human_false_positive_rate'])]], [3.3, 3.4])
        doc.add_paragraph("The detector labeled all 100 human-like sessions correctly and detected 63 Falcon3 sessions. The blind result shows that the detector transferred to an unseen model while meeting the selected human-like false-block target.")

        _new_page(doc, "Online defense results")
        _add_figure(doc, figures["defense"], "Figure 5. Session outcomes from the frozen online policy on all 300 blind sessions", 5.15)
        defense = blind["online_defense"]
        _add_table(doc, ["Online outcome", "Result"], [["Falcon3 sessions blocked", f"85/100 ({_percent(defense['ai_block_rate'], 0)})"], ["95% Wilson interval for AI block rate", f"{_percent(defense['ai_block_rate_95_ci']['lower_95'])} to {_percent(defense['ai_block_rate_95_ci']['upper_95'])}"], ["Browser-like script sessions blocked", f"13/100 ({_percent(defense['script_block_rate'], 0)})"], ["Human-like sessions falsely blocked", f"0/100 ({_percent(defense['human_false_block_rate'], 0)})"], ["95% Wilson interval for human false blocks", f"0.0% to {_percent(defense['human_false_block_rate_95_ci']['upper_95'])}"], ["Mean Falcon3 detection request", f"{defense['mean_detection_request_count']['ai_agent']:.2f}"], ["Mean virtual delay to Falcon3 block", f"{defense['mean_detection_to_block_latency_ms']['ai_agent'] / 1000:.2f} seconds"]], [3.65, 3.05])
        doc.add_heading("Breakdown by prompt and task", level=2)
        _add_table(doc, ["Blind subgroup", "Offline AI recall", "Online AI block rate"], [["Adaptive prompt", "24%", "52%"], ["Cautious prompt", "60%", "100%"], ["Exploratory prompt", "80%", "96%"], ["Methodical prompt", "88%", "92%"], ["SQL injection", "75%", "94.2%"], ["Blind SQL injection", "50%", "75%"]], [2.7, 2.0, 2.0])
        doc.add_paragraph("The online prefix model performed better than the full-session model on Falcon3 because an early repeated pattern could trigger a block before later actions made the session look more varied. Adaptive prompts remained the hardest subgroup.")

        _new_page(doc, "Discussion")
        doc.add_heading("Supported inference", level=2)
        doc.add_paragraph("The blind evaluation establishes practical utility within the stated experimental scope. Although Falcon3 was absent from development, the online gateway blocked 85% of its sessions and produced no observed false blocks on the new human-like profiles. This result relied on request behavior alone; model names, prompt profiles, generator profiles, and action rationales were excluded from the representation.")
        doc.add_paragraph("Detection performance varied across model families and prompt strategies, with adaptive Falcon3 sessions producing the lowest recall. The human-like false-block result was 0/100, with a 95% Wilson upper bound of 3.7%.")
        doc.add_heading("Answers to the research questions", level=2)
        _add_table(doc, ["Question", "Answer from this experiment"], [["RQ1", "Yes in the controlled data. Development offline macro F1 was 0.851 and blind macro F1 was 0.808."], ["RQ2", "Partly. Falcon3 transferred well enough for 85% online blocking, while SmolLM2 remained difficult during development."], ["RQ3", "Yes for the blind sample. The online policy blocked 85% of AI sessions with 0 observed human-like false blocks."]], [1.0, 5.7])

        doc.add_heading("10. Conclusion", level=1)
        doc.add_paragraph("I built this project to test whether network behavior can reveal AI-controlled web attacks. A bounded AI agent interacts with DVWA, and a separate gateway records, classifies, and blocks the traffic inside an isolated laboratory. The early 100% result led me to redesign the data collection and evaluation around multiple models, grouped holdouts, and a frozen blind test.")
        doc.add_paragraph("The formal study used seven model families, 2,100 sessions, and 14,139 requests. With all detector choices fixed in advance, Falcon3 blind testing produced offline macro F1 0.808 and an online AI block rate of 85%. No false blocks occurred in 100 blind human-like simulations, although the confidence interval allows a true rate up to 3.7%. The gateway usually made its Falcon3 decision after about four requests.")
        doc.add_paragraph("Across the completed experiments, observable request behavior supported detection of attacks from an unseen local AI model. Detection performance varied by model family and prompt strategy, while the online system still blocked 85% of Falcon3 sessions after about four requests. The same experimental framework can now be extended to additional models, targets, and attack families.")
        doc.add_heading("References", level=1)
        references = ["[1] DVWA Team, Damn Vulnerable Web Application README, https://github.com/digininja/DVWA/blob/master/README.md, accessed 2 November 2026.", "[2] Docker, Networking in Compose, https://docs.docker.com/compose/how-tos/networking/, accessed 2 November 2026.", "[3] LangChain, LangGraph overview, https://langchain-ai.github.io/langgraph/, accessed 2 November 2026.", "[4] Ollama, Docker documentation, https://docs.ollama.com/docker, accessed 2 November 2026.", "[5] L. Breiman, Random Forests, Machine Learning, vol. 45, pp. 5-32, 2001, https://doi.org/10.1023/A:1010933404324.", "[6] scikit-learn developers, RandomForestClassifier documentation, https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html, accessed 2 November 2026."]
        for reference in references:
            paragraph = doc.add_paragraph(reference)
            paragraph.paragraph_format.left_indent = Inches(0.18)
            paragraph.paragraph_format.first_line_indent = Inches(-0.18)
            paragraph.paragraph_format.space_after = Pt(1)
            paragraph.paragraph_format.line_spacing = 0.95
            for reference_run in paragraph.runs:
                _set_font(reference_run, 9.2)

        _new_page(doc, "Appendix")
        doc.add_heading("A. Data fields", level=2)
        doc.add_paragraph("Raw events contain session_id, true_label, timestamp, virtual_timestamp_ms, sequence, path_id, payload_id, status_code, response_marker, response_bytes, user_agent_profile, agent_action, action_rationale, gateway_decision, model_id, prompt_profile, attack_task, and generator_profile. Rationale and group identity fields are retained for auditing but excluded from model input.")
        doc.add_heading("B. Frozen feature list", level=2)
        feature_names = [name.replace("_", " ") for name in development["feature_columns"]]
        doc.add_paragraph(", ".join(feature_names) + ".")
        doc.save(output_path)
    return output_path
