# -*- coding: utf-8 -*-
"""Build the FusedSparseAttentionOverlap operator integration guide (.docx, later converted to .doc)."""
import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = Path(__file__).resolve().parent
OUT_DOCX = HERE / "FusedSparseAttentionOverlap算子接入指导.docx"

NAVY = RGBColor(0x1F, 0x38, 0x64)
BLUE = RGBColor(0x2E, 0x75, 0xB6)
TEAL = RGBColor(0x1F, 0x7A, 0x6B)
GRAY = RGBColor(0x59, 0x59, 0x59)
DARK = RGBColor(0x26, 0x26, 0x26)

BODY_CN = "宋体"
BODY_EN = "Calibri"
HEAD_CN = "微软雅黑"
CODE_FONT = "Consolas"


# --------------------------------------------------------------------------- #
# low level helpers
# --------------------------------------------------------------------------- #
def set_run_font(run, cn=BODY_CN, en=BODY_EN, size=10.5, bold=None, color=None):
    run.font.name = en
    run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), en)
    rfonts.set(qn("w:hAnsi"), en)
    rfonts.set(qn("w:eastAsia"), cn)


CODE_INLINE = RGBColor(0x1A, 0x3A, 0x5E)


def add_rich(paragraph, text, size=10.5, color=None, bold_all=False, en=None):
    """Render **bold** and `code` spans as runs."""
    for seg in re.split(r"(\*\*.+?\*\*|`[^`]+`)", text):
        if not seg:
            continue
        if seg.startswith("**") and seg.endswith("**") and len(seg) > 4:
            run = paragraph.add_run(seg[2:-2])
            set_run_font(run, en=en or BODY_EN, size=size, bold=True,
                         color=color if color is not None else DARK)
        elif seg.startswith("`") and seg.endswith("`") and len(seg) > 2:
            run = paragraph.add_run(seg[1:-1])
            set_run_font(run, cn=CODE_FONT, en=CODE_FONT, size=size - 1.0,
                         bold=True if bold_all else None, color=CODE_INLINE)
        else:
            run = paragraph.add_run(seg)
            set_run_font(run, en=en or BODY_EN, size=size,
                         bold=True if bold_all else None, color=color)


def shade(element, fill):
    """Apply w:shd to a paragraph or table cell element."""
    pr = element.get_or_add_tcPr() if element.tag.endswith("}tc") else element.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    pr.append(shd)


def cell_shade(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def para_border(paragraph, color="D5DBE3", size=6, space=4):
    p_pr = paragraph._p.get_or_add_pPr()
    borders = OxmlElement("w:pBdr")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(size))
        el.set(qn("w:space"), str(space))
        el.set(qn("w:color"), color)
        borders.append(el)
    p_pr.append(borders)


def add_page_number(paragraph):
    run = paragraph.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)
    set_run_font(run, size=9, color=GRAY)


def add_field(paragraph, instr_text, placeholder=""):
    run = paragraph.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instr_text
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = placeholder
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_sep)
    run._r.append(text)
    run._r.append(fld_end)


def set_cell_margins(table, top=60, bottom=60, left=100, right=100):
    tbl_pr = table._tbl.tblPr
    mar = OxmlElement("w:tblCellMar")
    for name, value in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
        el = OxmlElement(f"w:{name}")
        el.set(qn("w:w"), str(value))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tbl_pr.append(mar)


def set_table_borders(table, color="C9D3E0", size=4):
    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(size))
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), color)
        borders.append(el)
    tbl_pr.append(borders)


def repeat_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    tr_pr.append(header)


def no_split(row):
    tr_pr = row._tr.get_or_add_trPr()
    cant = OxmlElement("w:cantSplit")
    cant.set(qn("w:val"), "true")
    tr_pr.append(cant)


# --------------------------------------------------------------------------- #
# document level helpers
# --------------------------------------------------------------------------- #
class Doc:
    def __init__(self):
        self.d = Document()
        self._setup_page()
        self._setup_styles()
        self.figure_no = 0
        self.table_no = 0
        self.example_no = 0
        self.chapter = 0

    def set_chapter(self, number):
        self.chapter = number
        self.figure_no = 0
        self.table_no = 0

    # ---- setup ----
    def _setup_page(self):
        sec = self.d.sections[0]
        sec.page_width = Cm(21.0)
        sec.page_height = Cm(29.7)
        sec.top_margin = Cm(2.4)
        sec.bottom_margin = Cm(2.2)
        sec.left_margin = Cm(2.2)
        sec.right_margin = Cm(2.2)
        sec.header_distance = Cm(1.2)
        sec.footer_distance = Cm(1.1)
        sec.different_first_page_header_footer = True

        header_p = sec.header.paragraphs[0]
        header_p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        run = header_p.add_run("FusedSparseAttentionOverlap 算子接入指导")
        set_run_font(run, cn=HEAD_CN, en=HEAD_CN, size=8.5, color=GRAY)
        p_pr = header_p._p.get_or_add_pPr()
        borders = OxmlElement("w:pBdr")
        bottom = OxmlElement("w:bottom")
        bottom.set(qn("w:val"), "single")
        bottom.set(qn("w:sz"), "4")
        bottom.set(qn("w:space"), "1")
        bottom.set(qn("w:color"), "BFBFBF")
        borders.append(bottom)
        p_pr.append(borders)

        footer_p = sec.footer.paragraphs[0]
        footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r1 = footer_p.add_run("第 ")
        set_run_font(r1, size=9, color=GRAY)
        add_page_number(footer_p)
        r2 = footer_p.add_run(" 页")
        set_run_font(r2, size=9, color=GRAY)

    def _setup_styles(self):
        styles = self.d.styles

        normal = styles["Normal"]
        normal.font.name = BODY_EN
        normal.font.size = Pt(10.5)
        normal.font.color.rgb = DARK
        normal.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_CN)
        pf = normal.paragraph_format
        pf.line_spacing = 1.4
        pf.space_before = Pt(0)
        pf.space_after = Pt(5)

        specs = [
            ("Heading 1", 17, NAVY, 20, 10),
            ("Heading 2", 13.5, NAVY, 14, 6),
            ("Heading 3", 11.5, BLUE, 10, 4),
        ]
        for name, size, color, before, after in specs:
            st = styles[name]
            st.font.name = HEAD_CN
            st.font.size = Pt(size)
            st.font.bold = True
            st.font.color.rgb = color
            st.element.rPr.rFonts.set(qn("w:eastAsia"), HEAD_CN)
            st.paragraph_format.space_before = Pt(before)
            st.paragraph_format.space_after = Pt(after)
            st.paragraph_format.keep_with_next = True

        # code style
        code = styles.add_style("FSCODE", 1)
        code.font.name = CODE_FONT
        code.font.size = Pt(7.9)
        code.element.rPr.rFonts.set(qn("w:eastAsia"), CODE_FONT)
        code.paragraph_format.space_before = Pt(0)
        code.paragraph_format.space_after = Pt(0)
        code.paragraph_format.line_spacing = 1.12

    # ---- text ----
    def h1(self, text):
        p = self.d.add_heading(text, level=1)
        for run in p.runs:
            set_run_font(run, cn=HEAD_CN, en=HEAD_CN, size=17, bold=True, color=NAVY)
        p_pr = p._p.get_or_add_pPr()
        borders = OxmlElement("w:pBdr")
        bottom = OxmlElement("w:bottom")
        bottom.set(qn("w:val"), "single")
        bottom.set(qn("w:sz"), "10")
        bottom.set(qn("w:space"), "3")
        bottom.set(qn("w:color"), "1F3864")
        borders.append(bottom)
        p_pr.append(borders)
        return p

    def h2(self, text):
        p = self.d.add_heading(text, level=2)
        for run in p.runs:
            set_run_font(run, cn=HEAD_CN, en=HEAD_CN, size=13.5, bold=True, color=NAVY)
        return p

    def h3(self, text):
        p = self.d.add_heading(text, level=3)
        for run in p.runs:
            set_run_font(run, cn=HEAD_CN, en=HEAD_CN, size=11.5, bold=True, color=BLUE)
        return p

    def p(self, text, size=10.5, color=None, align=None, space_after=None, indent=None):
        para = self.d.add_paragraph()
        if align is not None:
            para.alignment = align
        if space_after is not None:
            para.paragraph_format.space_after = Pt(space_after)
        if indent is not None:
            para.paragraph_format.first_line_indent = Pt(indent)
        add_rich(para, text, size=size, color=color)
        return para

    def bullets(self, items, size=10.5):
        for item in items:
            para = self.d.add_paragraph(style="List Bullet")
            para.paragraph_format.space_after = Pt(2)
            para.paragraph_format.line_spacing = 1.32
            add_rich(para, item, size=size)

    def note(self, title, lines, accent="1F7A6B", fill="F1F8F6"):
        table = self.d.add_table(rows=1, cols=1)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = table.cell(0, 0)
        no_split(table.rows[0])
        cell_shade(cell, fill)
        set_cell_margins(table, top=100, bottom=100, left=160, right=160)
        tbl_pr = table._tbl.tblPr
        borders = OxmlElement("w:tblBorders")
        for edge in ("top", "left", "bottom", "right"):
            el = OxmlElement(f"w:{edge}")
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), "4" if edge != "left" else "24")
            el.set(qn("w:color"), accent)
            borders.append(el)
        tbl_pr.append(borders)

        first = cell.paragraphs[0]
        first.paragraph_format.space_after = Pt(3)
        run = first.add_run(title)
        set_run_font(run, cn=HEAD_CN, en=HEAD_CN, size=10, bold=True, color=RGBColor.from_string(accent))
        for line in lines:
            para = cell.add_paragraph()
            para.paragraph_format.space_after = Pt(1)
            para.paragraph_format.line_spacing = 1.32
            add_rich(para, line, size=9.5)
        self.d.add_paragraph().paragraph_format.space_after = Pt(2)

    def code(self, code_text, caption=None):
        if caption:
            self.example_no += 1
            cap = self.d.add_paragraph()
            cap.paragraph_format.space_before = Pt(6)
            cap.paragraph_format.space_after = Pt(3)
            run = cap.add_run(f"示例 {self.example_no}  {caption}")
            set_run_font(run, cn=HEAD_CN, en=HEAD_CN, size=9, color=GRAY)
        lines = code_text.strip("\n").split("\n")
        para = self.d.add_paragraph(style="FSCODE")
        para.paragraph_format.left_indent = Cm(0.15)
        para.paragraph_format.right_indent = Cm(0.05)
        para_border(para, color="DCE3EA", size=4, space=3)
        shade(para._p, "F7F9FB")
        for idx, line in enumerate(lines):
            run = para.add_run(line if line else " ")
            set_run_font(run, cn=CODE_FONT, en=CODE_FONT, size=7.9, color=RGBColor(0x1B, 0x1F, 0x24))
            if idx != len(lines) - 1:
                run.add_break()
        self.d.add_paragraph().paragraph_format.space_after = Pt(2)

    def figure(self, path, caption, width_cm=16.0):
        self.figure_no += 1
        label = f"{self.chapter}-{self.figure_no}" if self.chapter else str(self.figure_no)
        para = self.d.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.space_before = Pt(4)
        para.paragraph_format.space_after = Pt(3)
        para.add_run().add_picture(str(path), width=Cm(width_cm))
        cap = self.d.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.space_after = Pt(8)
        run = cap.add_run(f"图 {label}  {caption}")
        set_run_font(run, cn=HEAD_CN, en=HEAD_CN, size=9, color=GRAY)

    def table(self, caption, headers, rows, widths, font_size=8.8, first_col_bold=False, keep=None):
        self.table_no += 1
        label = f"{self.chapter}-{self.table_no}" if self.chapter else str(self.table_no)
        if caption:
            cap = self.d.add_paragraph()
            cap.alignment = WD_ALIGN_PARAGRAPH.LEFT
            cap.paragraph_format.space_before = Pt(6)
            cap.paragraph_format.space_after = Pt(3)
            cap.paragraph_format.keep_with_next = True
            run = cap.add_run(f"表 {label}  {caption}")
            set_run_font(run, cn=HEAD_CN, en=HEAD_CN, size=9, color=GRAY)

        table = self.d.add_table(rows=1, cols=len(headers))
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        tbl_pr = table._tbl.tblPr
        layout = OxmlElement("w:tblLayout")
        layout.set(qn("w:type"), "fixed")
        tbl_pr.append(layout)
        set_table_borders(table)
        set_cell_margins(table, top=50, bottom=50, left=80, right=80)

        for i, w in enumerate(widths):
            table.columns[i].width = Cm(w)

        hdr = table.rows[0]
        repeat_header(hdr)
        no_split(hdr)
        for i, h in enumerate(headers):
            cell = hdr.cells[i]
            cell.width = Cm(widths[i])
            cell_shade(cell, "1F3864")
            para = cell.paragraphs[0]
            para.paragraph_format.space_after = Pt(0)
            para.paragraph_format.line_spacing = 1.15
            run = para.add_run(h)
            set_run_font(run, cn=HEAD_CN, en=HEAD_CN, size=font_size, bold=True,
                         color=RGBColor(0xFF, 0xFF, 0xFF))

        for r_idx, row in enumerate(rows):
            tr = table.add_row()
            no_split(tr)
            for i, value in enumerate(row):
                cell = tr.cells[i]
                cell.width = Cm(widths[i])
                if r_idx % 2 == 1:
                    cell_shade(cell, "F4F6FA")
                para = cell.paragraphs[0]
                para.paragraph_format.space_after = Pt(0)
                para.paragraph_format.line_spacing = 1.2
                add_rich(para, str(value), size=font_size,
                         bold_all=first_col_bold and i == 0, en=HEAD_CN if first_col_bold and i == 0 else None)
        if keep if keep is not None else len(rows) <= 8:
            for row in table.rows[:-1]:
                for cell in row.cells:
                    for para in cell.paragraphs:
                        para.paragraph_format.keep_with_next = True
        self.d.add_paragraph().paragraph_format.space_after = Pt(2)

    def page_break(self):
        self.d.add_page_break()

    def save(self, path):
        self.d.save(str(path))


# --------------------------------------------------------------------------- #
# content
# --------------------------------------------------------------------------- #
def cover(doc: Doc):
    for _ in range(3):
        doc.d.add_paragraph()

    p = doc.d.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("FusedSparseAttentionOverlap")
    set_run_font(run, cn=HEAD_CN, en=HEAD_CN, size=30, bold=True, color=NAVY)
    p.paragraph_format.space_after = Pt(4)

    p = doc.d.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("算子接入指导")
    set_run_font(run, cn=HEAD_CN, en=HEAD_CN, size=30, bold=True, color=NAVY)
    p.paragraph_format.space_after = Pt(18)

    p = doc.d.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("Atlas A3  ·  vLLM Ascend  ·  PR #20927")
    set_run_font(run, cn=HEAD_CN, en=HEAD_CN, size=13, color=TEAL)
    p.paragraph_format.space_after = Pt(6)

    line = doc.d.add_paragraph()
    line.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = line.add_run("—" * 28)
    set_run_font(run, cn=HEAD_CN, en=HEAD_CN, size=10, color=RGBColor(0xC9, 0xD3, 0xE0))

    for _ in range(6):
        doc.d.add_paragraph()

    info = [
        ("文档版本", "V1.0"),
        ("发布日期", "2026-11-09"),
        ("适用产品", "Atlas A3（ascend910_9382 / SOC ascend910_93）"),
        ("适用算子", "fused_sparse_attention_overlap（PR #20927）"),
        ("文档密级", "内部公开"),
    ]
    table = doc.d.add_table(rows=0, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for k, v in info:
        row = table.add_row()
        c0, c1 = row.cells
        c0.width = Cm(3.6)
        c1.width = Cm(8.4)
        p0 = c0.paragraphs[0]
        p0.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        r0 = p0.add_run(k)
        set_run_font(r0, cn=HEAD_CN, en=HEAD_CN, size=10, color=GRAY)
        p1 = c1.paragraphs[0]
        r1 = p1.add_run("  " + v)
        set_run_font(r1, size=10, color=DARK)

    doc.page_break()


def revision(doc: Doc):
    p = doc.d.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run("修订记录")
    set_run_font(run, cn=HEAD_CN, en=HEAD_CN, size=13.5, bold=True, color=NAVY)
    doc.table(
        "",
        ["版本", "日期", "修订内容", "修订人"],
        [["V1.0", "2026-11-09", "首版。覆盖算子概述、环境依赖、编译参数、框架内接入、结构验证。", "—"]],
        [2.0, 3.0, 9.5, 2.3],
        first_col_bold=True,
    )

    toc_title = doc.d.add_paragraph()
    toc_title.paragraph_format.space_before = Pt(8)
    run = toc_title.add_run("目录")
    set_run_font(run, cn=HEAD_CN, en=HEAD_CN, size=14, bold=True, color=NAVY)

    toc_p = doc.d.add_paragraph()
    add_field(toc_p, 'TOC \\o "1-3" \\h \\z \\u', "（打开文档后右键“更新域”生成目录）")
    doc.page_break()


def chapter1(doc: Doc):
    doc.set_chapter(1)
    doc.h1("一、概述")

    doc.h2("1.1 算子简介")
    doc.h3("1.1.1 算子定位")
    doc.p(
        "FusedSparseAttentionOverlap（下文简称 Pooling SFA）是 DSA（DeepSeek Sparse Attention）稀疏注意力方案里的核心算子。"
        "DSA 在 decode 时只对 Indexer 选出的 TopK 个历史 Token 算注意力；这些 Token 的 KV 往往不在片上，"
        "需要按索引搬进 NPU 侧的 Selection KV Cache。Pooling SFA 把这两件事合成一次调用：搬和算在同一个 kernel 里"
        "重叠执行，KV offload 的搬运也一并融了进来。"
    )
    doc.p(
        "命中项直接从 Selection KV Cache 读，不再重复搬；未命中项边搬边算。"
        "算子会原地改 Selection KV Cache 和状态表，下一个 step 接着用。"
    )

    doc.h3("1.1.2 解决什么问题")
    doc.p(
        "长上下文推理里，decode 每个 step 都要读历史 KV，全量注意力的开销随序列长度线性涨。DSA 的思路是"
        "先用 Indexer 给每个 query 选 TopK 个 Token，只算这些，计算量从 O(L) 降到 O(topk)。"
        "GLM-5.x 这类带 `index_topk` 的模型用的是同一思路（SFA）。"
    )
    doc.p(
        "DSA 只回答了“算哪些 Token”，没回答“KV 放哪、怎么搬”。长序列的 KV 总量通常远超 HBM，实际部署会再配"
        "一层 KV offload：主副本放在 host 侧大内存池，NPU 上只保留 TopK 的紧凑 Selection Cache。于是每个 decode "
        "step 都要按索引把未驻留的 KV 搬进 Selection Cache，再算注意力。Pooling SFA 融合的就是这一段搬运和后面的计算。"
    )
    doc.p(
        "如果拆成两个算子串行跑（Gather 一个、Sparse Flash Attention 一个），会有三个问题："
    )
    doc.bullets([
        "搬运和计算不重叠。拷贝发射和 Attention 串行，拷贝时 AI Core 空转。",
        "同一批 KV 反复搬。相邻 step 的 TopK 高度重叠，这些 KV 每个 step 都要再从 Full KV Cache 搬一遍。",
        "小 batch 时核用不满。请求数少、行数不够，大量 subcore 闲着。",
    ])
    doc.p(
        "Pooling SFA 用 status / membership 记下哪些 Token 已在 Selection Cache 里，只搬没在的，搬运和注意力计算流水重叠。"
        "PR #20927 在 arch22（A2/A3）上又补了几处优化，代码在 `.../op_kernel/arch22` 下。生效条件和收益见下表："
    )
    doc.table(
        "PR #20927 在 arch22 上的优化点",
        ["优化点", "作用", "生效条件"],
        [
            ["空闲核 helper", "小 batch 时调度空闲核参与搬运与回写", "batch ≤ 12"],
            ["配对排序（Opt01）", "将命中/未命中项配对排序，压缩搬运发射", "存在 selection update"],
            ["回写路径优化（Opt02~04）", "减少状态回写的数据量与指令数", "存在 selection update"],
            ["UB 直写（Opt05）", "搬运算子直接写 UB，省去中间中转", "命中率 0%~70% 时收益明显"],
            ["窗口化两阶段 MergeKv", "搬运与注意力计算流水重叠", "所有 decode 调用"],
            ["按行数划分 subcore（P0）", "行数足够时优先于 helper，充分利用 48 个 subcore", "batch > 24"],
        ],
        [3.8, 7.8, 5.0],
        first_col_bold=True,
    )
    doc.p(
        "单算子测试里，selection-update 路径耗时降了 19%~42%，拷贝发射开销降了约 33%，"
        "命中率 0%~70% 区间端到端提升 10%~12%。这是单算子的数，整网能拿到多少要看 4.3 节的配置，别直接外推。",
        size=9.8,
        color=GRAY,
    )

    doc.h3("1.1.3 典型应用场景")
    doc.bullets([
        "GLM-5.x 的 decode：权重 config 里带 `index_topk`，主 KV 不是 C8。",
        "长序列稀疏注意力：上下文 32K / 64K，真正参与计算的只有 TopK 个 Token（TopK ≤ 2048）。",
        "配合 Sparse KV Offload：KV 主副本放在 host 侧大内存池，NPU 上只留 TopK 紧凑缓存。",
        "单算子精度和性能评估：验证 kernel 改动，或做优化前后的对比。",
    ])

    doc.h3("1.1.4 算子在大模型中的位置")
    doc.p(
        "在 GLM-5.x 的一层 decode 里，算子在 Indexer 之后、O 投影之前。Indexer 给出 TopK 索引，"
        "算子据此更新 Selection Cache、算稀疏注意力，结果送进 O 投影，再进下一层。"
        "Full KV Cache 是 prefill 阶段写好的，decode 只读。"
    )
    doc.p(
        "两套 Cache 各有一张 Block Table。Full KV Cache 的容量随请求序列增长；Selection KV Cache 的容量由 topk 定，"
        "block_size = 128、topk = 128 时只占 1 个逻辑块，另加 1 个空闲物理块。算子只管在两者之间搬，不负责分配。"
    )
    doc.p(
        "接入时要同时准备这两套分页元数据，还得保证 TopK 里每个 Token 都能通过 Full KV 的 Block Table 找到物理块。"
        "找不到，Miss 分支就没数据可搬。"
    )
    doc.h3("1.1.5 单次调用的执行流程")
    doc.p(
        "算子拿 status 和 membership 判断 TopK Token 在不在 Selection Cache 里。不在的，按 Block Table 从 "
        "Full KV Cache Gather 出来写进去；在的，直接读。搬运计划先做配对排序，再用窗口化两阶段 MergeKv "
        "跟注意力计算叠着跑。"
    )
    doc.p(
        "以 topk=128 为例：prefill 之后 Selection Cache 还是空的，第一次调用全部未命中，要搬 128 个 Token；"
        "第二次用同样的索引，全部命中，搬运量为 0，输出和状态都和第一次一样。"
    )

    doc.h3("1.1.6 计算语义")
    doc.p("注意力输出按下式计算，K、V 只取 TopK 索引选中的那些 Token：")
    p = doc.d.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(8)

    def math(text, italic=True, sup=False):
        run = p.add_run(text)
        set_run_font(run, cn=BODY_CN, en="Cambria Math", size=11, color=NAVY)
        run.font.italic = italic
        if sup:
            run.font.superscript = True

    math("Attention(Q, K, V) = softmax(")
    math("α")
    math(" · QK")
    math("T", italic=False, sup=True)
    math(") · V")
    doc.p(
        "其中 α 一般取 1/√(D+Dr)，D 和 Dr 分别是 NoPE 与 RoPE 的维度，取 512 和 64。"
        "算法上 Q 由 NoPE 与 RoPE 两部分拼接后与 K 做点积，两部分各自对应一套缓存。"
        "除了输出的注意力结果，算子还会原地更新四个状态量：`selection_k_rope`、`selection_kv_cache`、"
        "`selection_kv_block_status`、`selection_membership_map`，并给出本次驻留的有效 Token 数"
        "（`selection_kv_actual_seq`）。"
    )

    doc.h2("1.2 算子能力范围与不支持能力")
    doc.table(
        "算子支持的能力范围",
        ["维度", "支持范围", "说明"],
        [
            ["产品", "Atlas A3 / Atlas A2 / Ascend 950", "A3 对应 SOC `ascend910_93`、芯片 `ascend910_9382`、DAV_2201（arch22）"],
            ["数据类型", "FLOAT16、BFLOAT16", "query、Full KV Cache、Selection KV Cache 三者必须一致"],
            ["query 排布", "TND", "shape 为 (T1, N1, D+Dr)，仅支持 MLA-absorb 形态"],
            ["KV 排布", "PA_BSND", "分页存储，block_size = 128"],
            ["结构维度", "D = 512（NoPE）、Dr = 64（RoPE）", "query 最后一维 576，attentionOut 最后一维 512"],
            ["头配置", "N1 = 4、N2 = 1", "已验证的组合；N2 为 KV 头数"],
            ["稀疏度", "topk ≤ 2048，sparseBlockSize = 1，topkBlockSize = 1", "融合路径只支持 Token-wise 稀疏"],
            ["稀疏模式", "sparseMode = 3（rightDownCausal）", "Decode 场景使用"],
            ["运行模式", "推理、支持图模式", "不支持训练"],
        ],
        [2.4, 6.2, 8.0],
        first_col_bold=True,
    )
    doc.table(
        "不支持能力总览",
        ["不支持项", "表现 / 后果"],
        [
            ["BSND 排布的 query", "layoutQuery 只接受 \"TND\"，其它取值报错或结果未定义"],
            ["非 MLA 结构（MHA/GQA 普通注意力）", "算子按 D=512、Dr=64 的 MLA-absorb 结构实现，不适用"],
            ["空 Tensor、非连续输入", "算子要求所有输入非空且连续，需在调用前 contiguous()"],
            ["topkBlockSize > 1", "融合路径仅支持 selection_topk_block_size = 1"],
            ["C8 / TQ4 packed 主 KV 缓存", "sparse offload 直接拒绝：does not support packed SFA main caches"],
            ["带 compress_ratios 的权重（DeepSeek-V4 等）", "Sparse KV offload don't support compress now"],
            ["Context Parallel（enable_dsa_cp）", "Sparse KV offload 要求 TP without context parallelism"],
            ["Atlas 200I/500 A2、推理系列、训练系列", "算子未在该产品上编译，调用会报算子不存在"],
            ["Ascend 950（arch35）", "可运行，但 PR #20927 的优化只改 arch22，无对应收益"],
        ],
        [6.4, 10.2],
        first_col_bold=True,
    )


def chapter2(doc: Doc):
    doc.set_chapter(2)
    doc.h1("二、环境依赖和约束")

    doc.p(
        "这套组合在 Atlas A3 单机上跑通过。换硬件、换镜像、换 CANN 之后别默认还能跑，先按 2.6 节自检一遍。"
        "最容易出问题的是 vLLM 提交号，以及 CANN 和 torch_npu 的匹配。"
    )

    doc.h2("2.1 硬件依赖")
    doc.table(
        "硬件要求",
        ["项目", "要求 / 取值", "说明"],
        [
            ["硬件产品", "Atlas A3", "已在 A3 单机（8 卡 × 2 chip）验证"],
            ["CPU 架构", "aarch64（ARM64）", "与镜像内 aarch64 wheel 一致"],
            ["芯片型号", "`ascend910_9382`（SOC `ascend910_93`）", "编译与运行期 SOC 必须对应"],
            ["达芬奇架构", "DAV_2201，代码路径 `op_kernel/arch22`", "PR #20927 的优化仅对 arch22 生效"],
            ["单算子测试卡数", "1 张（`ASCEND_RT_VISIBLE_DEVICES=0`）", "仓内精度用例单卡即可"],
            ["整网测试卡数", "8 卡 × 2 chip = 16 个逻辑 NPU，每 chip 64GB", "TP16 + EP16；合计 1TB HBM"],
            ["Host 内存", "≥ 512GB（建议 1TB）", "承载权重 page cache 与 sparse offload host pool"],
            ["共享内存 / 磁盘", "`/dev/shm` ≥ 1GB（建议 8~16GB）；构建磁盘 ≥ 100GB", "全量算子构建与权重加载需要"],
            ["NPU 驱动", "宿主机安装，容器挂载 `/usr/local/Ascend/driver`", "`npu-smi info` 正常显示设备即合格"],
        ],
        [3.0, 7.0, 6.6],
        first_col_bold=True,
    )

    doc.h2("2.2 软件依赖")
    doc.table(
        "软件要求",
        ["组件", "要求 / 取值", "说明"],
        [
            ["容器镜像", "`vllm-ascend:30.0-a3`", "镜像仓中的标签为 `v0.30.0-a3`；镜像内已含 CANN、torch、torch_npu"],
            ["操作系统", "镜像自带 Linux（aarch64）", "不要替换系统自带的 glibc / gcc"],
            ["Python", "镜像内置", "与镜像保持一致，不要另装一套"],
            ["CANN", "镜像内置", "统一通过 `/usr/local/Ascend/ascend-toolkit` 加载，不要硬编码带版本号的路径"],
            ["vllm-ascend", "官方仓库 PR #20927 head", "`git fetch origin pull/20927/head:pr20927`"],
            ["vLLM", "`.github/vllm-main-verified.commit` 指定的提交", "必须严格对齐，不能沿用镜像预装版本"],
            ["torch / torch_npu", "镜像内置", "与 CANN 绑定，不要单独升级"],
            ["transformers / triton-ascend", "以仓库 requirements 为准", "整网路径需要"],
            ["memfabric_hybrid / memcache_hybrid", "以仓库 requirements 为准", "sparse KV offload 必需"],
        ],
        [4.0, 8.2, 4.4],
        first_col_bold=True,
    )

    doc.h2("2.3 拉取镜像并启动容器")
    doc.p(
        "本机已有 `vllm-ascend:30.0-a3` 的可以跳过拉取，直接用下面的 docker run 启动。"
    )
    doc.code(
        """
export IMAGE=registry-cbu.huawei.com/ascend/vllm-ascend:v0.30.0-a3
docker images | grep vllm-ascend
docker pull "$IMAGE"
""",
        "查看并拉取镜像",
    )
    doc.p("启动容器，把 NPU 设备、驱动和代码目录挂进去：")
    doc.code(
        """
docker run --name <容器名> -it -d --net=host --shm-size=500g \\
  --privileged=true \\
  -w /home \\
  --device=/dev/davinci0 --device=/dev/davinci1 \\
  --device=/dev/davinci2 --device=/dev/davinci3 \\
  --device=/dev/davinci4 --device=/dev/davinci5 \\
  --device=/dev/davinci6 --device=/dev/davinci7 \\
  --device=/dev/davinci_manager \\
  --device=/dev/hisi_hdc \\
  --device=/dev/devmm_svm \\
  --entrypoint=bash \\
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \\
  -v /usr/local/dcmi:/usr/local/dcmi \\
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \\
  -v /usr/local/sbin:/usr/local/sbin \\
  -v /home:/home \\
  -v /tmp:/tmp \\
  -v /data:/data \\
  -v /mnt/:/mnt/ \\
  -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \\
  -v /etc/hccn.conf:/etc/hccn.conf \\
  -e http_proxy=$http_proxy \\
  -e https_proxy=$https_proxy \\
  "$IMAGE"
""",
        "启动容器",
    )
    doc.p(
        "容器启动后执行 `docker exec -it <容器名> bash` 进入。先跑 `npu-smi info` 确认能看到设备，"
        "再看 `python3 --version`；`npu-smi info` 失败就先查宿主机驱动和容器设备挂载，别继续往下走。"
    )

    doc.h2("2.4 其他依赖库")
    doc.table(
        "其他依赖",
        ["依赖", "来源", "用途"],
        [
            ["fla_npu", "当前分支要求的 aarch64 wheel", "pytest 公共 conftest 的导入链需要"],
            ["modelscope", "随 requirements 安装", "conftest 会 import，测试不下载模型；不要设置 `VLLM_USE_MODELSCOPE`"],
            ["pytest", "随 requirements 安装", "运行仓内单算子用例"],
            ["setuptools-rust / ninja / pybind11", "pip 可安装的最新版本", "编译 Python 扩展与算子包"],
            ["ais-bench-benchmark + pandas", "独立 venv 安装", "整网压测（可选）"],
        ],
        [4.0, 8.6, 4.0],
        first_col_bold=True,
    )
    doc.p(
        "编译和测试依赖按仓库的 `requirements-dev.txt` 安装。FLA 的 wheel 不随 pip 源发布，"
        "需要从 releases 页面下载当前分支要求的 aarch64 版本单独安装。"
    )

    doc.h2("2.5 版本不兼容带来的风险")
    doc.table(
        "常见版本不兼容风险",
        ["风险项", "典型现象", "规避方式"],
        [
            ["vLLM 提交与分支不匹配", "`ImportError: cannot import name 'CircularBufferSpec'` 或缺少 `prepare_dcp_local_seq_lens`", "切到 `.github/vllm-main-verified.commit` 指定的提交"],
            ["vLLM 比分支新", "`No module named 'vllm.v1.core.kv_cache_planning'`", "同样切回指定提交，不要用最新 main"],
            ["CANN 与 torch_npu 不匹配", "`libruntime_common.so: undefined symbol`", "使用镜像内置组合；新终端重新 source 正确的 setenv"],
            ["编译期与运行期 CANN 不一致", "`libcust_opmaster_rt2.0.so` 加载失败", "清理 `csrc/build` 等缓存后全量重编，不要用 --force 跳过校验"],
            ["ninja 指向旧 Python", "报错路径不是当前 Python 的安装目录", "重装 ninja 并调整 PATH，清缓存重编"],
            ["SOC_VERSION 与芯片不一致", "运行时找不到 kernel 或精度异常", "编译 `--soc=ascend910_93`，运行 `SOC_VERSION=ascend910_9382`"],
            ["主 KV 为 C8 / fp8", "offload 初始化直接拒绝", "使用非 c8 权重，去掉 `--kv-cache-dtype`"],
        ],
        [4.6, 7.4, 4.6],
        first_col_bold=True,
    )

    doc.h2("2.6 环境自检")
    doc.p("进入容器后先执行以下检查，全部通过再开始编译：")
    doc.code(
        """
npu-smi info                       # 必须正常显示设备
python3 --version                  # 与镜像内置版本一致
python3 -c "import platform; print(platform.machine())"   # 必须输出 aarch64

python3 - <<'PY'
import torch, torch_npu, vllm
print("torch    :", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("vllm     :", vllm.__version__, vllm.__file__)
print("npu count:", torch.npu.device_count())
PY

readlink -f /usr/local/Ascend/ascend-toolkit/latest
""",
        "硬件与软件环境自检",
    )
    doc.p(
        "先别急着编译。`npu-smi info` 失败先查 NPU 驱动与容器挂载；`platform.machine()` 不是 `aarch64`，"
        "说明镜像或节点不对；`vllm.__file__` 为 `None`，说明当前目录的 `vllm/` 挡住了安装包。"
    )


def chapter3(doc: Doc):
    doc.set_chapter(3)
    doc.h1("三、编译指导")
    doc.p(
        "本节列的是官方预编译包（run 包）用的编译参数。集成方一般直接用官方包，不用自己编；"
        "只有改了 kernel、适配新 SOC，或者官方包没覆盖你用的版本，才需要照本节重编，"
        "重编时照抄这些参数就行，产物和官方包一致。"
    )

    doc.h2("3.1 编译环境准备")
    doc.p(
        "先把代码拉下来。代理和 pip 镜像按实际网络环境配，不需要就跳过前几行；"
        "PR #20927 的代码直接从官方仓库拉取引用，不必 fork："
    )
    doc.code(
        """
export PIP_EXTRA_INDEX_URL=https://mirrors.huaweicloud.com/ascend/repos/pypi
export PIP_TRUSTED_HOST="repo.huaweicloud.com mirrors.huaweicloud.com"

# 走公司代理时按需设置，<工号> 和 <密码> 换成自己的
export http_proxy=http://<工号>:<密码>@proxy.huawei.com:8080
export https_proxy=http://<工号>:<密码>@proxy.huawei.com:8080
export no_proxy=127.0.0.1,localhost,.huawei.com,10.*.*.*
export GIT_SSL_NO_VERIFY=1
export CMAKE_TLS_VERIFY=0

git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git fetch origin pull/20927/head:pr20927
git checkout pr20927
""",
        "配置代理并拉取 PR 代码",
    )
    doc.p("进入仓库根目录后配置编译环境，变量只对当前 shell 生效：")
    doc.code(
        """
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit
source "$ASCEND_HOME_PATH/set_env.sh"     # 部分版本为 bin/setenv.bash

unset LD_PRELOAD
export DRIVER_LIBS=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver
export LD_LIBRARY_PATH="$DRIVER_LIBS:/usr/local/Ascend/driver/lib64/common:${LD_LIBRARY_PATH:-}"

export SOC_VERSION=ascend910_9382          # 运行期芯片型号
export MAX_JOBS=8
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
""",
        "编译环境变量",
    )
    doc.p(
        "确认 ninja 没有被其它 Python 环境劫持，否则会出现“编译到一半找不到 ninja”或链接到旧版本的问题："
    )
    doc.code(
        """
which python3
which ninja
ninja --version          # 必须正常打印版本号
""",
        "检查 ninja 指向",
    )

    doc.h2("3.2 编译参数清单")
    doc.p(
        "vllm-ascend 有两种构建入口：`pip install -e .`（同时构建 Python 扩展与全部自定义算子）"
        "与 `csrc/build.sh` / `csrc/build_aclnn.sh`（只构建算子 run 包）。官方预编译包使用全量参数。"
    )
    doc.table(
        "编译参数清单",
        ["参数 / 变量", "官方预编译包取值", "作用"],
        [
            ["`COMPILE_CUSTOM_KERNELS`", "`1`", "`pip install -e .` 时编译全部 CANN 自定义算子；`0` 只装 Python 包，不重编算子"],
            ["`SOC_VERSION`（构建）", "`ascend910_9382`", "传给 build_aclnn.sh，决定编译的芯片型号"],
            ["`--soc`（build.sh）", "`ascend910_93`", "SoC 简称，对应芯片 `ascend910_9382`；不要写成全名"],
            ["`--pkg`", "启用", "生成带 kernel bin 的 run 包；不加则为 JIT 包"],
            ["`--ops`", "缺省（全量）或 `fused_sparse_attention_overlap`", "逗号分隔的算子 snake 名；单算子包仅用于精度验证"],
            ["`-j` / `MAX_JOBS`", "`8`", "并行编译任务数"],
            ["`-O` / `--build-type`", "`O2`（Release）", "编译优化级别"],
            ["`USE_NINJA` / `VLLM_ASCEND_USE_NINJA`", "`auto`", "构建器选择，auto 时优先 ninja"],
            ["`VLLM_BATCH_INVARIANT`", "`0`", "为 1 时额外构建 batch invariant 算子包"],
            ["`--vendor_name`", "`custom`（缺省）", "生成 `custom_transformer` vendor 目录"],
        ],
        [5.4, 5.6, 5.6],
        first_col_bold=True,
    )

    doc.h3("3.2.1 编译选项（kernel 级）")
    doc.p(
        "本算子在 `op_host/CMakeLists.txt` 中显式设置了以下编译选项，二次编译时不要随意去掉，"
        "否则可能出现同步错误或性能退化："
    )
    doc.table(
        "FusedSparseAttentionOverlap 的 kernel 编译选项",
        ["选项", "作用"],
        [
            ["`--cce-auto-sync=off`", "关闭编译器自动同步，由 kernel 手工管理事件同步"],
            ["`-Wno-deprecated-declarations`", "屏蔽 CANN 头文件中的弃用告警"],
            ["`-Werror`", "告警视为错误，保证代码质量"],
            ["`-fpermissive`", "放宽部分 C++ 语法检查，兼容 AscendC 头文件"],
            ["`-mllvm -cce-vf-remove-membar=false`", "保留向量化过程中的内存屏障，保证跨核可见性"],
            ["`-mllvm -cce-aicore-hoist-movemask=false`", "禁止上提 movemask，避免破坏流水时序"],
        ],
        [7.0, 9.6],
    )

    doc.h2("3.3 编译步骤")
    doc.h3("3.3.1 安装当前分支配套的 vLLM")
    doc.p(
        "镜像里预装的 vLLM 不一定和分支匹配，必须切到仓库指定的提交。这一步只动 "
        "`/vllm-workspace/vllm`，不会切换或修改当前的 vllm-ascend 分支。"
    )
    doc.code(
        """
export VLLM_COMMIT="$(tr -d '[:space:]' < .github/vllm-main-verified.commit)"
export VLLM_SRC=/vllm-workspace/vllm

if [[ ! -d "$VLLM_SRC/.git" ]]; then
  git clone https://github.com/vllm-project/vllm.git "$VLLM_SRC"
fi

git -C "$VLLM_SRC" fetch --depth 1 origin "$VLLM_COMMIT"
git -C "$VLLM_SRC" checkout --detach FETCH_HEAD
test "$(git -C "$VLLM_SRC" rev-parse HEAD)" = "$VLLM_COMMIT"

VLLM_TARGET_DEVICE=empty python3 -m pip install -e "$VLLM_SRC" --no-build-isolation --no-deps
""",
        "安装配套 vLLM",
    )

    doc.h3("3.3.2 全量包（官方包方式，推荐）")
    doc.code(
        """
cd /path/to/vllm-ascend
rm -rf csrc/build csrc/build_out csrc/output build

COMPILE_CUSTOM_KERNELS=1 python3 -m pip install -e . --no-build-isolation
""",
        "全量编译并安装 vllm-ascend",
    )
    doc.p(
        "该命令会编译约 40 个 A3 自定义算子与 Python 扩展。首次构建通常需要 30 分钟以上，"
        "建议放在 tmux / nohup 中执行。看到以下信息表示成功："
    )
    doc.code(
        """
Successfully built vllm_ascend
Successfully installed ... vllm_ascend-...
""",
        "安装成功标志",
    )

    doc.h3("3.3.3 单算子包（仅用于精度验证）")
    doc.code(
        """
cd csrc
bash build.sh --pkg --ops=fused_sparse_attention_overlap --soc=ascend910_93 -j8
cd ..

chmod +x csrc/build/cann-ops-transformer*.run
csrc/build/cann-ops-transformer*.run --install-path="$PWD/vllm_ascend/_cann_ops_custom"
""",
        "只编译并安装 fused_sparse_attention_overlap",
    )
    doc.p(
        "单算子包不能启动整网服务。整网模型会调用 `mla_preprocess`、`sparse_flash_attention`、"
        "`moe_gating_top_k`、`grouped_matmul_swiglu_quant`、`dispatch_ffn_combine` 等约 40 个 A3 自定义算子，"
        "只装单算子包会让服务启动失败（operator not found）。单算子包仅用于 4.4.1 的精度测试。"
    )

    doc.h3("3.3.4 换算子 / 换分支后的重编")
    doc.p(
        "`git checkout` 之后必须重新编译算子。`COMPILE_CUSTOM_KERNELS=0` 只重装 Python 包，不会重编算子；"
        "增量构建不需要删除 `csrc/build`，保留它、`touch` 改动文件后重跑即可。只有 CMake 配置或算子列表变化时才需要清理。"
    )

    doc.h2("3.4 产物清单")
    doc.table(
        "编译产物与安装位置",
        ["产物", "路径", "说明"],
        [
            ["算子 run 包", "`csrc/build/cann-ops-transformer*.run`", "可单独分发、单独安装"],
            ["算子包根目录", "`vllm_ascend/_cann_ops_custom/vendors/custom_transformer/`", "运行时通过 set_env.bash 挂载"],
            ["kernel 二进制", "`.../custom_transformer/op_impl/ai_core/tbe/kernel/`\n（按 SOC / arch 分目录）", "按 SOC / arch 分目录，arch22 对应 A2/A3"],
            ["环境脚本", "`.../custom_transformer/bin/set_env.bash`", "设置 `ASCEND_CUSTOM_OPP_PATH` 与库路径"],
            ["Python 扩展", "`vllm_ascend/vllm_ascend_C*.so`", "torch 自定义算子注册入口"],
        ],
        [3.6, 9.4, 3.6],
        first_col_bold=True,
    )

    doc.h2("3.5 编译结果验收")
    doc.code(
        """
cd /path/to/vllm-ascend
source "$PWD/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash"

python3 - <<'PY'
import torch, torch_npu, vllm_ascend
from vllm_ascend.utils import enable_custom_op

enable_custom_op()
assert hasattr(torch.ops._C_ascend, "npu_fused_sparse_attention_overlap")
print("算子加载成功")
PY
""",
        "确认算子已注册",
    )
    doc.p("必须输出 `算子加载成功`。随后可运行 4.4.1 的精度用例，期望 `4 passed`。")

    doc.h2("3.6 编译失败排查")
    doc.table(
        "编译阶段常见失败",
        ["现象", "原因", "处理"],
        [
            ["`ModuleNotFoundError: No module named 'setuptools_rust'`", "缺少编译依赖", "`python3 -m pip install setuptools-rust` 后重跑"],
            ["`No module named 'ninja'`，或 ninja 路径指向其它 Python", "ninja 来自其它 Python 环境", "重装 ninja，调整 PATH 后 `hash -r`"],
            ["`libgraph_base.so: undefined symbol`", "复用了其它 CANN 版本留下的 CMake / ninja 缓存", "清理 `csrc/build`、`csrc/build_out`、`csrc/output`、`build` 后全量重编"],
            ["`build_aclnn.sh` 非零退出", "环境变量、SOC 或依赖不满足", "查看日志第一处 `[ERROR]`，逐项核对 3.1"],
            ["`catlass is missing, try to fetch it` 后失败", "未初始化子模块 / 代理不可用", "`git submodule update --init --recursive` 后重试"],
            ["`ERROR: Failed building wheel for vllm_ascend`", "算子编译失败", "向上翻日志找到第一个 error，而不是只看最后一行"],
        ],
        [6.8, 5.0, 4.8],
        first_col_bold=True,
        keep=False,
    )
    doc.p(
        "构建末尾可能出现 `skip ...`（文档、示例或当前 SOC 不涉及的项），这是正常的；"
        "以 `op ok` 与单算子 `4 passed` 为准。若跳过的是 vendor 包安装，则需要排查。",
        size=9.8,
        color=GRAY,
    )


def chapter4(doc: Doc):
    doc.set_chapter(4)
    doc.h1("四、接入指导")

    doc.h2("4.1 调用整体流程")
    doc.p(
        "接入分两步：先用 `enable_custom_op()` 把算子注册进来，再通过 "
        "`torch.ops._C_ascend.npu_fused_sparse_attention_overlap` 调用。vLLM 整网走的就是这条路径，"
        "单点调用和整网调用用的是同一个入口。"
    )

    doc.h3("4.1.1 调用时序（以整网 decode 为例）")
    doc.p("整网接入按下面的顺序走。前两步是框架的事，跟算子无关：")
    doc.bullets([
        "Prefill 阶段，模型把每层的 K/V 写进 Full KV Cache（PA_BSND 分页布局），同时记下 Block Table 和有效长度。",
        "Indexer 阶段，每个 decode step 输出 TopK Token 索引，shape 为 (T1, N2, topk)。",
        "准备 Selection 状态：申请 Selection KV Cache 和 Block Table；首次调用前把 `selection_kv_block_status`、"
        "`selection_membership_map` 全填 -1。",
        "调用算子：14 个输入、5 个属性、2 个输出，执行一次。",
        "原地更新：算子把未命中的 KV 写进 Selection Cache，同时更新 status / membership，输出 attention 和有效数量。",
        "复用：下一个 step 用同一份 Selection Cache。TopK 命中率高时，搬运量接近 0。",
        "回收：请求结束或 KV Block 回收时，重置对应的 status / membership 行。",
    ])

    doc.table(
        "调用前置条件检查清单",
        ["检查项", "要求"],
        [
            ["算子已加载", "`hasattr(torch.ops._C_ascend, \"npu_fused_sparse_attention_overlap\")` 为 True"],
            ["环境已加载", "已 source `custom_transformer/bin/set_env.bash`"],
            ["输入连续", "所有 Tensor `is_contiguous()`，空 Tensor 不允许"],
            ["dtype 一致", "query、full KV、selection KV 同为 fp16 或同为 bf16"],
            ["形状约束", "D=512、Dr=64、N1=4、N2=1、block_size=128、topk ≤ 2048"],
            ["状态已初始化", "status / membership 为 -1；Selection Cache 首次调用前无有效数据"],
            ["索引合法", "TopK 索引都在对应 batch 的 `fullKvActualSeq` 范围内"],
            ["容量足够", "Selection Cache 容量 ≥ topk；两张 Block Table 覆盖所有被访问的逻辑块"],
        ],
        [4.6, 12.0],
        first_col_bold=True,
    )

    doc.h2("4.2 参数与张量契约")
    doc.p(
        "参数按算子定义（`op_host/fused_sparse_attention_overlap_def.cpp`）列在表 4-2。"
        "单点调用和整网调用用的都是这一套参数。"
    )
    doc.table(
        "算子输入输出与属性",
        ["参数", "方向", "shape", "dtype", "初始化", "说明"],
        [
            ["query", "输入", "(T1, N1, D)", "fp16/bf16", "—", "仅 NoPE 部分，D=512；不是 576 维"],
            ["key", "输入", "(F, bs, 1, D)", "同 query", "—", "Full KV Cache 加一个 head 维（unsqueeze(2)）"],
            ["value", "输入", "同 key", "同 query", "—", "与 key 同一份数据"],
            ["sparse_indices", "输入", "(T1, N2, topk)", "int32", "—", "TopK Token 索引，不支持空 Tensor"],
            ["block_table", "输入（可选）", "(B, F_logical)", "int32", "—", "Full KV 逻辑块 → 物理块"],
            ["actual_seq_lengths_query", "输入（可选）", "(B,)", "int32", "—", "Decode 场景每项为 1"],
            ["actual_seq_lengths_kv", "输入（可选）", "(B,)", "int32", "—", "Full KV 的有效长度"],
            ["query_rope", "输入（可选）", "(T1, N1, Dr)", "fp16/bf16", "—", "Dr=64；无 RoPE 时传 nullptr"],
            ["key_rope", "输入（可选）", "(F, bs, 1, Dr)", "fp16/bf16", "—", "Full K RoPE，加 head 维"],
            ["selection_k_rope", "输入/输出", "(S, bs, Dr)", "fp16/bf16", "NaN/任意", "原地更新"],
            ["selection_kv_cache", "输入/输出", "(S, bs, D)", "fp16/bf16", "NaN/任意", "原地更新"],
            ["selection_kv_block_table", "输入", "(B, S_logical)", "int32", "—", "Selection Cache 逻辑块 → 物理块"],
            ["selection_kv_block_status", "输入/输出", "(B, N2, stride)", "int32", "-1", "驻留状态，stride = align8(topk+1)"],
            ["selection_membership_map", "输入/输出", "(B, N2, 16400)", "int16", "-1", "Token→Slot 映射与搬运计划"],
            ["attention_out", "输出", "(T1, N1, D)", "同 query", "—", "不含 RoPE 维"],
            ["selection_kv_actual_seq", "输出", "(B,)", "int32", "—", "本次驻留的有效 Token 数"],
            ["scale_value", "属性", "标量", "float", "1/sqrt(D+Dr)", "缩放系数"],
            ["sparse_block_size", "属性", "标量", "int64", "1", "融合路径只支持 1"],
            ["selection_topk_block_size", "属性", "标量", "int64", "1", "仅 torch 包装暴露，用于校验其为 1"],
            ["layout_query", "属性", "字符串", "string", "\"TND\"", "只支持 TND"],
            ["layout_kv", "属性", "字符串", "string", "\"PA_BSND\"", "只支持 PA_BSND"],
            ["sparse_mode", "属性", "标量", "int64", "3", "rightDownCausal"],
        ],
        [4.5, 1.9, 3.1, 1.9, 2.0, 3.2],
        font_size=8.2,
        first_col_bold=True,
    )
    doc.p(
        "注意：`selection_topk_block_size` 只存在于 torch 包装层，用于强制校验其为 1。",
        size=9.5,
        color=GRAY,
    )

    doc.h3("4.2.1 状态张量布局")
    doc.table(
        "status 与 membership 的布局",
        ["张量", "布局", "接入方需要做什么"],
        [
            ["`selection_kv_block_status`",
             "每行前 topk 个 int32 记录驻留 Token ID；索引 topk 处记录有效数量；整行按 8 个 int32 对齐（topk=128 时 stride=136）",
             "按 stride 申请，整行初始化为 -1；不要改写内容"],
            ["`selection_membership_map`",
             "行宽 16400：偏移 [0, 16376) 为 Token→Slot 映射；[16384, 16392) 为 8 个控制字段（含 ready / plan 标记）；[16392] 为计划偏移",
             "按 16400 申请，整行初始化为 -1；仅在外部计划模式下才写入控制字段"],
        ],
        [4.2, 8.2, 4.2],
        first_col_bold=True,
    )
    doc.p(
        "status / membership 的字段含义属于算子内部实现，接入方只负责按对齐要求分配、初始化并保持原样。"
        "测试代码中出现的 0x5A4D、0x5A50、0x5A45 等标记，不能作为对外接口使用。"
    )

    doc.h2("4.3 框架内接入（vLLM / PyTorch 自定义算子）")
    doc.h3("4.3.1 单点调用")
    doc.code(
        """
import math
import torch
import torch_npu
import vllm_ascend
from vllm_ascend.utils import enable_custom_op

enable_custom_op()
assert hasattr(torch.ops._C_ascend, "npu_fused_sparse_attention_overlap")

out = torch.ops._C_ascend.npu_fused_sparse_attention_overlap(
    query=query,                                  # (T1, N1, 576)
    selection_k_rope=selection_k_rope,
    selection_kv_cache=selection_kv_cache,
    selection_kv_block_table=selection_kv_block_table,
    selection_kv_block_status=selection_kv_block_status,
    selection_membership_map=selection_membership_map,
    selection_topk_indices=selection_topk_indices,
    full_k_rope=full_k_rope,
    full_kv_cache=full_kv_cache,
    full_kv_block_table=full_kv_block_table,
    full_kv_actual_seq=full_kv_actual_seq,
    full_q_actual_seq=full_q_actual_seq,
    scale_value=1.0 / math.sqrt(576),
    sparse_block_size=1,
    selection_topk_block_size=1,
    layout_query="TND",
    layout_kv="PA_BSND",
    sparse_mode=3,
)
torch.npu.synchronize()
""",
        "torch 自定义算子调用",
    )

    doc.h3("4.3.2 整网接入配置")
    doc.p(
        "整网只有同时满足三个条件，才会走到本算子：模型带 `index_topk`（GLM-5.x）、"
        "开启 `sparse_kv_offload_config.enabled`、且 `use_fused_overlap=true`。"
        "缺少 offload 配置时服务只是普通推理，算子一次都不会执行。"
    )
    doc.code(
        """
vllm serve "$MODEL_PATH" \\
  --tensor-parallel-size 16 \\
  --enable-expert-parallel \\
  --max-num-seqs 8 \\
  --max-model-len 40960 \\
  --max-num-batched-tokens 4096 \\
  --quantization ascend \\
  --gpu-memory-utilization 0.90 \\
  --enable-chunked-prefill \\
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \\
  --additional-config '{
    "sparse_kv_offload_config": {
      "enabled": true,
      "use_fused_overlap": true,
      "keep_device_kv_cache": true,
      "dram_size_per_dp_GB": 48
    }
  }'
""",
        "整网启动参数（单机调试形态）",
    )
    doc.table(
        "整网参数约束",
        ["约束", "说明"],
        [
            ["不要设置 `--kv-cache-dtype int8/fp8`", "主 KV 会变成 C8 packed cache，sparse offload 直接拒绝"],
            ["权重不能是 c8", "`w8a8c8` / `w4a8c8` 均被拒绝"],
            ["不要开 `enable_dsa_cp`", "sparse offload 不支持 context parallel"],
            ["`dram_size_per_dp_GB` 按主机内存调整", "host 连续内存不足时先下调（例如 48）"],
            ["`use_fused_overlap=false` 不是旧 kernel 基线", "关闭后走 Copy-SFA 路径，A/B 结论不能混用"],
            ["`tensor-parallel-size` 与权重匹配", "391GB 权重在 TP16 下约 24.5GB/chip，TP8 会挤占 KV 空间"],
        ],
        [6.4, 9.8],
        first_col_bold=True,
    )

    doc.h2("4.4 结构验证")
    doc.p(
        "进程不崩，不代表接入成功。结构验证看的是算子确实被调用、输出和状态都符合契约。"
        "下面分单算子和整网两层给判据。"
    )

    doc.h3("4.4.1 单算子精度测试（仓内用例）")
    doc.code(
        """
cd /path/to/vllm-ascend
source "$PWD/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash"
export ASCEND_RT_VISIBLE_DEVICES=0

pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_sparse_attention_overlap.py
""",
        "运行仓内精度用例",
    )
    doc.p("通过标准是最后一行输出：")
    doc.code("4 passed", "单算子测试通过标志")
    doc.p(
        "用例包含 BF16 / FP16 与 topk=32 / 128 的 4 个参数化组合，每个组合都会执行两遍："
    )
    doc.table(
        "单算子用例的断言内容",
        ["阶段", "断言", "判据说明"],
        [
            ["首次调用", "输出 shape / dtype / 无 NaN/Inf", "`attention_out` 为 (1, 4, 512)，与参考同 dtype"],
            ["首次调用", "`torch.testing.assert_close(out, reference, rtol=1e-2, atol=1e-2)`", "与 CPU 参考实现（gather → softmax → matmul）一致"],
            ["首次调用", "`status[topk] == topk`", "有效驻留数量等于 TopK"],
            ["首次调用", "驻留 ID 与 TopK 索引排序后完全相等", "未命中项全部被搬入"],
            ["首次调用", "Selection Cache 每个 Slot 等于 Full KV 按索引 Gather 的结果", "搬运内容逐元素相等（rtol=atol=0）"],
            ["首次调用", "membership 的 ready / plan 标记合法", "内部计划状态自洽"],
            ["再次调用", "输出与首次一致", "All-Hit 路径结果不变"],
            ["再次调用", "status / Selection Cache 与首次结束后逐元素相同", "All-Hit 不产生重复搬运与改写"],
        ],
        [2.4, 7.6, 6.2],
        first_col_bold=True,
        keep=False,
    )

    doc.h3("4.4.2 整网结构验证")
    doc.bullets([
        "先确认算子被调用。`use_fused_overlap=true` 时 decode 走 `_execute_fused_overlap_offload_decode`，"
        "内部通过 `_require_custom_op(\"npu_fused_sparse_attention_overlap\")` 取算子，缺了会直接抛错。"
        "所以请求能正常返回、日志没异常，就说明算子跑过。",
        "再看 offload 路径是否生效：`curl /metrics | grep prefix_cache` 能查到 `vllm:prefix_cache_*` 指标。",
        "服务本身要可用：`/health` 返回 200，`/v1/chat/completions` 能返回结果。",
        "最后看性能：TPOT 随命中率提升而下降。和 `use_fused_overlap=false` 比，能确认走的是不同路径，但后者不是旧 kernel 基线，别拿来做 A/B。",
    ])
    doc.code(
        """
curl -s http://127.0.0.1:8000/health; echo
curl -s http://127.0.0.1:8000/metrics | grep prefix_cache | head
curl -s http://127.0.0.1:8000/v1/chat/completions \\
  -H 'Content-Type: application/json' \\
  -d '{"model":"glm-5.1","messages":[{"role":"user","content":"hi"}],"max_tokens":16,"temperature":0}'
""",
        "整网验收命令",
    )

    doc.h3("4.4.3 验证清单")
    doc.table(
        "结构验证清单",
        ["层次", "验证项", "通过标准"],
        [
            ["编译", "算子注册", "`hasattr(torch.ops._C_ascend, \"npu_fused_sparse_attention_overlap\")` 为 True"],
            ["单算子", "仓内精度用例", "`4 passed`"],
            ["单算子", "状态正确性", "`status[topk] == topk`，Cache 与 Gather 结果一致"],
            ["整网", "服务可用", "`/health` 200，decode 请求正常返回"],
            ["整网", "路径生效", "`/metrics` 有 `vllm:prefix_cache_*`"],
            ["整网", "性能", "TPOT 稳定，命中率提升时搬运开销下降"],
        ],
        [3.0, 6.0, 7.6],
        first_col_bold=True,
    )


def appendix(doc: Doc):
    doc.set_chapter(5)
    doc.h1("五、附录")

    doc.h2("附录 A  单算子测试速查")
    doc.code(
        """
cd /path/to/vllm-ascend
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit
source "$ASCEND_HOME_PATH/set_env.sh"
source "$PWD/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash"

unset LD_PRELOAD
export DRIVER_LIBS=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver
export LD_LIBRARY_PATH="$DRIVER_LIBS:/usr/local/Ascend/driver/lib64/common:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=0
export SOC_VERSION=ascend910_9382

pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_sparse_attention_overlap.py
""",
        "重新进入容器后的完整复测命令",
    )

    doc.h2("附录 B  整网最小启动命令")
    doc.code(
        """
export HCCL_IF_IP=<本机 IP>
export HCCL_SOCKET_IFNAME=<网卡名>
export GLOO_SOCKET_IFNAME=<网卡名>
export HCCL_BUFFSIZE=512
export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve "$MODEL_PATH" --host 0.0.0.0 --port 8000 \\
  --served-model-name glm-5.1 \\
  --data-parallel-size 1 --tensor-parallel-size 16 --enable-expert-parallel \\
  --max-num-seqs 8 --max-model-len 40960 --max-num-batched-tokens 4096 \\
  --trust-remote-code --quantization ascend --gpu-memory-utilization 0.90 \\
  --enable-chunked-prefill --enable-prefix-caching \\
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \\
  --additional-config '{
    "sparse_kv_offload_config": {
      "enabled": true,
      "use_fused_overlap": true,
      "keep_device_kv_cache": true,
      "dram_size_per_dp_GB": 48
    }
  }'
""",
        "单机整网启动",
    )

    doc.h2("附录 C  运行期常见问题")
    doc.table(
        "运行期问题排查",
        ["现象", "原因", "处理"],
        [
            ["`operator not found` / 算子缺失", "只装了单算子包，或没有重新加载环境", "按 3.3.2 安装全量包；新终端 source set_env.bash；重启进程"],
            ["`libcust_opapi.so` 加载失败", "环境变量未生效或库路径不对", "source 算子包的 set_env.bash，检查 `ASCEND_CUSTOM_OPP_PATH`"],
            ["`does not support packed SFA main caches (C8 or TQ4)`", "权重为 c8 或设置 kv-cache-dtype", "换非 c8 权重，去掉 `--kv-cache-dtype`"],
            ["`Sparse KV offload don't support compress now`", "权重带 `compress_ratios`", "换标准 DSA 权重（GLM-5.1/5.2）"],
            ["`requires TP without context parallelism`", "开启了 `enable_dsa_cp`", "从 additional-config 删除该项"],
            ["`Sparse KV offload offload.initialize failed`", "host 连续内存不足", "下调 `dram_size_per_dp_GB`，检查 `free -g`"],
            ["`HCCL_BUFFSIZE_EP is too SMALL`", "MoE AllToAll 缓冲区不足", "`export HCCL_BUFFSIZE=512` 后重启"],
            ["`HCCL_INTRA_PCIE_ENABLE or HCCL_INTRA_ROCE_ENABLE is invalid`", "只设置了一个 intra 变量", "两个都 unset，或显式 PCIE=1 + ROCE=0"],
            ["加载到 `Loading safetensors checkpoint shards: 0%` 长时间不动", "大权重首次加载", "正常等待，可看 `/proc/<pid>/io` 的 read_bytes 是否增长"],
        ],
        [6.8, 4.8, 5.0],
        first_col_bold=True,
    )

    doc.h2("附录 D  术语表")
    doc.table(
        "术语与缩写",
        ["术语", "含义"],
        [
            ["Pooling SFA", "FusedSparseAttentionOverlap 在本文中的简称"],
            ["SFA / DSA", "Sparse Flash Attention / DeepSeek Sparse Attention，带 Indexer 的稀疏注意力"],
            ["TopK 索引", "Indexer 为每个 Query Token 选出的 TopK 个历史 Token 编号"],
            ["Full KV Cache", "完整的 KV 缓存，PA_BSND 分页存储，prefill 写入"],
            ["Selection KV Cache", "只驻留 TopK Token 的紧凑缓存，位于 NPU 侧，算子原地维护"],
            ["Miss / Hit", "TopK Token 是否已驻留在 Selection KV Cache 中"],
            ["All-Hit", "本次 TopK 全部已驻留，搬运量为 0 的情形"],
            ["PA_BSND", "PageAttention 分页布局，逻辑块经 Block Table 映射到物理块"],
            ["TND", "以 Token 数、Head 数、HeadDim 组织的 query 排布"],
            ["All-Hit 幂等", "同样输入下第二次调用不改变输出与状态"],
        ],
        [4.2, 12.0],
        first_col_bold=True,
    )

    doc.h2("附录 E  参考资料")
    doc.bullets([
        "PR #20927：https://github.com/vllm-project/vllm-ascend/pull/20927",
        "算子 README：`csrc/attention/fused_sparse_attention_overlap/README.md`",
        "精度用例：`tests/e2e/nightly/single_node/ops/singlecard_ops/` 下的 `test_fused_sparse_attention_overlap.py`",
        "整网接入：`FSAO_整网接入指导手册.md`（同仓库）",
        "单算子编译：`Pooling_SFA算子编译指导.md`（同仓库）",
        "vLLM Ascend 文档：https://docs.vllm.ai/projects/ascend/en/latest/",
    ])

    p = doc.d.add_paragraph()
    p.paragraph_format.space_before = Pt(18)
    run = p.add_run("文档结束")
    set_run_font(run, cn=HEAD_CN, en=HEAD_CN, size=9, color=GRAY)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER


def build():
    doc = Doc()
    cover(doc)
    revision(doc)
    chapter1(doc)
    chapter2(doc)
    chapter3(doc)
    chapter4(doc)
    appendix(doc)
    doc.save(OUT_DOCX)
    print("docx written:", OUT_DOCX)


if __name__ == "__main__":
    build()
