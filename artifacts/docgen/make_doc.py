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
def chapter1(doc: Doc):
    doc.set_chapter(1)
    doc.h1("一、概述")

    doc.h2("1.1 算子简介")
    doc.h3("1.1.1 算子定位")
    doc.p(
        "FusedSparseAttentionOverlap（下文简称 Pooling SFA）是 DSA（DeepSeek Sparse Attention）"
        "稀疏注意力方案里的核心算子。DSA 在 decode 时只对 Indexer 选出的 TopK 个历史 Token 算注意力；"
        "这些 Token 的 KV 往往不在片上，需要按索引搬进 NPU 侧的 Selection KV Cache。"
        "Pooling SFA 把这两件事合成一次调用：搬和算在同一个 kernel 里重叠执行，KV offload 的搬运也一并融了进来。"
    )
    doc.p(
        "命中项直接从 Selection KV Cache 读，不再重复搬；未命中项边搬边算。"
        "算子会原地改 Selection KV Cache 和状态表，下一个 step 接着用。"
    )

    doc.h3("1.1.2 解决什么问题")
    doc.p(
        "长上下文推理里，decode 每个 step 都要读历史 KV，全量注意力的开销随序列长度线性涨。"
        "DSA 的思路是先用 Indexer 给每个 query 选 TopK 个 Token，只算这些，计算量从 O(L) 降到 O(topk)。"
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
        "Pooling SFA 用 status / membership 记下哪些 Token 已在 Selection Cache 里，只搬没在的，"
        "搬运和注意力计算流水重叠。PR #20927 在 arch22（A2/A3）上又补了几处优化，代码在 `.../op_kernel/arch22` 下。"
    )
    doc.p(
        "单算子测试里，selection-update 路径耗时降了 19%~42%，拷贝发射开销降了约 33%，"
        "命中率 0%~70% 区间端到端提升 10%~12%。这是单算子的数，整网能拿到多少要看 4.2 节的配置，别直接外推。",
        size=9.8,
        color=GRAY,
    )

    doc.h3("1.1.3 典型应用场景")
    doc.p("它解决的是 decode 阶段“KV 太多、只能挑着算”的场景，主要有下面几类：")
    doc.bullets([
        "长上下文稀疏推理。上下文 32K / 64K，全量注意力算不动，Indexer 每个 step 只选 TopK"
        "（≤ 2048）个 Token；本算子负责把这批 Token 的 KV 备好并算完，DSA/SFA 模型（如 GLM-5.x）都走这条路。",
        "PD 分离的 decode 节点。prefill 产生的 KV 下放到 host 侧大内存池，decode 节点按 TopK 把要用的 KV 拉回片上，"
        "Selection Cache 留在 NPU 上跨 step 复用，这是生产部署的典型形态。",
        "单机调试与功能验证。prefill 和 decode 放在同一台机器上，用 `keep_device_kv_cache` 让 KV 走一遍下放和回搬，"
        "验证接入是否正常；这种形态的绝对性能不代表生产。",
        "命中率高的连续 decode。相邻 step 的 TopK 大量重叠，命中率越高，算子省掉的搬运越多，收益越明显。",
        "kernel 改动后的单算子评估。改完 arch22 的 kernel，直接跑单算子用例和压测，做改动前后的对比。",
    ])

    doc.h2("1.2 算子能力范围")
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


def chapter2(doc: Doc):
    doc.set_chapter(2)
    doc.page_break()
    doc.h1("二、准备环境")

    doc.p(
        "下面的配置在 Atlas A3 单机上跑通过。开工前先确认硬件和镜像版本，自检不过就先别往下走。"
    )

    doc.h2("2.1 硬件要求")
    doc.table(
        "硬件要求",
        ["项目", "要求 / 取值", "说明"],
        [
            ["硬件产品", "Atlas A3", "已在 A3 单机（8 卡 × 2 chip）验证"],
            ["CPU 架构", "aarch64（ARM64）", "与镜像内 aarch64 wheel 一致"],
            ["芯片型号", "`ascend910_9382`（SOC `ascend910_93`）", "编译与运行期 SOC 必须对应"],
            ["达芬奇架构", "DAV_2201，代码路径 `op_kernel/arch22`", "PR #20927 的优化仅对 arch22 生效"],
            ["整网测试卡数", "8 卡 × 2 chip = 16 个逻辑 NPU，每 chip 64GB", "TP16 + EP16；合计 1TB HBM"],
            ["Host 内存", "≥ 512GB（建议 1TB）", "承载权重 page cache 与 sparse offload host pool"],
            ["共享内存 / 磁盘", "`/dev/shm` ≥ 1GB；构建磁盘 ≥ 100GB", "全量算子构建与权重加载需要"],
            ["NPU 驱动", "宿主机安装，容器挂载 `/usr/local/Ascend/driver`", "`npu-smi info` 正常显示设备即合格"],
        ],
        [3.0, 7.0, 6.6],
        first_col_bold=True,
    )

    doc.h2("2.2 软件要求")
    doc.table(
        "软件要求",
        ["组件", "要求 / 取值", "说明"],
        [
            ["容器镜像", "`vllm-ascend:30.0-a3`", "镜像仓中的标签为 `v0.30.0-a3`；镜像内已含 CANN、torch、torch_npu"],
            ["操作系统", "镜像自带 Linux（aarch64）", "不要替换系统自带的 glibc / gcc"],
            ["Python", "镜像内置", "与镜像保持一致，不要另装一套"],
            ["CANN", "镜像内置，已验证 9.2.1", "统一通过 `/usr/local/Ascend/ascend-toolkit` 加载，不要硬编码带版本号的路径"],
            ["vllm-ascend", "官方仓库 PR #20927 head", "`git fetch origin pull/20927/head:pr20927`"],
            ["vLLM", "`.github/vllm-main-verified.commit`，当前为 `a98247ab4db686ee03c66d5feb3c761e52a2f8ab`", "必须严格对齐，不能沿用镜像预装版本"],
            ["torch / torch_npu", "2.10.0 / 2.10.0.post4", "与 CANN 绑定，不要单独升级"],
            ["transformers / triton-ascend", "5.19.0 / 3.2.2", "整网路径需要"],
            ["memfabric_hybrid / memcache_hybrid", "1.2.0 / 1.2.0", "sparse KV offload 必需"],
        ],
        [4.0, 8.2, 4.4],
        first_col_bold=True,
    )

    doc.h2("2.3 其他依赖")
    doc.table(
        "其他依赖",
        ["依赖", "来源", "用途"],
        [
            ["modelscope", "单独安装，`<1.38`", "conftest 需要；测试不下载模型"],
            ["pytest", "单独安装", "运行 `per_test` 单算子精度用例"],
            ["setuptools-rust / ninja / pybind11", "pip 可安装的最新版本", "编译扩展与算子包"],
            ["ais-bench-benchmark + pandas", "独立 venv 安装", "整网压测（可选）"],
        ],
        [4.0, 8.6, 4.0],
        first_col_bold=True,
    )

    doc.h2("2.4 开工前自检")
    doc.p(
        "进容器后先跑下面几条，四条都符合再开始。任何一条不对，先看附录 B.1，不要硬着头皮编译。"
    )
    doc.code(
        r"""
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
""",
        "环境自检",
    )


def chapter3(doc: Doc):
    doc.set_chapter(3)
    doc.page_break()
    doc.h1("三、从零跑通（按顺序执行）")

    doc.p(
        "这一章是从零到算子可用的完整流程，命令按顺序执行即可。尖括号里的内容（如 `<容器名>`、`<本机 IP>`）"
        "要换成自己的实际值，其余保持原样。没有特别说明的话，命令都在 vllm-ascend 仓库根目录执行。"
        "每一步末尾都有检查点：看到期望结果再进下一步，没通过就翻附录 B。"
    )

    doc.h2("3.1 第 1 步：拉镜像、启动容器")
    doc.p(
        "在宿主机上执行，不是在容器里。机器上已有 `vllm-ascend:30.0-a3` 的可以跳过拉取。"
    )
    doc.code(
        r"""
export IMAGE=registry-cbu.huawei.com/ascend/vllm-ascend:v0.30.0-a3
docker images | grep vllm-ascend
docker pull "$IMAGE"
""",
        "拉取镜像",
    )
    doc.p(
        "启动容器。下面这条命令把 8 张 NPU 卡、驱动和常用目录都挂进去；`<容器名>` 自己起一个，比如 `fsao`。"
    )
    doc.code(
        r"""
docker run --name <容器名> -it -d --net=host --shm-size=500g \
  --privileged=true \
  -w /home \
  --device=/dev/davinci0 --device=/dev/davinci1 \
  --device=/dev/davinci2 --device=/dev/davinci3 \
  --device=/dev/davinci4 --device=/dev/davinci5 \
  --device=/dev/davinci6 --device=/dev/davinci7 \
  --device=/dev/davinci_manager \
  --device=/dev/hisi_hdc \
  --device=/dev/devmm_svm \
  --entrypoint=bash \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/sbin:/usr/local/sbin \
  -v /home:/home \
  -v /tmp:/tmp \
  -v /data:/data \
  -v /mnt/:/mnt/ \
  -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
  -v /etc/hccn.conf:/etc/hccn.conf \
  -e http_proxy=$http_proxy \
  -e https_proxy=$https_proxy \
  "$IMAGE"
""",
        "启动容器",
    )
    doc.p("进入容器：")
    doc.code("docker exec -it <容器名> bash", "进入容器")
    doc.p("检查点：能看到 8 张卡、Python 版本正常、架构是 aarch64。")
    doc.code(
        r"""
npu-smi info
python3 -c "import platform; print(platform.machine())"
""",
        "确认容器里的设备",
    )

    doc.h2("3.2 第 2 步：拉代码、配环境")
    doc.p(
        "代理和 pip 镜像按实际网络环境配，不需要就跳过前几行。PR #20927 的代码直接从官方仓库拉取引用，不必 fork。"
    )
    doc.code(
        r"""
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
    doc.p(
        "代码会拉到 `vllm-ascend` 目录。后面没有特别说明的命令，都在这个仓库根目录执行；"
        "换终端时先 `cd vllm-ascend`，并重新跑一遍下面的环境变量。"
    )
    doc.p("配置编译环境变量（只对当前终端生效）：")
    doc.code(
        r"""
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit
source "$ASCEND_HOME_PATH/set_env.sh"     # 部分版本为 bin/setenv.bash

unset LD_PRELOAD
export DRIVER_LIBS=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver
export LD_LIBRARY_PATH="$DRIVER_LIBS:/usr/local/Ascend/driver/lib64/common:${LD_LIBRARY_PATH:-}"

export SOC_VERSION=ascend910_9382
export MAX_JOBS=8
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
""",
        "编译环境变量",
    )
    doc.p(
        "安装编译和测试依赖。不要直接 `pip install -r requirements-dev.txt`：它会连带安装 `requirements.txt` 里的 "
        "torch / torch-npu / triton-ascend，和镜像里按 CANN 配好的版本不一定一致，容易把环境搞坏。只装下面这几项："
    )
    doc.code(
        r"""
python3 -m pip install setuptools-rust ninja pybind11
python3 -m pip install 'modelscope<1.38' 'pytest>=6,<9'
""",
        "安装编译与测试依赖",
    )
    doc.p("检查点：`ninja --version` 要能正常打印版本号，且 `which ninja` 在当前 Python 环境里。")
    doc.code(
        r"""
which python3
which ninja
ninja --version
""",
        "检查 ninja",
    )

    doc.h2("3.3 第 3 步：安装配套 vLLM")
    doc.p(
        "镜像里预装的 vLLM 不一定和分支匹配，必须切到仓库指定的提交。这一步只动 `/vllm-workspace/vllm`，"
        "不会切换或修改当前的 vllm-ascend 分支。"
    )
    doc.code(
        r"""
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
    doc.p("检查点：vLLM 的提交号和 `.github/vllm-main-verified.commit` 一致，且能从当前 Python 导入。")
    doc.code(
        r"""
git -C /vllm-workspace/vllm rev-parse HEAD
cat .github/vllm-main-verified.commit
python3 -c "import vllm; print(vllm.__file__)"
""",
        "确认 vLLM 提交",
    )

    doc.h2("3.4 第 4 步：编译并安装 vllm-ascend")
    doc.p(
        "全量编译约 40 个 A3 自定义算子与 Python 扩展，首次通常要 30 分钟以上，建议放在 tmux 里跑。"
    )
    doc.code(
        r"""
cd vllm-ascend
rm -rf csrc/build csrc/build_out csrc/output build

COMPILE_CUSTOM_KERNELS=1 python3 -m pip install -e . --no-build-isolation
""",
        "全量编译安装",
    )
    doc.p("检查点：最后出现下面两行就说明装好了。")
    doc.code(
        r"""
Successfully built vllm_ascend
Successfully installed ... vllm_ascend-...
""",
        "安装成功的标志",
    )

    doc.h2("3.5 第 5 步：确认算子能加载")
    doc.p("加载自定义算子包，确认算子已经注册：")
    doc.code(
        r"""
cd vllm-ascend
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
    doc.p("检查点：输出 `算子加载成功`。到这一步，编译和安装就完成了。")


def chapter6_service(doc: Doc):
    doc.set_chapter(6)
    doc.page_break()
    doc.h1("六、接入指导：启动推理服务")

    doc.p(
        "单算子精度和性能都通过后，再做整网接入。本章只调整部署参数，不再改算子二进制。"
    )

    doc.h2("6.1 能接哪些模型")
    doc.p("只有带 Indexer 的稀疏注意力模型才走这条路径，落地时要同时满足模型结构和 KV 缓存两方面的条件：")
    doc.bullets([
        "权重 config 里有 `index_topk`，没有 `compress_ratios`。GLM-5.x 属于这一类；DeepSeek-V4 系列带 CSA2 "
        "压缩，不走这条路径。",
        "主 KV 不是 C8 / TQ4 packed 缓存。`w8a8c8`、`w4a8c8` 这类带 c8 的权重会被 sparse offload 直接拒绝。",
        "部署时开启 `sparse_kv_offload_config`，并设置 `use_fused_overlap=true`。否则走的是普通推理或 "
        "Copy-SFA 路径，算子不会执行。",
        "不要开 `enable_dsa_cp`，也不要用 `--kv-cache-dtype int8/fp8`，两者都和 sparse offload 互斥。",
    ])
    doc.table(
        "常见权重的接入结论",
        ["权重", "结论", "说明"],
        [
            ["GLM-5.1-w4a8", "可用", "约 391GB，TP16 下每 chip 约 24.5GB"],
            ["GLM-5.2-w4a8", "可用，余量小", "约 755GB，TP16 下每 chip 约 47GB"],
            ["GLM-5.2-w8a8", "可用，余量很小", "权重文档下限 1TB，刚好达到"],
            ["任何 c8 权重", "不可用", "C8 主 KV 与 sparse offload 互斥"],
            ["DeepSeek-V4 / V4.1-Flash", "不可用", "带 `compress_ratios`（CSA2），不走 SFA/offload 路径"],
        ],
        [4.2, 4.0, 8.4],
        first_col_bold=True,
    )

    doc.h2("6.2 启动服务")
    doc.p(
        "先准备好模型权重（本文用 GLM-5.1-w4a8，约 391GB），把 `MODEL_PATH` 指到权重目录。网卡名和 IP 用"
        "下面第一条命令查，挑一个不是 127.0.0.1 的，第二列就是网卡名。服务参数按 16 卡、TP16 写的。"
    )
    doc.code(
        r"""
export MODEL_PATH=<权重目录>

# 查看本机网卡和 IP，第二列是网卡名
ip -o -4 addr show | awk '$4 !~ /^127/ {print $2, $4}'

export HCCL_IF_IP=<本机 IP>
export HCCL_SOCKET_IFNAME=<网卡名>
export GLOO_SOCKET_IFNAME=<网卡名>
export TP_SOCKET_IFNAME=<网卡名>
export HCCL_BUFFSIZE=512
export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"
unset HCCL_INTRA_PCIE_ENABLE
unset HCCL_INTRA_ROCE_ENABLE
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 --port 8000 \
  --served-model-name glm-5.1 \
  --data-parallel-size 1 \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --max-num-seqs 8 \
  --max-model-len 40960 \
  --max-num-batched-tokens 4096 \
  --trust-remote-code \
  --quantization ascend \
  --gpu-memory-utilization 0.90 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --additional-config '{
    "sparse_kv_offload_config": {
      "enabled": true,
      "use_fused_overlap": true,
      "keep_device_kv_cache": true,
      "dram_size_per_dp_GB": 48
    }
  }'
""",
        "启动推理服务",
    )
    doc.p(
        "首次启动要从磁盘加载权重，日志停在 loading 阶段十几分钟是正常的。看到 "
        "`Application startup complete` 就说明服务起来了。另开一个终端执行："
    )
    doc.code(
        r"""
curl -s http://127.0.0.1:8000/health; echo
""",
        "检查服务是否就绪",
    )
    doc.p("检查点：返回 `{\"status\":\"ok\"}`，或者直接进入 6.3 发一条请求看结果。")

    doc.h2("6.3 发一个请求")
    doc.code(
        r"""
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.1","messages":[{"role":"user","content":"hi"}],"max_tokens":16,"temperature":0}'
""",
        "发一条对话请求",
    )
    doc.p(
        "检查点：返回 JSON 里有 `choices` 字段和模型输出。能返回结果，说明服务通、算子也被调用过了。"
    )


def chapter7_verify(doc: Doc):
    doc.set_chapter(7)
    doc.page_break()
    doc.h1("七、验证算子确实生效")

    doc.p("进程不崩、请求能回，不代表走的是这个算子。下面两件事按顺序确认。")

    doc.h2("7.1 看指标")
    doc.code(
        r"""
curl -s http://127.0.0.1:8000/metrics | grep prefix_cache | head
""",
        "查看 prefix cache 指标",
    )
    doc.p("检查点：能查到 `vllm:prefix_cache_*`，说明 sparse offload 路径整体生效。")

    doc.h2("7.2 验证清单")
    doc.table(
        "结构验证清单",
        ["层次", "验证项", "通过标准"],
        [
            ["编译", "算子注册", "`hasattr(torch.ops._C_ascend, \"npu_fused_sparse_attention_overlap\")` 为 True"],
            ["单算子", "状态正确性", "`status[topk] == topk`，Cache 与 Gather 结果一致"],
            ["整网", "服务可用", "`/health` 200，对话请求正常返回"],
            ["整网", "路径生效", "`/metrics` 有 `vllm:prefix_cache_*`"],
            ["整网", "性能", "TPOT 稳定，命中率提升时搬运开销下降"],
        ],
        [3.0, 6.0, 7.6],
        first_col_bold=True,
    )


def chapter4_precision(doc: Doc):
    doc.set_chapter(4)
    doc.page_break()
    doc.h1("四、单算子精度测试")

    doc.p(
        "单算子编译通过后，先过精度门禁，再进入性能测试。性能只回答“快不快”，"
        "不回答“对不对”；精度挂了，性能数据直接作废。"
    )

    doc.h2("4.1 运行精度门禁")
    doc.p(
        "`per_test` 自带小内存精度用例，不需要额外测试依赖，也不依赖整网服务。"
        "它会覆盖内部 planner 和外部计划（LRU）两条路，以及 miss / hit 两种状态。"
    )
    doc.code(
        r"""
cd vllm-ascend
DSA_CPU_KV_BACKEND=offload bash per_test/run.sh precision
""",
        "单算子精度门禁",
    )
    doc.p(
        "检查点：最后一行是 `74 passed, 27 deselected, 0 failed`（A3 小内存预设）；"
        "出现 failed 先停下，不要继续性能测试。"
    )

    doc.h2("4.2 官方精度用例（可选补充）")
    doc.p(
        "仓库内的官方 e2e 精度用例带 fp32 golden，可作为结构补充；它不是本章的门禁，"
        "门禁仍以 4.1 的 `per_test` 结果为准。该用例的 conftest 依赖见附录 B.3。"
    )
    doc.code(
        r"""
cd vllm-ascend
source "$PWD/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash"
export ASCEND_RT_VISIBLE_DEVICES=0

pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_sparse_attention_overlap.py
""",
        "运行官方精度用例",
    )
    doc.p("检查点：最后一行是 `4 passed`。")

    doc.h2("4.3 完整 per_test 精度用例（可选）")
    doc.p(
        "小内存预设会跳过 `baseline_batch_sizes` / `baseline_hit_ratios` 两条重尾巴用例，"
        "它们固定 msl=65536，需要更大的宿主池。机器内存/大页充足时跑完整版："
    )
    doc.code(
        r"""
DSA_CPU_KV_BACKEND=offload bash per_test/run.sh precision-full
""",
        "完整精度用例",
    )

    doc.note(
        "为什么精度必须先于性能",
        [
            "本算子有内部 planner 和外部计划两条路，只测一条会漏掉真问题。",
            "先把数值正确性钉住，后面性能 A/B 的差异才有意义。",
        ],
    )


def chapter5_perf(doc: Doc):
    doc.set_chapter(5)
    doc.page_break()
    doc.h1("五、单算子性能测试")

    doc.p(
        "本章说明如何用仓库根目录的 `per_test/` 工程测量 FusedSparseAttentionOverlap "
        "的单算子 kernel 性能，并比较优化前后的构建。整网 TPOT / 吞吐的 A/B 属于另一套流程，"
        "不在本章范围。"
    )

    doc.note(
        "先记住三条判据",
        [
            "**只看 `kernel_p50_us`**：端到端包含约 160 μs 框架开销且会漂移，不作为判据。",
            "**差别小于门槛的不算改进**：门槛不是拍脑袋的百分比，而是这一趟测量自己的 A/A 误差棒。",
            "**精度先过**：性能变快但精度挂了，是“错的更快”，这组性能数据直接作废。",
        ],
    )

    doc.h2("5.1 测试工程 per_test")

    doc.p(
        "`per_test/` 放在仓库根目录，入口是 `run.sh`。所有内部路径按脚本自身位置解析，"
        "不依赖目录名，也不需要先手动 `source` 环境；日志默认写到 `per_test/logs/`。"
    )

    doc.table(
        "per_test 目录",
        ["文件", "作用"],
        [
            ["`run.sh`", "一键入口：env / build / precision / perf / compare / check"],
            ["`env.sh`", "CANN、SOC_VERSION、memfabric、vendor 算子包、PYTHONPATH、ulimit"],
            ["`build_ops.sh`", "编译 vendor 算子包 + `vllm_ascend_C` .so"],
            ["`sweep_cases.sh`", "标准曲线入口（hit / hit4 / bs / bs0 / seq / frag / ablate / validate）"],
            ["`sweep_median.py`", "多进程跑 bench，逐点取中位数"],
            ["`bench_fuse_sparse_attention_overlap.py`", "单进程计时 bench，产出 KERNEL / RESULT 行"],
            ["`cmp_sweeps.py`", "两趟结果逐点比较，按 A/A 自动算门槛"],
            ["`fake_bench.py`", "SMOKE 假 bench，秒回，不碰 NPU"],
            ["`precision_small.sh` + `tests/test_fsao_precision.py`", "小内存精度门禁"],
            ["`sweep_small.sh`", "小内存性能冒烟，只用于内存受限机器"],
            ["`.gitattributes`", "强制测试文件 LF，避免 Windows 拷到 Linux 后 CRLF 报错"],
        ],
        [7.0, 9.4],
        first_col_bold=True,
    )

    doc.p("常用入口：")
    doc.code(
        r"""
bash per_test/run.sh env          # 自检：仓库 / SOC / memfabric / vendor 包 / python / pytest
bash per_test/run.sh build        # 编译（可加算子名只编单个）
bash per_test/run.sh precision    # 精度门禁
bash per_test/run.sh perf hit4    # 标准性能曲线（本章主用）
bash per_test/run.sh compare A.log B.log
""",
        "per_test 一键入口",
    )

    doc.h2("5.2 环境与构建")

    doc.p("跑测试前先自检，确认输出里的 SOC、memfabric、vendor 包三项：")
    doc.code(
        r"""
bash per_test/run.sh env
""",
        "环境自检",
    )
    doc.bullets([
        "`SOC` 应为实际芯片，例如 A3 的 `ascend910_9382`、A5 的 `ascend950pr_xxx`；",
        "`memfabric offload 可用（1.3.0 ...）`：镜像内的 MemFabric 1.3 已就绪；",
        "`vendor op` 指向本仓库 `vllm_ascend/_cann_ops_custom/vendors/custom_transformer`；",
        "`ulimit -l = unlimited`：host KV 池需要；脚本会尝试自动抬高。",
    ])

    doc.p("只编本算子（A5 的全量列表不含 FSAO，必须单算子；A3 单算子也更快）：")
    doc.code(
        r"""
bash per_test/run.sh build fused_sparse_attention_overlap
""",
        "编译 FSAO 算子包",
    )

    doc.p(
        "编译结束会打印 kernel 指纹。**换构建后必须确认指纹变了**；指纹没变说明优化没有进二进制。"
    )
    doc.code(
        r"""
find vllm_ascend/_cann_ops_custom -type f \
  -path '*fused_sparse_attention_overlap*' -print0 \
  | sort -z | xargs -0 md5sum | md5sum
""",
        "记录 FSAO 产物指纹",
    )

    doc.note(
        "增量编译不可信",
        [
            "时间戳、文件名里的哈希、日志里出现算子名，三样都可能骗人；只有 `.o` 的内容 md5 不骗人。",
            "改过 `csrc/` 下任何头文件，必须全量重编（`build_ops.sh` 默认会删 build）。",
            "只改 kernel 时不必重装 `vllm_ascend_C.so`；如果改了算子 schema，才需要重编 .so。",
        ],
    )

    doc.h2("5.3 测试矩阵")

    doc.table(
        "per_test 标准模式",
        ["模式", "形状", "用途", "点数/进程"],
        [
            ["`hit`", "bs=16，msl=64K，扫描命中率", "命中率曲线", "44"],
            ["`hit4`", "bs=4，msl=64K，扫描命中率", "helper 主战场（本章默认）", "44"],
            ["`bs`", "80% 命中，扫描 batch size", "batch size 曲线", "40"],
            ["`bs0`", "0% 命中，扫描 batch size", "搬运受限端", "40"],
            ["`seq`", "80% 命中，扫描 msl", "序列长度曲线", "4/档"],
            ["`frag`", "命中名次连续 vs 碎片", "碎片化归因", "16"],
            ["`ablate`", "单点消融", "归因", "2"],
            ["`validate`", "串行 vs 并行", "并行等价性复验", "20"],
        ],
        [2.2, 7.2, 5.2, 2.2],
        first_col_bold=True,
    )

    doc.p(
        "为什么优化 A3 主要看 `hit4`：bs=16 时 24 个核已经用满，“用上空闲核”的收益恒为零，"
        "`hit` 是零对照臂（曲线必须平）；bs≤8 才能看到 helper 机制的真实收益。"
        "`hit4` 与其只差 batch size，其余参数完全一致。"
    )

    doc.h2("5.4 hit4 标准参数")

    doc.p("`hit4` 实际展开后的命令如下（`sweep_cases.sh` 固定参数 + 环境变量覆盖）：")
    doc.code(
        r"""
python3 per_test/sweep_median.py --processes 5 --e2e --echo -- \
  --batch-size 4 \
  --max-seq-len 65536 \
  --topk 2048 \
  --q-heads 16 \
  --topk-buffer-size 4096 \
  --plan-direct on \
  --warmup 10 --iters 50 --repeat 1 --kernel-time 200 --disable-gc \
  --hit-layout strided \
  --cpu-kv-backend offload \
  [--dram-size-gb <DRAM_GB>] \
  --plan-slots off,scattered \
  --hit-ratio 0.0000,0.1000,0.2000,0.3000,0.4000,0.5000,0.6000,0.7000,0.8000,0.9000,1.0000,0.0001,0.1001,0.2001,0.2999,0.4001,0.5001,0.6001,0.7001,0.7999,0.9001,0.9999
""",
        "hit4 完整命令",
    )

    doc.table(
        "hit4 参数含义",
        ["参数", "值", "说明"],
        [
            ["`--batch-size`", "4", "“hit4” 的 4；`hit` 是 16"],
            ["`--max-seq-len`", "65536", "64K 上下文"],
            ["`--topk`", "2048", "FSAO topk"],
            ["`--q-heads`", "16", "MLA q heads"],
            ["`--topk-buffer-size`", "4096", "槽位数，生产默认不跟着 topk 走"],
            ["`--plan-direct`", "on", "控制区直接寻址，与生产一致"],
            ["`--hit-layout`", "strided", "碎片化最重的一档"],
            ["`--plan-slots`", "off,scattered", "同一趟里交错测两条 plan 路"],
            ["`--hit-ratio`", "11 个整点 + 11 个 A/A 标签", "0,10%,…,100% 及噪声对照"],
            ["`--warmup` / `--iters` / `--repeat`", "10 / 50 / 1", "每点 50 次正式测量"],
            ["`--kernel-time`", "200", "触发 KERNEL 行，不是超时"],
            ["`--disable-gc`", "开", "减少 host 侧 Python GC 噪声"],
            ["`--cpu-kv-backend`", "offload", "MemFabric 宿主 KV 池"],
            ["`--dram-size-gb`", "可选", "不传按 msl 自动算（64K→1.5 GiB，对齐 2 GiB）；建议统一 `DRAM_GB=1`"],
            ["`--processes`", "5", "5 个独立进程，逐点取中位数"],
        ],
        [5.2, 6.0, 5.4],
        first_col_bold=True,
    )

    doc.p(
        "为什么这些参数不能随手改：`topk-buffer-size` 量成 topk 会让命中显得比实际连续；"
        "`plan-direct off` 会让 `slot >= topk` 的命中被静默判成未命中；"
        "`hit-layout prefix` 最友好，结果偏乐观；"
        "两条 plan 路必须一趟里交错测，分两趟中间横着 2~4%，正好是要测的差值本身的量级。"
    )

    doc.h2("5.5 测量方法：A/A、进程与门槛")

    doc.p(
        "`hit4` 默认 22 个命中率（11 个真实点 + 11 个 A/A 标签）× 2 条 `plan-slots` "
        "= **44 点/进程**；`--processes 5` 表示 5 个进程各跑一遍，逐点取中位数。"
    )
    doc.bullets([
        "**A/A 标签**：`round(r×topk)` 与对应整点完全相同，只是名字不同、排在队尾；"
        "两次读数的差就是这一趟的误差棒。",
        "**门槛**：取**基线臂** A/A 的两倍（下限 0.5%）；不能取两臂最大值，否则一个臂越烂判据越松。",
        "**硬闸**：任一臂自己的 A/A 超过 2%，这一趟直接不合格——连自己都复现不出来的数，跟谁比都没意义。",
        "**进度检查**：每个进程的点数必须一样；进程分成了几团，说明中位数落在空档里，不能用一个数概括。",
        "**只测一次的点**：没有 A/A 标签，绝对值不可用；正式曲线不要关 `AA_LABELS`。",
    ])

    doc.p(
        "实测时如果出现“进程间数值差很多但代码没动”，用下面的脚本查这趟是否被别的任务干扰："
    )
    doc.code(
        r"""
bash per_test/run.sh check 本趟.log 干净的那趟.log
""",
        "检查干扰",
    )

    doc.h2("5.6 基线 vs 优化：A/B 流程")

    doc.h3("5.6.1 基线版本：vllm-ascend 0.30.0")

    doc.p(
        "基线固定使用 **vllm-ascend 0.30.0**（镜像 `vllm-ascend:0.30.0-a3` / "
        "`vllm-ascend:0.30.0-a5`）。这个版本还没有 PR #20927 的 FSAO 优化，"
        "作为**未优化对照**；优化版本使用包含 #20927 的 main。"
    )

    doc.p("基线源码准备（二选一）：")
    doc.code(
        r"""
# 方式一（推荐）：直接用 0.30.0 镜像里自带的源码树
cd /vllm-workspace/vllm-ascend
# 把我们的 per_test/ 拷到仓库根目录

# 方式二：在机器上单独 clone 与镜像一致的 0.30.0 源码
#   以镜像内实际版本为准（例如 v0.30.0rc2）
git clone -b <0.30.0-tag> https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
# 把我们的 per_test/ 拷进来
""",
        "准备 v0.30.0 基线源码",
    )

    doc.p("确认这个版本里没有 #20927 的优化提交，然后单算子编译：")
    doc.code(
        r"""
# 没有输出 “Optimize fused_sparse_attention_overlap” 才是基线
git log --oneline | grep -i "Optimize fused_sparse_attention_overlap" \
  || echo "no #20927 (baseline)"

export SOC_VERSION=ascend910_9382      # A3；A5 用 ascend950pr_xxx
bash per_test/run.sh build fused_sparse_attention_overlap
""",
        "编译 v0.30.0 基线算子",
    )

    doc.p(
        "编译后记录 kernel 指纹。基线跑完先把日志存成稳定名字，后面优化版本必须用完全相同的命令："
    )
    doc.code(
        r"""
DRAM_GB=1 bash per_test/run.sh perf hit4
DSA_CPU_KV_BACKEND=offload bash per_test/run.sh precision

cp "$(ls -t per_test/logs/case_hit4_curve_*.log | head -1)" \
   per_test/logs/baseline_hit4.log
""",
        "跑 v0.30.0 基线并保存日志",
    )

    doc.note(
        "基线测试的两个前提",
        [
            "基线必须和优化版本在同一台机器、同一张卡上跑；跨机器/跨卡不可比。",
            "0.30.0 与 main 之间还隔着若干 upstream 提交；如果只想证明 #20927 本身，"
            "应在同一个 main 底座上回退 #20927 作为对照。",
        ],
        accent="B45309",
        fill="FFF7E6",
    )

    doc.p("推荐按下面顺序做，两个构建只允许算子二进制不同：")
    doc.bullets([
        "**基线**：用 vllm-ascend 0.30.0（未含 #20927）编译 FSAO，"
        "记录 kernel 指纹，跑 `DRAM_GB=1 bash per_test/run.sh perf hit4`，保存日志。",
        "**保存基线包（可选 ABBA）**：`cp -a vllm_ascend/_cann_ops_custom /path/to/ops_base`。",
        "**优化**：切到优化源码（例如 `origin/main`，含 PR #20927），"
        "重新单算子编译，确认 kernel 指纹与基线不同。",
        "**同一命令重跑**：`DRAM_GB=1 bash per_test/run.sh perf hit4`，"
        "参数、卡、进程数、后端、池大小必须与基线完全一致。",
        "**精度门禁**：`DSA_CPU_KV_BACKEND=offload bash per_test/run.sh precision`，必须 0 failed。",
        "**逐点比较**：`python3 per_test/cmp_sweeps.py 基线.log 优化后.log --label-a 基线 --label-b 优化后`。",
    ])

    doc.note(
        "A/B 公平性检查",
        [
            "同一台机器、同一张卡、同一批进程数；跑的过程中不要 `git pull`、不要 `| head`。",
            "一次只跑一趟：host 内存带宽共用，两趟叠跑会偏向性污染 miss 重的一端（实测约 +6%）。",
            "两臂的 `DRAM_GB`、`DSA_CPU_KV_BACKEND`、`hit-layout`、`plan-slots`、`kernel-time` 必须一致。",
            "正式结论建议做 ABBA（base → opt → opt → base），用同一套 vendor 包来回切，抵消时间漂移。",
        ],
        accent="B45309",
        fill="FFF7E6",
    )

    doc.p(
        "如果两个分支的上游底座不同（例如一个基于旧 main、一个基于新 main），"
        "那 A/B 的差异里除了目标优化，还混着其它 upstream 提交；"
        "要严格比较单个 PR，应在**同一个底座**上回退/应用该 PR，再加 `per_test/`。"
    )

    doc.h2("5.7 结果怎么读")

    doc.p("比较脚本只比 `=== kernel duration：绝对值（μs）===` 这张表，输出带判定：")
    doc.code(
        r"""
python3 per_test/cmp_sweeps.py baseline.log optimized.log \
  --label-a 基线 --label-b 优化后
""",
        "逐点比较",
    )
    doc.bullets([
        "先看两臂 A/A：任一臂超过 2%，这组数据作废，先查干扰或重跑。",
        "再看每个命中率点、两条 plan 路是否都变快；只在一个点快不等于整条曲线快。",
        "变化小于门槛的标 `within noise`，不算改进。",
        "最后看曲线形状：优化不应把某一段明显抬高。",
        "精度结果与性能结果一起归档；同一次构建的精度日志和性能日志放在一起。",
    ])

    doc.h2("5.8 常见问题")

    doc.table(
        "性能测试常见问题",
        ["现象", "原因", "处理"],
        [
            ["`$'\\r': command not found`", "Windows 拷贝时脚本是 CRLF", "只拷 `per_test/`（已强制 LF），或 `git ls-files '*.sh' | xargs sed -i 's/\\r$//'`"],
            ["`OffloadConfig` 没有 `reserve_size/alloc_size`", "memfabric 不是 1.3", "升级 memfabric 1.3.0；脚本已按 1.3 API 编写"],
            ["`offload.initialize returned -1 ... Map mem failed:-6`", "宿主大页/平台映射问题", "A3 通常无需配大页；A5 需配 2M 大页或平台配置，conn-based 兜底不适用于本 kernel"],
            ["kernel 指纹没变", "增量/缓存没重编", "删 build 全量重编，检查 build-cache HIT/MISS"],
            ["`DRAM_GB` 残留导致池过大", "shell 里 export 了旧值", "`unset DRAM_GB` 或显式 `DRAM_GB=1`"],
            ["进程点数不一致 / 告警 `SWEEP NOT SELF-CONSISTENT`", "有进程少测/被干扰", "看告警，重跑；一次只跑一趟"],
            ["精度通过但性能异常快/慢", "构建混了旧 `.so`/旧 vendor 包", "对比 kernel 指纹和 `.so` 时间，重新构建"],
        ],
        [7.0, 4.8, 4.6],
        first_col_bold=True,
        keep=False,
    )

    doc.h2("5.9 最小命令清单（A3）")

    doc.code(
        r"""
cd /home/z00943842/vllm-ascend
git checkout <优化分支>
unset DRAM_GB DSA_OFFLOAD_GIANT_PAGE MF_HYBM_ENABLE_4K_PAGE

bash per_test/run.sh env
bash per_test/run.sh build fused_sparse_attention_overlap

SMOKE=1 bash per_test/run.sh perf hit
DRAM_GB=1 PROCS=1 RATIOS_MAIN=1.0 AA_LABELS=0 bash per_test/run.sh perf hit

# 精度门禁：见第四章

DRAM_GB=1 bash per_test/run.sh perf hit4

python3 per_test/cmp_sweeps.py \
  per_test/logs/baseline_hit4.log \
  "$(ls -t per_test/logs/case_hit4_curve_*.log | head -1)" \
  --label-a 基线 --label-b 优化后
""",
        "A3 性能测试最小流程",
    )

    doc.p(
        "快速验证用的 `PROCS=1 RATIOS_MAIN=1.0 AA_LABELS=0` 只用于确认 kernel 能跑，"
        "绝对值不能用于 A/B；正式数据必须回到默认 5 进程和 A/A 标签。"
    )


def appendix(doc: Doc):
    doc.set_chapter("A")
    doc.page_break()
    doc.h1("八、附录")

    doc.h2("附录 A  编译参数与产物")
    doc.p("主流程用的是全量编译命令，一般不用改。需要调整构建方式时看下面两张表。")
    doc.table(
        "编译参数清单",
        ["参数 / 变量", "官方预编译包取值", "作用"],
        [
            ["`COMPILE_CUSTOM_KERNELS`", "`1`", "`pip install -e .` 时编译全部 CANN 自定义算子；`0` 只装 Python 包，不重编算子"],
            ["`SOC_VERSION`（构建）", "`ascend910_9382`", "传给 build_aclnn.sh，决定编译的芯片型号"],
            ["`--soc`（build.sh）", "`ascend910_93`", "SoC 简称，对应芯片 `ascend910_9382`；不要写成全名"],
            ["`--pkg`", "启用", "生成带 kernel bin 的 run 包；不加则为 JIT 包"],
            ["`--ops`", "缺省（全量）", "逗号分隔的算子 snake 名，只编指定算子"],
            ["`-j` / `MAX_JOBS`", "`8`", "并行编译任务数"],
            ["`-O` / `--build-type`", "`O2`（Release）", "编译优化级别"],
            ["`USE_NINJA` / `VLLM_ASCEND_USE_NINJA`", "`auto`", "构建器选择，auto 时优先 ninja"],
            ["`VLLM_BATCH_INVARIANT`", "`0`", "为 1 时额外构建 batch invariant 算子包"],
        ],
        [5.4, 5.6, 5.6],
        first_col_bold=True,
    )
    doc.table(
        "本算子的 kernel 编译选项",
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
    doc.table(
        "编译产物与安装位置",
        ["产物", "路径", "说明"],
        [
            ["算子 run 包", "`csrc/build/cann-ops-transformer*.run`", "可单独分发、单独安装"],
            ["算子包根目录", "`vllm_ascend/_cann_ops_custom/vendors/custom_transformer/`", "运行时通过 set_env.bash 挂载"],
            ["kernel 二进制", "`.../custom_transformer/op_impl/ai_core/tbe/kernel/`\n（按 SOC / arch 分目录）", "arch22 对应 A2/A3"],
            ["环境脚本", "`.../custom_transformer/bin/set_env.bash`", "设置 `ASCEND_CUSTOM_OPP_PATH` 与库路径"],
            ["Python 扩展", "`vllm_ascend/vllm_ascend_C*.so`", "torch 自定义算子注册入口"],
        ],
        [3.6, 9.4, 3.6],
        first_col_bold=True,
    )
    doc.p(
        "只改了 kernel、不想重装 vllm-ascend 时，`touch` 改动文件后重跑编译脚本即可，"
        "`COMPILE_CUSTOM_KERNELS=0` 只重装 Python 包、不会重编算子："
    )
    doc.code('bash csrc/build_aclnn.sh "$PWD" "$SOC_VERSION"', "增量编译算子（不重装 vllm-ascend）")

    doc.h2("附录 B  常见报错与处理")

    doc.h3("B.1 环境自检不过")
    doc.bullets([
        "`npu-smi info` 失败：先查宿主机 NPU 驱动和容器设备挂载（`--device=/dev/davinci*` 等），"
        "容器里没有设备时后面的步骤都会失败。",
        "`platform.machine()` 不是 `aarch64`：拉错了镜像，或者跑在 x86 节点上。",
        "`torch.npu.device_count()` 是 0：容器启动时没挂 `--device`，或者宿主机驱动异常。",
        "`vllm.__file__` 为 `None`：当前目录下有名为 `vllm/` 的目录遮蔽了安装包，换目录执行。",
    ])

    doc.h3("B.2 编译阶段")
    doc.table(
        "编译阶段常见失败",
        ["现象", "原因", "处理"],
        [
            ["`ModuleNotFoundError: No module named 'setuptools_rust'`", "缺少编译依赖", "`python3 -m pip install setuptools-rust` 后重跑"],
            ["`No module named 'ninja'`，或 ninja 路径指向其它 Python", "ninja 来自其它 Python 环境", "重装 ninja，调整 PATH 后 `hash -r`"],
            ["`libgraph_base.so: undefined symbol`", "复用了其它 CANN 版本留下的 CMake / ninja 缓存", "清理 `csrc/build`、`csrc/build_out`、`csrc/output`、`build` 后全量重编"],
            ["`build_aclnn.sh` 非零退出", "环境变量、SOC 或依赖不满足", "查看日志第一处 `[ERROR]`，逐项核对 3.2"],
            ["`catlass is missing, try to fetch it` 后失败", "未初始化子模块 / 代理不可用", "`git submodule update --init --recursive` 后重试"],
            ["`ERROR: Failed building wheel for vllm_ascend`", "算子编译失败", "向上翻日志找到第一个 error，而不是只看最后一行"],
        ],
        [6.8, 5.0, 4.8],
        first_col_bold=True,
        keep=False,
    )

    doc.h3("B.3 运行与推理")
    doc.table(
        "运行期常见问题",
        ["现象", "原因", "处理"],
        [
            ["`operator not found` / 算子缺失", "算子包没装上，或环境没重新加载", "按 3.4 重新安装；新终端 source set_env.bash；重启进程"],
            ["`libcust_opapi.so` 加载失败", "环境变量未生效或库路径不对", "source 算子包的 set_env.bash，检查 `ASCEND_CUSTOM_OPP_PATH`"],
            ["`does not support packed SFA main caches (C8 or TQ4)`", "权重为 c8 或设置 kv-cache-dtype", "换非 c8 权重，去掉 `--kv-cache-dtype`"],
            ["`Sparse KV offload don't support compress now`", "权重带 `compress_ratios`", "换标准 DSA 权重（GLM-5.1/5.2）"],
            ["`requires TP without context parallelism`", "开启了 `enable_dsa_cp`", "从 additional-config 删除该项"],
            ["`Sparse KV offload offload.initialize failed`", "host 连续内存不足", "下调 `dram_size_per_dp_GB`，检查 `free -g`"],
            ["`HCCL_BUFFSIZE_EP is too SMALL`", "MoE AllToAll 缓冲区不足", "`export HCCL_BUFFSIZE=512` 后重启"],
            ["`HCCL_INTRA_PCIE_ENABLE or HCCL_INTRA_ROCE_ENABLE is invalid`", "只设置了一个 intra 变量", "两个都 unset，或显式 PCIE=1 + ROCE=0"],
            ["`No module named 'fla_npu'` / 官方精度用例 conftest 报错", "官方 e2e 用例的 conftest 依赖 FLA", "当前镜像一般已预装；缺失时从 https://github.com/flashserve/flash-linear-attention-npu/releases 下载 aarch64 wheel 安装"],
            ["加载到 `Loading safetensors checkpoint shards: 0%` 长时间不动", "大权重首次加载", "正常等待，可看 `/proc/<pid>/io` 的 read_bytes 是否增长"],
            ["`RuntimeError: HCCL ... error code is 19`", "网卡选择失败或 hostname 解析不到", "设置 `HCCL_IF_IP` / `HCCL_SOCKET_IFNAME`，检查 `/etc/hosts`"],
        ],
        [6.8, 4.8, 5.0],
        first_col_bold=True,
        keep=False,
    )

    doc.h2("附录 C  术语表")
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
        ],
        [4.2, 12.0],
        first_col_bold=True,
    )

    doc.h2("附录 D  参考资料")
    doc.bullets([
        "PR #20927：https://github.com/vllm-project/vllm-ascend/pull/20927",
        "算子 README：`csrc/attention/fused_sparse_attention_overlap/README.md`",
        "精度用例：`tests/e2e/nightly/single_node/ops/singlecard_ops/` 下的 `test_fused_sparse_attention_overlap.py`",
        "整网接入：`FSAO_整网接入指导手册.md`（同仓库）",
        "vLLM Ascend 文档：https://docs.vllm.ai/projects/ascend/en/latest/",
    ])

    p = doc.d.add_paragraph()
    p.paragraph_format.space_before = Pt(18)
    run = p.add_run("文档结束")
    set_run_font(run, cn=HEAD_CN, en=HEAD_CN, size=9, color=GRAY)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER


def build():
    doc = Doc()
    chapter1(doc)
    chapter2(doc)
    chapter3(doc)
    chapter4_precision(doc)
    chapter5_perf(doc)
    chapter6_service(doc)
    chapter7_verify(doc)
    appendix(doc)
    doc.save(OUT_DOCX)
    print("docx written:", OUT_DOCX)


if __name__ == "__main__":
    build()
