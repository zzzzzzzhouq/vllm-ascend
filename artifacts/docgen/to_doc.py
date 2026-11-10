# -*- coding: utf-8 -*-
"""Convert the generated docx to .doc (Word 97-2003) and export a PDF preview."""
import sys
from pathlib import Path

import win32com.client as win32

HERE = Path(__file__).resolve().parent
DOCX = HERE / "FusedSparseAttentionOverlap算子接入指导.docx"
DOC = HERE / "FusedSparseAttentionOverlap算子接入指导.doc"
PDF = HERE / "preview.pdf"

word = win32.gencache.EnsureDispatch("Word.Application")
word.Visible = False
word.DisplayAlerts = 0

doc = word.Documents.Open(str(DOCX), ReadOnly=False, AddToRecentFiles=False)
try:
    doc.Fields.Update()
    if doc.TablesOfContents.Count > 0:
        toc = doc.TablesOfContents(1)
        toc.Update()
        for para in toc.Range.Paragraphs:
            para.Format.SpaceBefore = 0
            para.Format.SpaceAfter = 0
            para.Format.LineSpacingRule = 0
    doc.Repaginate()
    doc.SaveAs2(str(DOC), FileFormat=0)          # wdFormatDocument (Word 97-2003)
    doc.ExportAsFixedFormat(str(PDF), 17, False, 0, 0, 1, 0, False,
                            True, 0, False, True, False)  # wdExportFormatPDF
    print("pages:", doc.ComputeStatistics(2))    # wdStatisticPages
finally:
    doc.Close(SaveChanges=0)
    word.Quit()

print("doc :", DOC, DOC.stat().st_size, "bytes")
print("pdf :", PDF, PDF.stat().st_size, "bytes")
