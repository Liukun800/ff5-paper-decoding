# -*- coding: utf-8 -*-
"""从原文 PDF 裁出用于 PPT 的论文截图，并拼一张接触表便于一次性校验。"""
import os, pymupdf
from PIL import Image, ImageDraw, ImageFont

PDF = r"D:/xwechat_files/wxid_wrd26f1wyigw22_df24/msg/file/2026-09/Fama-French五因子模型在中国股票市场的实证检验_李志冰.pdf"
ROOT = r"C:/Users/liuko/WorkBuddy/2026-09-13-07-29-45/解析材料"
CROPS = os.path.join(ROOT, "crops")
os.makedirs(CROPS, exist_ok=True)

X0, X1 = 86.0, 512.0          # 正文栏宽度（含少量留白）

# (名称, 页码, y0, y1, 说明)
SPEC = [
    ("cover",    1, 148, 472, "P191 标题/作者/摘要"),
    ("intro",    1, 478, 600, "P191 引言首段"),
    ("auth",     1, 606, 716, "P191 作者简介+基金"),
    ("gap",      2, 174, 260, "P192 国内研究稀少"),
    ("split",    2, 259, 410, "P192 股权分置改革背景"),
    ("lit_early",3, 174, 342, "P193 早期文献分歧"),
    ("lit_2016", 3, 341, 539, "P193 2016撞车+本文切入"),
    ("data",     4, 341, 411, "P194 数据处理口径"),
    ("def",      4, 538, 660, "P194 Size/BM/OP/Inv定义"),
    ("foot1",    4, 672, 716, "P194 脚注1 分位点"),
    ("t1",       5, 314, 524, "P195 表1 因子构建"),
    ("t2",       6, 143, 316, "P196 表2 因子描述统计"),
    ("meanarg",  6, 316, 512, "P196 均值不可靠论证"),
    ("t3",       6, 504, 632, "P196 表3 风格/冗余检验"),
    ("t4",       7, 506, 702, "P197 表4 Size-Inv回归"),
    ("infer",    7, 421, 508, "P197 CMA冗余猜测"),
    ("t4b",      8, 124, 382, "P198 表4续表"),
    ("grs",      9, 193, 302, "P199 GRS与四指标"),
    ("t5",       9, 470, 622, "P199 表5 模型比较"),
    ("chow",    10, 318, 492, "P200 股改分界+Chow"),
    ("chowfn",  10, 674, 716, "P200 邹至庄检验脚注"),
    ("t6",      11, 124, 292, "P201 表6 市场效率"),
    ("t7",      12, 124, 296, "P202 表7 分时期冗余"),
    ("mech",    12, 478, 700, "P202 机制包装段"),
    ("t8",      13, 314, 500, "P203 表8 动量检验"),
    ("rev",     13, 504, 576, "P203 反转效应结论"),
    ("method",  14, 124, 212, "P204 方法论建议"),
    ("conc",    14, 389, 592, "P204 结论五条"),
    ("handoff", 15, 126, 262, "P205 自我设限(交棒)"),
    ("refs_cn", 15, 276, 560, "P205 参考文献(中文)"),
    ("refs_en", 15, 560, 716, "P205 参考文献(英文起)"),
    ("abstract",16, 96, 300, "P206 英文摘要"),
]

doc = pymupdf.open(PDF)
ZOOM = 300 / 72.0
mat = pymupdf.Matrix(ZOOM, ZOOM)
manifest = []
for name, pno, y0, y1, desc in SPEC:
    pg = doc[pno - 1]
    clip = pymupdf.Rect(X0, y0, X1, y1)
    pix = pg.get_pixmap(matrix=mat, clip=clip)
    path = os.path.join(CROPS, name + ".png")
    pix.save(path)
    manifest.append((name, pno, y0, y1, desc, pix.width, pix.height))
    print(f"{name:10s} p{pno:2d} y{y0:5.0f}-{y1:5.0f}  {pix.width}x{pix.height}  {desc}")

# ---------- 接触表 ----------
try:
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 15)
except Exception:
    font = ImageFont.load_default()

CELL_W = 460
LABEL_H = 24
cells = []
for name, pno, y0, y1, desc, w, h in manifest:
    im = Image.open(os.path.join(CROPS, name + ".png")).convert("RGB")
    nh = max(1, int(im.height * (CELL_W / im.width)))
    im = im.resize((CELL_W, nh), Image.LANCZOS)
    cells.append((im, f"{name}  p{pno}  {desc}"))

COLS = 3
rows = (len(cells) + COLS - 1) // COLS
col_h = [0] * COLS
for i, (im, _) in enumerate(cells):
    c = i % COLS
    col_h[c] += im.height + LABEL_H + 10
H = max(col_h)
sheet = Image.new("RGB", (COLS * (CELL_W + 12) + 12, H + 12), (240, 238, 232))
draw = ImageDraw.Draw(sheet)
ys = [6] * COLS
for i, (im, lab) in enumerate(cells):
    c = i % COLS
    x = 6 + c * (CELL_W + 12)
    y = ys[c]
    draw.rectangle([x, y, x + CELL_W, y + LABEL_H - 4], fill=(27, 42, 74))
    draw.text((x + 6, y + 3), lab, fill=(255, 255, 255), font=font)
    sheet.paste(im, (x, y + LABEL_H))
    ys[c] = y + LABEL_H + im.height + 10
sheet.save(os.path.join(CROPS, "_contact.png"))
print("\ncontact sheet:", sheet.size)

# ---------- 整页渲染（用于侧栏与页速览条） ----------
zfull = 150 / 72.0
mfull = pymupdf.Matrix(zfull, zfull)
for pno in range(1, doc.page_count + 1):
    pix = doc[pno - 1].get_pixmap(matrix=mfull)
    pix.save(os.path.join(CROPS, f"full_p{pno:02d}.png"))
print("full pages rendered:", doc.page_count, "->", f"{pix.width}x{pix.height}")
