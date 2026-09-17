# -*- coding: utf-8 -*-
"""
三维结构法 · 第三步 跑通灵魂代码
=====================================================================
目标文献：李志冰、杨光艺、冯永昌、景亮（2017）
          《Fama-French 五因子模型在中国股票市场的实证检验》
          《金融研究》2017 年第 6 期（总第 444 期），pp.191-206

【先读这段，否则会误用】
1) 本脚本用【合成数据】端到端跑通原文全链路：
   因子构建(2x3) -> 25 组合 -> 时序回归 -> GRS 检验 -> 截距项四指标 -> 面板 Chow 断点检验。
   合成数据数值【与原文不可直接比较】，只验证链路与口径是否自洽。
2) 脚本内置三组【原文没做的对照实验】，这三组对照本身即创新点候选：
   对照A：全样本 vs 剔除最小 30% 市值（壳价值污染，Liu-Stambaugh-Yuan 2019）
   对照B：价值因子构建口径赛马 B/M vs E/P（LSY 2019：中国由 E/P 主导）
   对照C：单一已知断点 Chow(2007-06) vs 未知断点扫描（Bai-Perron 思路）
3) 脚本末尾打印【原文真实数字对照表】与【Gap 清单】，每条均标注原文页码。

复用：把 make_panel() 换成真实面板（保持 date/code/size/bm/ep/op/inv/ret 列名），
      全链路无需改动。

运行：python ff5_3d_repro.py
"""
import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(201706)
np.set_printoptions(suppress=True)

MONTHS = pd.period_range("1994-07", "2015-08", freq="M")   # 原文样本：254 个月
T = len(MONTHS)
N_STOCK = 600
BREAK_TRUE = 156        # 2007-06，原文 Chow 检验所用断点
SHELL_N = int(N_STOCK * 0.30)


# =====================================================================
# 0. 合成 A 股特征面板
# =====================================================================
def make_panel():
    """生成 (月份 x 股票) 面板。刻意植入四条与原文结论对应的机制：
       - 股改前市场风险主导：RMW/CMA 载荷为 0（对应"股改前这两个因子冗余"）
       - 股改后盈利与投资被定价：RMW/CMA 载荷由 0 变为非零
       - 最小 30% 市值含"壳期权"：收益与经营基本面脱钩（LSY 2019 的核心批评）
       - 价值溢价通过 E/P 传导而非 B/M（LSY 2019 的赛马结论）"""
    codes = np.array([f"S{i:04d}" for i in range(N_STOCK)])
    logsize = RNG.normal(0, 1.3, N_STOCK)
    size = np.exp(logsize)
    order = np.argsort(size)
    is_shell = np.zeros(N_STOCK, dtype=bool)
    is_shell[order[:SHELL_N]] = True

    op = RNG.normal(0.10, 0.06, N_STOCK)
    inv = RNG.normal(0.12, 0.10, N_STOCK)
    bm = np.exp(RNG.normal(-1.0, 0.6, N_STOCK))
    # E/P 与 B/M 的相关性刻意做低（模拟 LSY 的赛马结论：E/P 主导、B/M 被吸收）；
    # 并让 E/P 与 OP 基本独立，避免价值因子退化成盈利因子
    ep = np.clip(0.045 + RNG.normal(0, 0.060, N_STOCK), -0.10, 0.40)

    # ---- 因子真实过程（月度，单位：小数）----
    mkt = np.concatenate([RNG.normal(0.012, 0.095, BREAK_TRUE + 1),
                          RNG.normal(0.005, 0.070, T - BREAK_TRUE - 1)])
    smb = RNG.normal(0.008, 0.032, T)
    hml = RNG.normal(0.002, 0.030, T)
    post_mask = np.arange(T) > BREAK_TRUE
    rmw = np.where(post_mask, RNG.normal(0.005, 0.018, T), RNG.normal(0.000, 0.020, T))
    cma = np.where(post_mask, RNG.normal(0.004, 0.016, T), RNG.normal(0.000, 0.017, T))
    mom = np.where(post_mask, RNG.normal(-0.006, 0.032, T), RNG.normal(0.000, 0.030, T))

    # ---- 个股载荷 ----
    ls_n = (logsize - logsize.min()) / (logsize.max() - logsize.min())
    b_smb = 0.90 - 0.55 * ls_n
    b_hml = 0.30 + 0.50 * (ep - ep.mean()) / ep.std()    # 价值载荷走 E/P，不走 B/M
    b_rmw = 0.40 + 0.60 * (op - op.mean()) / op.std()
    b_cma = -0.40 - 0.60 * (inv - inv.mean()) / inv.std()
    beta = 0.85 + RNG.normal(0, 0.25, N_STOCK)
    shell_shock = np.where(is_shell, 1.8, 1.0)

    tile = lambda a: np.tile(a, T)
    P = lambda a: np.repeat(a, N_STOCK)          # 因子序列：按 date 为主序 repeat
    f_mkt, f_smb, f_hml = P(mkt), P(smb), P(hml)
    f_rmw, f_cma, f_mom = P(rmw), P(cma), P(mom)
    post = P(post_mask.astype(float))

    eps = RNG.normal(0, 0.050, T * N_STOCK) * tile(shell_shock)
    # 载荷口径：股改前市场风险主导 —— 其余因子载荷压缩到 25%（对应原文
    # "股改前市场风险占据主导地位，盈利能力、投资风格及动量因子冗余"）
    PRE = 0.25
    gate = PRE + (1 - PRE) * post
    ret = (tile(beta) * f_mkt
           + tile(b_smb) * gate * f_smb
           + tile(b_hml) * gate * f_hml
           + tile(b_rmw) * post * f_rmw       # 盈利/投资载荷仅股改后被定价
           + tile(b_cma) * post * f_cma
           + 0.15 * f_mom + eps + 0.0015 * (1 - post))

    df = pd.DataFrame({
        "date": np.repeat(MONTHS.values, N_STOCK),
        "code": np.tile(codes, T),
        "size": tile(size), "bm": tile(bm), "ep": tile(ep),
        "op": tile(op), "inv": tile(inv),
        "ret": ret, "mkt_excess": f_mkt, "post": post,
    })
    df["date"] = pd.PeriodIndex(df["date"], freq="M")
    ti = df["date"].map({p: i for i, p in enumerate(MONTHS)}).values
    df["rf"] = 0.0021 + 0.0004 * np.sin(ti / 40.0)
    df["ret_ex"] = df["ret"] - df["rf"]
    return df


# =====================================================================
# 1. 横截面分组工具（按月度截面分位点）
# =====================================================================
def xrank(df, col):
    return df.groupby("date")[col].rank(pct=True)


def assign3(df, col, labels=("L", "N", "H"), lo=0.30, hi=0.70):
    """按 30%/70% 分位点三组。BM 用 (L,N,H)；OP 用 (W,N,R)；Inv 用 (A,N,C)。"""
    r = xrank(df, col)
    lo_l, mid, hi_l = labels
    return np.where(r <= lo, lo_l, np.where(r <= hi, mid, hi_l))


def assign2(df, col):
    return np.where(xrank(df, col) <= 0.5, "S", "B")


def assign5(df, col, labels):
    r = xrank(df, col)
    idx = np.clip((r * 5).apply(np.ceil).astype(int) - 1, 0, 4).values
    return np.asarray(labels)[idx]


def vw_return(df, port_col, ret_col="ret_ex", w_col="size"):
    """市值加权组合收益率（对应原文"以流通股本计算市值加权权重"口径）。"""
    num = df.assign(_wm=df[ret_col] * df[w_col]).groupby(["date", port_col], observed=True)[["_wm", w_col]].sum()
    return (num["_wm"] / num[w_col]).unstack(port_col).sort_index()


# =====================================================================
# 2. FF(2015a) 2x3 因子构建 + MOM 因子
# =====================================================================
def build_factors_2x3(df, value_var="bm"):
    """value_var 指定价值排序变量："bm"（原文口径）或 "ep"（LSY 口径）。"""
    d = df.copy()
    d["sz2"] = assign2(d, "size")
    d["val3"] = assign3(d, value_var, ("L", "N", "H"))
    d["op3"] = assign3(d, "op", ("W", "N", "R"))
    d["inv3"] = assign3(d, "inv", ("A", "N", "C"))
    d["p_val"] = d["sz2"] + d["val3"]
    d["p_op"] = d["sz2"] + d["op3"]
    d["p_inv"] = d["sz2"] + d["inv3"]

    P = {k: vw_return(d, k) for k in ["p_val", "p_op", "p_inv"]}
    col = lambda p, n: P[p][n]

    SMB = sum(
        (col(p, "S" + a) + col(p, "S" + b) + col(p, "S" + c)) / 3
        - (col(p, "B" + a) + col(p, "B" + b) + col(p, "B" + c)) / 3
        for p, (a, b, c) in [("p_val", ("H", "N", "L")),
                             ("p_op", ("R", "N", "W")),
                             ("p_inv", ("C", "N", "A"))]
    ) / 3

    HML = (col("p_val", "SH") + col("p_val", "BH")) / 2 - (col("p_val", "SL") + col("p_val", "BL")) / 2
    RMW = (col("p_op", "SR") + col("p_op", "BR")) / 2 - (col("p_op", "SW") + col("p_op", "BW")) / 2
    CMA = (col("p_inv", "SC") + col("p_inv", "BC")) / 2 - (col("p_inv", "SA") + col("p_inv", "BA")) / 2

    F = pd.DataFrame({"Mkt-RF": d.groupby("date")["mkt_excess"].first(),
                      "SMB": SMB, "HML": HML, "RMW": RMW, "CMA": CMA}).sort_index()

    # ---- MOM：按 FF(2016)，用 t-12 ~ t-2 累计历史收益率做 2x3 分组 ----
    R = df.pivot(index="date", columns="code", values="ret").sort_index()
    prior = np.expm1(np.log1p(R).rolling(11, min_periods=11).sum().shift(2))
    long = prior.stack().rename("prior").reset_index()
    sub = df.merge(long, on=["date", "code"], how="left").dropna(subset=["prior"]).copy()
    if len(sub):
        sub["sz2"] = assign2(sub, "size")
        sub["pr3"] = assign3(sub, "prior", ("L", "N", "H"))
        sub["p_pr"] = sub["sz2"] + sub["pr3"]
        Pp = vw_return(sub, "p_pr")
        F["MOM"] = ((Pp["SH"] + Pp["BH"]) / 2 - (Pp["SL"] + Pp["BL"]) / 2).reindex(F.index).fillna(0.0)
    else:
        F["MOM"] = 0.0
    return F.dropna()


# =====================================================================
# 3. 25 组合 + 时序回归 + GRS + 截距项四指标
# =====================================================================
def build_25(df, c2="op"):
    d = df.copy()
    d["sz5"] = assign5(d, "size", ["Small", "2", "3", "4", "Big"])
    d["c5"] = assign5(d, c2, ["Low", "2", "3", "4", "High"])
    d["port"] = d["sz5"].astype(str) + "-" + d["c5"].astype(str)
    return vw_return(d, "port").dropna(axis=1, how="any")


def ols(port_ret, factors):
    X = np.column_stack([np.ones(len(factors)), factors.values])
    alphas, resid = [], []
    for c in port_ret.columns:
        y = port_ret[c].values
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        alphas.append(b[0])
        resid.append(y - X @ b)
    return np.array(alphas), np.array(resid).T


def grs_stat(alphas, resid, factors):
    """标准 GRS：H0 = 所有截距项联合为 0。服从 F(N, T-N-L)。"""
    Tn, N = resid.shape
    L = factors.shape[1]
    a = alphas.reshape(-1, 1)
    sigma = resid.T @ resid / Tn
    fm = factors.values - factors.values.mean(axis=0)
    omega = fm.T @ fm / Tn
    mu = factors.values.mean(axis=0).reshape(-1, 1)
    num = (a.T @ np.linalg.pinv(sigma) @ a)[0, 0]
    den = 1 + (mu.T @ np.linalg.pinv(omega) @ mu)[0, 0]
    g = ((Tn - N - L) / N) * num / den
    return g, 1 - stats.f.cdf(g, N, Tn - N - L)


def evaluate(port_ret, F, models):
    """原文 4 个截距项指标（越小越好）：GRS、A|a|、A|a|/A|r|、Aa²/Ar²"""
    rows = []
    for name, cols in models.items():
        if not all(c in F.columns for c in cols):
            continue
        fa = F[cols]
        a, e = ols(port_ret, fa)
        g, p = grs_stat(a, e, fa)
        r = port_ret.mean(axis=0).values
        rd = r - r.mean()
        rows.append({"model": name, "GRS": g, "p值": p, "A|a|": np.mean(np.abs(a)),
                     "A|a|/A|r|": np.mean(np.abs(a)) / np.mean(np.abs(rd)),
                     "Aa²/Ar²": np.mean(a ** 2) / np.mean(rd ** 2)})
    return pd.DataFrame(rows)


# =====================================================================
# 4. 断点检验：面板 Chow -> Fisher 合并 -> 未知断点扫描 -> 顺序多重断点
# =====================================================================
def chow_reg(y, X, bp):
    """对含截距的回归做单一断点 Chow 检验：H0 = 两子样本回归系数相同。"""
    k = X.shape[1]
    n = len(y)

    def rss(yy, XX):
        b, *_ = np.linalg.lstsq(XX, yy, rcond=None)
        return float(((yy - XX @ b) ** 2).sum())

    r_all, r1, r2 = rss(y, X), rss(y[:bp], X[:bp]), rss(y[bp:], X[bp:])
    df2 = n - 2 * k
    if df2 <= 0 or (r1 + r2) <= 0:
        return np.nan, np.nan
    f = ((r_all - (r1 + r2)) / k) / ((r1 + r2) / df2)
    return f, 1 - stats.f.cdf(f, k, df2)


def panel_chow(port_ret, factors, bp):
    """原文 p.200 脚注的做法：对全部 25 个 panel 分别做 Chow，再 Fisher 合并 p 值。"""
    X = np.column_stack([np.ones(len(factors)), factors.values])
    ps, fs, nsig = [], [], 0
    for c in port_ret.columns:
        f, p = chow_reg(port_ret[c].values, X, bp)
        if np.isfinite(p):
            ps.append(p); fs.append(f)
            nsig += int(p < 0.01)
    ps = np.array(ps)
    fisher = -2 * np.log(np.clip(ps, 1e-300, 1)) .sum()
    return nsig, len(ps), float(np.mean(fs)), float(fisher), float(1 - stats.chi2.cdf(fisher, 2 * len(ps)))


def scan_unknown_break(port_ret, factors, min_frac=0.15, max_breaks=4):
    """未知断点顺序扫描（Bai-Perron 思路）：用面板平均 Chow F 作扫描统计量。"""
    n = len(factors)
    k = 1 + factors.shape[1]
    X = np.column_stack([np.ones(n), factors.values])
    Y = port_ret.values
    crit = stats.f.ppf(0.99, k, max(n - 2 * k, 1))

    def mean_f(lo, hi):
        cand = np.arange(lo + int((hi - lo) * min_frac), hi - int((hi - lo) * min_frac))
        if len(cand) == 0:
            return None
        best = (-np.inf, None)
        for c in cand:
            if c - lo <= k or hi - c <= k:
                continue
            fs = [chow_reg(Y[lo:hi, j], X[lo:hi], c - lo)[0] for j in range(Y.shape[1])]
            mm = float(np.nanmean(fs))
            if mm > best[0]:
                best = (mm, c)
        return best

    breaks, queue = [], [(0, n)]
    while queue and len(breaks) < max_breaks:
        lo, hi = queue.pop(0)
        r = mean_f(lo, hi)
        if r is None:
            continue
        m, c = r
        if m > crit:
            breaks.append((c, m))
            queue += [(lo, c), (c, hi)]
    return sorted(breaks), crit


def price_assets(targets, F, cols):
    """用模型 cols 去定价测试资产 targets，返回逐资产 alpha/t 与联合 GRS。"""
    fa = F[cols]
    a, e = ols(targets, fa)
    g, p = grs_stat(a, e, fa)
    n = len(fa)
    X = np.column_stack([np.ones(n), fa.values])
    ts = []
    for c in targets.columns:
        y = targets[c].values
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        s2 = ((y - X @ b) ** 2).sum() / (n - X.shape[1])
        se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * s2)
        ts.append(b[0] / se[0])
    return pd.DataFrame({"测试资产": list(targets.columns),
                         "alpha(%)": a * 100, "t值": ts}), g, p


# =====================================================================
# 5. 主流程
# =====================================================================
MODELS = {"CAPM": ["Mkt-RF"], "FF3": ["Mkt-RF", "SMB", "HML"],
          "FF3+MOM": ["Mkt-RF", "SMB", "HML", "MOM"],
          "FF5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]}
BAR = "=" * 80


def main():
    print(BAR)
    print("三维结构法 · 第三步｜FF5 在中国：链路复现 + 三组原文未做的对照实验")
    print(BAR)

    df = make_panel()
    print(f"[数据] 合成面板 {T} 个月 x {N_STOCK} 只 = {len(df):,} 行；"
          f"区间 {MONTHS[0]} ~ {MONTHS[-1]}（对齐原文 199407-201508）")

    # ---- 5.1 因子描述性统计 ----
    F = build_factors_2x3(df, "bm")
    print("\n【1】2x3 因子构建（全样本）。仅验证链路，数值不可与原文比较")
    print(pd.DataFrame({"Mean(%)": F.mean() * 100,
                        "t值": F.mean() / (F.std(ddof=1) / np.sqrt(len(F)))}).round(3).to_string())

    # ---- 5.2 模型赛马 ----
    P25 = build_25(df, "op")
    print(f"\n【2】Size-OP 25 组合，四指标模型赛马（n={P25.shape[1]}，四指标越小越好）")
    print(evaluate(P25, F, MODELS).round(3).to_string(index=False))

    # ---- 5.3 对照A ----
    print("\n" + "-" * 80)
    print("【对照A】壳价值污染：全样本 vs 剔除最小 30% 市值（LSY 2019）")
    print("-" * 80)
    thr = df.groupby("date")["size"].transform(lambda s: s.quantile(0.30))
    df_ex = df[df["size"] > thr].copy()
    F_ex, P25_ex = build_factors_2x3(df_ex, "bm"), build_25(df_ex, "op")
    print(evaluate(P25_ex, F_ex, MODELS).round(3).to_string(index=False))
    print("\n  观察点：比较两表 FF5 的 A|a|/A|r|。若剔除壳池后明显下降，说明原文"
          "\n          「全样本」结论中有相当部分由壳污染驱动，而非真实盈利/投资效应。")

    # ---- 5.4 对照B：价值因子构建口径赛马（复刻 LSY 的"模型能否定价对方的因子"设计）----
    print("\n" + "-" * 80)
    print("【对照B】价值因子构建口径：HML 由 B/M 建（原文） vs 由 E/P 建（LSY 2019）")
    print("-" * 80)
    F_ep = build_factors_2x3(df, "ep")
    tstat = lambda s: s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))
    print(f"  corr(HML_BM, HML_EP) = {F['HML'].corr(F_ep['HML']):.3f}")
    print(f"  HML_BM：均值 {F['HML'].mean()*100:6.3f}%  (t={tstat(F['HML']):5.2f})")
    print(f"  HML_EP：均值 {F_ep['HML'].mean()*100:6.3f}%  (t={tstat(F_ep['HML']):5.2f})")

    Fcmp = pd.DataFrame({"Mkt-RF": F["Mkt-RF"], "RMW": F["RMW"], "CMA": F["CMA"],
                         "SMB_BM": F["SMB"], "HML_BM": F["HML"],
                         "SMB_EP": F_ep["SMB"], "HML_EP": F_ep["HML"]})
    mdl_bm = ["Mkt-RF", "SMB_BM", "HML_BM", "RMW", "CMA"]   # 相当于 FF-3 的类比物
    mdl_ep = ["Mkt-RF", "SMB_EP", "HML_EP", "RMW", "CMA"]   # 相当于 CH-3 的类比物

    for mname, mcols, tgt_name, tgt_cols in [
        ("FF5(HML_BM)", mdl_bm, "E/P 模型自有因子 SMB_EP/HML_EP", ["SMB_EP", "HML_EP"]),
        ("FF5(HML_EP)", mdl_ep, "B/M 模型自有因子 SMB_BM/HML_BM", ["SMB_BM", "HML_BM"]),
    ]:
        tbl, g, p = price_assets(Fcmp[tgt_cols], Fcmp, mcols)
        print(f"\n  用 {mname} 定价「{tgt_name}」")
        print("  " + tbl.round(3).to_string(index=False).replace("\n", "\n  "))
        print(f"  联合 GRS = {g:.3f}   (p = {p:.3g})")

    # 再以标准 25 组合为测试资产做一次横向对照
    Fcmp2 = Fcmp.copy()
    for aname, P in [("Size-BM 组合", build_25(df, "bm")), ("Size-EP 组合", build_25(df, "ep"))]:
        r = pd.concat([
            evaluate(P, Fcmp2, {"FF5(HML_BM)": mdl_bm}),
            evaluate(P, Fcmp2, {"FF5(HML_EP)": mdl_ep})])
        print(f"\n  测试资产 = {aname}（四指标越小越好）")
        print("  " + r.round(3).to_string(index=False).replace("\n", "\n  "))
    print("\n  观察点：LSY(2019) 在真实数据上的结果是 —— B/M 模型留下约 17%/年的 alpha")
    print("          无法解释 E/P 价值因子，而 E/P 模型可以把对方的因子定价掉。")
    print("          合成数据的方向由植入机制决定，此处只为搭好可迁移的检验框架。")

    # ---- 5.5 对照C：断点 ----
    print("\n" + "-" * 80)
    print("【对照C】制度断点：已知单一 Chow(2007-06) vs 未知断点扫描（Bai-Perron 思路）")
    print("-" * 80)
    n1, ntot, mf, fish, fp = panel_chow(P25, F[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]], BREAK_TRUE)
    print(f"  已知断点 2007-06｜对 {ntot} 个 panel 分别做 Chow(截距+5 个因子系数)：")
    print(f"     1% 水平显著 panel 数 = {n1}/{ntot}；平均 Chow F = {mf:.3f}")
    print(f"     Fisher 合并统计量 = {fish:.2f}  (合并 p = {fp:.3g})")
    print("     对应原文 p.200 脚注：全部 25 个 panel 的邹至庄检验 p 值几乎全在 1% 显著")
    bps, crit = scan_unknown_break(P25, F[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]])
    print(f"\n  未知断点顺序扫描（1% 临界 F = {crit:.3f}，统计量为面板平均 Chow F）：")
    if bps:
        for c, m in bps:
            print(f"     估计断点 {MONTHS[c]}   平均 Chow F = {m:.3f}")
        hit = any(abs(c - BREAK_TRUE) <= 6 for c, _ in bps)
        print(f"     是否命中原文 2007-06 附近（±6 个月）：{'是' if hit else '否'}")
    else:
        print("     未识别出显著断点（统计量未超过临界值）")
    print("\n  观察点：真实制度冲击不止一次（2005 股改 / 2017 注册制）。")
    print("          单一 Chow -> 多重未知断点，可同时给出断点个数、位置与置信区间。")

    # ---- 5.6 原文真实数字对照表 ----
    print("\n" + BAR)
    print("【原文真实数字对照表】全部可溯源到页码 / 表号")
    print(BAR)
    orig = [
        ["表2 PanelA  Rm-Rf 均值", "1.38%  (t=1.92，10% 显著)", "p.196 表2"],
        ["表2 PanelA  SMB 均值", "0.86%  (t=2.70，1% 显著)", "p.196 表2"],
        ["表2 PanelA  RMW / CMA 均值", "-0.07 (t=-0.21) / -0.02 (t=-0.09)，均不显著", "p.196 表2"],
        ["表2 PanelB  美国同期 SMB / HML", "0.20 (t=0.99) / 0.18 (t=0.90)，均不显著", "p.196 表2"],
        ["表2 脚注：HML 显著性依赖构建方法", "2x3 下不显著；2x2 与 2x2x2x2 下 HML 均值显著为正", "p.196 脚注"],
        ["表3  风格效应：RMW 对 FF3 回归 alpha", "0.82%  (3.39) 显著", "p.196 表3"],
        ["表3  风格效应：CMA 对 FF3 回归 alpha", "0.41%  (2.52) 显著", "p.196 表3"],
        ["表3  风格效应：MOM 对 FF3 回归 alpha", "0.00  (-0.01) 不显著", "p.196 表3"],
        ["表3  冗余检验：CMA 对另外四因子", "0.05  (0.42) 不显著 → CMA 冗余", "p.196 表3"],
        ["表5  Size-OP，FF3 四指标", "GRS 2.23*** / A|a| 0.30 / 0.76 / 0.64", "p.199 表5"],
        ["表5  Size-OP，FF3+MOM 四指标", "GRS 2.35*** / A|a| 0.31 / 0.78 / 0.72", "p.199 表5"],
        ["表5  Size-OP，FF5 四指标", "GRS 1.55*  / A|a| 0.20 / 0.51 / 0.27", "p.199 表5"],
        ["表6  股改前 FF5（三种构建法）", "一阶 0.53 / 0.69 / 0.55；二阶 0.30 / 0.56 / 0.32", "p.201 表6"],
        ["表6  股改后 FF5（三种构建法）", "一阶 0.35 / 0.40 / 0.38；二阶 0.13 / 0.17 / 0.15", "p.201 表6"],
        ["表6  股改后 CAPM 反而变差", "一阶 1.53（股改前 0.65）→ 单因子模型失效", "p.201 表6"],
        ["表7  股改前冗余：RMW / CMA / MOM", "0.34(1.25) / -0.17(-0.99) / -0.04(-0.12) 均不显著", "p.202 表7"],
        ["表7  股改后冗余：RMW / CMA / MOM", "0.44***(2.75) / 0.23*(1.91) / -0.76**(-2.04)", "p.202 表7"],
        ["表8  股改后 MOM 截距（三种分组）", "-0.63** / -0.76** / -0.68** → 显著反转效应", "p.203 表8"],
        ["表8  FF5 vs FF5+MOM（size-prior）", "股改后 GRS 2.24*** / 2.00** → 加 MOM 提升", "p.203 表8"],
        ["全文自我设限（交棒句）", "FF5 与 FF6 在某些分组下仍无法通过 GRS，股改后尤为突出", "p.205 结论"],
    ]
    for a, b, c in orig:
        print(f"  {a:<36} | {b:<44} | {c}")

    # ---- 5.7 Gap 清单 ----
    print("\n" + BAR)
    print("【Gap 清单】论文做法 → 留出的口子 → 可写成创新点的改写方向")
    print(BAR)
    gaps = [
        ("市值口径用【流通市值】加权，Size 与 B/M 分母均为流通市值",
         "流通市值只覆盖部分股本；国有控股比例高时与总市值排序差异大，小盘股被系统性错分。"
         "LSY(2019) 用总市值构造",
         "做【流通市值 vs 总市值】双口径对照，量化 SMB 载荷错分对"
         "「股改后规模效应增强」这一结论的贡献度"),
        ("直接照搬 FF(2015a) 分组流程，采用【全体 A 股分位点】",
         "未剔除最小 30% 市值（壳价值污染）。借壳上市的壳期权价值进入小盘股收益，"
         "且 83% 的借壳标的来自最小 30%",
         "剔除最小 30% 或以「高壳概率」筛选（Li & Rao 2022）后重估，"
         "检验「CMA 冗余」结论是否翻转"),
        ("OP 用「营业利润 / 股东权益合计」替代 FF 的「营业利润 / 账面权益」",
         "原文以中美会计准则差异为由直接替换，但【未论证两种口径的等价性】",
         "构造两套 OP 并做等价性检验（相关系数 + 排序一致性 + 因子收益对比），"
         "量化口径替换带来的因子载荷漂移"),
        ("B/M 作为唯一价值变量，未与竞争性估值指标赛马",
         "LSY(2019) 的 Fama-MacBeth 赛马表明，中国市场 E/P 吸收 B/M 的全部价值效应；"
         "原文照搬美国结论选用 B/M",
         "以 E/P 重建 HML，做 B/M vs E/P 赛马，检验 HML 冗余性与"
         "「股改后市场趋于有效」结论的稳健性"),
        ("股改断点使用单一已知点 Chow 检验（2007-06）",
         "只有一次结构化检验，无法识别 2017 注册制等后续断点，也无法给出断点位置的置信区间",
         "单一 Chow → 多重未知断点（Bai-Perron 或顺序检验），输出断点个数、位置与置信区间"),
        ("MOM 因子按 FF(2016) 方向构建（高历史收益减低历史收益）",
         "文章已发现中国市场是【反转】效应，却仍沿用动量方向的因子；"
         "截距显著为负，正说明因子方向与实际效应相反",
         "显式构建反转方向因子，并检验换手率 / 情绪（PMO）能否解释股改后反转效应"),
        ("模型评估以月度频率、GRS + 截距项四指标为主",
         "未考虑交易成本；未做多重检验校正（同一数据集上数百个因子的检验）",
         "加入换手率与成本约束后的净 alpha 排序；"
         "用 Harvey-Liu(2020) 双重自举做多重检验校正"),
        ("作者在结论中交棒：FF5 与 FF6 在某些分组下仍无法通过 GRS",
         "未回答「是哪些分组、为什么失败、缺哪个因子」",
         "定位 GRS 失败的分组切片，反推缺失因子的经济学属性，"
         "构造候选新因子并做样本外验证"),
    ]
    for i, (a, b, c) in enumerate(gaps, 1):
        print(f"\n  Gap {i}")
        print(f"    论文做法 → {a}")
        print(f"    留出口子 → {b}")
        print(f"    改写方向 → {c}")

    print("\n" + BAR)
    print("【复用】把 make_panel() 替换为真实面板即可全链路复用：")
    print("       必需列 date / code / size / bm / ep / op / inv / ret；")
    print("       真实数据建议额外处理：剔 ST/*ST、剔上市 120 个交易日、剔负净资产。")
    print(BAR)


if __name__ == "__main__":
    main()
