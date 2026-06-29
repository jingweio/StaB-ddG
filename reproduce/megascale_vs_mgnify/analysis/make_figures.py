#!/usr/bin/env python3
"""Generate Megascale-vs-MGnify comparison figures from the analysis arrays."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

D="/tmp/claude-224072"
OUT="/home/guoj0f/repos/StaB-ddG/.claude/worktrees/MGnify-replace/reproduce/megascale_vs_mgnify/figures"

mega_per_wt = np.load(f"{D}/mega_per_wt.npy")
mega_sat    = np.load(f"{D}/mega_sat.npy")
mega_dG     = np.load(f"{D}/mega_dG.npy")
mega_len    = np.load(f"{D}/mega_wt_len.npy")
mg_dG       = np.load(f"{D}/mgnify_dG.npy")
mg_len      = np.load(f"{D}/mgnify_len.npy")
mg_per      = np.load(f"{D}/mgnify_per_pdb.npy")
mg_indel    = np.load(f"{D}/mgnify_indel_per_base.npy")

MEGA_C="#E4572E"   # orange-red
MG_C="#C42D6B"     # magenta (paper palette)
plt.rcParams.update({"font.size":11,"axes.linewidth":0.8,
                     "savefig.dpi":150,"figure.dpi":110})

# ----------------------------------------------------------------------------
# Figure 1: Breadth vs Depth (the core trade-off)
# ----------------------------------------------------------------------------
fig,ax=plt.subplots(1,2,figsize=(11,4.4))

# (a) Breadth
labels=["WT domains","Seq. clusters\n(30% id)","Structural clusters\n(TM<0.5)","Total\nmeasurements"]
mega_vals=[479, 199, 1, 776298]          # struct clusters for Megascale not reported -> ~ small; shown as n/a marker
mg_vals  =[704424, 211020, 129303, 1859498]
x=np.arange(len(labels)); w=0.38
b1=ax[0].bar(x-w/2, mega_vals, w, color=MEGA_C, label="Megascale (Tsuboyama 2023)")
b2=ax[0].bar(x+w/2, mg_vals, w, color=MG_C, label="MGnify Stability (Cho 2026)")
ax[0].set_yscale("log"); ax[0].set_ylim(0.5,5e6)
ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=9)
ax[0].set_ylabel("count (log scale)")
ax[0].set_title("(a) Breadth: sequence & structural scope", fontsize=11, loc="left")
for rects,vals in [(b1,mega_vals),(b2,mg_vals)]:
    for r,v in zip(rects,vals):
        txt="n/r" if (v==1) else (f"{v:,}" if v<1000 else f"{v/1e3:.0f}k" if v<1e6 else f"{v/1e6:.2f}M")
        ax[0].text(r.get_x()+r.get_width()/2, r.get_height()*1.25, txt,
                   ha="center", va="bottom", fontsize=7.5, rotation=0)
ax[0].legend(fontsize=8.5, loc="upper left")
# fold-change annotations
for i,(a,b) in enumerate(zip(mega_vals,mg_vals)):
    if a>1:
        ax[0].text(i, 2.2e6, f"×{b/a:,.0f}", ha="center", fontsize=8, color="#444", style="italic")

# (b) Depth: variants measured per WT domain (log)
data=[np.clip(mega_per_wt,1,None), np.clip(mg_per,1,None)]
parts=ax[1].violinplot(data, positions=[0,1], widths=0.8, showextrema=False)
for pc,c in zip(parts['bodies'],[MEGA_C,MG_C]):
    pc.set_facecolor(c); pc.set_alpha(0.55); pc.set_edgecolor(c)
# overlay medians
med=[np.median(d) for d in data]
ax[1].scatter([0,1], med, color="k", zorder=5, s=30)
for i,m in enumerate(med):
    ax[1].text(i+0.07, m, f"median {m:.0f}" if m>=10 else f"median {m:.1f}", va="center", fontsize=9)
ax[1].set_yscale("log"); ax[1].set_ylim(0.7,2e4)
ax[1].set_xticks([0,1]); ax[1].set_xticklabels(["Megascale","MGnify"])
ax[1].set_ylabel("variants measured per WT domain (log)")
ax[1].set_title("(b) Depth: mutational scanning per domain", fontsize=11, loc="left")
ax[1].text(0,7000,f"site-sat. DMS\n~{mega_sat.mean()*100:.0f}% of 19·L\n(mean {mega_per_wt.mean():.0f}/domain)",
           ha="center",fontsize=8,color=MEGA_C)
ax[1].text(1,7000,f"breadth-first\nmean {mg_per.mean():.2f}/domain\n68% singletons",
           ha="center",fontsize=8,color=MG_C)
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_breadth_vs_depth.png", bbox_inches="tight")
print("saved fig1_breadth_vs_depth.png")

# ----------------------------------------------------------------------------
# Figure 2: ΔG and length distributions
# ----------------------------------------------------------------------------
fig,ax=plt.subplots(1,2,figsize=(11,4.2))
bins=np.linspace(-6,8,80)
ax[0].hist(mega_dG, bins=bins, density=True, color=MEGA_C, alpha=0.55, label=f"Megascale (n={len(mega_dG)/1e3:.0f}k)")
ax[0].hist(mg_dG,   bins=bins, density=True, color=MG_C,   alpha=0.55, label=f"MGnify (n={len(mg_dG)/1e6:.2f}M)")
ax[0].axvspan(-1,5,color="grey",alpha=0.10)
ax[0].set_xlabel("ΔG (kcal/mol)"); ax[0].set_ylabel("density")
ax[0].set_title("(a) Folding stability (ΔG) distribution", fontsize=11, loc="left")
ax[0].legend(fontsize=9)
ax[0].text(2,0.01,"resolved\nrange -1..5",fontsize=7,color="#555",ha="center")

lb=np.arange(28,84)
ax[1].hist(mega_len, bins=lb, density=True, color=MEGA_C, alpha=0.55, label="Megascale WT (32–74 aa)")
ax[1].hist(mg_len,   bins=lb, density=True, color=MG_C,   alpha=0.55, label="MGnify (60–80 aa)")
ax[1].set_xlabel("domain length (aa)"); ax[1].set_ylabel("density")
ax[1].set_title("(b) Domain length distribution", fontsize=11, loc="left")
ax[1].legend(fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_distributions.png", bbox_inches="tight")
print("saved fig2_distributions.png")

# ----------------------------------------------------------------------------
# Figure 3: Mutation-scanning DEPTH — the two flavours
# ----------------------------------------------------------------------------
fig,ax=plt.subplots(1,3,figsize=(14,4.2))

# (a) MGnify: how many measurements each WT domain receives (substitution depth)
hist={1:361635,2:6238,3:68715,4:96400}
xs=list(hist.keys()); ys=list(hist.values())
bars=ax[0].bar(xs, ys, color=MG_C, alpha=0.8, width=0.65)
ax[0].set_yscale("log"); ax[0].set_ylim(1e3,1e6)
ax[0].set_xticks(xs); ax[0].set_xlabel("measurements per WT domain\n(WT + its substitution mutants)")
ax[0].set_ylabel("number of WT domains (log)")
ax[0].set_title("(a) MGnify: substitution depth is shallow", fontsize=11, loc="left")
for b,y in zip(bars,ys):
    ax[0].text(b.get_x()+b.get_width()/2, y*1.15, f"{y:,}", ha="center", fontsize=8)
ax[0].text(2.5, 4e5, "67.9% of domains\nhave ONLY the WT\n(no mutant at all)", fontsize=9,
           color=MG_C, ha="center", va="top")

# (b) single-substitution site-saturation: Megascale vs MGnify
sat=[99.1, 0.075]
b=ax[1].bar([0,1], sat, color=[MEGA_C,MG_C], alpha=0.8, width=0.6)
ax[1].set_ylim(0,108); ax[1].set_xticks([0,1]); ax[1].set_xticklabels(["Megascale","MGnify"])
ax[1].set_ylabel("single-substitution site-saturation\n(% of 19·L measured per domain)")
ax[1].set_title("(b) Substitution saturation: 99% vs ~0%", fontsize=11, loc="left")
ax[1].text(0, 101, "99.1%", ha="center", fontsize=11, color=MEGA_C, fontweight="bold")
ax[1].text(1, 6,  "≈0.07%\n(~1 sub ÷ 19·L)", ha="center", fontsize=9, color=MG_C, fontweight="bold")
ax[1].text(0, 58, "every position,\n~18.9/19 subs\n(479 domains)", ha="center",
           fontsize=8.5, color="white", fontweight="bold")

# (c) MGnify indels ARE deep — but only on a tiny subset of domains
ax[2].hist(mg_indel, bins=np.arange(43,52), color="#7B2D8E", alpha=0.8, rwidth=0.9)
ax[2].set_xlabel("indel variants per scanned domain")
ax[2].set_ylabel("number of domains")
ax[2].set_title("(c) MGnify indels: deep, but narrow", fontsize=11, loc="left")
ax[2].text(0.5,0.95,f"only {len(mg_indel):,} domains got indels\n"
           f"(~0.9% of all), but each was\nnear-saturation scanned\n(~49 indels/domain)",
           transform=ax[2].transAxes, fontsize=9, va="top", ha="left", color="#7B2D8E")
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_mutation_depth.png", bbox_inches="tight")
print("saved fig3_mutation_depth.png")
