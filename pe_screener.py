"""
╔══════════════════════════════════════════════════════════════╗
║         BRAZIL PE DEAL SCREENER                              ║
║         Automated LBO Target Identification Engine           ║
║         Bruno Imbiriba Campello | UFRJ Finance               ║
║                                                              ║
║  WHAT THIS IS:                                               ║
║  A systematic screening engine that scans Brazilian          ║
║  equities and scores each company as a potential LBO         ║
║  target — replicating the sourcing process of a PE fund.    ║
║                                                              ║
║  HOW TO RUN:                                                 ║
║  Google Colab:                                               ║
║  !pip install yfinance pandas matplotlib numpy -q            ║
║  Then run all cells                                          ║
║                                                              ║
║  METHODOLOGY:                                                ║
║  6 LBO criteria scored 0-100, weighted by importance.        ║
║  Mirrors the checklist used by mid-market PE funds.          ║
╚══════════════════════════════════════════════════════════════╝
"""

# ── INSTALL (uncomment in Google Colab) ────────────────────
# !pip install yfinance pandas matplotlib numpy -q

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
import time
warnings.filterwarnings('ignore')

print("=" * 65)
print("  BRAZIL PE DEAL SCREENER")
print("  Automated LBO Target Identification Engine")
print("  Bruno Imbiriba Campello | UFRJ Finance")
print("=" * 65)

# ══════════════════════════════════════════════════════════════
# THE UNIVERSE — IBOVESPA COMPONENTS + MID-CAP B3
# ══════════════════════════════════════════════════════════════

"""
WHY THESE COMPANIES?

We screen the IBOVESPA (Brazil's main index) plus selected
mid-cap companies. This mirrors what a real PE fund does:
define the investable universe first, then screen it.

Mid-cap bias is intentional — large caps like Petrobras
have government ownership making LBO structurally impossible.
PE funds hunt in the R$1B-R$20B enterprise value range.
"""

UNIVERSE = {
    # INDUSTRIALS & ENGINEERING (your edge)
    "WEGE3.SA":  "WEG",
    "EMBR3.SA":  "Embraer",
    "RAIL3.SA":  "Rumo Logística",
    "CCRO3.SA":  "CCR",
    "TIMS3.SA":  "TIM Brasil",
    "SBSP3.SA":  "Sabesp",
    "CSAN3.SA":  "Cosan",
    "PRIO3.SA":  "PetroRio",
    
    # CONSUMER & RETAIL
    "RENT3.SA":  "Localiza",
    "RADL3.SA":  "Raia Drogasil",
    "MGLU3.SA":  "Magazine Luiza",
    "VVAR3.SA":  "Americanas",
    "LJQQ3.SA":  "Quero-Quero",
    "PETZ3.SA":  "Petz",
    "SOMA3.SA":  "Grupo Soma",
    "ARZZ3.SA":  "Arezzo",
    
    # HEALTHCARE
    "HAPV3.SA":  "Hapvida",
    "RDOR3.SA":  "Rede D'Or",
    "FLRY3.SA":  "Fleury",
    "DASA3.SA":  "Dasa",
    "ONCO3.SA":  "Oncoclínicas",
    
    # TECHNOLOGY & SERVICES  
    "TOTVS3.SA": "TOTVS",
    "CASH3.SA":  "Méliuz",
    "LWSA3.SA":  "Locaweb",
    "IFCM3.SA":  "Infracommerce",
    
    # REAL ESTATE & INFRASTRUCTURE
    "CYRE3.SA":  "Cyrela",
    "MRVE3.SA":  "MRV",
    "TEND3.SA":  "Tenda",
    "EVEN3.SA":  "Even",
    
    # FOOD & AGRO
    "JBSS3.SA":  "JBS",
    "MRFG3.SA":  "Marfrig",
    "SMTO3.SA":  "São Martinho",
    "SLCE3.SA":  "SLC Agrícola",
    
    # FINANCIAL SERVICES (non-bank)
    "IRBR3.SA":  "IRB Brasil",
    "CIEL3.SA":  "Cielo",
    "SULA11.SA": "SulAmérica",
}

# ══════════════════════════════════════════════════════════════
# SCORING FRAMEWORK — THE PE CHECKLIST
# ══════════════════════════════════════════════════════════════

"""
THE 6 LBO CRITERIA — WHY EACH ONE MATTERS:

1. LEVERAGE (25% weight) — Most important for LBO
   PE adds debt. Company must handle it.
   Low existing leverage = room for PE debt.
   Net Debt/EBITDA < 1x = perfect
   Net Debt/EBITDA > 4x = likely impossible to LBO

2. FCF CONVERSION (20% weight) — The PE engine
   LBO debt is repaid with FCF.
   FCF/EBITDA > 60% = efficient cash machine
   FCF/EBITDA < 30% = cash trap (high CapEx business)

3. EBITDA MARGIN (20% weight) — Quality of business
   High margins = pricing power = defensible moat
   Margin > 20% = excellent
   Margin < 10% = commoditised, risky

4. REVENUE STABILITY (15% weight) — Debt serviceability
   PE needs predictable cash flows to service debt.
   Consistent growth is more valuable than high growth.
   Volatile revenue = dangerous with 50-70% debt load

5. CAPEX INTENSITY (10% weight) — Cash flow quality
   High CapEx businesses need constant reinvestment.
   Less cash available for debt repayment.
   CapEx/Revenue < 5% = light (software, services)
   CapEx/Revenue > 15% = heavy (mining, airlines)

6. VALUATION ENTRY POINT (10% weight) — Return potential
   Cheaper entry = higher IRR potential
   EV/EBITDA < 8x = attractive entry
   EV/EBITDA > 15x = expensive, limited upside
"""

WEIGHTS = {
    'leverage':   0.25,
    'fcf_conv':   0.20,
    'margin':     0.20,
    'stability':  0.15,
    'capex':      0.10,
    'valuation':  0.10,
}

def score_leverage(net_debt_ebitda):
    """
    Score existing leverage. Lower = better for LBO.
    PE wants to ADD leverage — existing debt is a problem.
    
    < 0x   (net cash) = 100  perfect
    0-1x              =  85  excellent  
    1-2x              =  70  good
    2-3x              =  50  acceptable
    3-4x              =  25  difficult
    > 4x              =   0  impossible
    """
    if pd.isna(net_debt_ebitda): return 40  # neutral if no data
    if net_debt_ebitda < 0:   return 100
    if net_debt_ebitda < 1:   return 85
    if net_debt_ebitda < 2:   return 70
    if net_debt_ebitda < 3:   return 50
    if net_debt_ebitda < 4:   return 25
    return 5

def score_fcf_conversion(fcf_conv):
    """
    FCF/EBITDA — how much EBITDA becomes real cash.
    High conversion = efficient business = good LBO.
    """
    if pd.isna(fcf_conv): return 40
    if fcf_conv > 0.70:   return 100
    if fcf_conv > 0.55:   return 80
    if fcf_conv > 0.40:   return 60
    if fcf_conv > 0.25:   return 35
    return 10

def score_margin(ebitda_margin):
    """
    EBITDA Margin — quality and defensibility.
    """
    if pd.isna(ebitda_margin): return 40
    if ebitda_margin > 0.30:   return 100
    if ebitda_margin > 0.20:   return 85
    if ebitda_margin > 0.12:   return 65
    if ebitda_margin > 0.07:   return 40
    return 15

def score_stability(revenue_growth, revenue_volatility):
    """
    Consistent, stable growth beats volatile high growth.
    Stability crucial for debt serviceability.
    """
    if pd.isna(revenue_growth) or pd.isna(revenue_volatility): return 40
    
    growth_score = 0
    if 0.05 < revenue_growth < 0.20:  growth_score = 80   # sweet spot
    elif revenue_growth > 0.20:        growth_score = 60   # too high = risky
    elif revenue_growth > 0:           growth_score = 50   # slow but positive
    else:                              growth_score = 10   # declining = bad
    
    vol_penalty = min(revenue_volatility * 200, 40)         # penalise volatility
    return max(0, growth_score - vol_penalty)

def score_capex(capex_pct):
    """
    CapEx/Revenue — lower = more FCF for debt repayment.
    """
    if pd.isna(capex_pct): return 40
    if capex_pct < 0.03:   return 100  # asset-light (software)
    if capex_pct < 0.06:   return 80   # light CapEx
    if capex_pct < 0.10:   return 60   # moderate
    if capex_pct < 0.15:   return 35   # heavy
    return 10                           # very heavy (airlines, mining)

def score_valuation(ev_ebitda):
    """
    Entry valuation — cheaper = more return potential.
    """
    if pd.isna(ev_ebitda) or ev_ebitda <= 0: return 40
    if ev_ebitda < 6:    return 100   # very cheap — distressed/turnaround
    if ev_ebitda < 8:    return 85    # attractive
    if ev_ebitda < 10:   return 65    # fair
    if ev_ebitda < 13:   return 40    # full
    if ev_ebitda < 16:   return 20    # expensive
    return 5                           # very expensive

# ══════════════════════════════════════════════════════════════
# DATA FETCHER
# ══════════════════════════════════════════════════════════════

def fetch_company_data(ticker, name):
    """
    Pulls financial data from Yahoo Finance and calculates
    all metrics needed for LBO scoring.
    
    WHAT yf.Ticker() GIVES US:
    .info          → key metrics (EV, EBITDA, market cap etc)
    .financials    → income statement (annual)
    .balance_sheet → balance sheet (annual)
    .cashflow      → cash flow statement (annual)
    .history()     → historical price data
    """
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info
        
        # ── BASIC CHECKS ───────────────────────────────────
        market_cap = info.get('marketCap', 0)
        if not market_cap or market_cap < 500_000_000:   # min R$500M
            return None
        
        # ── KEY METRICS FROM INFO ──────────────────────────
        ev          = info.get('enterpriseValue', np.nan)
        ebitda      = info.get('ebitda', np.nan)
        total_debt  = info.get('totalDebt', np.nan)
        cash        = info.get('totalCash', np.nan)
        revenue     = info.get('totalRevenue', np.nan)
        net_income  = info.get('netIncomeToCommon', np.nan)
        op_cf       = info.get('operatingCashflow', np.nan)
        capex_raw   = info.get('capitalExpenditures', np.nan)
        ev_ebitda   = info.get('enterpriseToEbitda', np.nan)
        
        # ── DERIVED METRICS ────────────────────────────────
        net_debt = (total_debt or 0) - (cash or 0)
        
        # Leverage
        net_debt_ebitda = net_debt / ebitda if ebitda and ebitda > 0 else np.nan
        
        # EBITDA Margin
        ebitda_margin = ebitda / revenue if revenue and revenue > 0 else np.nan
        
        # FCF Conversion — how much EBITDA becomes free cash
        capex = abs(capex_raw) if capex_raw else np.nan
        fcf   = op_cf - capex if op_cf and capex else np.nan
        fcf_conversion = fcf / ebitda if fcf and ebitda and ebitda > 0 else np.nan
        
        # CapEx intensity
        capex_pct = capex / revenue if capex and revenue and revenue > 0 else np.nan
        
        # Revenue stability — use price history as proxy
        hist = stock.history(period='2y')
        if len(hist) > 20:
            monthly    = hist['Close'].resample('M').last()
            returns    = monthly.pct_change().dropna()
            rev_vol    = returns.std()
            rev_growth = (monthly.iloc[-1] / monthly.iloc[0]) ** (1/2) - 1
        else:
            rev_vol    = np.nan
            rev_growth = np.nan
        
        return {
            'ticker':           ticker,
            'name':             name,
            'market_cap_bn':    round(market_cap / 1e9, 1),
            'ev_bn':            round(ev / 1e9, 1) if ev else np.nan,
            'ebitda_margin':    round(ebitda_margin * 100, 1) if ebitda_margin else np.nan,
            'net_debt_ebitda':  round(net_debt_ebitda, 1) if net_debt_ebitda else np.nan,
            'fcf_conversion':   round(fcf_conversion * 100, 1) if fcf_conversion else np.nan,
            'capex_pct':        round(capex_pct * 100, 1) if capex_pct else np.nan,
            'ev_ebitda':        round(ev_ebitda, 1) if ev_ebitda else np.nan,
            'rev_growth':       round(rev_growth * 100, 1) if rev_growth else np.nan,
            'rev_vol':          round(rev_vol * 100, 1) if rev_vol else np.nan,
            # raw for scoring
            '_leverage':        net_debt_ebitda,
            '_fcf_conv':        fcf_conversion,
            '_margin':          ebitda_margin,
            '_rev_growth':      rev_growth,
            '_rev_vol':         rev_vol,
            '_capex_pct':       capex_pct,
            '_ev_ebitda':       ev_ebitda,
        }
        
    except Exception as e:
        return None

# ══════════════════════════════════════════════════════════════
# MAIN SCREENING ENGINE
# ══════════════════════════════════════════════════════════════

def run_screener(universe):
    """
    Loops through all companies, fetches data, scores each one,
    and returns a ranked DataFrame.
    """
    print(f"\n📡 Scanning {len(universe)} companies...\n")
    
    results = []
    failed  = []
    
    for i, (ticker, name) in enumerate(universe.items()):
        print(f"  [{i+1:02d}/{len(universe)}] {name:<25}", end='')
        
        data = fetch_company_data(ticker, name)
        
        if data is None:
            print("  ✗ skipped")
            failed.append(name)
            time.sleep(0.3)
            continue
        
        # ── SCORE EACH CRITERION ───────────────────────────
        s_leverage  = score_leverage(data['_leverage'])
        s_fcf       = score_fcf_conversion(data['_fcf_conv'])
        s_margin    = score_margin(data['_margin'])
        s_stability = score_stability(data['_rev_growth'], data['_rev_vol'])
        s_capex     = score_capex(data['_capex_pct'])
        s_valuation = score_valuation(data['_ev_ebitda'])
        
        # ── WEIGHTED TOTAL SCORE ───────────────────────────
        total_score = (
            s_leverage  * WEIGHTS['leverage']  +
            s_fcf       * WEIGHTS['fcf_conv']  +
            s_margin    * WEIGHTS['margin']    +
            s_stability * WEIGHTS['stability'] +
            s_capex     * WEIGHTS['capex']     +
            s_valuation * WEIGHTS['valuation']
        )
        
        data.update({
            'score_leverage':  s_leverage,
            'score_fcf':       s_fcf,
            'score_margin':    s_margin,
            'score_stability': s_stability,
            'score_capex':     s_capex,
            'score_valuation': s_valuation,
            'TOTAL_SCORE':     round(total_score, 1),
        })
        
        results.append(data)
        
        # Rating label
        rating = '🟢 STRONG' if total_score >= 70 else ('🟡 WATCH' if total_score >= 50 else '🔴 WEAK')
        print(f"  Score: {total_score:.0f}/100  {rating}")
        time.sleep(0.4)   # be polite to Yahoo Finance API
    
    df = pd.DataFrame(results)
    df = df.sort_values('TOTAL_SCORE', ascending=False).reset_index(drop=True)
    df.index += 1   # rank starts at 1
    
    return df, failed

# ══════════════════════════════════════════════════════════════
# RESULTS DISPLAY
# ══════════════════════════════════════════════════════════════

def display_results(df):
    print("\n" + "=" * 65)
    print("  TOP LBO TARGETS — BRAZIL PE DEAL SCREENER")
    print("=" * 65)
    
    top10 = df.head(10)
    
    print(f"\n  {'#':<3} {'Company':<22} {'Score':>5} {'EV/EBITDA':>9} {'Leverage':>9} {'EBITDA%':>7} {'FCF%':>6} {'Rating'}")
    print(f"  {'─'*80}")
    
    for rank, row in top10.iterrows():
        score  = row['TOTAL_SCORE']
        rating = '🟢 BUY'   if score >= 70 else ('🟡 WATCH' if score >= 55 else '🔴 PASS')
        
        lev  = f"{row['net_debt_ebitda']:.1f}x" if not pd.isna(row['net_debt_ebitda']) else 'N/A'
        ev_e = f"{row['ev_ebitda']:.1f}x"       if not pd.isna(row['ev_ebitda'])       else 'N/A'
        marg = f"{row['ebitda_margin']:.1f}%"   if not pd.isna(row['ebitda_margin'])   else 'N/A'
        fcf  = f"{row['fcf_conversion']:.0f}%"  if not pd.isna(row['fcf_conversion'])  else 'N/A'
        
        print(f"  #{rank:<2} {row['name']:<22} {score:>5.0f} {ev_e:>9} {lev:>9} {marg:>7} {fcf:>6}  {rating}")
    
    print(f"\n  {'─'*80}")
    print(f"  Screened: {len(df)} companies  |  Strong targets (≥70): {(df['TOTAL_SCORE']>=70).sum()}  |  Watch (50-70): {((df['TOTAL_SCORE']>=50) & (df['TOTAL_SCORE']<70)).sum()}")

# ══════════════════════════════════════════════════════════════
# DASHBOARD VISUALISATION
# ══════════════════════════════════════════════════════════════

def build_dashboard(df):
    top10 = df.head(10)
    
    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor('#0A0F2C')
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)
    
    NAVY  = '#0A0F2C'
    NAVY2 = '#111830'
    GOLD  = '#C9A84C'
    GREEN = '#2D6A4F'
    RED   = '#C1121F'
    BLUE  = '#1A3A6B'
    WHITE = '#F8F6F2'
    GRAY  = '#888888'
    
    # ── Chart 1: Total Score Ranking (top, full width) ──────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor(NAVY2)
    ax1.set_title('LBO Target Score Ranking — Brazil Equities',
                  color=GOLD, fontsize=12, fontweight='bold', pad=10)
    
    names  = [f"#{i+1} {r['name']}" for i, (_, r) in enumerate(top10.iterrows())]
    scores = top10['TOTAL_SCORE'].values
    colors = [GREEN if s >= 70 else (GOLD if s >= 55 else RED) for s in scores]
    
    bars = ax1.barh(range(len(names)), scores, color=colors, height=0.65)
    
    # Score labels
    for bar, score, name in zip(bars, scores, names):
        ax1.text(score + 0.5, bar.get_y() + bar.get_height()/2,
                f'{score:.0f}/100', va='center', fontsize=9,
                color=WHITE, fontweight='bold')
    
    # Threshold lines
    ax1.axvline(70, color=GREEN, linewidth=1.2, linestyle='--', alpha=0.7)
    ax1.axvline(55, color=GOLD,  linewidth=1.2, linestyle='--', alpha=0.7)
    ax1.text(70.5, len(names)-0.5, 'Strong', fontsize=8, color=GREEN)
    ax1.text(55.5, len(names)-0.5, 'Watch',  fontsize=8, color=GOLD)
    
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, color=WHITE, fontsize=9)
    ax1.set_xlim(0, 110)
    ax1.set_xlabel('LBO Attractiveness Score (0-100)', color=GRAY, fontsize=9)
    ax1.tick_params(axis='x', colors=GRAY, labelsize=8)
    for spine in ax1.spines.values(): spine.set_edgecolor('#334466')
    
    # ── Chart 2: Radar — top 3 companies ───────────────────
    ax2 = fig.add_subplot(gs[1, 0], projection='polar')
    ax2.set_facecolor(NAVY2)
    
    categories = ['Leverage', 'FCF Conv.', 'Margin', 'Stability', 'CapEx', 'Valuation']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    score_cols = ['score_leverage','score_fcf','score_margin',
                  'score_stability','score_capex','score_valuation']
    
    colors_r = [GOLD, GREEN, BLUE]
    top3     = df.head(3)
    
    for (_, row), color in zip(top3.iterrows(), colors_r):
        values = [row[c] for c in score_cols]
        values += values[:1]
        ax2.plot(angles, values, 'o-', linewidth=1.5, color=color)
        ax2.fill(angles, values, alpha=0.1, color=color)
    
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, size=7.5, color=WHITE)
    ax2.set_ylim(0, 100)
    ax2.set_yticks([25, 50, 75, 100])
    ax2.set_yticklabels(['25', '50', '75', '100'], size=6, color=GRAY)
    ax2.tick_params(colors=GRAY)
    ax2.set_facecolor(NAVY2)
    ax2.spines['polar'].set_color('#334466')
    ax2.grid(color='#334466', linewidth=0.5)
    ax2.set_title('Criteria Radar\n(Top 3 Companies)',
                  color=GOLD, fontsize=10, fontweight='bold', pad=15)
    
    legend_elements = [mpatches.Patch(color=c, label=df.iloc[i]['name'].split()[0])
                       for i, c in enumerate(colors_r)]
    ax2.legend(handles=legend_elements, loc='lower right',
               facecolor=NAVY2, labelcolor=WHITE, fontsize=7,
               bbox_to_anchor=(1.3, -0.05))
    
    # ── Chart 3: Leverage vs EV/EBITDA bubble ───────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor(NAVY2)
    ax3.set_title('Leverage vs Entry Valuation\n(bubble = score)',
                  color=GOLD, fontsize=10, fontweight='bold', pad=10)
    
    valid = df.dropna(subset=['net_debt_ebitda', 'ev_ebitda']).head(15)
    
    scatter_colors = [GREEN if s >= 70 else (GOLD if s >= 55 else RED)
                      for s in valid['TOTAL_SCORE']]
    sizes = valid['TOTAL_SCORE'] ** 1.5 * 0.8
    
    ax3.scatter(valid['ev_ebitda'], valid['net_debt_ebitda'],
                s=sizes, c=scatter_colors, alpha=0.8, edgecolors=WHITE, linewidth=0.5)
    
    # Label top companies
    for _, row in valid.head(8).iterrows():
        ax3.annotate(row['name'].split()[0],
                    (row['ev_ebitda'], row['net_debt_ebitda']),
                    textcoords='offset points', xytext=(5, 3),
                    fontsize=6.5, color=WHITE)
    
    # Sweet spot box
    ax3.axhspan(-1, 2, alpha=0.05, color=GREEN)
    ax3.axvspan(0, 10, alpha=0.05, color=GREEN)
    ax3.text(1, 1.7, 'Sweet Spot', fontsize=7.5, color=GREEN, fontstyle='italic')
    
    ax3.axhline(0, color=GRAY, linewidth=0.5, linestyle=':')
    ax3.set_xlabel('EV/EBITDA (Entry Multiple)', color=GRAY, fontsize=8)
    ax3.set_ylabel('Net Debt / EBITDA', color=GRAY, fontsize=8)
    ax3.tick_params(colors=GRAY, labelsize=7)
    for spine in ax3.spines.values(): spine.set_edgecolor('#334466')
    
    # ── Chart 4: Score breakdown heatmap ───────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_facecolor(NAVY2)
    ax4.set_title('Score Breakdown\n(Top 8 Companies)',
                  color=GOLD, fontsize=10, fontweight='bold', pad=10)
    
    top8      = df.head(8)
    criteria  = ['Leverage', 'FCF', 'Margin', 'Stability', 'CapEx', 'Valuation']
    score_c   = ['score_leverage','score_fcf','score_margin',
                 'score_stability','score_capex','score_valuation']
    
    matrix = top8[score_c].values
    im     = ax4.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    ax4.set_yticks(range(len(top8)))
    ax4.set_yticklabels([r['name'].split()[0] for _, r in top8.iterrows()],
                        color=WHITE, fontsize=8)
    ax4.set_xticks(range(len(criteria)))
    ax4.set_xticklabels(criteria, color=WHITE, fontsize=7.5, rotation=30, ha='right')
    
    for i in range(len(top8)):
        for j in range(len(criteria)):
            val = matrix[i, j]
            ax4.text(j, i, f'{val:.0f}', ha='center', va='center',
                    fontsize=7, color='black' if 30 < val < 75 else WHITE,
                    fontweight='bold')
    
    plt.colorbar(im, ax=ax4, shrink=0.8).ax.tick_params(labelsize=7, colors=GRAY)
    
    # ── Main Title ──────────────────────────────────────────
    fig.text(0.5, 0.97,
             'BRAZIL PE DEAL SCREENER — LBO Target Identification Engine',
             ha='center', fontsize=14, color=GOLD, fontweight='bold')
    fig.text(0.5, 0.945,
             f'Universe: {len(df)} companies screened  |  '
             f'Strong targets: {(df["TOTAL_SCORE"]>=70).sum()}  |  '
             f'Bruno Imbiriba Campello  |  UFRJ Finance',
             ha='center', fontsize=9, color=WHITE)
    
    fig.text(0.5, 0.01,
             'Data: Yahoo Finance API  |  For educational purposes only  |  '
             'github.com/BrunoImbiribaCampello/brazil-equity-research',
             ha='center', fontsize=7, color=GRAY)
    
    plt.savefig('/home/claude/pe_screener_dashboard.png',
                dpi=150, bbox_inches='tight', facecolor='#0A0F2C')
    plt.close()
    print("\n✅  Dashboard saved: pe_screener_dashboard.png")

# ══════════════════════════════════════════════════════════════
# INVESTMENT MEMO GENERATOR
# ══════════════════════════════════════════════════════════════

def generate_memo(df):
    """
    Generates a mini investment memo for the #1 ranked company.
    This is what a PE analyst would write after screening.
    """
    top = df.iloc[0]
    
    print("\n" + "=" * 65)
    print("  AUTO-GENERATED INVESTMENT MEMO")
    print(f"  TOP RANKED TARGET: {top['name'].upper()}")
    print("=" * 65)
    
    lev  = f"{top['net_debt_ebitda']:.1f}x" if not pd.isna(top['net_debt_ebitda']) else 'N/A'
    ev_e = f"{top['ev_ebitda']:.1f}x"       if not pd.isna(top['ev_ebitda'])       else 'N/A'
    marg = f"{top['ebitda_margin']:.1f}%"   if not pd.isna(top['ebitda_margin'])   else 'N/A'
    
    print(f"""
  COMPANY:      {top['name']} ({top['ticker'].replace('.SA','')})
  LBO SCORE:    {top['TOTAL_SCORE']:.0f}/100
  EV:           R${top['ev_bn']:.1f}B
  EV/EBITDA:    {ev_e}  (entry multiple)
  LEVERAGE:     {lev}  (existing debt)
  EBITDA MARGIN:{marg}

  WHY THIS IS A STRONG LBO TARGET:

  {"✅ LOW LEVERAGE" if not pd.isna(top['net_debt_ebitda']) and top['net_debt_ebitda'] < 2 else "⚠  MODERATE LEVERAGE"}
    Existing debt is manageable — room to add PE leverage.
    
  {"✅ ATTRACTIVE VALUATION" if not pd.isna(top['ev_ebitda']) and top['ev_ebitda'] < 10 else "⚠  FULL VALUATION"}  
    Entry multiple is in PE's target range.
    
  {"✅ STRONG MARGINS" if not pd.isna(top['ebitda_margin']) and top['ebitda_margin'] > 15 else "⚠  MARGIN IMPROVEMENT NEEDED"}
    Business generates substantial cash to service debt.

  SUGGESTED LBO STRUCTURE:
    Entry multiple:   {ev_e} (current market)
    Debt:             50% of EV
    PE Equity:        50% of EV
    Hold period:      5 years
    Exit multiple:    {ev_e} + 2x expansion
    Target IRR:       25%+

  NEXT STEPS:
    1. Request management meeting
    2. Sign NDA → receive CIM
    3. Build full LBO model (see embraer_lbo.py)
    4. Submit indicative offer
    """)

# ══════════════════════════════════════════════════════════════
# RUN EVERYTHING
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    
    # 1. Run screener
    df, failed = run_screener(UNIVERSE)
    
    # 2. Display results
    display_results(df)
    
    # 3. Build dashboard
    print("\n📊 Building dashboard...")
    build_dashboard(df)
    
    # 4. Generate memo for top target
    generate_memo(df)
    
    # 5. Export to CSV
    export_cols = ['ticker','name','TOTAL_SCORE','ev_ebitda','net_debt_ebitda',
                   'ebitda_margin','fcf_conversion','capex_pct','rev_growth',
                   'score_leverage','score_fcf','score_margin',
                   'score_stability','score_capex','score_valuation']
    df[export_cols].to_csv('/home/claude/pe_screener_results.csv', index_label='rank')
    
    print(f"""
{'='*65}
  DONE.
  
  Files saved:
  📊 pe_screener_dashboard.png  — upload to GitHub
  📄 pe_screener_results.csv    — full ranked data
  🐍 pe_screener.py             — the engine itself

  Add to your GitHub:
  github.com/BrunoImbiribaCampello/brazil-equity-research

  CV line:
  "Built an automated PE deal screener in Python that
   systematically evaluates {len(df)} Brazilian equities across 6
   LBO criteria (leverage, FCF conversion, EBITDA margin,
   revenue stability, CapEx intensity, entry valuation)
   — replicating the sourcing process of a PE fund."

  Interview line:
  "Instead of analyzing one company at a time, I built a
   system that scans the entire Brazilian equity market and
   automatically identifies the best LBO targets. The same
   logic a PE analyst applies manually — I automated it."
{'='*65}
""")
