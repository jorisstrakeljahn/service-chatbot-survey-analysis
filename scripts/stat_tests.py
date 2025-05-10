#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical Tests für Segment-Analysen der Service-Chatbot-Umfrage.
- Für A301 Top-2-Box: Mann-Whitney-U (2 Gruppen) oder Kruskal-Wallis (>2 Gruppen)
- Für A305 Erfüllt: Chi-Quadrat-Test und Cramér's V
Erstellt CSV mit Teststatistiken, p-Werten und Effektstärken.
"""
import os
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, kruskal, chi2_contingency
import matplotlib.pyplot as plt


def cramers_v(confusion_matrix):
    """Berechnet Cramér's V aus einer Kontingenztafel"""
    chi2 = chi2_contingency(confusion_matrix, correction=False)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / (min(k-1, r-1)))


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(base, 'data', 'data_chatbot_quality_2025-05-09.xlsx')
    out_dir = os.path.join(base, 'figures', 'stats')
    os.makedirs(out_dir, exist_ok=True)

    # Daten einlesen
    raw = pd.read_excel(data_file, engine='openpyxl')
    df = raw.iloc[1:].reset_index(drop=True)
    df['A101'] = pd.to_numeric(df['A101'], errors='coerce')
    users = df[df['A101'] == 1]

    # Variablen
    segments = ['A601', 'A602', 'A202']
    a301_vars = [f'A301_{i:02d}' for i in range(1,9)]
    a305_vars = [f'A305_{i:02d}' for i in range(1,9)]

    results = []
    # Loop über Segmente
    for seg in segments:
        users[seg] = users[seg].astype(str)
        cats = users[seg].dropna().unique()
        # A301 Tests
        for var in a301_vars:
            # Rangdaten
            data_by_cat = [pd.to_numeric(users[users[seg]==c][var], errors='coerce').dropna() for c in cats]
            # mindestens zwei Gruppen mit Werten
            valid = [d for d in data_by_cat if len(d)>0]
            if len(valid) < 2:
                continue
            if len(cats) == 2:
                # Mann-Whitney-U
                stat, p = mannwhitneyu(valid[0], valid[1], alternative='two-sided')
                # Effektgröße r = Z/sqrt(N)
                # Approx Z from U: Z = (U - n1*n2/2)/sqrt(n1*n2*(n1+n2+1)/12)
                n1, n2 = len(valid[0]), len(valid[1])
                u = stat
                mean_u = n1*n2/2
                sd_u = np.sqrt(n1*n2*(n1+n2+1)/12)
                z = (u - mean_u) / sd_u
                r = abs(z) / np.sqrt(n1+n2)
                results.append({'Segment': seg, 'Variable': var, 'Test': 'Mann-Whitney-U',
                                'Stat': round(stat,2), 'p-value': round(p,3), 'Effect_r': round(r,3)})
            else:
                # Kruskal-Wallis
                stat, p = kruskal(*valid)
                # Effektgröße eta^2: (H - k +1)/(n - k)
                k = len(valid)
                n = sum(len(d) for d in valid)
                eta2 = (stat - k + 1)/(n - k) if n>k else np.nan
                results.append({'Segment': seg, 'Variable': var, 'Test': 'Kruskal-Wallis',
                                'Stat': round(stat,2), 'p-value': round(p,3), 'Effect_eta2': round(eta2,3)})
        # A305 Tests
        for var in a305_vars:
            # Kreuztabelle 2xG
            tab = pd.crosstab(users[seg], pd.to_numeric(users[var], errors='coerce')==2)
            if tab.shape[0] < 2 or tab.shape[1] < 2:
                continue
            chi2, p, dof, exp = chi2_contingency(tab, correction=False)
            v = cramers_v(tab)
            results.append({'Segment': seg, 'Variable': var, 'Test': 'Chi-Quadrat',
                            'Stat': round(chi2,2), 'p-value': round(p,3), 'CramersV': round(v,3)})

    # Ergebnisse speichern
    res_df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, 'stat_tests_results.csv')
    res_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print('Statistische Tests gespeichert:', csv_path)

    # Zusatzplots: Effektgrößen-Barplot und p-Werte-Heatmap
    df_plot = res_df.copy()
    # Zusammenführen aller Effektgrößen in eine Spalte
    df_plot['EffectSize'] = df_plot.get('Effect_r').fillna(df_plot.get('Effect_eta2')).fillna(df_plot.get('CramersV'))

    # Barplot: Effektgrößen pro Variable und Segment
    pivot_eff = df_plot.pivot(index='Variable', columns='Segment', values='EffectSize').fillna(0)
    plt.figure(figsize=(12,6))
    pivot_eff.plot(kind='bar')
    plt.title('Effektgrößen (r, η² oder Cramér\'s V) nach Segment und Variable')
    plt.ylabel('Effektgröße')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    bar_png = os.path.join(out_dir, 'effect_sizes_barplot.png')
    plt.savefig(bar_png)
    plt.close()
    print('Barplot Effektgrößen gespeichert:', bar_png)

    # Heatmap: p-Werte nach Variable und Segment
    pivot_p = df_plot.pivot(index='Variable', columns='Segment', values='p-value')
    plt.figure(figsize=(10,6))
    im = plt.imshow(pivot_p, aspect='auto', cmap='viridis', interpolation='none')
    plt.colorbar(im, label='p-value')
    plt.xticks(range(len(pivot_p.columns)), pivot_p.columns, rotation=45, ha='right')
    plt.yticks(range(len(pivot_p.index)), pivot_p.index)
    plt.title('Heatmap p-Werte nach Segment und Variable')
    plt.tight_layout()
    heatmap_png = os.path.join(out_dir, 'pvalues_heatmap.png')
    plt.savefig(heatmap_png)
    plt.close()
    print('Heatmap p-Werte gespeichert:', heatmap_png)

if __name__ == '__main__':
    main()
