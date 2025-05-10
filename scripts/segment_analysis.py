#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segment- und Subgruppenanalyse für Service-Chatbot-Umfrage.
Erstellt pro Segment (Alter A601, Geschlecht A602, Nutzung A202) gruppierte Balkendias mit einheitlicher Optik.
"""
import os
import sys
# sicherstellen, dass das Skript-Verzeichnis im Import-Pfad ist
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import plot_settings  # zentrale RC-Params und PGF-Backend


def read_csv_enc(path, sep=','):
    """
    Lese CSV mit verschiedenen Encodings.
    """
    for enc in ['utf-8-sig', 'utf-8', 'utf-16', 'ISO-8859-1']:
        try:
            return pd.read_csv(path, sep=sep, encoding=enc, engine='python')
        except Exception:
            continue
    raise IOError(f"Kann Datei {path} nicht einlesen.")


def normalize_label(opt):
    """ Entferne Gruppen-Präfix bis ': ' """
    return opt.split(': ', 1)[1] if isinstance(opt, str) and ': ' in opt else opt


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(base, 'data', 'data_chatbot_quality_2025-05-09.xlsx')
    vars_file = os.path.join(base, 'data', 'variables_chatbot_quality_2025-05-09.csv')
    values_file = os.path.join(base, 'data', 'values_chatbot_quality_2025-05-09.csv')
    out_dir = os.path.join(base, 'figures', 'segments')
    os.makedirs(out_dir, exist_ok=True)

    # Daten einlesen und User-Subset (A101==1)
    raw = pd.read_excel(data_file, engine='openpyxl')
    df = raw.iloc[1:].reset_index(drop=True)
    df['A101'] = pd.to_numeric(df['A101'], errors='coerce')
    df = df[df['A101'] == 1].copy()

    # Variablenlisten
    a301_vars = [f'A301_{i:02d}' for i in range(1, 9)]
    a305_vars = [f'A305_{i:02d}' for i in range(1, 9)]
    segments = ['A601', 'A602', 'A202']

    # Codebooks
    var_df = read_csv_enc(vars_file, sep='\t')
    val_df = read_csv_enc(values_file, sep='\t')

    # Mappings
    value_map = {}
    for _, row in val_df.iterrows():
        value_map.setdefault(row['VAR'], {})[str(row['RESPONSE'])] = row['MEANING']
    label_map = {row['VAR']: row['LABEL'] for _, row in var_df.iterrows()}

    # Kürzere Merkmal-Namen (Best-Practice-Begriffe)
    short_labels = {
        'Schnelligkeit der Antworten': 'Geschwindigkeit',
        'Genauigkeit der Antworten': 'Genauigkeit',
        'Fähigkeit, Probleme effektiv zu lösen': 'Problemlösung',
        'Natürliche, menschliche Interaktion': 'Menschlichkeit',
        'Einfache Bedienung (Auswahloptionen)': 'Bedienbarkeit',
        'Kontextverständnis (Chatverlauf wird genutzt)': 'Kontext',
        'Freundlichkeit / Tonalität der Antwort': 'Tonalität',
        'Datenschutz & Vertrauenswürdigkeit': 'Datenschutz'
    }

    # Fixe Reihenfolge der Merkmale
    label_order = [
        'Geschwindigkeit', 'Genauigkeit', 'Problemlösung', 'Menschlichkeit',
        'Bedienbarkeit', 'Kontext', 'Tonalität', 'Datenschutz'
    ]

    # Spezielle Reihenfolge für A202-Kategorien
    a202_order = [
        'Mehrmals pro Woche', 'Einmal pro Woche', 'Einmal pro Monat',
        'Seltener als einmal im Monat', 'Ich nutze sie nur, wenn ich keine Alternative finde'
    ]

    # Pro Segment
    for seg in segments:
        df[seg] = df[seg].astype(str)
        df[seg + '_lbl'] = df[seg].map(value_map.get(seg, {})).fillna(df[seg])
        records = []
        for cat in sorted(df[seg + '_lbl'].unique()):
            sub = df[df[seg + '_lbl'] == cat]
            n = len(sub)
            if n == 0:
                continue
            # Top-2-Box A301
            for v in a301_vars:
                vals = pd.to_numeric(sub[v], errors='coerce').dropna()
                pct = vals.isin([1, 2]).sum() / len(vals) * 100 if len(vals) > 0 else 0
                records.append({
                    'Kategorie': cat,
                    'Merkmal': normalize_label(label_map.get(v, v)),
                    'Top2_Wichtigkeit': round(pct, 2)
                })
            # Erfüllt A305
            for idx, v in enumerate(a305_vars, 1):
                vals = pd.to_numeric(sub[v], errors='coerce')
                cnt = (vals == 2).sum()
                pct = cnt / n * 100
                records.append({
                    'Kategorie': cat,
                    'Merkmal': normalize_label(label_map.get(f'A305_{idx:02d}', v)),
                    'Erfüllt': round(pct, 2)
                })

        res_df = pd.DataFrame(records)
        # Kürze Merkmalbezeichnungen und setze Reihenfolge
        res_df['Merkmal'] = res_df['Merkmal'].map(short_labels).fillna(res_df['Merkmal'])

        csv_path = os.path.join(out_dir, f'segment_{seg}.csv')
        res_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f'Segment {seg}: CSV geschrieben: {csv_path}')

        # Plot pro Metrik
        for metric in ['Top2_Wichtigkeit', 'Erfüllt']:
            df_met = res_df[['Merkmal', 'Kategorie', metric]].dropna()
            pivot = df_met.pivot(index='Merkmal', columns='Kategorie', values=metric).fillna(0)
            pivot = pivot.reindex(index=label_order)
            # A202: Spalten in definierter Reihenfolge
            if seg == 'A202':
                pivot = pivot.reindex(columns=a202_order)

            fig, ax = plt.subplots(figsize=(7, 4.5))
            pivot.plot(kind='bar', ax=ax, color=plt.get_cmap('Set2').colors)
            ax.yaxis.grid(True, linestyle='--', linewidth=0.5)
            ax.set_axisbelow(True)
            ax.set_xlabel('')
            ax.set_xticklabels(pivot.index, rotation=45, ha='right', fontsize=mpl.rcParams['axes.labelsize'])
            plt.ylabel('Prozent', labelpad=10)
            # Unterschiedliche Legende für A202: vertikal über dem Plot
            if seg == 'A202':
                ax.legend(
                    loc='lower center',
                    bbox_to_anchor=(0.5, 1.02),
                    ncol=1,
                    frameon=False
                )
            else:
                ax.legend(
                    loc='lower center',
                    bbox_to_anchor=(0.5, 1.02),
                    ncol=len(pivot.columns),
                    frameon=False
                )
            plt.tight_layout()

            out_pdf = os.path.join(out_dir, f'{seg}_{metric}.pdf')
            fig.savefig(out_pdf, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)
            print(f'Segment {seg} Plot {metric} gespeichert: {out_pdf}')


if __name__ == '__main__':
    main()
