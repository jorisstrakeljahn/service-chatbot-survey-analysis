#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deskriptive Statistiken und PDF-Scalable Plots für die Nutzerumfrage zu Service-Chatbots.
Spezialmethode für A501 (Hauptgründe für Nichtnutzung) mit horizontalem Balkendiagramm,
Kombinierung von "Andere"-Kategorien, Prozentannotationen und Legende oberhalb.
Andere Mehrfachauswahlfragen nutzen generische Methode.
A301-Fragen werden nicht ausgewertet.
"""
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import plot_settings  # zentrale RC-Params und PGF-Backend laden

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

#-----------------------------------------------------------------------------------------

def read_codebook(path, sep='\t'):
    encodings = ['utf-8-sig', 'utf-16-le', 'utf-16', 'ISO-8859-1']
    for enc in encodings:
        try:
            return pd.read_csv(path, sep=sep, encoding=enc, engine='python')
        except Exception:
            continue
    raise IOError(f"Konnte Codebook {os.path.basename(path)} nicht einlesen.")

#-----------------------------------------------------------------------------------------

def normalize_label(label):
    if isinstance(label, str) and ': ' in label:
        return label.split(': ', 1)[1]
    return label

#-----------------------------------------------------------------------------------------

def plot_vertical_bar(categories, counts, percents, out_path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    # Vertikales Säulendiagramm mit numerischer x-Achse
    indices = list(range(1, len(categories) + 1))
    bars = ax.bar(indices, counts, color=plt.get_cmap('Set2').colors)

    # Achsen und Gitter
    ax.set_xticks(indices)
    ax.set_xticklabels(indices, fontsize=mpl.rcParams['xtick.labelsize'])
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_ylabel('Anzahl der Nennungen', labelpad=10)

    # Platz für Prozent-Annotationen schaffen
    # Bestimme max_val sicher aus counts
    if len(counts) > 0:
        max_val = counts.max()
    else:
        max_val = 1
    ax.set_ylim(0, max_val * 1.15)

    # Prozent-Annotations über jeder Säule
    for idx, (cnt, pct) in zip(indices, zip(counts, percents)):
        ax.text(idx, cnt + max_val * 0.02, f'{pct:.1f}', ha='center', va='bottom', fontsize=mpl.rcParams['xtick.labelsize'])

    # Legende oberhalb mit Nummern und Beschreibung
    legend_labels = [f'{i}: {cat}' for i, cat in zip(indices, categories)]
    ax.legend(
        bars,
        legend_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=1,
        frameon=False
    )

    # Layout und Export
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    basename = os.path.basename(out_path).replace('_group.pdf','')
    print(f"{basename}-Diagramm gespeichert: {out_path}")

#-----------------------------------------------------------------------------------------

def plot_pie_chart(categories, counts, percents, out_path):
    # gleiche Farbpalette wie bei plot_bar/-vertical_bar
    colors = plt.get_cmap('Set2').colors[:len(categories)]

    fig, ax = plt.subplots(figsize=(4.5, 4))
    total = counts.sum()

    def autopct_format(pct):
        absolute = int(np.round(pct / 100. * total))
        return f"{pct:.1f}%\n({absolute})"

    # Kreis etwas kleiner (radius), Labels über Legende
    wedges, texts, autotexts = ax.pie(
        counts,
        radius=0.2,
        labels=None,             # wir holen uns die Beschriftung über die Legende
        autopct=autopct_format,
        startangle=90,
        counterclock=False,
        colors=colors
    )

    # Prozent-Texte etwas größer
    for t in autotexts:
        t.set_fontsize(mpl.rcParams['xtick.labelsize'] + 2)

    ax.axis('equal')  # Kreisform

    # Legende rechts außen
    ax.legend(
        wedges,
        categories,
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        frameon=False
    )

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    basename = os.path.basename(out_path).replace('_distribution.pdf', '')
    print(f"{basename}-Kreisdiagramm gespeichert: {out_path}")


#-----------------------------------------------------------------------------------------

def describe_A501(df, variables, out_dir):
    df_sub = df[df['A101'] == 2]
    n = len(df_sub)
    if n == 0:
        print("Keine Nicht-Nutzer (A501) vorhanden.")
        return
    subvars = [v for v in variables['VAR'] if v.startswith('A501_')]
    labels, counts = [], []
    for var in subvars:
        cnt = (pd.to_numeric(df_sub[var], errors='coerce') == 2).sum()
        raw_lbl = variables.loc[variables['VAR'] == var, 'LABEL'].iloc[0]
        core = normalize_label(raw_lbl)
        label = 'Andere' if 'Andere' in core else short_labels.get(core, core)
        labels.append(label)
        counts.append(cnt)
    df_tmp = pd.DataFrame({'Option': labels, 'Häufigkeit': counts})
    df_grouped = df_tmp.groupby('Option', as_index=False).sum()
    df_grouped['Prozent'] = (df_grouped['Häufigkeit'] / df_grouped['Häufigkeit'].sum() * 100).round(1)

    # Reihenfolge der Kategorien festlegen
    desired_order = [
        'Ich bevorzuge persönlichen Kontakt (z.B. Hotline, E-Mail)',
        'Ich finde Service-Chatbots zu kompliziert oder unpraktisch',
        'Ich traue Service-Chatbots nicht zu, mir wirklich helfen zu können',
        'Ich habe bisher keine Service-Chatbots auf Webseiten gesehen',
        'Datenschutzbedenken oder mangelndes Vertrauen',
        'Ich habe negative Erfahrungen selber gemacht',
        'Ich habe negative Erfahrungen von anderen gehört',
        'Ich wusste nicht, dass Service-Chatbots hilfreich sein können',
        'Andere'
    ]
    # Reindex DataFrame nach gewünschter Reihenfolge
    df_grouped = df_grouped.set_index('Option').reindex(desired_order).reset_index()

    # CSV export
    csv_path = os.path.join(out_dir, 'A501_group.csv')
    df_grouped.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"Deskriptives A501 exportiert: {csv_path} (n={n})")

    # Vertikales Säulendiagramm mit geänderter Reihenfolge
    pdf_path = os.path.join(out_dir, 'A501_group.pdf')
    plot_vertical_bar(df_grouped['Option'], df_grouped['Häufigkeit'], df_grouped['Prozent'], pdf_path)

#-----------------------------------------------------------------------------------------

def describe_A203(df, variables, out_dir):
    df_sub = df[df['A101'] == 1]
    n = len(df_sub)
    if n == 0:
        print("Keine Nicht-Nutzer (A203) vorhanden.")
        return
    subvars = [v for v in variables['VAR'] if v.startswith('A203_')]
    labels, counts = [], []
    for var in subvars:
        cnt = (pd.to_numeric(df_sub[var], errors='coerce') == 2).sum()
        raw_lbl = variables.loc[variables['VAR'] == var, 'LABEL'].iloc[0]
        core = normalize_label(raw_lbl)
        label = 'Andere' if 'Andere' in core else short_labels.get(core, core)
        labels.append(label)
        counts.append(cnt)
    df_tmp = pd.DataFrame({'Option': labels, 'Häufigkeit': counts})
    df_grouped = df_tmp.groupby('Option', as_index=False).sum()
    df_grouped['Prozent'] = (df_grouped['Häufigkeit'] / df_grouped['Häufigkeit'].sum() * 100).round(1)

    # Reihenfolge der Kategorien festlegen
    desired_order = [
        'Online-Shops / E-Commerce',
        'Kundenservice-Portale (Telekommunikation, Banken, Versicherungen)',
        'Buchungsseiten (Reisen, Hotels, Tickets)',
        'Webseiten von Behörden oder öffentlichen Einrichtungen',
        'Andere'
    ]
    # Reindex DataFrame nach gewünschter Reihenfolge
    df_grouped = df_grouped.set_index('Option').reindex(desired_order).reset_index()

    # CSV export
    csv_path = os.path.join(out_dir, 'A203_group.csv')
    df_grouped.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"Deskriptives A203 exportiert: {csv_path} (n={n})")

    # Vertikales Säulendiagramm mit geänderter Reihenfolge
    pdf_path = os.path.join(out_dir, 'A203_group.pdf')
    plot_vertical_bar(df_grouped['Option'], df_grouped['Häufigkeit'], df_grouped['Prozent'], pdf_path)

#-----------------------------------------------------------------------------------------

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    out_dir = os.path.join(base_dir, 'figures', 'descriptives')
    os.makedirs(out_dir, exist_ok=True)

    data_file = os.path.join(data_dir, 'data_chatbot_quality_2025-05-09.xlsx')
    vars_file = os.path.join(data_dir, 'variables_chatbot_quality_2025-05-09.csv')
    values_file = os.path.join(data_dir, 'values_chatbot_quality_2025-05-09.csv')

    raw = pd.read_excel(data_file, engine='openpyxl')
    df = raw.iloc[1:].reset_index(drop=True) if not raw.empty else raw
    df['A101'] = pd.to_numeric(df['A101'], errors='coerce')

    variables = read_codebook(vars_file)
    values = read_codebook(values_file)

    # Mapping response -> meaning
    value_map = {}
    for _, row in values.iterrows():
        value_map.setdefault(row['VAR'], {})[str(row['RESPONSE'])] = row['MEANING']

    # Spezial A501 und A203
    describe_A501(df, variables, out_dir)
    describe_A203(df, variables, out_dir)

    # Gruppen für Mehrfachauswahl
    multi_groups = {
        'A203': 'Arten von Webseiten',
        'A302': 'Wichtigkeit',
        'A305': 'Ist-Erfüllung',
        'A501': 'Hauptgründe für Nichtnutzung'
    }

    #-----------------------------------------------------------------------------------------

    def plot_bar(categories, counts, percents, out_path):
        fig, ax = plt.subplots(figsize=(6, 3.3))
        bars = ax.bar(categories, counts, color=plt.get_cmap('Set2').colors)

        # Gitter, Labels, Drehung etc. (wie gehabt)
        ax.yaxis.grid(True, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=mpl.rcParams['xtick.labelsize'])
        ax.set_ylabel('Anzahl der Nennungen', labelpad=10)

        # Platz für Prozent-Annotationen schaffen
        # Bestimme max_val sicher aus counts
        if len(counts) > 0:
            max_val = counts.max()
        else:
            max_val = 1
        ax.set_ylim(0, max_val * 1.15)

        # Prozent-Annotations über jeder Säule
        for idx, (cnt, pct) in enumerate(zip(counts, percents)):
            ax.text(idx, cnt + max_val * 0.02, f'{pct:.1f}', ha='center', va='bottom', fontsize=mpl.rcParams['xtick.labelsize'])

        plt.tight_layout()
        fig.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        print(f"Diagramm gespeichert: {out_path}")

    #-----------------------------------------------------------------------------------------

    def describe_multi(prefix, group_label):
        # Subset: für A501 Nicht-Nutzer (A101==2), sonst Nutzer (A101==1)
        df_sub = df[df['A101'] == (2 if prefix == 'A501' else 1)]
        n = len(df_sub)
        if n == 0:
            print(f"Keine Daten für Gruppe {prefix}")
            return

        # Untervariablen
        subvars = [v for v in variables['VAR'] if v.startswith(prefix + '_')]
        labels, counts = [], []
        for var in subvars:
            cnt = (pd.to_numeric(df_sub[var], errors='coerce') == 2).sum()
            raw_lbl = variables.loc[variables['VAR'] == var, 'LABEL'].iloc[0]
            core = normalize_label(raw_lbl)
            label = short_labels.get(core, core)
            labels.append(label)
            counts.append(cnt)

        total = sum(counts)
        df_desc = pd.DataFrame({'Option': labels, 'Häufigkeit': counts,
                                 'Prozent': [round(c / total * 100, 2) for c in counts]})
        # CSV export
        csv_path = os.path.join(out_dir, f'{prefix}_group.csv')
        df_desc.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"Deskriptives {prefix}: {csv_path} (Selektionen={total})")

        # Plot
        pdf_path = os.path.join(out_dir, f'{prefix}_group.pdf')
        plot_bar(
            df_desc['Option'],
            df_desc['Häufigkeit'],
            df_desc['Prozent'],
            pdf_path
        )

    #-----------------------------------------------------------------------------------------

    def describe_question(var, vartype):     
        # skip A301 und alle multi-group Subvars
        if var.startswith('A301_') \
            or any(var.startswith(p + '_') for p in multi_groups):
            return

        df_sub = df if var in ('A101', 'A601', 'A602') else df[df['A101'] == 1]
        n = len(df_sub)
        if n == 0:
            return

        raw = df_sub[var].fillna('-9').astype(str)
        labels = raw.map(value_map.get(var, {})).fillna('Missing')
        # Shorten labels
        labels = labels.map(lambda x: short_labels.get(normalize_label(x), normalize_label(x)))

        counts = labels.value_counts(dropna=False)
        props = (labels.value_counts(normalize=True, dropna=False) * 100).round(2)

        df_desc = pd.DataFrame({'Antwort': counts.index.astype(str),
                                 'Häufigkeit': counts.values,
                                 'Prozent': props.values})
        # Ordinale Kennzahlen
        if vartype == 'ORDINAL':
            nums = pd.to_numeric(df_sub[var], errors='coerce').dropna().astype(int)
            df_desc['Mittelwert'] = nums.mean()
            df_desc['Median'] = nums.median()
            df_desc['Std'] = nums.std()

        csv_path = os.path.join(out_dir, f'{var}_descriptives.csv')
        df_desc.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"Deskriptives {var}: {csv_path} (n={n})")

        pdf_path = os.path.join(out_dir, f'{var}_distribution.pdf')
        # Binary-Variablen als Kreisdiagramm
        if var in ('A101', 'A602'):
            plot_pie_chart(
                df_desc['Antwort'],
                df_desc['Häufigkeit'],
                df_desc['Prozent'],
                pdf_path
            )
        elif var == 'A202':
            # Spezielle Reihenfolge für A202
            desired_order = [
                'Mehrmals pro Woche', 'Einmal pro Woche', 'Einmal pro Monat',
                'Seltener als einmal im Monat', 'Ich nutze sie nur, wenn ich keine Alternative finde'
            ]
            df_plot = df_desc.set_index('Antwort').reindex(desired_order).reset_index()
            plot_vertical_bar(
                df_plot['Antwort'],
                df_plot['Häufigkeit'],
                df_plot['Prozent'],
                pdf_path
            )
        elif var == 'A401':
            # Spezielle Reihenfolge für A401 inkl. fehlender Option
            desired_order = [
                'Sehr zufrieden',
                'Zufrieden',
                'Neutral',
                'Unzufrieden',
                'Sehr unzufrieden'
            ]
            df_plot = df_desc.set_index('Antwort').reindex(desired_order, fill_value=0).reset_index()
            # Prozent neu berechnen, da wir fehlende Werte hinzugefügt haben
            total = df_plot['Häufigkeit'].sum()
            df_plot['Prozent'] = (df_plot['Häufigkeit'] / total * 100).round(2)
            plot_bar(
                df_plot['Antwort'],
                df_plot['Häufigkeit'],
                df_plot['Prozent'],
                pdf_path
            )
        elif var == 'A402':
            # Spezielle Reihenfolge für A401 inkl. fehlender Option
            desired_order = [
                'Ja, definitiv',
                'Eher ja',
                'Weiß nicht',
                'Eher nein',
                'Nein, definitiv nicht'
            ]
            df_plot = df_desc.set_index('Antwort').reindex(desired_order, fill_value=0).reset_index()
            # Prozent neu berechnen, da wir fehlende Werte hinzugefügt haben
            total = df_plot['Häufigkeit'].sum()
            df_plot['Prozent'] = (df_plot['Häufigkeit'] / total * 100).round(2)
            plot_bar(
                df_plot['Antwort'],
                df_plot['Häufigkeit'],
                df_plot['Prozent'],
                pdf_path
            )
        elif var == 'A601':
            # Spezielle Reihenfolge für A401 inkl. fehlender Option
            desired_order = [
                '18-25 Jahre',
                '26-35 Jahre',
                '36-45 Jahre',
                '46-55 Jahre',
                'Über 55 Jahre'
            ]
            df_plot = df_desc.set_index('Antwort').reindex(desired_order, fill_value=0).reset_index()
            # Prozent neu berechnen, da wir fehlende Werte hinzugefügt haben
            total = df_plot['Häufigkeit'].sum()
            df_plot['Prozent'] = (df_plot['Häufigkeit'] / total * 100).round(2)
            plot_bar(
                df_plot['Antwort'],
                df_plot['Häufigkeit'],
                df_plot['Prozent'],
                pdf_path
            )
        else:
            # Standard-Style
            plot_bar(
                df_desc['Antwort'],
                df_desc['Häufigkeit'],
                df_desc['Prozent'],
                pdf_path
            )

    #-----------------------------------------------------------------------------------------

    # Einzelne Fragen
    for _, row in variables.iterrows():
        if row.get('TYPE') in ['NOMINAL', 'ORDINAL', 'DICHOTOMOUS']:
            describe_question(row['VAR'], row['TYPE'])
    # Mehrfachauswahl-Gruppen
    for prefix, label in multi_groups.items():
        # A501 und A202 werden schon separat behandelt
        if prefix in ('A501', 'A203'):
            continue
        describe_multi(prefix, label)

if __name__ == '__main__':
    main()
