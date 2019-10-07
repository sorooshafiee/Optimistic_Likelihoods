import os
import glob
import numpy as np
import pandas as pd

DIR_SAVE = os.path.join(os.environ["HOME"], "Soroosh/results_full")
cwd = os.getcwd()
DIR_CSV = os.path.join(cwd, "csv")
DIR_TABLE = os.path.join(cwd, "table")

FINAL_CSV = os.path.join(DIR_CSV, "main.csv")
FILESET = glob.glob(DIR_SAVE + "/*.csv")
FILESET.sort()

columns = ['dataset', 'method', 'rho_1', 'rho_2',
           'DA acc val', 'QDA acc val', 'DA auc val', 'QDA auc val',
           'DA acc test', 'QDA acc test', 'DA auc test', 'QDA auc test']
data = []

def to_latex(df, rule, score, latex_file=None):
    val = ' '.join([rule, score, 'val'])
    test = ' '.join([rule, score, 'test'])
    df = df.loc[df.groupby(['dataset', 'method'])[val].idxmax()][test]
    df = df.reset_index()
    df = pd.pivot_table(df[['dataset','method', test]],
                        values=test,
                        index=['dataset'],
                        columns='method')
    if latex_file is not None:
        with open(latex_file, 'w') as f:
            f.write("\documentclass{article} \n\\usepackage{booktabs} \n\\begin{document} \n")
            f.write(df.to_latex())
            f.write("\n\\end{document}")
        os.chdir(DIR_TABLE)
        os.system('pdflatex {}'.format(latex_file))
        os.chdir(cwd)
    return df


if os.path.isfile(FINAL_CSV):
    df_main = pd.read_csv(FINAL_CSV)
else:
    for ind, fname in enumerate(FILESET):
        df = pd.read_csv(fname, header=None)
        result = df.mean(axis=0).round(2).get_values()
        name = fname.split('/')[-1][:-4]
        if any(char.isdigit() for char in name):
            sname = name.split('_')
            dataset = '_'.join(sname[:-3])
            method = 'RDA' if sname[-3].lower() == 'freg'\
                else sname[-3][0].upper() + 'QDA'
            rho_1 = sname[-2]
            rho_2 = sname[-1]
        else:
            if 'freg' in name:
                sname = name.split('_')
                dataset = '_'.join(sname[:-1])
                method = 'RDA'
            else:
                dataset = name
                method = 'QDA'
            rho_1 = '0'
            rho_2 = '0'
        row = np.append(np.array([dataset, method, rho_1, rho_2]), result)
        data.append(row)
    df_main = pd.DataFrame(data=data, columns=columns)
    df_main.to_csv(FINAL_CSV, index=False)

convert_dict = dict(list(zip(columns[2:], [float]*len(columns[2:]))))
df_main = df_main.astype(convert_dict)

df_ = df_main.set_index(['dataset', 'method', 'rho_1', 'rho_2'])
QDA_acc = to_latex(df_, 'QDA', 'acc', os.path.join(DIR_TABLE, "QDA_acc.tex"))
QDA_auc = to_latex(df_, 'QDA', 'auc', os.path.join(DIR_TABLE, "QDA_auc.tex"))
DA_acc = to_latex(df_, 'DA', 'acc', os.path.join(DIR_TABLE, "DA_acc.tex"))
DA_auc = to_latex(df_, 'DA', 'acc', os.path.join(DIR_TABLE, "DA_auc.tex"))

cleanupFiles = glob.glob(os.path.join(DIR_TABLE, "*.*"))
for cleanupFile in cleanupFiles:
    if '.tex' in cleanupFile or '.pdf' in cleanupFile:
        pass
    else:
        os.remove(cleanupFile)