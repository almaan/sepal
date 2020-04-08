#!/usr/bin/env python3


from gprofiler import GProfiler
import pandas as pd




# fam_pth = "/home/alma/w-projects/spatential/res/melanoma/m1/ST_mel1_rep1_counts-family-index.tsv"
fam_pth = "/tmp/mel/m1-new/ST_mel1_rep1_counts-family-index.tsv"
families = pd.read_csv(fam_pth, sep = '\t',header = 0, index_col = 0)

uni_fam = np.unique(families['family'].values)
organism = "hsapiens"

all_res = pd.DataFrame([])

for sel_fam in uni_fam:

    genes = families.index.values[families["family"].values == sel_fam]
    genes = [x.split(' ')[0] for x in genes]

    gp = GProfiler(return_dataframe = True)
    res = gp.profile(organism = organism,
                    query = genes,
                    )

    gobps = res.loc[res['source'] == "GO:BP",:]
    gobps = gobps.loc[gobps['significant'],:]
    gobps['family'] = (np.ones(gobps.shape[0]) * sel_fam).astype(int)

    all_res = pd.concat((all_res,gobps))

# new_order = np.append(all_res.shape[1]-1,np.arange(all_res.shape[1]-1))
keep_cols = ['family','native','name','p_value','intersection_size']
all_res = all_res.loc[:,keep_cols]
all_res['family'] += 1

all_res.index = pd.Index(np.arange(all_res.shape[0]) + 1)

with open("/tmp/test-latex.txt","w+") as f:
    f.write(all_res.to_latex())

