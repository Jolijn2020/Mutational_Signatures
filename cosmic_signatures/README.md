I read the files `COSMIC_v3.4.*txt` using the function below:


```
def read_signature_csvs(path):
    per_type = []
    for tp in glob.glob(f"{path}/*txt"):
        name = re.findall(r'_SBS_([A-Za-z0-9]+).txt', tp)[0]
        per_type.append(pd.read_csv(tp, sep='\t').set_index("Type").add_suffix(f"_{name}"))
    per_type = pd.concat(per_type, axis=1)
    
    all_sigs = {x.split("_")[0] for x in per_type.columns}
    
    per_sig = dict()
    for a in all_sigs:
        per_sig[a.lower()] = per_type.loc[:, [x for x in per_type.columns if re.match(f"{a}_", x)]].rename(columns=lambda x: re.sub(fr'{a}_', '',x)).rename_axis('mut')
    return per_sig
```

Called like this 
```
signature_dict = read_signature_csvs(path="COSMIC_catalogue-signatures_SBS96_v3.4")e
```
A small caveat, I renamed the signature names, I think from A[A>C]C to A.A-C.C? but possibly this was done at a later part in processing
I can't remember the resulting data structure exactly either :)
