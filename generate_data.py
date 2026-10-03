"""Generate a synthetic customer dataset (no real data was provided)."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
# (share, age, income, recency, freq, aov, discount, online, tenure)
A = {
 "champ":  (0.18,(38,8),(95,15),(15,8),(26,5),(120,25),(10,6),(70,15),(48,14)),
 "loyal":  (0.27,(42,10),(65,12),(40,20),(14,4),(70,15),(25,10),(55,18),(36,14)),
 "deal":   (0.25,(26,5),(35,8),(70,30),(8,3),(35,10),(60,12),(80,12),(18,10)),
 "lapsed": (0.30,(48,12),(50,15),(220,60),(3,2),(55,20),(30,15),(40,20),(30,18)),
}
cats = ["Electronics","Fashion","Home & Kitchen","Beauty","Groceries","Sports"]
pref = {"champ":[.30,.20,.15,.15,.10,.10],"loyal":[.10,.15,.30,.15,.20,.10],
        "deal":[.15,.35,.10,.20,.10,.10],"lapsed":[.15,.15,.20,.10,.30,.10]}
rows=[]
for k,(sh,*p) in A.items():
    n=int(1200*sh)
    g=lambda m,s,lo,hi: np.clip(rng.normal(m,s,n),lo,hi)
    age,inc,rec,fr,aov,dis,onl,ten=(g(*p[0],18,75),g(*p[1],15,200),g(*p[2],1,365),g(*p[3],1,60),
                                    g(*p[4],8,400),g(*p[5],0,95),g(*p[6],0,100),g(*p[7],1,96))
    d=pd.DataFrame(dict(Age=age.round(),Annual_Income_k=inc.round(1),Recency_Days=rec.round(),
        Frequency=fr.round(),Avg_Order_Value=aov.round(2),Discount_Usage_Pct=dis.round(1),
        Online_Share_Pct=onl.round(1),Tenure_Months=ten.round()))
    d["Total_Spend"]=(d.Frequency*d.Avg_Order_Value).round(2)
    d["Gender"]=rng.choice(["Female","Male"],n,p=[.52,.48])
    d["City_Tier"]=rng.choice(["Tier 1","Tier 2","Tier 3"],n,p=[.45,.35,.20])
    d["Preferred_Category"]=rng.choice(cats,n,p=pref[k])
    rows.append(d)
df=pd.concat(rows).sample(frac=1,random_state=1).reset_index(drop=True)
df.insert(0,"CustomerID",[f"C{i:04d}" for i in range(1,len(df)+1)])
df.to_csv("data/customers.csv",index=False); print(df.shape)
