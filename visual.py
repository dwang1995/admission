import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import cufflinks as cf
import numpy as np
import seaborn as sns
import sys
import os

# read data
df = pd.read_csv("Admission_Predict.csv",sep = ",")
# get basic information
df.info()
print("There are",len(df.columns),"columns, and", len(df),"cases in the dataset. The variables include:")
for x in df.columns:
    if x != "Serial No.":
        sys.stdout.write(str(x)+", ")

df.hist(bins=50, figsize=(20,15))
plt.show()

corr_matrix=df.corr()
#corr_matrix

mask = np.zeros_like(corr_matrix, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},
            xticklabels=corr_matrix.columns.values,
            yticklabels=corr_matrix.columns.values)
plt.show()

gre_class = pd.cut(np.array(df[df.columns[1]]),5, labels=["290-300", "300-310", "310-320","320-330","330-340"])

gre_class = pd.DataFrame(np.array(gre_class), columns =['GRE'])

sns.countplot(x="GRE", data=gre_class, palette="Set3")
plt.show()

print("The minimum TOEFL score is",min(df[df.columns[2]]),"and the maximum TOEFL score is", max(df[df.columns[2]]))

toefl_class = pd.cut(np.array(df[df.columns[2]]),bins=[90,100,110,120], labels=["90-100",'100-110','110-120'])
toefl_class = pd.DataFrame(np.array(toefl_class), columns =['TOEFL'])
sns.countplot(x="TOEFL", data=toefl_class, palette="Set3")
plt.show()

sns.set()
sns.scatterplot(y=df[df.columns[1]], x=df[df.columns[-3]],
                hue=df[df.columns[3]],
                data=df)
plt.show()

# Data to plot
labels = list(set(df[df.columns[3]]))
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','blue']
sizes = list(Counter(df[df.columns[3]]).values())
explode = (0.1, 0, 0, 0, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()

ranking_ = df[df[df.columns[-1]] >= 0.8]
sns.catplot(x=df.columns[3], kind="count", palette="ch:.25", data=ranking_);
plt.show()

sns.catplot(x=df.columns[4], y=df.columns[-3], data=df);
plt.show()

sns.catplot(x=df.columns[4], y=df.columns[1], data=df);
plt.show()

sns.regplot(x=df[df.columns[5]], y=df[df.columns[-1]], data=df)
plt.show()

sns.scatterplot(y=df[df.columns[1]], x=df[df.columns[-3]],
                hue=df[df.columns[5]],
                data=df)
plt.show()

f, ax = plt.subplots(figsize=(6,6))
ax = sns.countplot(x=df.columns[-2], data=df, palette="Set3")
total = float(len(df))
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/total),
            ha="center") 
plt.show()