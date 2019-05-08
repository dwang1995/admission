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
# get basic infor
df.info()
print("There are",len(df.columns),"columns, and", len(df),"cases in the dataset. The variables include:")
for x in df.columns:
    sys.stdout.write(str(x)+", ")

df.hist(bins=50, figsize=(20,15))
plt.show()
# from the graph, we can see that the dataset contains 3 continuous variable, which are GPA, GRE score and TOEFL score, 
# and 4 categorical variables, which are LOR, Research, SOP and university ranking. 
# The outcome, which is the chance of admittion, is also a continuous variable. 
# It is noticabel that the variable Serial No. only represents the case number of each student which does not 
# contain useful information for the prediction model. So we will omit this variable in the further analysis.
# This will also be proved in the correlation analysis in the next steps.

corr_matrix=df.corr()
corr_matrix
# From the correlation matrix, we can see that the serial no is barely correlated with the chance of admission or other features.
# This means that we can delete this variable in the futurue analysis because it does not provide useful information.

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
# from the graph, we can see that the features that are most positively correlated with the chance of admission are 
# GPA, GRE score and TOEFL score
# The features that are least positively correlated with the chance of admission are 
# Rearch, LOR and SOP. 

# The plot shows the frequency of GRE score among the candidates 
# From the graph, we can see that the most frequent GRE score is in the range from 310-320.
gre_class = pd.cut(np.array(df[df.columns[1]]),5, labels=["290-300", "300-310", "310-320","320-330","330-340"])
# print(np.array(gre_class))
gre_class = pd.DataFrame(np.array(gre_class), columns =['GRE'])

sns.countplot(x="GRE", data=gre_class, palette="Set3")
plt.show()

print("The minimum TOEFL score is",min(df[df.columns[2]]),"and the maximum TOEFL score is", max(df[df.columns[2]]))
#gre_class = pd.cut(np.array(df[df.columns[2]]),5, labels=["290-300", "300-310", "310-320","320-330","330-340"])
#gre_class = pd.DataFrame(gre_class, columns =['GRE'])

toefl_class = pd.cut(np.array(df[df.columns[2]]),bins=[90,100,110,120], labels=["90-100",'100-110','110-120'])
toefl_class = pd.DataFrame(np.array(toefl_class), columns =['TOEFL'])
sns.countplot(x="TOEFL", data=toefl_class, palette="Set3")
plt.show()

# the graph shows a positive linearity relationship between GPA and GRE
# better GPA is correlated with better GRE score
# also candidates with higher university ranking tend to have higher GPA and higher GRE score
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

# Among the candidates with high admission change, the most of them are coming from higher-ranked colleges 
ranking_ = df[df[df.columns[-1]] >= 0.8]
sns.catplot(x=df.columns[3], kind="count", palette="ch:.25", data=ranking_);
plt.show()

sns.catplot(x=df.columns[4], y=df.columns[-3], data=df);
plt.show()
# better SOP associated with higher G

sns.catplot(x=df.columns[4], y=df.columns[1], data=df);
plt.show()
# better SOP associated with better GRE score

# stronger recommendation letter associated with higher chance of admission
sns.regplot(x=df[df.columns[5]], y=df[df.columns[-1]], data=df)
plt.show()

# the graph shows a positive linearity relationship between GPA and GRE
# better GPA is correlated with better GRE score
# also candidates with stronger recommendation letter tend to have higher GPA and higher GRE score
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