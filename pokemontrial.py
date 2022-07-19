from matplotlib.image import pil_to_array
import pandas as pd
file = r'C:\Users\Alessandra Blasone\OneDrive\Desktop\Code\3-21-22 Pandas tutorial CSV example\pokemon_data1.csv'

df = pd.read_csv(file) #note that the name of the CSV file goes in the quotes
#note that when you want a specific section to be printed you can put print(df.head(*fill this in*)) for top rows to be printed.
#for bottom rows to be printed selectively, use print(df.tail(3))
#for Excel files: df_xlsx = pd.read_excel('**name of data sheet**') and then print(df)

# print(df.iloc[0:152])
# print(df[df.sort_values['HP', 'Attack']])


# #*FOR MULTIPLE SUBPLOTS*

# # import numpy as np
# import matplotlib.pyplot as plt

# x = df['HP'][0:151]
# y = df['Attack'][0:151]

# x2 = df['HP'][152:251]
# y2 = df['Attack'][152:251]

# x3 = df['HP'][252:386]
# y3 = df['Attack'][252:386]

# x4 = df['HP'][387:493]
# y4 = df['Attack'][387:493]

# x5 = df['HP'][494:649]
# y5 = df['Attack'][494:649]

# x6 = df['HP'][650:721]
# y6 = df['Attack'][650:721]

# import numpy as np

# corr_matrix = np.corrcoef(x, y)
# corr = corr_matrix[0,1]
# R_sq = corr**2
# print(R_sq)

# corr_matrix = np.corrcoef(x2, y2)
# corr = corr_matrix[0,1]
# R_sq2 = corr**2

# corr_matrix = np.corrcoef(x3, y3)
# corr = corr_matrix[0,1]
# R_sq3 = corr**2

# a, b = np.polyfit(x, y, 1)
# m, b = np.polyfit(x, y, 1)
# m2, b2 = np.polyfit(x2, y2, 1)
# m3, b3 = np.polyfit(x3, y3, 1)
# m4, b4 = np.polyfit(x4, y4, 1)
# m5, b5 = np.polyfit(x5, y5, 1)
# m6, b6 = np.polyfit(x6, y6, 1)

# fig, axs = plt.subplots(6)
# #plt.plot(x, y)
# color = ['r', 'b', 'g', 'c', 'k', 'y']

# size = 10

# axs[0].scatter(x, y, s = size, c = color[0])
# axs[1].scatter(x2, y2, s = size, c = color[1])
# axs[2].scatter(x3, y3, s = size, c = color[2])
# axs[3].scatter(x4, y4, s = size, c = color[3])
# axs[4].scatter(x5, y5, s = size, c = color[4])
# axs[5].scatter(x6, y6, s = size, c = color[5])

# xalign = 200

# # plt.plot(x, a*x+b, c = 'red')
# # plt.annotate('y=0.3x+54'.format(), xy = (200, 100))

# axs[0].plot(x, m*x+b, c = color[0])
# plt.annotate('y = {} x + {}'.format(str(round(m, 2)), str(round(b, 2))), xy = (xalign, 40))
# #plt.annotate('R\u00b2 = {}'.format(str(round(R_sq, 6))), xy = (xalign, 80))

# axs[1].plot(x2, m2*x2+b2, c = color[1])
# plt.annotate('y = {} x + {}'.format(str(round(m2, 2)), str(round(b2, 2))), xy = (xalign, 140))
# #plt.annotate('R\u00b2 = {}'.format(str(round(R_sq2, 6))), xy = (xalign, 180))

# axs[2].plot(x3, m3*x3+b3, c = color[2])
# plt.annotate('y = {} x + {}'.format(str(round(m3, 2)), str(round(b3, 2))), xy = (xalign, 240))
# #plt.annotate('R\u00b2 = {}'.format(str(round(R_sq3, 6))), xy = (xalign, 280))

# axs[3].plot(x4, m4*x4+b4, c = color[3])
# axs[4].plot(x5, m5*x5+b5, c = color[4])
# axs[5].plot(x6, m6*x6+b6, c = color[5])

# fig.suptitle('Attack vs. HP up to Generation 6 Pokemon')
# # plt.title('Attack vs. HP up to Generation 6 Pokemon')
# plt.xlabel("HP")
# plt.ylabel("Attack")
# plt.annotate('R\u00b2 = {}'.format(R_sq), xy = (120, 130))

# plt.show()
# #*END OF MULTIPLE SUBPLOTS

# #*FOR DATA ALL ON ONE PLOT*
# import matplotlib.pyplot as plt
# import numpy as np

# x = df['HP'][0:151]
# y = df['Attack'][0:151]
# x2 = df['HP'][152:251]
# y2 = df['Attack'][152:251]
# x3 = df['HP'][252:386]
# y3 = df['Attack'][252:386]
# x4 = df['HP'][387:493]
# y4 = df['Attack'][387:493]
# x5 = df['HP'][494:649]
# y5 = df['Attack'][494:649]
# x6 = df['HP'][650:721]
# y6 = df['Attack'][650:721]

# plt.scatter(x, y)
# plt.scatter(x2, y2)
# plt.scatter(x3, y3)
# plt.scatter(x4, y4)
# plt.scatter(x5, y5)
# plt.scatter(x6, y6)

# a, b = np.polyfit(x, y, 1)
# m, b = np.polyfit(x, y, 1)
# m2, b2 = np.polyfit(x2, y2, 1)
# m3, b3 = np.polyfit(x3, y3, 1)
# m4, b4 = np.polyfit(x4, y4, 1)
# m5, b5 = np.polyfit(x5, y5, 1)
# m6, b6 = np.polyfit(x6, y6, 1)

# corr_matrix = np.corrcoef(x, y)
# corr = corr_matrix[0,1]
# R_sq = corr**2

# corr_matrix = np.corrcoef(x2, y2)
# corr = corr_matrix[0,1]
# R_sq2 = corr**2

# corr_matrix = np.corrcoef(x3, y3)
# corr = corr_matrix[0,1]
# R_sq3 = corr**2

# corr_matrix = np.corrcoef(x4, y4)
# corr = corr_matrix[0,1]
# R_sq4 = corr**2

# corr_matrix = np.corrcoef(x5, y5)
# corr = corr_matrix[0,1]
# R_sq5 = corr**2

# corr_matrix = np.corrcoef(x6, y6)
# corr = corr_matrix[0,1]
# R_sq6 = corr**2

# fig, axs = plt.subplots()
# # plt.scatter(x, y)
# color = ['b', 'm', 'r', 'g', 'c', 'k']

# size = 7

# plt.scatter(x, y, s = size, c = color[0])
# plt.scatter(x2, y2, s = size, c = color[1])
# plt.scatter(x3, y3, s = size, c = color[2])
# plt.scatter(x4, y4, s = size, c = color[3])
# plt.scatter(x5, y5, s = size, c = color[4])
# plt.scatter(x6, y6, s = size, c = color[5])

# xalign = 225

# plt.plot(x, m*x+b, c = color[0])
# plt.annotate('y = {} x + {}'.format(str(round(m, 2)), str(round(b, 2))), xy = (xalign, 20))
# # plt.annotate('R\u00b2 = {}'.format(str(round(R_sq, 6))), xy = (xalign, 50))

# plt.plot(x2, m2*x2+b2, c = color[1])
# plt.annotate('y2 = {} x + {}'.format(str(round(m2, 2)), str(round(b2, 2))), xy = (xalign, 40))
# # plt.annotate('R\u00b2 = {}'.format(str(round(R_sq2, 6))), xy = (xalign, 70))

# plt.plot(x3, m3*x3+b3, c = color[2])
# plt.annotate('y3 = {} x + {}'.format(str(round(m3, 2)), str(round(b3, 2))), xy = (xalign, 60))
# # plt.annotate('R\u00b2 = {}'.format(str(round(R_sq3, 6))), xy = (xalign, 90))

# plt.plot(x4, m4*x4+b4, c = color[3])
# plt.annotate('y4 = {} x + {}'.format(str(round(m4, 2)), str(round(b4, 2))), xy = (xalign, 80))
# # plt.annotate('R\u00b2 = {}'.format(str(round(R_sq4, 6))), xy = (xalign, 110))

# plt.plot(x5, m5*x5+b5, c = color[4])
# plt.annotate('y5 = {} x + {}'.format(str(round(m5, 2)), str(round(b5, 2))), xy = (xalign, 100))
# # plt.annotate('R\u00b2 = {}'.format(str(round(R_sq5, 6))), xy = (xalign, 130))

# plt.plot(x6, m6*x6+b6, c = color[5])
# plt.annotate('y6 = {} x + {}'.format(str(round(m6, 2)), str(round(b6, 2))), xy = (xalign, 120))
# # plt.annotate('R\u00b2 = {}'.format(str(round(R_sq6, 6))), xy = (xalign, 150))

# plt.title('Attack vs. HP up to Generation 6 Pokemon')
# plt.xlabel("HP")
# plt.ylabel("Attack")

# plt.legend(["Generation 1: R²=0.079433", "Generation 2: R²=0.168722", "Generation 3: R²=0.11849", "Generation 4: R²=0.205574", "Generation 5: R²=0.316655", "Generation 6: R²=0.193636"])

# plt.show()
# #*END OF ONE PLOT DATA*


# #*FOR BOX AND WHISKER PLOT*
# import matplotlib.pyplot as plt
# import numpy as np
 
# boxplot = df.boxplot(figsize = (5,5), rot = 90, fontsize = '8', grid = False)

# data_1 = df['HP'][0:151]
# data_2 = df['HP'][152:251]
# data_3 = df['HP'][252:386]
# data_4 = df['HP'][387:493]
# data_5 = df['HP'][494:649]
# data_6 = df['HP'][650:721]
# data = [data_1, data_2, data_3, data_4, data_5, data_6]
 
# fig = plt.figure(figsize =(10, 7))

# ax = fig.add_axes([0.15, 0.15, 0.7, 0.8])

# bp = ax.boxplot(data)

# plt.title('HP Distribution up to Generation 6 Pokemon')
# plt.xlabel("Pokemon Generation")
# plt.ylabel("HP")

# plt.show()
# #**END OF BOXPLOT EXAMPLE**


#*Another one plot example*
# import matplotlib.pyplot as plt
# import numpy as np

# x = df['Sp. Def'][0:151]
# y = df['Sp. Atk'][0:151]
# x2 = df['Sp. Def'][152:251]
# y2 = df['Sp. Atk'][152:251]
# x3 = df['Sp. Def'][252:386]
# y3 = df['Sp. Atk'][252:386]
# x4 = df['Sp. Def'][387:493]
# y4 = df['Sp. Atk'][387:493]
# x5 = df['Sp. Def'][494:649]
# y5 = df['Sp. Atk'][494:649]
# x6 = df['Sp. Def'][650:721]
# y6 = df['Sp. Atk'][650:721]

# plt.scatter(x, y)
# plt.scatter(x2, y2)
# plt.scatter(x3, y3)
# plt.scatter(x4, y4)
# plt.scatter(x5, y5)
# plt.scatter(x6, y6)

# a, b = np.polyfit(x, y, 1)
# m, b = np.polyfit(x, y, 1)
# m2, b2 = np.polyfit(x2, y2, 1)
# m3, b3 = np.polyfit(x3, y3, 1)
# m4, b4 = np.polyfit(x4, y4, 1)
# m5, b5 = np.polyfit(x5, y5, 1)
# m6, b6 = np.polyfit(x6, y6, 1)

# corr_matrix = np.corrcoef(x, y)
# corr = corr_matrix[0,1]
# R_sq = corr**2

# corr_matrix = np.corrcoef(x2, y2)
# corr = corr_matrix[0,1]
# R_sq2 = corr**2

# corr_matrix = np.corrcoef(x3, y3)
# corr = corr_matrix[0,1]
# R_sq3 = corr**2

# corr_matrix = np.corrcoef(x4, y4)
# corr = corr_matrix[0,1]
# R_sq4 = corr**2

# corr_matrix = np.corrcoef(x5, y5)
# corr = corr_matrix[0,1]
# R_sq5 = corr**2

# corr_matrix = np.corrcoef(x6, y6)
# corr = corr_matrix[0,1]
# R_sq6 = corr**2

# fig, axs = plt.subplots()
# color = ['b', 'm', 'r', 'g', 'c', 'k']

# size = 7

# plt.scatter(x, y, s = size, c = color[0])
# plt.scatter(x2, y2, s = size, c = color[1])
# plt.scatter(x3, y3, s = size, c = color[2])
# plt.scatter(x4, y4, s = size, c = color[3])
# plt.scatter(x5, y5, s = size, c = color[4])
# plt.scatter(x6, y6, s = size, c = color[5])

# xalign = 205

# plt.plot(x, m*x+b, c = color[0])
# plt.annotate('y = {} x + {}'.format(str(round(m, 2)), str(round(b, 2))), xy = (xalign, 80))
# plt.annotate('R\u00b2 = {}'.format(str(round(R_sq, 6))), xy = (xalign, 90))

# plt.plot(x2, m2*x2+b2, c = color[1])
# plt.annotate('y2 = {} x + {}'.format(str(round(m2, 2)), str(round(b2, 2))), xy = (xalign, 100))
# plt.annotate('R\u00b2 = {}'.format(str(round(R_sq2, 6))), xy = (xalign, 110))

# plt.plot(x3, m3*x3+b3, c = color[2])
# plt.annotate('y3 = {} x + {}'.format(str(round(m3, 2)), str(round(b3, 2))), xy = (xalign, 120))
# plt.annotate('R\u00b2 = {}'.format(str(round(R_sq3, 6))), xy = (xalign, 130))

# plt.plot(x4, m4*x4+b4, c = color[3])
# plt.annotate('y4 = {} x + {}'.format(str(round(m4, 2)), str(round(b4, 2))), xy = (xalign, 140))
# plt.annotate('R\u00b2 = {}'.format(str(round(R_sq4, 6))), xy = (xalign, 150))

# plt.plot(x5, m5*x5+b5, c = color[4])
# plt.annotate('y5 = {} x + {}'.format(str(round(m5, 2)), str(round(b5, 2))), xy = (xalign, 160))
# plt.annotate('R\u00b2 = {}'.format(str(round(R_sq5, 6))), xy = (xalign, 170))

# plt.plot(x6, m6*x6+b6, c = color[5])
# plt.annotate('y6 = {} x + {}'.format(str(round(m6, 2)), str(round(b6, 2))), xy = (xalign, 180))
# plt.annotate('R\u00b2 = {}'.format(str(round(R_sq6, 6))), xy = (xalign, 190))

# plt.title('Sp. Attack vs. Sp. Defense up to Generation 6 Pokemon')
# plt.xlabel("Sp. Defense")
# plt.ylabel("Sp. Attack")

# plt.show()
# #**END OF pne plot example**




# #*FOR BARCHART*
# import matplotlib.pyplot as plt
# import numpy as np
 
# x = df.iloc[:, :-1]
# y = df['Type 1']

# y.value_counts().sort_index().plot.bar(x='Type 1', y='Number of Occurrences')

# plt.title('Amount of each Type 1 Pokemon')
# plt.xlabel("Type")
# plt.ylabel("Number of Pokemon")

# plt.show()
# #*END BARCHART*



#*FOR 2D HISTOGRAM AND SCATTER PLOT*
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19680801)

x = np.random.randn(1000)
y = np.random.randn(1000)

x = df['HP'][0:721]
y = df['Attack'][0:721]

def scatter_hist(x, y, ax, ax_histx, ax_histy):
    ax_histx.tick_params(axis="x", labelbottom=True)
    ax_histy.tick_params(axis="y", labelleft=True)

    ax.scatter(x, y)
    plt.xlim([0,300])
    plt.ylim([0,300])
    ax.set_xlim(left=None, right=None, emit=True, auto=False, xmin=0, xmax=None)
    ax.set_xlabel('HP', fontdict=None, labelpad=None, loc=None)
    ax.set_ylabel('Attack', fontdict=None, labelpad=None, loc=None)

    binwidth = 5
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')

left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
spacing = 0.005

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom + height + spacing, width, 0.2]
rect_histy = [left + width + spacing, bottom, 0.2, height]

fig = plt.figure(figsize=(8, 8))

fig.suptitle('Attack vs. HP up to Generation 6 Pokemon')

ax = fig.add_axes(rect_scatter)
ax_histx = fig.add_axes(rect_histx, sharex=ax)
ax_histy = fig.add_axes(rect_histy, sharey=ax)

scatter_hist(x, y, ax, ax_histx, ax_histy)

plt.show()








# fig, ax = plt.subplots()
# ax.plot(x, y, linewidth=1.0)
# ax.set(xlim=(1, 151), xticks=np.arange(1, 151),
#        ylim=(0, 300), yticks=np.arange(0, 300))

# plt.scatter(x, y)
# plt.show()

#prints HP's less than 50
# df[df['HP'] < 50]
# print(df[df['HP'] < 50])

# # #this reads columns: (in this case, just the legendary column)
# if 'Legendary' == True:
#     print(df.column('Legendary'))

# #this is for the top 4 rows
# print(df.head(4))

# #this reads rows
# print(df.iloc[0:4])

# #this prints entire row (an index)
# for index, row in df.iterrows():
#     print(index, row ['Legendary'])    

# #for a specific location R,C
# print(df.iloc[2,1])

# #this locates a specific genre
# df.loc[df['Type 1'] == "Fire"]

# #this summarizes datafram
# df.describe()

# #this alphabetizes
# df.sort_values('Name')

# #this reverse alphabetizes
# df.sort_values('Name', ascending=False)

# #these sort from low to high and high to low, respectively
# df.sort_values(['Type 1', 'HP'])
# df.sort_values(['Type 1', 'HP'], ascending=False)

# #this shows an overall score/overview as one rank
# df['Total'] = df['HP'] + df['Attack'] + df['Defense'] + df['Sp. Atk'] + df['Sp. Def'] + df['Speed'] 
# df.head(5)

# #df = df.drop(columns=['Total'])

# #this mathmatically adds specific rows
# #df['Total'] = df.iloc[:, 4:9].sub(axis=1)

# #this changes the order of the columns and displays only the desired ones
# cols = list(df.columns.values)
# df = df[cols[0:4] + [cols[-1]]+cols[4:12]]

# #TO SAVE:
# df.to_csv('modified.csv')

# #matplot.lib is good for excel

# #just grass type
# df.loc[df['Type 1'] == 'Grass']

# #just grass AND poison type
# #df.loc[df['Type 1'] == 'Grass' & df['Type 2'] == 'Poison']

# #just grass OR poison type
# #df.loc[df['Type 1'] == 'Grass' | df['Type 2'] == 'Poison']

# #just grass AND poison type AND HP > 70
# new_df = df.loc[(df['Type 1'] == 'Grass') & (df['Type 2'] == 'Poison') & (df['HP'] > 70)]
# new_df.to_csv('filtered.csv')

# #to call forth by name
# df.loc[df['Name'].str.contains('Mega')]

# #to rename a category
# df.loc[df['Type 1'] == 'Fire', 'Type 1'] = 'Flamer'

# #to classify
# df.loc[df['Total'] > 500, ['Generation', 'Legendary']] = 'TEST VALUE'
# df = pd.read_csv('modified.csv')

# #to groupby
# df.groupby(['Type 1']).mean().sort_values('Defense', ascending=False)
# df.groupby(['Type 1']).sum()

# #sort by counting up by ones
# df['count'] = 1
# df.groupby(['Type 1']).count()

# #for masssive data sets
# for df in pd.read_csv('modified.csv', chunksize=5):
#     print("CHUNK DF")
#     print(df)