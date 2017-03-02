"""
Bar chart demo with pairs of bars grouped for easy comparison.
"""
import numpy as np
import matplotlib.pyplot as plt


#4yo Mean,  3yo Mean
my_mean = (2.52, 1.4)
my_mean_std = (1.09, 0.8)

#4yo Mean Original, 3yo Mean Original
original_mean = (2.11, 1.19)
original_mean_std = (1.08, 1.03)


#index = (0,1,2,4,5,6,8,9,10)
#index = np.array([1, 2.5, 4, 5.5])
#index = np.array([2.5, 4])
index = np.array([1, 2])
bar_width = 0.35

error_config = dict(ecolor='black', lw=0.5, capsize=5, capthick=1, ls = 'dotted') #= {'ecolor': '0.3'}

rects1 = plt.bar(index, my_mean, bar_width,
                 alpha=1.0,
                 color='white',
                 error_kw=error_config,
                 yerr=my_mean_std,
                 hatch="/",
                 label='Robot')

rects2 = plt.bar(index + bar_width, original_mean, bar_width,
                 alpha=1.0,
                 color='white',
                 error_kw=error_config,
                 yerr=original_mean_std,
                 hatch="x",
                 label='Children')


#Set the labels
font_size = 15
#plt.title('Scores by group and gender')
plt.ylabel('Correct Answers (Mean)', size=font_size)
#plt.xlabel('Children Group (Age)', size=font_size)
plt.gca().yaxis.grid(True)
plt.xticks(index + bar_width, ('4-years-old', '3-years-old'), size=font_size)

#Hide the X ticks
ax = plt.axes()
ax.xaxis.set_ticks_position('none') 
#ax.xaxis.set_ticks(range(0, 11, 1))
#Set axis limit
border = 0.8 #defines the border space
ax.set_xlim([index[0]+bar_width-border, index[1]+bar_width+border])
ax.set_ylim([0, 4])

#Setup the legend
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),ncol=3, fancybox=True, shadow=True)

plt.tight_layout()

plt.savefig("fig_robot_children_comparison.jpg", dpi=500)


