import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import mpl_toolkits.axisartist as axisartist

# with open('./turned_dataset/sample_data.txt', 'r') as f:
#     y = np.array(f.read().splitlines(), dtype='int')
# x = np.array(range(1, y.shape[0] + 1))

rcParams['font.family'] = 'Times New Roman, simsun'
# fig = plt.figure()
# ax = axisartist.Subplot(fig, 111)
# fig.add_axes(ax)
# plt.plot(x[165:], y[165:])
# ax.axis["top"].set_visible(False)
# ax.axis["right"].set_visible(False)
# ax.axis["bottom"].set_axisline_style("-|>", size=1.5)
# ax.axis["left"].set_visible(False)
# # ax.yaxis.set_major_locator(plt.MaxNLocator(6))
# # ax.vlines([52, 125, 165], 0, np.max(y), linestyles='dashed', colors='green')
# # ax.set_xlabel('时间/s', x='right')
# # ax.set_xticks([240], ['时间/s'], minor=True)
# ax.tick_params(labelsize=15)
# # plt.grid(axis='y')
# plt.show()

# np.random.seed(1002)
# rand_series = [np.random.random(15)[:, None] for i in range(3)]
# rand_series[1] += 1
# rand_series[2] += 2
# rand_series = np.concatenate(rand_series, axis=-1)
# fig = plt.figure()
# ax = axisartist.Subplot(fig, 111)
# fig.add_axes(ax)
# plt.plot(rand_series)
# ax.axis["top"].set_visible(False)
# ax.axis["right"].set_visible(False)
# ax.axis["bottom"].set_axisline_style("-|>", size=1.5)
# ax.axis["left"].set_axisline_style("-|>", size=1.5)
# ax.set_xticks([])
# ax.set_yticks([])
# ax.vlines([2, 6, 14], 0, 3, linestyles='dashed', colors='black')
# plt.xlim([0, 14.1])
# plt.ylim([-0.2, 3])
# plt.show()

# np.random.seed(1002)
# num_series = 5
# len_series = 5
# rand_series = [np.random.random(len_series)[:, None] for i in range(num_series)]
# rand_series[-1][0] = 0.9
# rand_series[0][0] = 1
# rand_series[1][0] = 1
# for i in range(num_series):
#     rand_series[i] += i
# rand_series = np.concatenate(rand_series, axis=-1)
# fig = plt.figure()
# ax = axisartist.Subplot(fig, 111)
# fig.add_axes(ax)
# plt.plot(rand_series)
# ax.axis["top"].set_visible(False)
# ax.axis["right"].set_visible(False)
# ax.axis["bottom"].set_axisline_style("-|>", size=1.5)
# ax.axis["left"].set_axisline_style("-|>", size=1.5)
# ax.set_xticks([])
# ax.set_yticks([])
# # ax.vlines([2, 6, 14], 0, 3, linestyles='dashed', colors='black')
# plt.xlim([0, len_series - 1])
# plt.ylim([-0.2, num_series])
# plt.show()

# 多模型曲线图对比
# show_range = list(range(631-200, 715))
show_range = list(range(220, 320))
fig = plt.figure()
ax1 = fig.add_subplot(616, axes_class=axisartist.Axes)
stg = pd.read_csv('./experiment/main_exp/StateGraph/2024-03-20_22-33-31/Static_result.csv')
ax1.plot(show_range, stg['trues'].iloc[show_range], color='black')
ax1.plot(show_range, stg['preds'].iloc[show_range], color='purple')
ax1.set_xlabel('STGFAN')
ax2 = fig.add_subplot(612, axes_class=axisartist.Axes)
rf = pd.read_csv('./experiment/main_exp/RF/Static_result.csv')
ax2.plot(show_range, rf['trues'].iloc[show_range], color='black')
ax2.plot(show_range, rf['preds'].iloc[show_range], color='green')
ax2.set_xlabel('RF')
ax3 = fig.add_subplot(611, axes_class=axisartist.Axes)
svr = pd.read_csv('./experiment/main_exp/SVR/Static_result.csv')
ax3.plot(show_range, svr['trues'].iloc[show_range], color='black')
ax3.plot(show_range, svr['preds'].iloc[show_range], color='red')
ax3.set_xlabel('SVR')
ax4 = fig.add_subplot(615, axes_class=axisartist.Axes)
tr = pd.read_csv('./experiment/main_exp/StateGraph/2024-03-20_15-56-44_NoShuffle_Tr/Static_result.csv')
ax4.plot(show_range, tr['trues'].iloc[show_range], color='black')
ax4.plot(show_range, tr['preds'].iloc[show_range], color='blue')
ax4.set_xlabel('iTransformer')
ax5 = fig.add_subplot(614, axes_class=axisartist.Axes)
lstm = pd.read_csv('./experiment/main_exp/StateGraph/2024-03-20_16-22-30_NoShuffle_LSTM/Static_result.csv')
ax5.plot(show_range, lstm['trues'].iloc[show_range], color='black')
ax5.plot(show_range, lstm['preds'].iloc[show_range], color='orange')
ax5.set_xlabel('LSTM')
ax6 = fig.add_subplot(613, axes_class=axisartist.Axes)
dn = pd.read_csv('./experiment/main_exp/StateGraph/2024-03-20_16-46-26_NoShuffle_DenseNet/Static_result.csv')
ax6.plot(show_range, dn['trues'].iloc[show_range], color='black')
ax6.plot(show_range, dn['preds'].iloc[show_range], color='deeppink')
ax6.set_xlabel('DenseNet')
for i in range(1, 7):
    eval('ax' + str(i)).axis["top"].set_visible(False)
    eval('ax' + str(i)).axis["right"].set_visible(False)
    eval('ax' + str(i)).axis["bottom"].set_axisline_style("-|>", size=1)
    eval('ax' + str(i)).axis["left"].set_axisline_style("-|>", size=1)
plt.tight_layout()
plt.show()
