import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from utils import get_stat_func, CompressPDF

def get_cmap(n, name='tab20'):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)

font = {'size': 40}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.constrained_layout.use'] = True

RESULTS_PATH = '/home/ahmed/Documents/ICLR2022/studies/'
STUDY = 'rebuttal'
SAVE_PATH = '/home/ahmed/Documents/ICLR2022/studies/plots/' + STUDY + '/'
TO_PLOT = ['speed']

NB_CLASSES = 11

LINE = 'mean'
ERR = 'std'
DPI = 30
N_SEEDS = None
N_EPOCHS = None
LINEWIDTH = 7
MARKERSIZE = 10
ALPHA = 0.3
ALPHA_TEST = 0.05
MARKERS = ['o', 'v', 's', 'P', 'D', 'X', "*", 'v', 's', 'p', 'P', '1']
FREQ = 5
NB_CYCLES_PER_EPOCH = 1200
LAST_EP = 220
LIM = NB_CYCLES_PER_EPOCH * LAST_EP / 1000 + 2
line, err_min, err_plus = get_stat_func(line=LINE, err=ERR)
COMPRESSOR = CompressPDF(4)

def setup_figure(xlabel=None, ylabel=None, xlim=None, ylim=None):
    fig = plt.figure(figsize=(29, 20), frameon=False)
    ax = fig.add_subplot(111)
    ax.spines['top'].set_linewidth(6)
    ax.spines['right'].set_linewidth(6)
    ax.spines['bottom'].set_linewidth(6)
    ax.spines['left'].set_linewidth(6)
    ax.tick_params(width=10, direction='in', length=20, labelsize='40')
    artists = ()
    if xlabel:
        xlab = plt.xlabel(xlabel)
        artists += (xlab,)
    if ylabel:
        ylab = plt.ylabel(ylabel)
        artists += (ylab,)
    if ylim:
        plt.ylim(ylim)
    if xlim:
        plt.xlim(xlim)
    return artists, ax

def setup_n_figs(n, m, xlabels=None, ylabels=None, xlims=None, ylims=None):
    # fig, axs = plt.subplots(2, 3, figsize=(25, 10), frameon=False)
    fig, axs = plt.subplots(n, m, figsize=(24,6*m), frameon=False)
    axs = axs.ravel()
    # fig = plt.figure(figsize=(22, 7))
    # axs = np.array([plt.subplot(141, gridspec_kw={'width_ratios': [3, 1]}), plt.subplot(142, gridspec_kw={'width_ratios': [3, 1]}),
    #                 plt.subplot(143, gridspec_kw={'width_ratios': [3, 1]}), plt.subplot(144, gridspec_kw={'width_ratios': [3, 1]})])
    axs = axs.ravel()
    artists = ()
    for i_ax, ax in enumerate(axs):
        ax.spines['top'].set_linewidth(3)
        ax.spines['right'].set_linewidth(3)
        ax.spines['bottom'].set_linewidth(3)
        ax.spines['left'].set_linewidth(3)
        ax.tick_params(width=5, direction='in', length=15, labelsize='20', zorder=10)
        if xlabels[i_ax]:
            xlab = ax.set_xlabel(xlabels[i_ax])
            artists += (xlab,)
        if ylabels[i_ax]:
            ylab = ax.set_ylabel(ylabels[i_ax])
            artists += (ylab,)
        if ylims[i_ax]:
            ax.set_ylim(ylims[i_ax])
        if xlims[i_ax]:
            ax.set_xlim(xlims[i_ax])
    return artists, axs, fig

def save_fig(directory, filename, artists):
    if not os.path.exists(directory):
        os.mkdir(directory)
    path = directory + filename
    plt.savefig(os.path.join(path), bbox_extra_artists=artists, bbox_inches='tight', dpi=DPI)
    plt.close('all')
    # compress PDF
    try:
        COMPRESSOR.compress(path, path[:-4] + '_compressed.pdf')
        os.remove(path)
    except:
        pass


def check_length_and_seeds(experiment_path):
    conditions = os.listdir(experiment_path)
    # check max_length and nb seeds
    max_len = 0
    max_seeds = 0
    min_len = 1e6
    min_seeds = 1e6

    for cond in conditions:
        cond_path = experiment_path + cond + '/'
        list_runs = sorted(os.listdir(cond_path))
        if len(list_runs) > max_seeds:
            max_seeds = len(list_runs)
        if len(list_runs) < min_seeds:
            min_seeds = len(list_runs)
        for run in list_runs:
            try:
                run_path = cond_path + run + '/'
                data_run = pd.read_csv(run_path + 'progress.csv')
                nb_epochs = len(data_run)
                if nb_epochs > max_len:
                    max_len = nb_epochs
                if nb_epochs < min_len:
                    min_len = nb_epochs
            except:
                pass
    return max_len, max_seeds, min_len, min_seeds


def plot_classes(experiment_path, n_seeds=1):
    all_conditions = os.listdir(experiment_path)
    n = int((input('How many conditions to be tested ? ')))
    print('Select conditions from {}'.format(all_conditions))
    conditions = [input() for _ in range(n)]
    titles = [input('Enter label for condition {}: '.format(c)) for c in conditions]
    artists, axs, fig = setup_n_figs(n=len(conditions)//2,
                                     m=2,
                                     xlabels=[None, None] * (len(conditions)//2 - 1) + ['Cycles (x10³)', 'Cycles (x10³)'],
                                     ylabels=['Success Rate', None] * (len(conditions)//2),
                                     xlims=[(0, LIM)] * len(conditions),
                                     ylims=[(-0.01, 1.01)] * len(conditions))
    colors = get_cmap(6, name='tab10')
    for k, cond in enumerate(conditions):
        cond_path = experiment_path + cond + '/'
        list_runs = sorted(os.listdir(cond_path))
        # for run in list_runs:
        run = list_runs[0]
        print(run)
        # try:
        run_path = cond_path + run + '/'
        data_run = pd.read_csv(run_path + 'progress.csv')
        x_eps = np.arange(0, (LAST_EP + 1) * NB_CYCLES_PER_EPOCH, NB_CYCLES_PER_EPOCH * FREQ) / 1000
        x = np.arange(0, LAST_EP + 1, FREQ)
        for j, i in enumerate([4, 5, 6, 8, 9, 10]):
            axs[k].plot(x_eps, data_run['Eval_SR_{}'.format(i + 1)][x], color=colors(j), marker=MARKERS[j], markersize=MARKERSIZE,
                        linewidth=LINEWIDTH)
            axs[k].set_title(titles[k], fontweight='bold')
        axs[k].grid()
    # ['Close 1', 'Close 2', 'Close 3', 'Stack 2', 'Stack 3', 'Stacks 2&2', 'Stack 2&3', 'Pyramid', 'Pyramid&Stack2', 'Stack 4',
    #  'Stack 5']
    leg = fig.legend(['Stack 3', 'Stacks 2&2', 'Stack 2&3', 'Pyramid&Stack2', 'Stack 4', 'Stack 5'],
                     loc='upper center',
                     bbox_to_anchor=(0.5, 1.1),
                     ncol=6,
                     fancybox=True,
                     shadow=True,
                     prop={'size': 20, 'weight': 'bold'},
                     markerscale=1)
    artists += (leg,)
    plt.savefig(SAVE_PATH + '/goals_classes.pdf', bbox_inches='tight')
    plt.close('all')

def plot_counts_and_sr(experiment_path, conditions):
    # conditions = os.listdir(experiment_path)
    # titles = ['Close 3', 'Stack 2', 'Stack 3', 'Stack 2 & 2', 'Stack 2 & 3', 'Pyramid', 'Pyr 3 & Stack 2', 'Stack 4', 'Stack 5']
    titles = ['Stack 3', 'Stack 2 & 2', 'Stack 2 & 3', 'Pyr 3 & Stack 2', 'Stack 4', 'Stack 5']
    classes = [5, 6, 7, 9, 10, 11]
    # titles = ['Stack 3', 'Stack 2 & 2', 'Pyr 3 & Stack 2']
    # classes = [5, 6, 9]
    colors = get_cmap(9, name='tab10')
    for cond in conditions:
        cond_path = experiment_path + cond + '/'
        list_runs = sorted(os.listdir(cond_path))
        # for run in list_runs:
        run = list_runs[2]
        print(run)
        # try:
        run_path = cond_path + run + '/'
        data_run = pd.read_csv(run_path + 'progress.csv')
        x_eps = np.arange(0, (LAST_EP + 1) * NB_CYCLES_PER_EPOCH, NB_CYCLES_PER_EPOCH * FREQ) / 1000
        x = np.arange(0, LAST_EP + 1, FREQ)
        artists, axs, fig = setup_n_figs(n=3,
                                         m=2,
                                    xlabels=[None, None, None] * 1 + ['Cycles (x$10^3$)', 'Cycles (x$10^3$)','Cycles (x$10^3$)'],
                                    ylabels=['SP goals', None, None] * 2,
                                    xlims=[(0, LIM)] * 6,
                                    ylims=[(0, 20), (0, 10), (0, 10), (0, 10), (0, 20), (0, 30)])
        # artists, axs, fig = setup_n_figs(n=3,
        #                                  m=1,
        #                                  xlabels=[None, None, 'Cycles (x$10^3$)'],
        #                                  ylabels=['SP goals', 'SP goals', 'SP goals'],
        #                                  xlims=[(0, LIM)] * 3,
        #                                  ylims=[(0, 20), (0, 10), (0, 10)])
        for i, c in enumerate(classes):
            l1 = axs[i].plot(x_eps, data_run['# class_teacher {}'.format(c)][x], color=colors(0), marker=MARKERS[c], markersize=MARKERSIZE, linewidth=LINEWIDTH,
                        linestyle='dashed')
            l2= axs[i].plot(x_eps, data_run['# class_agent {}'.format(c)][x], color=colors(0), marker=MARKERS[c], markersize=MARKERSIZE, linewidth=LINEWIDTH,
                        linestyle='dotted')
            axs[i].grid()
            ax2 = axs[i].twinx()
            l3 = ax2.plot(x_eps, data_run['Eval_SR_{}'.format(c)][x], color=colors(0), marker=MARKERS[c], markersize=MARKERSIZE, linewidth=LINEWIDTH)
            if (i+1) % 3 == 0:
                ax2.set_ylabel("Success Rate")
            ax2.set_ylim(-0.01, 1.01)
            axs[i].set_title(titles[i], fontweight='bold')

        # l = l1 + l2 + l3
        leg = fig.legend(['# SP suggestions', '# reached configurations', 'Success Rate'],
                         loc='upper center',
                         bbox_to_anchor=(0.5, 1.1),
                         ncol=3,
                         fancybox=True,
                         shadow=True,
                         prop={'size': 20, 'weight': 'bold'},
                         markerscale=1)
        artists += (leg,)
        # ax.grid()
        # plt.show()
        # save_fig(path=run_path + 'goals_{}.pdf'.format(cond), artists=artists)
        # except:
        #     print('failed')
        plt.savefig(SAVE_PATH + '/goals_sr.pdf'.format(i), bbox_inches='tight')
        plt.close('all')

def plot_stepping_stones(experiment_path, max_len, max_seeds, conditions=None, labels=None):
    if conditions is None:
        conditions = os.listdir(experiment_path)
    st = np.zeros([max_seeds, len(conditions), LAST_EP + 1 ])
    st.fill(np.nan)
    colors = get_cmap(3*len(conditions))
    for i_cond, cond in enumerate(conditions):
        cond_path = experiment_path + cond + '/'
        list_runs = sorted(os.listdir(cond_path))
        for i_run, run in enumerate(list_runs):
            run_path = cond_path + run + '/'
            data_run = pd.read_csv(run_path + 'progress.csv')
            all_sr = np.mean(np.array([data_run['stepping_stones_len'][:LAST_EP + 1]]), axis=0)
            st[i_run, i_cond, :all_sr.size] = all_sr.copy()


    st_per_cond_stats = np.zeros([len(conditions), LAST_EP + 1, 3])
    st_per_cond_stats[:, :, 0] = line(st)
    st_per_cond_stats[:, :, 1] = err_min(st)
    st_per_cond_stats[:, :, 2] = err_plus(st)


    x_eps = np.arange(0, (LAST_EP + 1) * NB_CYCLES_PER_EPOCH, NB_CYCLES_PER_EPOCH * FREQ) / 1000
    x = np.arange(0, LAST_EP + 1, FREQ)

    artists, ax = setup_figure(xlabel='Cycles (x$10^3$)',
                               ylabel='# Stepping Stones',
                               xlim=[-1, LIM],
                               ylim=[-0.02, 8100])

    for i in range(len(conditions)):
        plt.plot(x_eps, st_per_cond_stats[i, x, 0], color=colors(3*i), marker=MARKERS[i], markersize=MARKERSIZE, linewidth=LINEWIDTH)
        plt.fill_between(x_eps, st_per_cond_stats[i, x, 1], st_per_cond_stats[i, x, 2], color=colors(3*i), alpha=ALPHA)

    if labels is None:
        labels = conditions
    leg = plt.legend(labels,
                     loc='upper center',
                     bbox_to_anchor=(0.5, 1.1),
                     ncol=4,
                     fancybox=True,
                     shadow=True,
                     prop={'size': 40, 'weight': 'bold'},
                     markerscale=1,
                     )
    for l in leg.get_lines():
        l.set_linewidth(7.0)
    artists += (leg,)
    ax.set_yticks([0, 2000, 4000, 6000, 8000])
    plt.grid()
    save_fig(directory=SAVE_PATH, filename=PLOT + '.pdf', artists=artists)
    return st_per_cond_stats.copy()

def plot_speed_discovery(experiment_path, max_len, max_seeds, conditions=None, labels=None):
    if conditions is None:
        conditions = os.listdir(experiment_path)
    st = np.zeros([max_seeds, len(conditions), LAST_EP + 1 ])
    st.fill(np.nan)
    colors = get_cmap(3*len(conditions))
    for i_cond, cond in enumerate(conditions):
        cond_path = experiment_path + cond + '/'
        list_runs = sorted(os.listdir(cond_path))
        for i_run, run in enumerate(list_runs):
            run_path = cond_path + run + '/'
            data_run = pd.read_csv(run_path + 'progress.csv')
            # all_sr = np.mean(np.array([data_run['stepping_stones_len'][:LAST_EP + 1]]), axis=0)
            l = [data_run['stepping_stones_len'][count+1] - data_run['stepping_stones_len'][count] for count in range(LAST_EP)]
            l2 = [l[count+1] - l[count] for count in range(LAST_EP-1)]
            all_sr = np.array(l2)
            st[i_run, i_cond, :all_sr.size] = all_sr.copy()


    st_per_cond_stats = np.zeros([len(conditions), LAST_EP + 1, 3])
    st_per_cond_stats[:, :, 0] = line(st)
    st_per_cond_stats[:, :, 1] = err_min(st)
    st_per_cond_stats[:, :, 2] = err_plus(st)


    x_eps = np.arange(0, (LAST_EP + 1) * NB_CYCLES_PER_EPOCH, NB_CYCLES_PER_EPOCH * FREQ) / 1000
    x = np.arange(0, LAST_EP + 1, FREQ)

    artists, ax = setup_figure(xlabel='Cycles (x$10^3$)',
                               ylabel='# Stepping Stones',
                               xlim=[-1, LIM],
                               ylim=[-1000, 1000])

    for i in range(len(conditions)):
        plt.plot(x_eps, st_per_cond_stats[i, x, 0], color=colors(i), marker=MARKERS[i], markersize=MARKERSIZE, linewidth=LINEWIDTH)
        plt.fill_between(x_eps, st_per_cond_stats[i, x, 1], st_per_cond_stats[i, x, 2], color=colors(i), alpha=ALPHA)

    if labels is None:
        labels = conditions
    leg = plt.legend(labels,
                     loc='upper center',
                     bbox_to_anchor=(0.5, 1.1),
                     ncol=4,
                     fancybox=True,
                     shadow=True,
                     prop={'size': 40, 'weight': 'bold'},
                     markerscale=1,
                     )
    for l in leg.get_lines():
        l.set_linewidth(7.0)
    artists += (leg,)
    # ax.set_yticks([0, 20, 40, 60, 80])
    plt.grid()
    save_fig(directory=SAVE_PATH, filename=PLOT + '.pdf', artists=artists)
    return st_per_cond_stats.copy()

def get_mean_sr(experiment_path, max_len, max_seeds):
    conditions = os.listdir(experiment_path)
    conditions.sort()
    labels = [input('Enter label for condition {}: '.format(c)) for c in conditions]
    ref = input('Enter reference agent: ')
    while ref not in conditions:
        print('Reference agent not recognized, please enter an agent within {}'.format(conditions))
        ref = input('Enter reference agent: ')
    sr = np.zeros([max_seeds, len(conditions), LAST_EP + 1 ])
    sr.fill(np.nan)
    colors = get_cmap(9, name='Set1')
    for i_cond, cond in enumerate(conditions):
        if cond == ref:
            ref_id = i_cond
        cond_path = experiment_path + cond + '/'
        list_runs = sorted(os.listdir(cond_path))
        for i_run, run in enumerate(list_runs):
            run_path = cond_path + run + '/'
            data_run = pd.read_csv(run_path + 'progress.csv')
            all_sr = np.mean(np.array([data_run['Eval_SR_{}'.format(i+1)][:LAST_EP + 1] for i in range(NB_CLASSES)]), axis=0)
            sr[i_run, i_cond, :all_sr.size] = all_sr.copy()


    sr_per_cond_stats = np.zeros([len(conditions), LAST_EP + 1, 3])
    sr_per_cond_stats[:, :, 0] = line(sr)
    sr_per_cond_stats[:, :, 1] = err_min(sr)
    sr_per_cond_stats[:, :, 2] = err_plus(sr)


    x_eps = np.arange(0, (LAST_EP + 1) * NB_CYCLES_PER_EPOCH, NB_CYCLES_PER_EPOCH * FREQ) / 1000
    x = np.arange(0, LAST_EP + 1, FREQ)
    # compute p value wrt ref id
    p_vals = dict()
    for i_cond in range(len(conditions)):
        if i_cond != ref_id:
            p_vals[i_cond] = []
            for i in x:
                ref_inds = np.argwhere(~np.isnan(sr[:, ref_id, i])).flatten()
                other_inds = np.argwhere(~np.isnan(sr[:, i_cond, i])).flatten()
                if ref_inds.size > 1 and other_inds.size > 1:
                    ref = sr[:, ref_id, i][ref_inds]
                    other = sr[:, i_cond, i][other_inds]
                    p_vals[i_cond].append(ttest_ind(ref, other, equal_var=False)[1])
                else:
                    p_vals[i_cond].append(1)

    artists, ax = setup_figure(xlabel='Cycles (x$10^3$)',
                               ylabel='Success Rate',
                               xlim=[-1, LIM],
                               ylim=[-0.02, 1. + 0.04 * (len(conditions)-1)])

    for i in range(len(conditions)):
        plt.plot(x_eps, sr_per_cond_stats[i, x, 0], color=colors(i), marker=MARKERS[i], markersize=MARKERSIZE, linewidth=LINEWIDTH)
        plt.fill_between(x_eps, sr_per_cond_stats[i, x, 1], sr_per_cond_stats[i, x, 2], color=colors(i), alpha=ALPHA)
    for i_cond in range(len(conditions)):
        if i_cond != ref_id:
            inds_sign = np.argwhere(np.array(p_vals[i_cond]) < ALPHA_TEST).flatten()
            if inds_sign.size > 0:
                plt.scatter(x=x_eps[inds_sign], y=np.ones([inds_sign.size]) + 0.05 * (i_cond), marker='*', color=colors(i_cond), s=1300)
    if labels is None:
        labels = conditions
    leg = plt.legend(labels,
                     loc='upper center',
                     bbox_to_anchor=(0.47, 1.1),
                     ncol=7,
                     fancybox=True,
                     shadow=True,
                     prop={'size': 30, 'weight': 'bold'},
                     markerscale=1,
                     )
    for l in leg.get_lines():
        l.set_linewidth(7.0)
    artists += (leg,)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    plt.grid()
    save_fig(directory=SAVE_PATH, filename=PLOT + '.pdf', artists=artists)
    return sr_per_cond_stats.copy()

if __name__ == '__main__':

    for PLOT in TO_PLOT:
        print('\n\tPlotting', PLOT)
        experiment_path = RESULTS_PATH + STUDY + '/'


        max_len, max_seeds, min_len, min_seeds = check_length_and_seeds(experiment_path=experiment_path)

        if PLOT == 'main':
            # conditions = ['SP_0.0%', 'SP_0.1%', 'SP_0.2%', 'SP_0.5%', 'SP_5%', 'SP_20%', 'SP_100%']
            # labels = ['SP_0.0%', 'SP_0.1%', 'SP_0.2%', 'SP_0.5%', 'SP_5%', 'SP_20%', 'SP_100%']
            # get_mean_sr(experiment_path, max_len, max_seeds, conditions, labels, ref='SP_0.2%')
            conditions = ['no_intern', 'intern_ss', 'intern_beyond', 'SP_0.2%']
            labels = ['no_intern', 'intern_ss', 'intern_beyond', 'SP_0.2%']
            get_mean_sr(experiment_path, max_len, max_seeds)
        elif PLOT == 'goals':
            # conditions = ['SP_5.0%', 'SP_5.0%_no_internalization']
            # labels = ['SP_5.0%', 'SP_5.0% B']
            # conditions = ['intern_beyond', 'SP_0.2%']
            # labels = ['intern_beyond', 'SP_0.2%']
            plot_classes(experiment_path)
        elif PLOT == 'ablations':
            # conditions = ['SP_0%', 'Frontier', 'Frontier_Beyond', 'SP_20%']
            # labels = ['SP 0%', 'SP 20% Frontier and Stop', 'SP 20% Frontier and Beyond', 'SP 20%']
            conditions = ['Beyond', 'Frontier_Beyond']
            labels = ['SP 20% Beyond', 'SP 20% Frontier and Beyond']
            get_mean_sr(experiment_path, max_len, max_seeds, conditions, labels, ref='Beyond')
        elif PLOT == 'learning_trajectory':
            conditions = ['SP_0.0%']
            plot_counts_and_sr(experiment_path, conditions)
        elif PLOT == 'stepping_stones':
            conditions = ['SP_0.2%', 'SP_0.0%']
            labels = ['SP_0.2%', 'SP_0.0%']
            plot_stepping_stones(experiment_path, max_len, max_seeds, conditions, labels)

        elif PLOT == 'speed':
            conditions = ['SP_20%', 'SP_0.2%', 'SP_0.0%']
            labels = ['SP_20%', 'SP_0.2%', 'SP_0.0%']
            plot_speed_discovery(experiment_path, max_len, max_seeds, conditions, labels)
