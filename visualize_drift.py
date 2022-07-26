import  matplotlib.pyplot as plt
import seaborn as sns

# globals:
n_divs = 30
n_repeats = 10
n_repeat_plot = 5
n_sessions = 30

def plot_corr(corrs, n_repeat_plot = 5):
    _, ax = plt.subplots(1, 1, figsize=(9.5,8))

    sns.heatmap(corrs[:n_repeat_plot*n_divs, :n_repeat_plot*n_divs], 
                cmap = 'PRGn', vmin = -1, vmax = 1,
                ax = ax)

    tick_locs = [n_divs/2 + i * n_divs for i in range(n_repeat_plot)]
    ax.set_xticks(tick_locs)
    ax.set_yticks(tick_locs)
    
    if n_repeat_plot == 5:
        tick_labels = [1, None, 3, None, 5]
        
    elif n_repeat_plot == 10:
        tick_labels = [1, None, 3, None, 5, None, 7, None, 9, None]

    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel('Repeat', fontsize = 14)
    ax.set_ylabel('Repeat', fontsize = 14)
    ax.set_title('Pearson Correlation', fontsize = 18)

    for repeat_idx in range(1, n_repeat_plot):
        ax.axhline((repeat_idx * n_divs)-0.5, color='k', linewidth=0.5)
        ax.axvline((repeat_idx * n_divs)-0.5, color='k', linewidth=0.5)
    
    plt.show()

def plot_angle(angles, n_repeat_plot = 5):

    _, ax = plt.subplots(1, 1, figsize=(9.5,8))

    sns.heatmap(angles[:n_repeat_plot*n_divs, :n_repeat_plot*n_divs], 
                cmap = 'PRGn', vmin = 0, vmax = 180,
                ax = ax)

    tick_locs = [n_divs/2 + i * n_divs for i in range(n_repeat_plot)]
    ax.set_xticks(tick_locs)
    ax.set_yticks(tick_locs)
    
    if n_repeat_plot == 5:
        tick_labels = [1, None, 3, None, 5]
        
    elif n_repeat_plot == 10:
        tick_labels = [1, None, 3, None, 5, None, 7, None, 9, None]

    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel('Repeat', fontsize = 14)
    ax.set_ylabel('Repeat', fontsize = 14)
    ax.set_title('Angles between response vectors', fontsize = 18)

    for repeat_idx in range(1, n_repeat_plot):
        ax.axhline((repeat_idx * n_divs)-0.5, color='k', linewidth=0.5)
        ax.axvline((repeat_idx * n_divs)-0.5, color='k', linewidth=0.5)
    
    plt.show()



def plot_cross_angle(across_session_angles, ordered_days): 

    fig, ax = plt.subplots(1, 1, figsize=(9.5 , 8))


    sns.heatmap(across_session_angles, 
                    cmap = 'PRGn', vmin = 0, vmax = 180,
                    ax = ax)

    tick_locs = [15, 45, 75]
    ax.set_xticks(tick_locs)
    ax.set_yticks(tick_locs)
    tick_labels = [str(age) for age in ordered_days]
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)

    ax.set_xlabel('Age (days)')
    ax.set_ylabel('Age (days)')

    for session_idx in range(1, n_sessions):
        ax.axhline((session_idx * n_divs), color='k', linewidth=0.5)
        ax.axvline((session_idx * n_divs), color='k', linewidth=0.5)



def plot_cross_corr(across_session_corrs, ordered_days): 

    fig, ax = plt.subplots(1, 1, figsize=(9.5 , 8))


    sns.heatmap(across_session_corrs, 
                    cmap = 'PRGn', vmin = -1, vmax = 1,
                    ax = ax)

    tick_locs = [15, 45, 75]
    ax.set_xticks(tick_locs)
    ax.set_yticks(tick_locs)
    tick_labels = [str(age) for age in ordered_days]
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)

    ax.set_xlabel('Age (days)')
    ax.set_ylabel('Age (days)')

    for session_idx in range(1, n_sessions):
        ax.axhline((session_idx * n_divs), color='k', linewidth=0.5)
        ax.axvline((session_idx * n_divs), color='k', linewidth=0.5)