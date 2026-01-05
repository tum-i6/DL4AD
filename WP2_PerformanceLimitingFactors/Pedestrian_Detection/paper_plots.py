import pickle
import copy
import fiftyone as fo
from fiftyone import ViewField as F
import matplotlib.pyplot as plt
import matplotlib.markers as mmark
import matplotlib.lines as mlines
import numpy as np
import yaml
import matplotlib.patches as patches

import statsmodels.api as sm

plt.rcParams["font.family"] = "Times New Roman"
cp_plfs=['brightness', 'contrast', 'edge_strength', 'bbox_height', 'bbox_aspect_ratio', 'visible_instance_pixels', 'occlusion_ratio', 'distance', 'foreground_brightness', 'contrast_to_background', 'entropy', 'background_edge_strength', 'boundary_edge_strength', 'crowdedness']
zero_plfs=['fog_intensity', 'lens_flare_intensity', 'vignette_intensity', 'occlusion_ratio', 'crowdedness']
colors=['#1e81fa','#c21d04','#cfc102', '#59ba04','#cfc102', '#59ba04','#c21d04','#cfc102', '#59ba04','#cfc102', '#59ba04','#c21d04','#cfc102', '#59ba04','#cfc102', '#59ba04','#c21d04','#cfc102', '#59ba04','#cfc102', '#59ba04','#c21d04','#cfc102', '#59ba04','#cfc102', '#59ba04']
plf_names={
'brightness':'Brightness',
'contrast':'Contrast',
'edge_strength': 'Edge Strength',
'bbox_height':'BBox Height',
'bbox_aspect_ratio':'BBox Asp. Ratio',
'visible_instance_pixels': 'Vis. Ins. Pix.',
'occlusion_ratio':'Occlusion',
'distance':'Distance',
'foreground_brightness':'Fore. Brightness',
'contrast_to_background': 'Contrast To BG',
'entropy': 'Entropy',
'background_edge_strength': 'BG Edge Str.',
'boundary_edge_strength': 'Boundary Edge Str.',
'crowdedness': 'Crowdedness',
'fog_intensity':'Fog',
'lens_flare_intensity':'Lens Flare',
'vignette_intensity':'Vignette',
'daytime_type': 'Daytime',
'sky_type': 'Sky Type',
'wetness_type': 'Wetness',
'truncated': 'Truncated'
}
markers={
'FasterRCNN-ResNet50-OD':'*',
'MaskRCNN-ResNet50-IS':'^',
'KeypointRCNN-ResNet50-KD':'o',
'FCOS-ResNet50-OD': 'h',
'RetinaNet-ResNet50-OD':'x',
'SSD300-ResNet50-OD':'D'
}

# Creates a new data structure for tracking the detection performance, averaged over all models and
# w.r. to the PLF values
#avg_plf_performance = copy.deepcopy(plf_performance[next(iter(plf_performance))])
def calculate_avg_recall(plf_performance, plf, plf_comp_source, plf_type, data_set='kia'):
    if plf_comp_source=='sample':
        metric='F1'
    else:
        metric='recall'
    values = []
    categories=''
    for _model in plf_performance:
        if plf_type=='numerical':

            recall = plf_performance[_model][plf_comp_source][plf_type][plf][metric]
            srtps = plf_performance[_model][plf_comp_source][plf_type][plf]['srtps']
            fns = plf_performance[_model][plf_comp_source][plf_type][plf]['fns']

            # Computes a histogram of occurences for each PLF value
            hist_values = (srtps + fns) / (np.sum(srtps) + np.sum(fns))
            # Defines the x axis, which contains values between 0 and 1
            x_axis_values = np.array([i / 100 for i in range(101)])


        else:
            categories = [cat for cat in plf_performance[_model][plf_comp_source][plf_type][plf]['tps']]
            if plf == 'daytime_type':
                categories=['day', 'medium', 'low']
            recall=[]
            srtps=[]
            fns=[]
            for cat in categories:
                recall.append(plf_performance[_model][plf_comp_source][plf_type][plf][metric][cat])
                srtps.append(plf_performance[_model][plf_comp_source][plf_type][plf]['srtps'][cat])
                fns.append(plf_performance[_model][plf_comp_source][plf_type][plf]['fns'][cat])

            srtps = np.array(srtps)
            fns = np.array(fns)
            recall = np.array(recall)

            # Computes a density histogram for each PLF category
            hist_values = (np.array(srtps) + np.array(fns)) / (np.sum(np.array(srtps)) + np.sum(np.array(fns)))
            # Defines the x axis, which will contain category names
            x_axis_values = np.arange(len(categories))

        # Removes all results for plf values that have a lower number of samples than specified in the configurations by the min_plf_samples variable
        if data_set == 'kia':
            valid_plf_values = np.where((srtps + fns) >= cfg['min_plf_samples'], 1, np.nan)
        else:
            valid_plf_values = np.where((srtps + fns) >= cfg['min_plf_samples']/2, 1, np.nan)
        hist_values *= valid_plf_values
        recall *= valid_plf_values

        values.append(recall)

    values = np.array(values)
    if plf in zero_plfs:
        values=values[:,1:]
        x_axis_values=x_axis_values[1:]
        hist_values=hist_values[1:]
    recalls_means = np.nanmean(values, axis=0)

    std = np.std(values, axis=0)
    recalls_mins = np.nanmin(values, axis=0)

    recalls_maxs = np.nanmax(values, axis=0)


    # Computes a local regression function (which will later serve as a trend line) to estimate the F1-score based on the PLF value
    isnan_mask = np.isnan(recalls_means)
    trend_line = sm.nonparametric.lowess(recalls_means[~isnan_mask], x_axis_values[~isnan_mask], frac=1 / 5)
    #recalls_maxs = trend_line[:,1]*0.5 + recalls_maxs[~isnan_mask]*0.5
    #recalls_mins = trend_line[:,1]*0.5 + recalls_mins[~isnan_mask]*0.5
    recalls_maxs =sm.nonparametric.lowess(recalls_maxs[~isnan_mask], x_axis_values[~isnan_mask], frac=1 / 7)[:,1]
    recalls_mins =sm.nonparametric.lowess(recalls_mins[~isnan_mask], x_axis_values[~isnan_mask], frac=1 / 7)[:,1]
    return trend_line, x_axis_values, hist_values, isnan_mask, recalls_mins, recalls_maxs, categories, recalls_means
def plot_single_lines(ax, plf_performance, plf, plf_comp_source, plf_type, data_set='kia', color='darkred'):
    recalls = []
    _markers=[]
    if plf_comp_source=='sample':
        metric='F1'
    else:
        metric='recall'
    for i, _model in enumerate(plf_performance):
        if plf_type == 'numerical':

            values = plf_performance[_model][plf_comp_source][plf_type][plf][metric]
            srtps = plf_performance[_model][plf_comp_source][plf_type][plf]['srtps']
            fns = plf_performance[_model][plf_comp_source][plf_type][plf]['fns']

            # Computes a histogram of occurences for each PLF value
            hist_values = (srtps + fns) / (np.sum(srtps) + np.sum(fns))
            # Defines the x axis, which contains values between 0 and 1
            x_axis_values = np.array([i / 100 for i in range(101)])


        else:
            categories = [cat for cat in plf_performance[_model][plf_comp_source][plf_type][plf]['tps']]
            if plf == 'daytime_type':
                categories=['day', 'medium', 'low']
            values = []
            srtps = []
            fns = []
            for cat in categories:
                values.append(plf_performance[_model][plf_comp_source][plf_type][plf][metric][cat])
                srtps.append(plf_performance[_model][plf_comp_source][plf_type][plf]['srtps'][cat])
                fns.append(plf_performance[_model][plf_comp_source][plf_type][plf]['fns'][cat])

            srtps = np.array(srtps)
            fns = np.array(fns)
            values = np.array(values)

            # Computes a density histogram for each PLF category
            hist_values = (np.array(srtps) + np.array(fns)) / (np.sum(np.array(srtps)) + np.sum(np.array(fns)))
            # Defines the x axis, which will contain category names
            x_axis_values = np.arange(len(categories))

        # Removes all results for plf values that have a lower number of samples than specified in the configurations by the min_plf_samples variable
        if data_set == 'kia':
            valid_plf_values = np.where((srtps + fns) >= cfg['min_plf_samples'], 1, np.nan)
        else:
            valid_plf_values = np.where((srtps + fns) >= cfg['min_plf_samples'] / 2, 1, np.nan)
        hist_values *= valid_plf_values
        values *= valid_plf_values




        if plf in zero_plfs:
            values = values[1:]
            x_axis_values = x_axis_values[1:]
            hist_values = hist_values[1:]






        # Computes a local regression function (which will later serve as a trend line) to estimate the F1-score based on the PLF value
        isnan_mask = np.isnan(values)
        trend_line = sm.nonparametric.lowess(values[~isnan_mask], x_axis_values[~isnan_mask], frac=1 / 5)
        if plf=='daytime_type':
            print('stop')
        _marker=ax.plot(trend_line[:, 0], trend_line[:, 1], color, linestyle='--',marker=markers[_model], label=_model.split('-')[0],
                 linewidth=1, alpha=0.5)
        _markers.append(_marker[0])

    return _markers


def single_plot_hist_per_model(plf, plf_comp_source, plf_type):


    trend_line_kia, x_axis_values_kia, hist_values_kia, isnan_mask_kia, recalls_mins_kia, recalls_maxs_kia, categories,recalls_means=calculate_avg_recall(plf_performance_kia, plf, plf_comp_source, plf_type, data_set='kia')
    if plf in cp_plfs:
        trend_line_cp, x_axis_values_cp, hist_values_cp, isnan_mask_cp, recalls_mins_cp, recalls_maxs_cp, _,recalls_means=calculate_avg_recall(plf_performance_cp, plf, plf_comp_source, plf_type, data_set='cp')

    x_values = np.linspace(0, 1, 100)

    # Generates the plot
    # fig = plt.figure(figsize=(2008/300, 1204.8/300), dpi=300)

    fig = plt.figure(figsize=(10, 4.8), dpi=300)
    #gs = fig.add_gridspec(bottom=0.05)
    fig.patch.set_facecolor((1, 1, 1))
    ax1 = fig.add_subplot(111)
    if plf_type=='numerical':
        ax1.set_xticks([r * 0.1 for r in range(11)], fontsize=15)
        bar_width=0.007
        ax1.set_xlabel(f"Normalized {' '.join(plf.split('_')).title()}", fontsize=15)
    else:
        ax1.set_xticks(x_axis_values_kia, fontsize=15)
        ax1.set_xticklabels(categories)
        bar_width = 0.7 / len(categories)
        ax1.set_xlabel(f"{' '.join(plf.split('_')).title()}", fontsize=15)
    ax1.set_yticks([r * 0.1 for r in range(11)], fontsize=15)
    # Plots the histogram values on the second y-axis
    ax2 = ax1.twinx()
    hist_ticks = np.nanmax(hist_values_kia)
    hist_bars_kia = ax2.bar(x_axis_values_kia, hist_values_kia, width=bar_width, color='orangered', alpha=0.8, label='Density Histogram KI-A')
    if plf in cp_plfs:
        hist_bars_cp = ax2.bar(x_axis_values_cp, hist_values_cp, width=0.007, color='deepskyblue', alpha=0.8, label='Density Histogram CP')
        hist_ticks = max(hist_ticks, np.nanmax(hist_values_cp))
    ax2.set_ylabel('Density Histogram', fontsize=15)
    #ax2.set_yticks([r * hist_ticks / 10 for r in range(11)], fontsize=15)
    #recall_plt = ax1.scatter(x_axis_values, recalls_means, color='#ff7f0e', marker='o', label='PLF Recall')

    #trend_line_plt1 = ax1.plot(trend_line_kia[:, 0], trend_line_kia[:, 1], 'darkred', linestyle='-', label='Recall Trend Line KI-A', linewidth=3)
    __markers=plot_single_lines(ax1, plf_performance_kia, plf, plf_comp_source, plf_type, data_set='kia', color='darkred')

    if plf in cp_plfs:
        #trend_line_plt2 = ax1.plot(trend_line_cp[:, 0], trend_line_cp[:, 1], 'dodgerblue', linestyle='-', label='Recall Trend Line CP', linewidth=3)
        plot_single_lines(ax1, plf_performance_cp, plf, plf_comp_source, plf_type, data_set='cp', color='dodgerblue')


    ax1.set_ylim([0, 1.03])
    if plf_comp_source=='sample':
        ax1.set_ylabel('F1 Score', fontsize=15)
    else:
        ax1.set_ylabel('Recall', fontsize=15)
    ax1.grid(zorder=0, color='gray', linestyle='dashed', alpha=0.5)
    box = ax1.get_position()
    box.y0 = box.y0 + 0.55
    box.y1 = box.y1 + 0.55
    ax1.set_position(box)
    ax1.set_zorder(2)
    ax2.set_zorder(1)
    ax1.patch.set_visible(False)


    #for model in plf_performance_kia:
    #    print(model)


    # Add a legend under the plot
    #if plf in cp_plfs:
        #lns = trend_line_plt1+trend_line_plt2  + [hist_bars_kia, hist_bars_cp]
   # else:
        #lns = trend_line_plt1+[hist_bars_kia]

    if plf in cp_plfs:

        lns = [hist_bars_kia, hist_bars_cp]+__markers
    else:
        lns = [hist_bars_kia]+__markers
    labs = [l.get_label() for l in lns]
    leg=plt.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=4)
    color='black'
    leg.legendHandles[2].set_color(color)
    leg.legendHandles[3].set_color(color)
    leg.legendHandles[4].set_color(color)
    leg.legendHandles[5].set_color(color)
    leg.legendHandles[6].set_color(color)
    if plf in cp_plfs:
        leg.legendHandles[7].set_color(color)
    fig.subplots_adjust(bottom=0.18)

    plt.title(f"{' '.join(plf.replace('bbox','bounding_box').split('_')).title()}", fontsize=20)
    #plt.grid(zorder=0, color='gray', linestyle='dashed', alpha=0.5)
    plt.locator_params(axis='y', nbins=10)
    plt.subplots_adjust(bottom=0.2, left=0.08, right=0.92, top=0.9)
    #plt.show()
    plt.savefig(f"/path/to/plfs/{'_'.join(plf.split('_')).title()}_per_model.png")
def single_plot_hist_min_max(plf, plf_comp_source, plf_type):


    trend_line_kia, x_axis_values_kia, hist_values_kia, isnan_mask_kia, recalls_mins_kia, recalls_maxs_kia, categories,recalls_means=calculate_avg_recall(plf_performance_kia, plf, plf_comp_source, plf_type, data_set='kia')
    if plf in cp_plfs:
        trend_line_cp, x_axis_values_cp, hist_values_cp, isnan_mask_cp, recalls_mins_cp, recalls_maxs_cp, _,recalls_means=calculate_avg_recall(plf_performance_cp, plf, plf_comp_source, plf_type, data_set='cp')

    x_values = np.linspace(0, 1, 100)

    # Generates the plot
    # fig = plt.figure(figsize=(2008/300, 1204.8/300), dpi=300)

    fig = plt.figure(figsize=(10, 4.8), dpi=300)

    fig.patch.set_facecolor((1, 1, 1))
    ax1 = fig.add_subplot(111)
    if plf_type=='numerical':
        ax1.set_xticks([r * 0.1 for r in range(11)], fontsize=15)
        bar_width=0.007
        ax1.set_xlabel(f"Normalized {' '.join(plf.split('_')).title()}", fontsize=15)
    else:
        ax1.set_xticks(x_axis_values_kia, fontsize=15)
        ax1.set_xticklabels(categories)
        bar_width = 0.7 / len(categories)
        ax1.set_xlabel(f"{' '.join(plf.split('_')).title()}", fontsize=15)
    ax1.set_yticks([r * 0.1 for r in range(11)], fontsize=15)
    # Plots the histogram values on the second y-axis
    ax2 = ax1.twinx()
    hist_ticks = np.nanmax(hist_values_kia)
    hist_bars_kia = ax2.bar(x_axis_values_kia, hist_values_kia, width=bar_width, color='orangered', alpha=0.8, label='Density Histogram KI-A')
    if plf in cp_plfs:
        hist_bars_cp = ax2.bar(x_axis_values_cp, hist_values_cp, width=0.007, color='deepskyblue', alpha=0.8, label='Density Histogram CP')
        hist_ticks = max(hist_ticks, np.nanmax(hist_values_cp))
    ax2.set_ylabel('Density Histogram', fontsize=15)
    #ax2.set_yticks([r * hist_ticks / 10 for r in range(11)], fontsize=15)
    #recall_plt = ax1.scatter(x_axis_values, recalls_means, color='#ff7f0e', marker='o', label='PLF Recall')

    trend_line_plt1 = ax1.plot(trend_line_kia[:, 0], trend_line_kia[:, 1], 'darkred', linestyle='-', label='Recall Trend Line KI-A', linewidth=3)
    ax1.fill_between(trend_line_kia[:, 0], trend_line_kia[:, 1], recalls_maxs_kia, facecolor='red', alpha=0.1)
    ax1.fill_between(trend_line_kia[:, 0], trend_line_kia[:, 1], recalls_mins_kia, facecolor='red', alpha=0.1)

    if plf in cp_plfs:
        trend_line_plt2 = ax1.plot(trend_line_cp[:, 0], trend_line_cp[:, 1], 'dodgerblue', linestyle='-',
                                  label='Recall Trend Line CP', linewidth=3)
        ax1.fill_between(trend_line_cp[:, 0], trend_line_cp[:, 1], recalls_maxs_cp,
                         facecolor='deepskyblue', alpha=0.1)
        ax1.fill_between(trend_line_cp[:, 0], trend_line_cp[:, 1], recalls_mins_cp,
                         facecolor='deepskyblue', alpha=0.1)


    ax1.set_ylim([0, 1.03])
    ax1.set_ylabel('Recall', fontsize=15)
    ax1.grid(zorder=0, color='gray', linestyle='dashed', alpha=0.5)






    # Add a legend under the plot
    if plf in cp_plfs:
        lns = trend_line_plt1+trend_line_plt2  + [hist_bars_kia, hist_bars_cp]
    else:
        lns = trend_line_plt1+[hist_bars_kia]
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    fig.subplots_adjust(bottom=0.18)

    plt.title(f"{' '.join(plf.replace('bbox','bounding_box').split('_')).title()}", fontsize=20)
    #plt.grid(zorder=0, color='gray', linestyle='dashed', alpha=0.5)
    plt.locator_params(axis='y', nbins=10)
    #plt.show()
    plt.savefig(f"/path/to//plfs/{'_'.join(plf.split('_')).title()}.png")

def compute_avg_plf_performance(plf_performance):
    """
    Uses the aggregated prediction performance values to compute the average prediction performance over all models.

    :param plf_performance: A dictionary for tracking the detection performance for each model with respect to the
    PLF values.
    :param avg_plf_performance: A dictionary for tracking the detection performance averaged over all models and with
    respect to the PLF values.
    :param prediction_fields: Dictionary, which maps the names of the detection models to the corresponding FiftyOne
    label field names.
    :return: Returns the avg_plf_performance dictionary with the final detection performance values with respect to the
    PLFs.
    """
    # Iterates over all specified detection models
    for model in plf_performance:
        # Iterates over all computation sources ('sample' or 'object')
        for plf_comp_source in plf_performance[model]:
            # Iterates over all PLF types ('numerical' or 'categorical')
            for plf_type in plf_performance[model][plf_comp_source]:
                # Iterates over all PLFs
                for plf in plf_performance[model][plf_comp_source][plf_type]:
                    #if plf_type == 'numerical':
                        # Plots the numerical PLFs
                        #if plf_comp_source == 'sample':
                            #plot_numerical_sample_plf(model, plf_comp_source, plf_type, plf, plf_performance)
                    print(plf, plf_comp_source, plf_type)
                    #single_plot_hist_min_max(plf, plf_comp_source, plf_type)
                    #if plf!='occlusion_ratio':
                    #   continue
                    single_plot_hist_per_model(plf, plf_comp_source, plf_type)


                        #elif plf_comp_source == 'object':
                            #plot_numerical_object_plf(model, plf_comp_source, plf_type, plf, plf_performance)
                    #elif plf_type == 'categorical':
                        #pass
                        # Plots the categorical PLFs
                        #if plf_comp_source == 'sample':
                            #plot_categorical_sample_plf(model, plf_comp_source, plf_type, plf, plf_performance)
                        #elif plf_comp_source == 'object':
                            #plot_categorical_object_plf(model, plf_comp_source, plf_type, plf, plf_performance)

        break

def generate_subplot(data, ax,title, plf, y_label='Recall'):
    ax.grid(zorder=1, linewidth=0.5, linestyle='dashed', alpha=0.5)
    #ax.set_zorder(1000)
    ax.set_xticks([r * 0.1 for r in range(11)], fontsize=15)
    bar_width = 0.007
    ax.set_yticks([r * 0.1 for r in range(11)], fontsize=15)
    # Plots the histogram values on the second y-axis
    ax2 = ax.twinx()
    hist_ticks = np.nanmax(data['hist_values_kia'])
    hist_bars1 = ax2.bar(data['x_axis_values_kia'], data['hist_values_kia'], width=bar_width, color='orangered', alpha=0.8,
                        label='PLF Density Histogram KI-A')
    if plf in cp_plfs:
        hist_bars2 = ax2.bar(data['x_axis_values_cp'], data['hist_values_cp'], width=0.007, color='deepskyblue', alpha=0.7,
                            label='PLF Density Histogram CP')
        hist_ticks = max(hist_ticks, np.nanmax(data['hist_values_cp']))
    ax2.set_ylabel('Density Histogram')

    trend_line_plt1 = ax.plot(data['trend_line_kia'][:, 0], data['trend_line_kia'][:, 1], 'darkred', linestyle='-',
                              label='Recall Trend Line KI-A', linewidth=3)
    ax.fill_between(data['trend_line_kia'][:, 0], data['trend_line_kia'][:, 1], data['recalls_maxs_kia'], facecolor='red', alpha=0.1)
    ax.fill_between(data['trend_line_kia'][:, 0], data['trend_line_kia'][:, 1], data['recalls_mins_kia'], facecolor='red', alpha=0.1)

    if plf in cp_plfs:
        trend_line_plt2=ax.plot(data['trend_line_cp'][:, 0], data['trend_line_cp'][:, 1], 'dodgerblue', linestyle='-',
                 label='Recall Trend Line CP', linewidth=3)
        ax.fill_between(data['trend_line_cp'][:, 0], data['trend_line_cp'][:, 1], data['recalls_maxs_cp'],
                         facecolor='deepskyblue', alpha=0.1)
        ax.fill_between(data['trend_line_cp'][:, 0], data['trend_line_cp'][:, 1], data['recalls_mins_cp'],
                         facecolor='deepskyblue', alpha=0.1)

    ax.set_xlabel(f"Normalized {title}")
    ax.set_ylim([0, 1.03])
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=20)
    if plf in cp_plfs:
        return [hist_bars1,hist_bars2], trend_line_plt1+ trend_line_plt2
    else:
        return [hist_bars1], trend_line_plt1



def paper_final_6x2():
    pro_plfs=['occlusion_ratio', 'distance', 'boundary_edge_strength']
    contra_plfs=['brightness', 'contrast', 'contrast_to_background']
    # Iterates over all specified detection models
    plfs_dict={}


    for model in plf_performance_kia:
        # Iterates over all computation sources ('sample' or 'object')
        for plf_comp_source in plf_performance_kia[model]:
            # Iterates over all PLF types ('numerical' or 'categorical')
            for plf_type in plf_performance_kia[model][plf_comp_source]:
                # Iterates over all PLFs
                for plf in plf_performance_kia[model][plf_comp_source][plf_type]:
                    if plf in pro_plfs or plf in contra_plfs:
                        trend_line_kia, x_axis_values_kia, hist_values_kia, isnan_mask_kia, recalls_mins_kia, recalls_maxs_kia, categories, recalls_kia = calculate_avg_recall(
                            plf_performance_kia, plf, plf_comp_source, plf_type, data_set='kia')
                        plfs_dict[plf]={'trend_line_kia':trend_line_kia,
                                            'x_axis_values_kia':x_axis_values_kia,
                                            'hist_values_kia':hist_values_kia,
                                            'isnan_mask_kia':isnan_mask_kia,
                                            'recalls_mins_kia':recalls_mins_kia,
                                            'recalls_maxs_kia':recalls_maxs_kia,
                                            'categories':categories}
                        if plf in cp_plfs:
                            trend_line_cp, x_axis_values_cp, hist_values_cp, isnan_mask_cp, recalls_mins_cp, recalls_maxs_cp, _, recalls_cp = calculate_avg_recall(
                                plf_performance_cp, plf, plf_comp_source, plf_type, data_set='cp')
                            plfs_dict[plf]['trend_line_cp']=trend_line_cp
                            plfs_dict[plf]['x_axis_values_cp']=x_axis_values_cp
                            plfs_dict[plf]['hist_values_cp']=hist_values_cp
                            plfs_dict[plf]['isnan_mask_cp']=isnan_mask_cp
                            plfs_dict[plf]['recalls_mins_cp']=recalls_mins_cp
                            plfs_dict[plf]['recalls_maxs_cp']=recalls_maxs_cp

        break

    plt.rcParams["figure.figsize"] = (3600 / 300, 2000 / 300)
    plt.rcParams["figure.dpi"] = 300
    plt.subplots_adjust(left=0.00, right=0.99, top=0.95, bottom=0.1)
    fig, axs = plt.subplots(2, 3)
    hist_bars, trend_line_plt=generate_subplot(plfs_dict['occlusion_ratio'], axs[0, 0], 'Occlusion', 'occlusion_ratio')
    hist_bars, trend_line_plt=generate_subplot(plfs_dict['distance'], axs[0, 1], 'Distance', 'distance')
    hist_bars, trend_line_plt=generate_subplot(plfs_dict['boundary_edge_strength'], axs[0, 2], 'Boundary Edge Strength', 'boundary_edge_strength')
    hist_bars, trend_line_plt=generate_subplot(plfs_dict['brightness'], axs[1, 0], 'Brightness', 'brightness', y_label='F1 Score')
    hist_bars, trend_line_plt=generate_subplot(plfs_dict['contrast'], axs[1, 1], 'Contrast', 'contrast', y_label='F1 Score')
    hist_bars, trend_line_plt=generate_subplot(plfs_dict['contrast_to_background'], axs[1, 2], 'Contrast To Background', 'contrast_to_background')
    handles, labels = axs[0, 0].get_legend_handles_labels()
    lns = trend_line_plt + hist_bars
    labs = [l.get_label() for l in lns]
    fig.legend(lns, labs, borderaxespad=-1, fontsize=13, loc='lower center', ncol=4,
               bbox_to_anchor=(0, 0.065, 1, 0))
    for ax in axs.flat:
        #pass
        #ax.label_outer()
        ax.yaxis.set_tick_params(labelleft=True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.16)
    plt.savefig(f"/path/to//plfs/final_6x2.pdf")
    #plt.show()
def paper_final_3x3():
    pro_plfs=['occlusion_ratio', 'distance', 'boundary_edge_strength', 'fog_intensity', 'edge_strength', 'crowdedness']
    contra_plfs=['brightness', 'contrast', 'contrast_to_background']
    # Iterates over all specified detection models
    plfs_dict={}


    for model in plf_performance_kia:
        # Iterates over all computation sources ('sample' or 'object')
        for plf_comp_source in plf_performance_kia[model]:
            # Iterates over all PLF types ('numerical' or 'categorical')
            for plf_type in plf_performance_kia[model][plf_comp_source]:
                # Iterates over all PLFs
                for plf in plf_performance_kia[model][plf_comp_source][plf_type]:
                    if plf in pro_plfs or plf in contra_plfs:
                        trend_line_kia, x_axis_values_kia, hist_values_kia, isnan_mask_kia, recalls_mins_kia, recalls_maxs_kia, categories, recalls_kia = calculate_avg_recall(
                            plf_performance_kia, plf, plf_comp_source, plf_type, data_set='kia')
                        plfs_dict[plf]={'trend_line_kia':trend_line_kia,
                                            'x_axis_values_kia':x_axis_values_kia,
                                            'hist_values_kia':hist_values_kia,
                                            'isnan_mask_kia':isnan_mask_kia,
                                            'recalls_mins_kia':recalls_mins_kia,
                                            'recalls_maxs_kia':recalls_maxs_kia,
                                            'categories':categories}
                        if plf in cp_plfs:
                            trend_line_cp, x_axis_values_cp, hist_values_cp, isnan_mask_cp, recalls_mins_cp, recalls_maxs_cp, _, recalls_cp = calculate_avg_recall(
                                plf_performance_cp, plf, plf_comp_source, plf_type, data_set='cp')
                            plfs_dict[plf]['trend_line_cp']=trend_line_cp
                            plfs_dict[plf]['x_axis_values_cp']=x_axis_values_cp
                            plfs_dict[plf]['hist_values_cp']=hist_values_cp
                            plfs_dict[plf]['isnan_mask_cp']=isnan_mask_cp
                            plfs_dict[plf]['recalls_mins_cp']=recalls_mins_cp
                            plfs_dict[plf]['recalls_maxs_cp']=recalls_maxs_cp

        break

    plt.rcParams["figure.figsize"] = (4500/300, 3600/300)
    plt.rcParams["figure.dpi"] = 300
    plt.subplots_adjust(left=0.00, right=0.99, top=0.95, bottom=0.1)
    fig, axs = plt.subplots(3, 3)
    hist_bars, trend_line_plt=generate_subplot(plfs_dict['occlusion_ratio'], axs[0, 0], 'Occlusion', 'occlusion_ratio')
    hist_bars, trend_line_plt=generate_subplot(plfs_dict['distance'], axs[0, 1], 'Distance', 'distance')
    hist_bars, trend_line_plt=generate_subplot(plfs_dict['boundary_edge_strength'], axs[0, 2], 'Boundary Edge Strength', 'boundary_edge_strength')
    hist_bars, trend_line_plt=generate_subplot(plfs_dict['fog_intensity'], axs[1, 0], 'Fog', 'fog_intensity')
    hist_bars, trend_line_plt=generate_subplot(plfs_dict['crowdedness'], axs[1, 1], 'Crowdedness', 'crowdedness')
    hist_bars, trend_line_plt=generate_subplot(plfs_dict['edge_strength'], axs[1, 2], 'Edge Strength','edge_strength')
    hist_bars, trend_line_plt=generate_subplot(plfs_dict['brightness'], axs[2, 0], 'Brightness', 'brightness')
    hist_bars, trend_line_plt=generate_subplot(plfs_dict['contrast'], axs[2, 1], 'Contrast', 'contrast')
    hist_bars, trend_line_plt=generate_subplot(plfs_dict['contrast_to_background'], axs[2, 2], 'Contrast To Background', 'contrast_to_background')
    handles, labels = axs[0, 0].get_legend_handles_labels()
    lns = trend_line_plt + hist_bars
    labs = [l.get_label() for l in lns]
    fig.legend(lns, labs, borderaxespad=-1, fontsize=18, loc='lower center', ncol=4,
               bbox_to_anchor=(0, 0.065, 1, 0))
    for ax in axs.flat:
        #pass
        #ax.label_outer()
        ax.yaxis.set_tick_params(labelleft=True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.16)
    plt.savefig(f"/path/to//plfs/final_3x3.pdf")
    #plt.show()

def plot_numerical_sample_plf(model, plf_comp_source, plf_type, plf, plf_performance):
    # Extracts the detection performance values from the plf_performance container
    srtps = plf_performance[model][plf_comp_source][plf_type][plf]['srtps']
    fns = plf_performance[model][plf_comp_source][plf_type][plf]['fns']
    precisions = plf_performance[model][plf_comp_source][plf_type][plf]['precision']
    recalls = plf_performance[model][plf_comp_source][plf_type][plf]['recall']
    f1s = plf_performance[model][plf_comp_source][plf_type][plf]['F1']

    # Computes a histogram of occurences for each PLF value
    hist_values = (srtps + fns) / (np.sum(srtps) + np.sum(fns))
    # Defines the x axis, which contains values between 0 and 1
    x_axis_values = np.array([i / 100 for i in range(101)])

    # Removes all results for plf values that have a lower number of samples than specified in the configurations by the min_plf_samples variable
    valid_plf_values = np.where((srtps + fns) >= cfg['min_plf_samples'], 1, np.nan)
    hist_values *= valid_plf_values
    precisions *= valid_plf_values
    recalls *= valid_plf_values
    f1s *= valid_plf_values

    # Computes a local regression function (which will later serve as a trend line) to estimate the F1-score based on the PLF value
    f1_isnan_mask = np.isnan(plf_performance[model][plf_comp_source][plf_type][plf]['F1'])
    trend_line = sm.nonparametric.lowess(f1s[~f1_isnan_mask], x_axis_values[~f1_isnan_mask], frac=1 / 5)
    x_values = np.linspace(0, 1, 100)

    # Generates the plot
    # fig = plt.figure(figsize=(2008/300, 1204.8/300), dpi=300)
    fig = plt.figure(figsize=(10, 6), dpi=300)
    fig.patch.set_facecolor((0.97, 0.97, 0.97))
    ax1 = fig.add_subplot(111)

    # Plots the PLF precision, PLF recall, PLF F1-score, baseline precision, baseline recall, baseline F1-score and F1-score trend line on the first y-axis
    precision_plt = ax1.scatter(x_axis_values, precisions, color='#1f77b4', marker='o', label='PLF Precision')
    recall_plt = ax1.scatter(x_axis_values, recalls, color='#ff7f0e', marker='o', label='PLF Recall')
    f1_plt = ax1.scatter(x_axis_values, f1s, linewidth=3.0, color='#2ca02c', marker='o', label='PLF F1-Score')
    baseline_precision_plt = ax1.axhline(y=cfg['model_baseline_performance'][cfg['dataset_name']][model]['Precision'],
                                         color='#1f77b4', alpha=0.5, linestyle='--', label='Baseline Precision')
    baseline_recall_plt = ax1.axhline(y=cfg['model_baseline_performance'][cfg['dataset_name']][model]['Recall'],
                                      color='#ff7f0e', alpha=0.5, linestyle='--', label='Baseline Recall')
    baseline_f1_plt = ax1.axhline(y=cfg['model_baseline_performance'][cfg['dataset_name']][model]['F1'],
                                  color='#2ca02c', alpha=0.5, linestyle='--', label='Baseline F1-Score')
    trend_line_plt = ax1.plot(trend_line[:, 0], trend_line[:, 1], 'black', linestyle='--', label='F1-Score Trend Line')
    ax1.set_xlabel(f"Normalized {' '.join(plf.split('_')).title()}")
    ax1.set_ylim([0, 1.03])
    ax1.set_ylabel('Precision / Recall / F1-Score')

    # Plots the histogram values on the second y-axis
    ax2 = ax1.twinx()
    hist_bars = ax2.bar(x_axis_values, hist_values, width=0.007, color='r', alpha=0.3, label='PLF Density Histogram')
    ax2.set_ylabel('Density Histogram')

    # Add a legend under the plot
    lns = [precision_plt] + [recall_plt] + [f1_plt] + trend_line_plt + [hist_bars]
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    fig.subplots_adjust(bottom=0.1)

    plt.title(f"{model} - {' '.join(plf.split('_')).title()}")
    plt.show()


def plot_numerical_object_plf(model, plf_comp_source, plf_type, plf, plf_performance):
    # Extracts the detection performance values from the plf_performance container
    srtps = plf_performance[model][plf_comp_source][plf_type][plf]['srtps']
    fns = plf_performance[model][plf_comp_source][plf_type][plf]['fns']
    recalls = plf_performance[model][plf_comp_source][plf_type][plf]['recall']

    # Computes a density histogram of occurences for each PLF value
    hist_values = (srtps + fns) / (np.sum(srtps) + np.sum(fns))
    # Defines the x axis, which contains values between 0 and 1
    x_axis_values = np.array([i / 100 for i in range(101)])

    # Removes all results for plf values that have a lower number of samples than specified in the configurations by the min_plf_samples variable
    valid_plf_values = np.where((srtps + fns) >= cfg['min_plf_samples'], 1, np.nan)
    hist_values *= valid_plf_values
    recalls *= valid_plf_values

    # Computes a local regression function (which will later serve as a trend line) to estimate the recall based on the PLF value
    recall_isnan_mask = np.isnan(plf_performance[model][plf_comp_source][plf_type][plf]['recall'])
    trend_line = sm.nonparametric.lowess(recalls[~recall_isnan_mask], x_axis_values[~recall_isnan_mask], frac=1 / 5)
    x_values = np.linspace(0, 1, 100)

    # Generates the plot
    # fig = plt.figure(figsize=(2008/300, 1204.8/300), dpi=300)
    fig = plt.figure(figsize=(10, 6), dpi=300)
    fig.patch.set_facecolor((0.97, 0.97, 0.97))
    ax1 = fig.add_subplot(111)

    # Plots the PLF recall, baseline recall and recall trend line on the first y-axis
    recall_plt = ax1.scatter(x_axis_values, recalls, linewidth=3.0, color='#ff7f0e', marker='o', label='PLF Recall')
    baseline_recall_plt = ax1.axhline(y=cfg['model_baseline_performance'][cfg['dataset_name']][model]['Recall'],
                                      color='#ff7f0e', alpha=0.5, linestyle='--', label='Baseline Recall')
    trend_line_plt = ax1.plot(trend_line[:, 0], trend_line[:, 1], 'black', linestyle='--', label='Recall Trend Line')
    ax1.set_xlabel(f"Normalized {' '.join(plf.split('_')).title()}")
    ax1.set_ylim([0, 1.03])
    ax1.set_ylabel('Recall')

    # Plots the histogram values on the second y-axis
    ax2 = ax1.twinx()
    hist_bars = ax2.bar(x_axis_values, hist_values, width=0.007, color='r', alpha=0.3, label='PLF Density Histogram')
    ax2.set_ylabel('Density Histogram')

    # Add a legend under the plot
    lns = [recall_plt] + [baseline_recall_plt] + trend_line_plt + [hist_bars]
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    fig.subplots_adjust(bottom=0.1)

    plt.title(f"{model} - {' '.join(plf.split('_')).title()}")
    plt.show()


def plot_categorical_sample_plf(model, plf_comp_source, plf_type, plf, plf_performance):
    # A list of all categories for the PLF
    categories = [cat for cat in plf_performance[model][plf_comp_source][plf_type][plf]['tps']]

    # Extracts the detection performance values from the plf_performance container
    srtps = []
    fns = []
    precisions = []
    recalls = []
    f1s = []
    for cat in categories:
        srtps.append(plf_performance[model][plf_comp_source][plf_type][plf]['srtps'][cat])
        fns.append(plf_performance[model][plf_comp_source][plf_type][plf]['fns'][cat])
        precisions.append(plf_performance[model][plf_comp_source][plf_type][plf]['precision'][cat])
        recalls.append(plf_performance[model][plf_comp_source][plf_type][plf]['recall'][cat])
        f1s.append(plf_performance[model][plf_comp_source][plf_type][plf]['F1'][cat])
    srtps = np.array(srtps)
    fns = np.array(fns)
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1s = np.array(f1s)

    # Computes a density histogram for each PLF category
    hist_values = (np.array(srtps) + np.array(fns)) / (np.sum(np.array(srtps)) + np.sum(np.array(fns)))
    # Defines the x axis, which will contain category names
    x_axis_values = np.arange(len(categories))

    # Removes all results for plf values that have a lower number of samples than specified in the configurations by the min_plf_samples variable
    valid_plf_values = np.where((srtps + fns) >= cfg['min_plf_samples'], 1, np.nan)
    hist_values *= valid_plf_values
    recalls *= valid_plf_values

    # Generates the plot
    # fig = plt.figure(figsize=(2008/300, 1204.8/300), dpi=300)
    fig = plt.figure(figsize=(10, 6), dpi=300)
    fig.patch.set_facecolor((0.97, 0.97, 0.97))
    ax1 = fig.add_subplot(111)

    # Plots the PLF precision, PLF recall, PLF F1-score, baseline precision, baseline recall and baseline F1-score on the first y-axis
    precision_plt = ax1.plot(x_axis_values, precisions, color='#1f77b4', marker='o', label='PLF Precision')
    recall_plt = ax1.plot(x_axis_values, recalls, color='#ff7f0e', marker='o', label='PLF Recall')
    f1_plt = ax1.plot(x_axis_values, f1s, linewidth=3.0, color='#2ca02c', marker='o', label='PLF F1-Score')
    baseline_precision_plt = ax1.axhline(y=cfg['model_baseline_performance'][cfg['dataset_name']][model]['Precision'],
                                         color='#1f77b4', alpha=0.5, linestyle='--', label='Baseline Precision')
    baseline_recall_plt = ax1.axhline(y=cfg['model_baseline_performance'][cfg['dataset_name']][model]['Recall'],
                                      color='#ff7f0e', alpha=0.5, linestyle='--', label='Baseline Recall')
    baseline_f1_plt = ax1.axhline(y=cfg['model_baseline_performance'][cfg['dataset_name']][model]['F1'],
                                  color='#2ca02c', alpha=0.5, linestyle='--', label='Baseline F1-Score')
    _ = ax1.axhline(y=0, alpha=0, label=' ')  # This is just a workaround to have the legend plotted in the correct way
    ax1.set_xticks(x_axis_values)
    ax1.set_xticklabels(categories)
    ax1.set_ylim([0, 1.03])
    ax1.set_ylabel('Precision / Recall / F1-Score')

    # Plots the histogram values on the second y-axis
    ax2 = ax1.twinx()
    hist_bars = ax2.bar(x_axis_values, hist_values, width=0.95 / len(categories), color='r', alpha=0.3,
                        label='PLF Density Histogram')
    ax2.set_ylabel('Density Histogram')

    # Add a legend under the plot
    lns = precision_plt + recall_plt + f1_plt + [baseline_precision_plt] + [baseline_recall_plt] + [baseline_f1_plt] + [
        hist_bars] + [_]
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    fig.subplots_adjust(bottom=0.1)

    plt.title(f"{model} - {' '.join(plf.split('_')).title()}")
    plt.show()


def plot_categorical_object_plf(model, plf_comp_source, plf_type, plf, plf_performance):
    # A list of all categories for the PLF
    categories = [cat for cat in plf_performance[model][plf_comp_source][plf_type][plf]['tps']]

    # Extracts the detection performance values from the plf_performance container
    srtps = []
    fns = []
    recalls = []
    for cat in categories:
        srtps.append(plf_performance[model][plf_comp_source][plf_type][plf]['srtps'][cat])
        fns.append(plf_performance[model][plf_comp_source][plf_type][plf]['fns'][cat])
        recalls.append(plf_performance[model][plf_comp_source][plf_type][plf]['recall'][cat])
    srtps = np.array(srtps)
    fns = np.array(fns)
    recalls = np.array(recalls)

    # Computes a density histogram for each PLF category
    hist_values = (np.array(srtps) + np.array(fns)) / (np.sum(np.array(srtps)) + np.sum(np.array(fns)))
    # Defines the x axis, which will contain category names
    x_axis_values = np.arange(len(categories))

    # Removes all results for PLF values that have a lower number of samples than specified in the configurations by the min_plf_samples variable
    valid_plf_values = np.where((srtps + fns) >= cfg['min_plf_samples'], 1, np.nan)
    hist_values *= valid_plf_values
    recalls *= valid_plf_values

    # Generates the plot
    # fig = plt.figure(figsize=(2008/300, 1204.8/300), dpi=300)
    fig = plt.figure(figsize=(10, 6), dpi=300)
    fig.patch.set_facecolor((0.97, 0.97, 0.97))
    ax1 = fig.add_subplot(111)

    # Plots the PLF recall and the baseline recall on the first y-axis
    recall_plt = ax1.plot(x_axis_values, recalls, linewidth=3.0, color='#ff7f0e', marker='o', label='PLF Recall')
    baseline_recall_plt = ax1.axhline(y=cfg['model_baseline_performance'][cfg['dataset_name']][model]['Recall'],
                                      color='#ff7f0e', alpha=0.5, linestyle='--', label='Baseline Recall')
    ax1.set_xticks(x_axis_values)
    ax1.set_xticklabels(categories)
    ax1.set_ylim([0, 1.03])
    ax1.set_ylabel('Recall')

    # Plots the histogram values on the second y-axis
    ax2 = ax1.twinx()
    hist_bars = ax2.bar(x_axis_values, hist_values, width=0.95 / len(categories), color='r', alpha=0.3,
                        label='PLF Density Histogram')
    ax2.set_ylabel('Density Histogram')

    # Add a legend under the plot
    lns = recall_plt + [baseline_recall_plt] + [hist_bars]
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    fig.subplots_adjust(bottom=0.1)

    plt.title(f"{model} - {' '.join(plf.split('_')).title()}")
    plt.show()

def per_model_per_class_plotting(plf_performance):
    # Iterates over all specified detection models
    for model in plf_performance:
        # Iterates over all computation sources ('sample' or 'object')
        for plf_comp_source in plf_performance[model]:
            # Iterates over all PLF types ('numerical' or 'categorical')
            for plf_type in plf_performance[model][plf_comp_source]:
                # Iterates over all PLFs
                for plf in plf_performance[model][plf_comp_source][plf_type]:
                    if plf_type == 'numerical':
                        # Plots the numerical PLFs
                        if plf_comp_source == 'sample':
                            plot_numerical_sample_plf(model, plf_comp_source, plf_type, plf, plf_performance)
                        elif plf_comp_source == 'object':
                            plot_numerical_object_plf(model, plf_comp_source, plf_type, plf, plf_performance)
                    elif plf_type == 'categorical':
                        # Plots the categorical PLFs
                        if plf_comp_source == 'sample':
                            plot_categorical_sample_plf(model, plf_comp_source, plf_type, plf, plf_performance)
                        elif plf_comp_source == 'object':
                            plot_categorical_object_plf(model, plf_comp_source, plf_type, plf, plf_performance)

def paper_final_bars():
    plfs_dict = {}
    for model in plf_performance_kia:
        # Iterates over all computation sources ('sample' or 'object')
        for plf_comp_source in plf_performance_kia[model]:
            # Iterates over all PLF types ('numerical' or 'categorical')
            for plf_type in plf_performance_kia[model][plf_comp_source]:
                # Iterates over all PLFs
                for plf in plf_performance_kia[model][plf_comp_source][plf_type]:

                    trend_line_kia, x_axis_values_kia, hist_values_kia, isnan_mask_kia, recalls_mins_kia, recalls_maxs_kia, categories, recalls_kia = calculate_avg_recall(
                        plf_performance_kia, plf, plf_comp_source, plf_type, data_set='kia')
                    plfs_dict[plf]={'trend_line_kia':trend_line_kia,
                                        'x_axis_values_kia':x_axis_values_kia,
                                        'hist_values_kia':hist_values_kia,
                                        'isnan_mask_kia':isnan_mask_kia,
                                        'recalls_mins_kia':recalls_mins_kia,
                                        'recalls_maxs_kia':recalls_maxs_kia,
                                        'categories':categories,
                                        'recalls_kia':recalls_kia}
                    if plf in cp_plfs:
                        trend_line_cp, x_axis_values_cp, hist_values_cp, isnan_mask_cp, recalls_mins_cp, recalls_maxs_cp, _, recalls_cp = calculate_avg_recall(
                            plf_performance_cp, plf, plf_comp_source, plf_type, data_set='cp')
                        plfs_dict[plf]['trend_line_cp']=trend_line_cp
                        plfs_dict[plf]['x_axis_values_cp']=x_axis_values_cp
                        plfs_dict[plf]['hist_values_cp']=hist_values_cp
                        plfs_dict[plf]['isnan_mask_cp']=isnan_mask_cp
                        plfs_dict[plf]['recalls_mins_cp']=recalls_mins_cp
                        plfs_dict[plf]['recalls_maxs_cp']=recalls_maxs_cp
                        plfs_dict[plf]['recalls_cp']=recalls_cp
        break
    corrs={
        'plf':[],
        'kia':[],
        'cp':[]
    }
    for plf in plf_names.keys():
        x=plfs_dict[plf]['x_axis_values_kia'][~plfs_dict[plf]['isnan_mask_kia']]
        y=plfs_dict[plf]['recalls_kia'][~plfs_dict[plf]['isnan_mask_kia']]
        corrs['plf'].append(plf)
        corrs['kia'].append(abs(np.corrcoef(x, y)[0][1]))
        print(plf, abs(np.corrcoef(x, y)[0][1]))
        if plf in cp_plfs:
            x = plfs_dict[plf]['x_axis_values_cp'][~plfs_dict[plf]['isnan_mask_cp']]
            y = plfs_dict[plf]['recalls_cp'][~plfs_dict[plf]['isnan_mask_cp']]
            corrs['cp'].append(abs(np.corrcoef(x, y)[0][1]))
            print(plf, abs(np.corrcoef(x, y)[0][1]))

    plt.rcParams["figure.figsize"] = (9400/300, 1900/300)
    plt.rcParams["figure.dpi"] = 300
    plt.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.1)
    barWidth = 0.3
    # Set position of bar on X axis
    #r1 = np.arange(len(plfs_dict.keys()))
    #r2 = [x + barWidth for x in r1]
    plt.grid(zorder=0, color='gray', linestyle='dashed', alpha=0.5)
    # Make the plot
    legend_elements = []


    r = [x + barWidth/2 if x<14 else x + barWidth for x in np.arange(len(corrs['kia']))]
    plt.bar(r, corrs['kia'], color='orangered', width=barWidth, label='KI-A', zorder=3, alpha=0.8)
    legend_elements.append(patches.Patch(facecolor='orangered', edgecolor='orangered', label='KI-Absicherung'))

    r = [x + barWidth*3/2 for x in np.arange(len(corrs['cp']))]
    plt.bar(r, corrs['cp'], color='deepskyblue', width=barWidth, label='CP', zorder=3, alpha=0.8)
    legend_elements.append(patches.Patch(facecolor='deepskyblue', edgecolor='deepskyblue', label='CityPersons'))


    # for index, _data in enumerate(baselines):
    #    plt.text(x=r[index] - 0.12, y=_data + 1, s=f"{_data}", fontdict=dict(fontsize=12))
    # Add xticks on the middle of the group bars
    plt.xlabel('Studied Factors', fontweight='bold',fontsize=15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.ylabel('Correlation Coefficient', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(plf_names.keys()))], plf_names.values(), fontsize=12, rotation=25, ha='right')
    #plt.yticks([r * 10 for r in range(11)])

    # Create legend & Show graphic

    #plt.legend(handles=legend_elements, fontsize=20, borderaxespad=-1, loc='lower center', ncol=2,bbox_to_anchor=(0, -0.45, 1, 0))
    plt.legend(handles=legend_elements, fontsize=20, borderaxespad=-1, loc='upper left',bbox_to_anchor=(0.02, 0.95, 1, 0))
    plt.subplots_adjust(bottom=0.2,left=0.025)
    plt.savefig(f"/Path/to/final_bars.pdf")

    #plt.show()

def main():


    config_file_path = './config/analyse_plfs.yaml'  # Path to the configuration yaml file

    # Reads the yaml file, which contains the parameters for the PLF analysis
    with open(config_file_path) as yaml_file:
        global cfg
        cfg= yaml.load(yaml_file, Loader=yaml.FullLoader)

    dataset = fo.load_dataset(cfg['dataset_name'])  # Loads the FiftyOne dataset instance

    # Filters out low confidence detections made by the models, based on the model specific confidence thresholds, which are specified in the configurations
    for model in cfg['prediction_fields']:
        dataset = dataset.filter_labels(
            cfg['prediction_fields'][model], F('confidence') >= cfg['conf_thresholds'][cfg['dataset_name']][model],
            only_matches=False
        )

    # Filters out all non-safety relevant ground truth labels from the dataset and stores the reference within a new variable
    sr_dataset = dataset.filter_labels('ground_truth', F('ignore') == False, only_matches=False)

    with open(f"{cfg['destination_path']}/{cfg['dataset_name']}_plf_analysis.p", 'rb') as openfile:
        plf_analysis_kia = pickle.load(openfile)

    with open(f"{cfg['destination_path']}/CityPersons_plf_analysis.p", 'rb') as openfile:
        plf_analysis_cp = pickle.load(openfile)

    # Unpacks the dictionary
    global plf_performance_kia, plf_performance_cp
    plf_performance_kia = plf_analysis_kia['plf_performance']
    plf_performance_cp = plf_analysis_cp['plf_performance']
    # avg_plf_performance = plf_analysis['avg_plf_performance']
    #num_plf_bounds = plf_analysis['num_plf_bounds']
    #per_model_per_class_plotting(plf_performance_kia)
    #compute_avg_plf_performance(plf_performance_kia)
    paper_final_6x2()
    #paper_final_3x3()
    #paper_final_bars()

if __name__ == '__main__':
    main()