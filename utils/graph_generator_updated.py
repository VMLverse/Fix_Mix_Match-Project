import json

import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

#setup styling
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
plt.style.use('seaborn')

def json_to_array(filename):
    input_file = open(filename)
    json_array = json.load(input_file)
    conv_list = []

    for item in json_array:
        conv_list.append(item[2])
    return conv_list

def json_to_array_relativeTime(filename):
    input_file = open(filename)
    json_array = json.load(input_file)
    conv_list = []
    time_list = []

    i = 0
    for item in json_array:
        conv_list.append(item[2])
        if i == 0:
            start_time = item[0]
            time_list.append(0)
        else:
            time_list.append(item[0] - start_time)
        i += 1
    return [conv_list, time_list]

def plot_learning_curve(input_arr, xlabel='epoch', ylabel='test loss', legend = ['Training', 'Validation'], title='Accuracy vs. No. of epochs', filename='as2_p2_2_learning_curve.pdf'):

    for results_array in input_arr:
        plt.plot(results_array, '-x')

    plt.xlabel(xlabel,fontsize = 14)
    plt.ylabel(ylabel,fontsize = 14)
    plt.legend(legend)
    plt.title(str(title),fontsize = 18)
    plt.savefig(filename)

def generate_cutout_results():
    #convert json to array
    regular_train = json_to_array('regular_train.json')
    regular_test = json_to_array('regular_test.json')
    cutoutonly_test = json_to_array('cutoutonly_test.json')
    nocutout_test = json_to_array('nocutout_test.json')
    plot_learning_curve(input_arr = [cutoutonly_test, nocutout_test, regular_test,], legend = ['cutout only', 'randaugment only', 'cutout+randaugment'],title='Effect of Cutout', filename='baseline.pdf' )

def generate_varying_labeled_results():
    num_labeled_70 = json_to_array("./json_results/VaryingLabeledData_test_acc/Experiments_VaryingLabeledData_fixmatch_cifar10_70_200epoch.json")
    num_labeled_250 = json_to_array("./json_results/VaryingLabeledData_test_acc/Experiments_VaryingLabeledData_fixmatch_cifar10_250_200epoch.json")
    num_labeled_1000 = json_to_array("./json_results/VaryingLabeledData_test_acc/Experiments_VaryingLabeledData_fixmatch_cifar10_1000_200epoch.json")
    num_labeled_4000 = json_to_array("./json_results/VaryingLabeledData_test_acc/Experiments_VaryingLabeledData_fixmatch_cifar10_4000_200epoch.json")
    num_labeled_50000 = json_to_array("./json_results/VaryingLabeledData_test_acc/Experiments_VaryingLabeledData_fixmatch_cifar10_50000_200epoch.json")

    num_labeled_array = [num_labeled_70, num_labeled_250, num_labeled_1000, num_labeled_4000,num_labeled_50000]
    legend_array = ["70 Labels", "250 Labels", "1000 Labels","4000 Labels","50000 Labels"]
    plot_learning_curve(input_arr =num_labeled_array, legend = legend_array ,title='Varying amount of labels', ylabel="Test Accuracy", filename='graphs/varying.pdf' )

def generate_varying_labeled_relativeTime_results():
    num_labeled_70 = json_to_array_relativeTime(
        "./json_results/VaryingLabeledData_test_acc/Experiments_VaryingLabeledData_fixmatch_cifar10_70_200epoch.json")
    num_labeled_250 = json_to_array_relativeTime(
        "./json_results/VaryingLabeledData_test_acc/Experiments_VaryingLabeledData_fixmatch_cifar10_250_200epoch.json")
    num_labeled_1000 = json_to_array_relativeTime(
        "./json_results/VaryingLabeledData_test_acc/Experiments_VaryingLabeledData_fixmatch_cifar10_1000_200epoch.json")
    num_labeled_4000 = json_to_array_relativeTime(
        "./json_results/VaryingLabeledData_test_acc/Experiments_VaryingLabeledData_fixmatch_cifar10_4000_200epoch.json")
    num_labeled_50000 = json_to_array_relativeTime(
        "./json_results/VaryingLabeledData_test_acc/Experiments_VaryingLabeledData_fixmatch_cifar10_50000_200epoch.json")

    num_labeled_70_time = num_labeled_70[1][99] / 60 / 60
    num_labeled_250_time = num_labeled_250[1][99] / 60 / 60
    num_labeled_1000_time = num_labeled_1000[1][99] / 60 / 60
    #num_labeled_1000_reduc_time = num_labeled_1000_reduc[1][99] / 60 / 60
    num_labeled_4000_time = num_labeled_4000[1][99] / 60 / 60
    num_labeled_50000_time = num_labeled_50000[1][99] / 60 / 60

    time_list = [num_labeled_70_time,num_labeled_250_time,num_labeled_1000_time,num_labeled_4000_time,num_labeled_50000_time]
    acc_list = [num_labeled_70[0][99],num_labeled_250[0][99],num_labeled_1000[0][99],num_labeled_4000[0][99],num_labeled_50000[0][99]]
    legend_array = ["70 Labels", "250 Labels", "1000 Labels","4000 Labels","50000 Labels"]

    time_df = pd.DataFrame({'Labeled_Data':legend_array, 'Time':time_list, 'Testing_Accuracy':acc_list})
    #sns.scatterplot(time_df, x = "Time", y = "Testing_Accuracy", hue = 'Labeled_Data')
    sns.barplot(time_df, x = "Labeled_Data", y = "Time")
    plt.xlabel('# of Labeled Data')
    plt.ylabel('100 Epoch Runtime (hours)')
    plt.title('Effect of the Amount of Labeled Data Used on Runtime')
    plt.savefig('graphs/LabeledData_RuntimePlot.pdf')
    plt.close()


def generate_threshold_results():
    thresh_default = json_to_array(
        "./json_results/Temp_test_acc/run-Default_cifar10red40_1000_baseline-tag-test_test_acc.json")
    threshold_75 = json_to_array("./json_results/Threshold_test_acc/Experiments_Threshold_config_fixmatch_cifar10red40_1000_0_75thresh.json")
    threshold_90 = json_to_array("./json_results/Threshold_test_acc/Experiments_Threshold_config_fixmatch_cifar10red40_1000_0.90thresh.json")
    threshold_100 = json_to_array("./json_results/Threshold_test_acc/Experiments_Threshold_config_fixmatch_cifar10red40_1000_1.0thresh.json")

    num_labeled_array = [thresh_default,threshold_75, threshold_90, threshold_100]
    legend_array = ["0.95 (Default","0.75", "0.90", "1.0"]
    plot_learning_curve(input_arr =num_labeled_array, legend = legend_array ,title='Pseudolabeling Threshold', ylabel="Test Accuracy", filename='graphs/threshold.pdf' )


def generate_temp_results():
    temp_default = json_to_array("./json_results/Temp_test_acc/run-Default_cifar10red40_1000_baseline-tag-test_test_acc.json")
    temp_95 = json_to_array('./json_results/Temp_test_acc/run-config_fixmatch_cifar10red40_1000_0.95Temp-tag-test_test_acc.json')
    temp_85 = json_to_array(
        './json_results/Temp_test_acc/run-config_fixmatch_cifar10red40_1000_0.85Temp-tag-test_test_acc.json')
    temp_75 = json_to_array(
        './json_results/Temp_test_acc/run-config_fixmatch_cifar10red40_1000_0.75Temp-tag-test_test_acc.json')
    num_labeled_array = [temp_default, temp_95, temp_85, temp_75]
    legend_array = ['1.0 (Default)', '0.95', '0.85','0.75']
    plot_learning_curve(input_arr =num_labeled_array, legend = legend_array ,title='Temperature', ylabel="Test Accuracy", filename='graphs/temperature.pdf' )


def generate_misc_results():
    misc_default = json_to_array("./json_results/Misc_Runs/run-Default_cifar10red40_1000_baseline-tag-test_test_acc.json")
    misc_adam = json_to_array(
        "./json_results/Misc_Runs/run-config_fixmatch_cifar10red40_1000_adam-tag-test_test_acc.json")
    misc_nesterov = json_to_array(
        "./json_results/Misc_Runs/run-config_fixmatch_cifar10red40_1000_nesterov_false-tag-test_test_acc.json")
    misc_mixup = json_to_array(
        "./json_results/Misc_Runs/run-fixmatch_cifar10_mixup-tag-test_test_acc.json")

    num_labeled_array = [misc_default, misc_adam, misc_nesterov, misc_mixup]
    legend_array = ['Default', 'Adam','Nesterov=F',"Mixup"]
    plot_learning_curve(input_arr=num_labeled_array, legend=legend_array, title='Model Perturbations', ylabel="Test Accuracy",
                        filename='graphs/MiscExperiments_errorplots.pdf')


def reduction_comparison():
    default = json_to_array("./json_results/VaryingLabeledData_test_acc/Experiments_VaryingLabeledData_fixmatch_cifar10_1000_200epoch.json")
    reduction = json_to_array("./json_results/Misc_Runs/run-Default_cifar10red40_1000_baseline-tag-test_test_acc.json")
    num_labeled_array = [default[0:100], reduction]
    legend_array = ['1000 Labels', '1000 Labels-40% Reduction']
    plot_learning_curve(input_arr=num_labeled_array, legend=legend_array, title='Effect of Data Reduction', ylabel="Test Accuracy",
                        filename='graphs/ReductionComparison_errorplots.pdf')

if __name__ == "__main__":
    generate_threshold_results()
    generate_temp_results()
    generate_varying_labeled_relativeTime_results()
    generate_misc_results()

