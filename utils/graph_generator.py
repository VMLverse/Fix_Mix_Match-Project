import json

import matplotlib as mpl
import matplotlib.pyplot as plt

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

def generate_threshold_results():
    threshold_75 = json_to_array("./json_results/Threshold_test_acc/Experiments_Threshold_config_fixmatch_cifar10red40_1000_0_75thresh.json")
    threshold_90 = json_to_array("./json_results/Threshold_test_acc/Experiments_Threshold_config_fixmatch_cifar10red40_1000_0.90thresh.json")
    threshold_100 = json_to_array("./json_results/Threshold_test_acc/Experiments_Threshold_config_fixmatch_cifar10red40_1000_1.0thresh.json")

    num_labeled_array = [threshold_75, threshold_90, threshold_100]
    legend_array = ["0.75", "0.90", "1.0"]
    plot_learning_curve(input_arr =num_labeled_array, legend = legend_array ,title='Pseudolabeling Threshold', ylabel="Test Accuracy", filename='graphs/threshold.pdf' )

def generate_barely_supervised_results():
    one_label = json_to_array("./json_results/Barely_supervised/Experiments_BarelySupervised_fixmatch_cifar10_10.json")
    seven_labels = json_to_array("./json_results/VaryingLabeledData_test_acc/Experiments_VaryingLabeledData_fixmatch_cifar10_70_200epoch.json")
    seven_labels = seven_labels[0:82]
    data_array = [one_label, seven_labels]
    legend_array = ["One label per class", "Seven labels per class"]
    plot_learning_curve(input_arr =data_array, legend = legend_array ,title='Barely Supervised', ylabel="Test Accuracy", filename='graphs/barely_supervised.pdf' )


if __name__ == "__main__":
    generate_barely_supervised_results()

