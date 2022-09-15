%% Visulize Effect of Regularization
clear all;
close all;

load("out_Nom_singleArray.mat")
load("out_L1_singleArray.mat")
load("training_input_singleArray.mat")
load("training_output_singleArray.mat")
load("x_plotting_array.mat")
lw = 1.5;

hold on;
grid on;
scatter(training_input_singleArray,training_output_singleArray, 'filled', 'Color','#7E2F8E')
plot(x_plotting_array, x_plotting_array.^2, 'LineWidth',lw, 'Color', 'k');
plot(x_plotting_array, out_Nom_singleArray, 'LineWidth',lw, 'Color', '#77AC30');
plot(x_plotting_array, out_L1_singleArray, 'LineWidth',lw, 'Color', '#A2142F');

lgd = legend('training data','f*','without regularization','L2 regularized');
lgd.Location = 'north';
lgd.FontSize = 12;
xlabel('x');
ylabel('f(x)')
ylim([-1.2,7])
xlim([-2.5,2.5])