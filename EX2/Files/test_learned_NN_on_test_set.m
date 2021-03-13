clc;
clearvars;
close all;

load('WS_after_5min_run.mat');

% performance of the trained network on the test set
% norm_test = norm_test(50:end);
in = norm_test';
[MSE_out,pred] = test_performance(in, net1, lag_value);
fprintf('MSE_out normalized: %d \n',MSE_out);

figure;  
plot(norm_test(lag_value+1:end))
hold on
plot(pred)
title('Standardized learning result','fontsize',font,'fontweight','bold');
set(gca,'fontsize',font_ax,'fontweight','bold')
set(gca,'fontsize',font_ax,'fontweight','bold')
xlabel('Discrete time','fontsize',font,'fontweight','bold')
% ylabel('Forecast','fontsize',font,'fontweight','bold')
legend('Test data','Forecast','Location', 'northeast');

original_test = norm_test(lag_value+1:end)*std_training + mean_training;
original_test = original_test';
original_pred = pred*std_training + mean_training;
MSE_original = 1/size(original_test,2)*sum((original_test - original_pred).^2);
fprintf('MSE_out original: %d \n',MSE_original);

figure;  
plot(norm_test(lag_value+1:end)*std_training + mean_training,'LineWidth',2.0)
hold on
plot(pred*std_training + mean_training,'LineWidth',2.0)
title('Learning result','fontsize',font,'fontweight','bold');
set(gca,'fontsize',font_ax,'fontweight','bold')
set(gca,'fontsize',font_ax,'fontweight','bold')
xlabel('Discrete time','fontsize',font,'fontweight','bold')
% ylabel('Forecast','fontsize',font,'fontweight','bold')
legend('Test data','Forecast','Location', 'northeast');


function [MSE,pred]=test_performance(data,net,lag)
% data is inputed as 1xn array
    MSE = 0;
    pred = cell(1,size(data,2) - lag);
    current_inputs = data(1,1:lag)'; %3x1 array
    current_inputs = flip(current_inputs,2); % oldest value is at the top
    for p = lag+1:1:size(data,2)
       prediction = sim(net,current_inputs);
       pred{1,p-lag} = prediction;
       MSE = MSE + (prediction - data(1,p))^2; 
       current_inputs = circshift(current_inputs,-1);
       current_inputs(lag,1) = prediction;       
    end
    amount_predictions = size(data,2) - lag;
    MSE = MSE/amount_predictions;
    pred = cell2mat(pred);
end