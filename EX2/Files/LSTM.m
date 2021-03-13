clc
clearvars
close all
%%%%%%%%%

font_ax = 14;
font = 16;
lasertrain = load('lasertrain.dat');
lasertest = load('laserpred.dat');

% standardizing dataset - all data should use the same standardization
mean_training = mean(lasertrain);
std_training = std(lasertrain);
norm_training = (lasertrain - mean_training)./std_training;
norm_test = (lasertest - mean_training)./std_training;
XTest = norm_test';

% Feedforward neural network with one hidden layer
fileID = fopen('Performance_LSTM.txt','w');

for lag_value = 3
    for numHiddenUnits = 200
        [XTrain,YTrain] = getTimeSeriesTrainData(norm_training, lag_value);
        numFeatures = lag_value;
        numResponses = 1;

        layers = [ ...
        sequenceInputLayer(numFeatures)
        lstmLayer(numHiddenUnits)
        fullyConnectedLayer(numResponses)
        regressionLayer];

    options = trainingOptions('adam', ...
        'MaxEpochs',3000, ...
        'GradientThreshold',1, ...
        'InitialLearnRate',0.005, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropPeriod',125, ...
        'LearnRateDropFactor',0.2, ...
        'Verbose',0, ...
        'Plots','training-progress');

        net = trainNetwork(XTrain,YTrain,layers,options);    
        net = predictAndUpdateState(net,XTrain); 
        % should make transition from the training set to the test set smooth
        % --> not to disturb the hidden state vector.
        start = size(norm_training,1) - lag_value + 1;
        current_inputs = norm_training(start:end); %3x1 array    
        store_prediction = cell(1,size(norm_test,1));
        for i = 1:size(norm_test,1)       
            [net,prediction] = predictAndUpdateState(net,current_inputs,'ExecutionEnvironment','cpu');
            current_inputs = circshift(current_inputs,-1);
            current_inputs(lag_value,1) = prediction;
            store_prediction{1,i} = prediction;    
        end
    
    end      
end

store_prediction = cell2mat(store_prediction);
original_test_data = norm_test*std_training + mean_training;
original_prediction = store_prediction*std_training + mean_training;

figure;  
plot(original_test_data,'LineWidth',2.0)
hold on
plot(original_prediction,'LineWidth',2.0)
title('Learning result','fontsize',font,'fontweight','bold');
set(gca,'fontsize',font_ax,'fontweight','bold')
set(gca,'fontsize',font_ax,'fontweight','bold')
xlabel('Discrete time','fontsize',font,'fontweight','bold')
% ylabel('Forecast','fontsize',font,'fontweight','bold')
legend('Test data','Forecast','Location', 'northeast');

fclose(fileID);

