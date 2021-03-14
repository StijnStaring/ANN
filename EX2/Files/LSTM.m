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
original_test_data = lasertest';

% Feedforward neural network with one hidden layer
fileID = fopen('Performance_next.txt','w');
iterations = 1;
number = 1; 
best_setting = 1;
current_best_mse = inf;
for lag_value = 16
    for numHiddenUnits = 200
        MSE_average = 0;
        for iter = 1:1:iterations
            close all
            [XTrain,YTrain] = getTimeSeriesTrainData(norm_training, lag_value);
            numFeatures = lag_value;
            numResponses = 1;

            layers = [ ...
            sequenceInputLayer(numFeatures)
            lstmLayer(numHiddenUnits)
            fullyConnectedLayer(numResponses)
            regressionLayer];

            options = trainingOptions('adam', ...
            'MaxEpochs',250, ...
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
            current_inputs = norm_training(start:end); %lag_valuex1 array    
            store_prediction = cell(1,size(norm_test,1));
            for i = 1:size(norm_test,1)       
                [net,prediction] = predictAndUpdateState(net,current_inputs,'ExecutionEnvironment','cpu');
                current_inputs = circshift(current_inputs,-1);
                current_inputs(lag_value,1) = prediction;
                store_prediction{1,i} = prediction;    
            end

            store_prediction = cell2mat(store_prediction);
            original_prediction = store_prediction*std_training + mean_training;
            MSE_original = 1/size(store_prediction,2)*sum((original_prediction - original_test_data).^2);
            MSE_average = MSE_average + MSE_original;
        end
        MSE_average = MSE_average/iterations;
        fprintf(fileID,'######################################\n');
        fprintf(fileID,'Nummer of setting: %d. \n', number);    
        fprintf(fileID,'The lag value: %d \n',lag_value);                   
        fprintf(fileID,'An average MSE on the test set of %d with %d iterations. \n',round(MSE_average,3),iterations);
        
        if MSE_average < current_best_mse
           current_best_mse = MSE_average;
           best_setting = number;
        end
                    
        number = number + 1;
        
       
    end      
end

fprintf(fileID,'Best setting: %d. \n', best_setting);

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

