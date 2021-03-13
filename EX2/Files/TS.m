clc
clearvars
close all
%%%%%%%%%%%%%

font_ax = 14;
font = 16;
lasertrain = load('lasertrain.dat');
lasertest = load('laserpred.dat');

% standardizing dataset - all data should use the same standardization
mean_training = mean(lasertrain);
std_training = std(lasertrain);
norm_training = (lasertrain - mean_training)./std_training;
norm_test = (lasertest - mean_training)./std_training;

% Feedforward neural network with one hidden layer
fileID = fopen('Performance_nothing.txt','w');
iterations = 5;
number = 1; 
best_setting = 1;
current_best_mse = inf;
for regulation_constant = 0.4 % 3 options
    for amountLayers = 1
        for H = 5:5:50 % 5 options
             for alg1 = {'traingd'}
                 for lag_value = 5:5:30 % 7 options
                    average_MSE = 0;
                    for iter = 1:1:iterations
                        [TrainData,TrainTarget] = getTimeSeriesTrainData(norm_training,lag_value); % you take two lag variables into account
                        [trainInd,valInd] = divideblock(size(TrainData,2),0.90,0.10,0.0); % 10 procent of the training is used as validation     
                        layers = ones(1,amountLayers)*H;
                        net1=feedforwardnet(H,string(alg1));
                        net1=configure(net1,TrainData,TrainTarget);% Set the input and output sizes of the net
                        net1.divideFcn = 'divideind';% setting manually the train, validation and testset
                        Ltrain = size(trainInd,2);
                        Lval = size(valInd,2);
                        net1.divideParam.trainInd = 1:1:Ltrain;
                        net1.divideParam.valInd = Ltrain+1:1:Ltrain + Lval;
                        net1.trainParam.max_fail = 15;
                        net1.trainParam.time = 5;
                        net1.trainParam.epochs = 10^9;
                        net1.trainParam.goal = 1e-3;
                        net1.performParam.regularization = regulation_constant;
                        net1=init(net1);% Initialize the weights (randomly)
                        a = net1.layers(1);
                        a{1,1}.transferFcn = 'tanh';
                        [net1,tr_descr] = train(net1,TrainData,TrainTarget);   % train the networks --> using a batch learning approach. (ofline learning) 
                        valData = norm_training';
                        valData = valData(Ltrain+1:1:Ltrain + Lval);                    
                        MSE_out = test_performance(valData,net1,lag_value);
                        average_MSE = average_MSE + MSE_out;
                    end
                    
                    average_MSE = average_MSE/iterations;
                    
                    fprintf(fileID,'######################################\n');
                    fprintf(fileID,'Nummer of setting: %d. \n', number);
                    fprintf(fileID,'The Learning method: %s. \n', string(alg1));      
                    fprintf(fileID,'The regulation parameter: %d \n', regulation_constant); 
                    fprintf(fileID,'The amount of hidden neurons: %d \n',H);
                    fprintf(fileID,'The lag value: %d \n',lag_value);                   
                    fprintf(fileID,'An average MSE on the test set of %d with %d iterations. \n',round(average_MSE,3),iterations);

                    if average_MSE < current_best_mse
                       current_best_mse = average_MSE;
                       best_setting = number;
                    end
                    
                    number = number + 1;
            
                 end
             end
        end
    end
end
fprintf(fileID,'--------------------------- \n');
fprintf(fileID,'The best settings are number: %d. \n',best_setting);
fclose(fileID);


% performance of the trained network on the test set
MSE_out = test_performance(norm_test',net1,lag_value);


function [MSE]=test_performance(data,net, lag)
% data is inputed as 1xn array
    MSE = 0;
    current_inputs = data(1,1:lag)'; %3x1 array
    current_inputs = flip(current_inputs,2); % oldest value is at the top
    for p = lag+1:1:size(data,2)
       prediction = sim(net,current_inputs);
       MSE = MSE + (prediction - data(1,p))^2; 
       current_inputs = circshift(current_inputs,-1);
       current_inputs(lag,1) = prediction;       
    end
    amount_predictions = size(data,2) - lag;
    MSE = MSE/amount_predictions;
end


% figure;
% plot(norm_training);  % plot the final point with a green circle
% title('Standardized time serie','fontsize',font,'fontweight','bold');
% set(gca,'fontsize',font_ax,'fontweight','bold')
% set(gca,'fontsize',font_ax,'fontweight','bold')
% xlabel('Discrete time','fontsize',font,'fontweight','bold')
% ylabel('Forecast','fontsize',font,'fontweight','bold')
% legend('Stand training','Location', 'northeast');
               
