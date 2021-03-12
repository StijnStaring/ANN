clc
clearvars
close all
%%%%%%%%%%%%%
font_ax = 14;
font = 16;
lasertrain = load('lasertrain.dat');
lasertest = load('laserpred.dat');


% standardizing dataset
mean_training = mean(lasertrain);
std_training = std(lasertrain);
norm_training = (lasertrain - mean_training)./std_training;
mean_test = mean(lasertest);
std_test = std(lasertest);
norm_test = (lasertest - mean_test)./std_test;

% figure;
% plot(norm_training);  % plot the final point with a green circle
% title('Standardized time serie','fontsize',font,'fontweight','bold');
% set(gca,'fontsize',font_ax,'fontweight','bold')
% set(gca,'fontsize',font_ax,'fontweight','bold')
% xlabel('Discrete time','fontsize',font,'fontweight','bold')
% ylabel('Forecast','fontsize',font,'fontweight','bold')
% legend('Stand training','Location', 'northeast');

% Feedforward neural network with one hidden layer
lag_value = 3;
[TrainData,TrainTarget] = getTimeSeriesTrainData(lasertrain,lag_value); % you take two lag variables into account
alg1 = 'trainbfg';% First training algorithm to use
H = 5;% Number of neurons in the hidden layer
net1=feedforwardnet(H,alg1);
net1=configure(net1,TrainData,TrainTarget);% Set the input and output sizes of the net
net1.divideFcn = 'dividetrain';
net1.trainParam.time = 60;
net1=init(net1);% Initialize the weights (randomly)
% a = net1.layers(1);
% a{1,1}.transferFcn = 'tanh';
[net1,tr_descr] = train(net1,TrainData,TrainTarget);   % train the networks --> using a batch learning approach (ofline learning)                  
               
