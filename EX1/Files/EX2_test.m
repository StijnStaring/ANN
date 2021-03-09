%% Development of a Feedforward NN (EX 2  in ANN course)
clc
clearvars
close all

font_ax = 16;
font = 18;



% loaded_data = load('Data_Problem1_regression.mat');
% T1 = loaded_data.T1;
% T2 = loaded_data.T2;
% T3 = loaded_data.T3;
% T4 = loaded_data.T4;
% T5 = loaded_data.T5;
% X1 = loaded_data.X1;
% X2 = loaded_data.X2;
% 
% % building new data
% [d1,d2,d3,d4,d5] = deal(6,3,2,0,0);
% 
% Tnew = (d1*T1 + d2*T2 + d3*T3 + d4*T4 + d5*T5)/(d1 + d2 + d3 + d4 + d5);
% clear d1 d2 d3 d4 d5 loaded_data T1 T2 T3 T4 T5
% 
% 
% 
% % loaded_data = load('newData.mat');
% 
% % building training, validation and test sets
% [Ltrain,Lval,Ltest] = deal(1000,1000,1000);
% training = build_set(X1,X2,Tnew,Ltrain);
% validation = build_set(X1,X2,Tnew,Lval);
% test = build_set(X1,X2,Tnew,Ltest);
% 
% save('newData.mat','training','validation','test');

loaded_data = load('newData.mat');
[Ltrain,Lval,Ltest] = deal(1000,1000,1000);
training = loaded_data.training;
validation = loaded_data.validation;
test = loaded_data.test;
clear loaded_data

F = scatteredInterpolant(training.X1,training.X2,training.Tnew);
[X1mesh,X2mesh] = meshgrid(0:0.01:1,0:0.01:1);
Tmesh = F(X1mesh,X2mesh);
figure;
mesh(X1mesh,X2mesh,Tmesh);
hold on;
plot3(X1mesh,X2mesh,Tmesh,'o');
pause(1)
close()

% building neural network
% fileID = fopen('performance_test.txt','w');
iterations = 1;
number = 1; 
best_setting = 1;
current_best_mse = inf;
for regulation_constant = 0
    for amountLayers = 2
        for H = 5
             for alg1 = {'traingd'}
%             for alg1 = {'traingd','traingda','traincgf','traincgp','trainbfg','trainlm'}
                average_MSE = 0;
                for iter = 1:1:iterations
            %         disp(iter);
        %             H = 50;% Number of neurons in the hidden layer
                    layers = ones(1,amountLayers)*H;
                    net1=feedforwardnet(layers,string(alg1));% Define the feedfoward net (hidden layers)
                    input_training = horzcat(training.X1,training.X2);
                    input_validation = horzcat(validation.X1,validation.X2);
                    input_test = horzcat(test.X1,test.X2);
                    input = [input_training;input_validation;input_test]';
                    target_training = training.Tnew;
                    target_validation = validation.Tnew;
                    target_test = test.Tnew;
                    target = [target_training;target_validation;target_test]';
                    net1.divideFcn = 'divideind';% setting manually the train, validation and testset
                    net1.divideParam.trainInd = 1:1:Ltrain;
                    net1.divideParam.valInd = Ltrain+1:1:Ltrain + Lval;
                    net1.divideParam.testInd = Ltrain + Lval+1:1:Ltrain + Lval + Ltest;
                    net1.trainParam.max_fail = 10;
                    net1=configure(net1,input,target);% Set the input and output sizes of the net
                    net1.trainParam.epochs = 10^10;
                    net1.trainParam.time = 300;
                    net1.trainParam.goal = 1e-3;
                    net.performParam.regularization = regulation_constant; % The regularization constant is a parameter --> using L2 norm for parameters
                    net1=init(net1);% Initialize the weights (randomly)
                    [net1,tr_descr] = train(net1,input,target);   % train the networks --> using a batch learning approach (ofline learning)

%                     prediction_test = sim(net1,input_validation');
%                     mse_test = sum((prediction_test - target_test').^2);
%                     average_MSE = average_MSE + mse_test;
                end
%             average_MSE = average_MSE/iterations;
%             fprintf(fileID,'######################################\n');
%             fprintf(fileID,'Nummer of iteration: %d. \n', number);
%             fprintf(fileID,'The Learning method: %s. \n', string(alg1));
%             fprintf(fileID,'The amount of hidden neurons: %d \n',H);
%             fprintf(fileID,'The amount of hidden layers + output layer: %d \n',length(net1.layers));
%             fprintf(fileID,'The regulation parameter: %d \n', net.performParam.regularization); 
%             fprintf(fileID,'An average MSE on the test set of %d with %d iterations. \n',average_MSE,iterations);
% 
%             if average_MSE < current_best_mse
%                current_best_mse = average_MSE;
%                best_setting = number;
%             end
            number = number + 1;
            end
        end
    end
end
% fclose(fileID);
% fprintf('The best settings are number: %d. \n',best_setting);

% performance of learned plane
prediction_test = sim(net1,input_test');
mse_test = 1/Ltest*sum((prediction_test - target_test').^2);

prediction_training = sim(net1,input_training');
mse_training = 1/Ltrain*sum((prediction_training - target_training').^2);


% figure(1)
% plot(x_ref,abs(y_ref-y_mpc),'g','LineWidth',1.0)
% hold on
% 
% title('Error [m]','fontsize',font,'fontweight','bold')
% set(gca,'fontsize',font_ax,'fontweight','bold')
% set(gca,'fontsize',font_ax,'fontweight','bold')
% xlabel('x [m]','fontsize',font,'fontweight','bold')
% ylabel('position error Y [m]','fontsize',font,'fontweight','bold')
% legend('V22.22-L3.47','V25.00-L6.94')

figure;
postregm(prediction_test,target_test');
% title('fontsize',font,'fontweight','bold')
% set(gca,'fontsize',font_ax,'fontweight','bold')
% set(gca,'fontsize',font_ax,'fontweight','bold')
% xlabel('fontsize',font,'fontweight','bold')
% ylabel('fontsize',font,'fontweight','bold')


[X1mesh,X2mesh] = meshgrid(0:0.01:1,0:0.01:1);
F_target = scatteredInterpolant(test.X1,test.X2,test.Tnew);
Ttarget = F_target(X1mesh,X2mesh);

F_prediction = scatteredInterpolant(test.X1,test.X2,prediction_test');
Tprediction = F_prediction(X1mesh,X2mesh);

figure;

mesh(X1mesh,X2mesh,Ttarget,'edgecolor','r');
hold on;
mesh(X1mesh,X2mesh,Tprediction,'edgecolor','b');
legend('Test Set','Learned Set');

figure;
Tdiff = test.Tnew - prediction_test';
F_error = scatteredInterpolant(test.X1,test.X2,Tdiff);
Terror = F_error(X1mesh,X2mesh);
mesh(X1mesh,X2mesh,Terror);
legend('Learning Error');

figure;
x =categorical({'Training error', 'Test error'});
y = [mse_training,mse_test];
bar(x,y);
title('MSE comparison Training and Test set');
ylabel('MSE [-]'); 




% Functions
function output = build_set(X1,X2,Tnew,L)
    output = struct();
    random_index = randi([1 length(Tnew)],1,L);
    output.X1 = X1(random_index);
    output.X2 = X2(random_index);
    output.Tnew = Tnew(random_index);
end