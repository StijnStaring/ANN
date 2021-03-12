clear
clc
close all

%%%%%%%%%%%
%algorlm.m
% A script comparing performance of 'trainlm' and 'trainbfg'
% trainbfg - BFGS (quasi Newton)
% trainlm - Levenberg - Marquardt
%%%%%%%%%%%

% Configuration:
H = 100;% Number of neurons in the hidden layer
delta_epochs = [1,14,985];% Number of epochs to train in each step
epochs = cumsum(delta_epochs);

%generation of examples and targets
dx=0.05;% Decrease this value to increase the number of data points
x=0:dx:3*pi;y=sin(x.^2);
sigma=0.2;% Standard deviation of added noise
yn=y+sigma*randn(size(y));% Add gaussian noise
t=yn;% Targets. Change to yn to train on noisy data

% generation of the test data
dx_t=0.003;% Decrease this value to increase the number of data points
x_t=0:dx_t:3*pi;y_t=sin(x_t.^2);

%creation of networks
amount_iter = 10;
init_weight_storage = struct();
init_weight_storage.IW = {};
init_weight_storage.LW = {};
init_weight_storage.B1 = {};
init_weight_storage.B2 = {};
for i = 1:1:amount_iter
    net1=feedforwardnet(H,'traingd');
    net1=configure(net1,x,t);
    net1=init(net1);
    init_weight_storage.IW{end+1} = net1.iw{1,1};
    init_weight_storage.LW{end+1} = net1.lw{2,1};
    init_weight_storage.B1{end+1} = net1.b{1};
    init_weight_storage.B2{end+1} = net1.b{2};
end
fileID = fopen('Table3.txt','w');
for method = {'traingd','traingda','traincgf','traincgp','trainbfg','trainlm','trainbr'}
    average_crit = [0,0,0,0];
    for iter = 1:1:amount_iter
        net1=feedforwardnet(H,string(method));% Define the feedfoward net (hidden layers)
        net1=configure(net1,x,t);% Set the input and output sizes of the net
        net1.divideFcn = 'dividetrain';% Use training set only (no validation and test split)
        net1.iw{1,1}=init_weight_storage.IW{1,iter};% Set the same weights and biases for the networks 
        net1.lw{2,1}=init_weight_storage.LW{1,iter};
        net1.b{1}=init_weight_storage.B1{1,iter};
        net1.b{2}=init_weight_storage.B2{1,iter};

        %training and simulation
        net1.trainParam.epochs=delta_epochs(1);  % set the number of epochs for the training 
        net1=train(net1,x,t);   % train the networks --> using a batch learning approach (ofline learning)
        a11=sim(net1,x_t);

        net1.trainParam.epochs=delta_epochs(2);
        net1=train(net1,x,t);
        a12=sim(net1,x_t); 

        net1.trainParam.epochs=delta_epochs(3);
        net1=train(net1,x,t); % training on the training set
        a13=sim(net1,x_t);

        % obtaining the performance on a test set. 
        [~,~,r1] = postregm(a11,y_t);
        average_crit(1) = average_crit(1) + r1;
        [~,~,r2] = postregm(a12,y_t);
        average_crit(2) = average_crit(2) + r2;
        [~,~,r3] = postregm(a13,y_t);
        average_crit(3) = average_crit(3) + r3;
        mse_test_a13 = 1/size(x_t,2)*sum((y_t - a13).^2);
        average_crit(4) = average_crit(4) + mse_test_a13;
             
        
    end
    average_crit = average_crit/amount_iter;
    fprintf(fileID,'######################################\n');
    fprintf(fileID,'Nummer of iterations: %d. \n', iter);
    fprintf(fileID,'The Learning method: %s. \n', string(method));
    fprintf(fileID,'R one iter: %d. \n', round(average_crit(1),3));
    fprintf(fileID,'R 15 iter: %d. \n', round(average_crit(2),3));
    fprintf(fileID,'R 1000 iter: %d. \n', round(average_crit(3),3));
    fprintf(fileID,'The mse_test_a13 of method %s has training error: %d.\n',string(method),round(average_crit(4),3));
    
end


% fprintf('The mse_test_a23 of method %s has training error: %d.\n',alg2,round(mse_test_a23,3));


