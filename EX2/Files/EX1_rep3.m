%%%%%%%%%%%
% rep3.m
% A script which generates n random initial points for
% and visualise results of simulation of a 3d Hopfield network net
%%%%%%%%%%
clc
clearvars
close all

font_ax = 14;
font = 16;

T = [1 1 1; -1 -1 1; 1 -1 -1]';
net = newhop(T);
candidates = {[0;0;0],[1;1;0],[-1;-1;0],[1;-1;0],[-1;1;0],[0.5;0.5;0],[0.5;-0.5;0],[-0.5;0.5;0],[-0.5;-0.5;0],[0;-1;0],[0;1;0],[-1;0;0],[1;0;0],[-1;-0.5;0],[-1;-0.2;0],...
    [-1;0.5;0],[1;-0.5;0],[1;0.5;0],[0.5;-1;0],[-0.5;1;0],[0.5;1;0],[-0.5;-1;0],[-0.5;0;0],[0.5;0;0],[0;-0.5;0],[0;0.5;0],[0.0001;0;0],[-0.0001;0;0],[0;0.0001;0],...
    [0;-0.0001;0],[-0.0001;-1;0],[-1;-10^-20;0],[1.000000e-04;1.000000e-04;0],[0;1.000000e-04;0],[1.000000e-20;1.000000e-20;0],...
    [-1;-1;-1],[1;1;-1],[-1;1;-1],[0;0;-1],[-1;1;1],[0;1;1]};

iter = 1000;
for a = candidates                   
    [y,Pf,Af] = sim(net,{1 iter},{},a);       % simulation of the network  for 50 timesteps
    record=[cell2mat(a) cell2mat(y)];       % formatting results
    start=cell2mat(a);                      % formatting results 
    plot3(start(1,1),start(2,1),start(3,1),'bx',record(1,:),record(2,:),record(3,:),'r');  % plot evolution
    hold on;
    plot3(record(1,iter),record(2,iter),record(3,iter),'gO');  % plot the final point with a green circle
    attractor = cell2mat(Af);
    fprintf('-------------------------------\n');
    fprintf('Start point: [%d,%d,%d].\n',start.');
    fprintf('The attractor: [%d,%d,%d].\n',attractor.');
    
    amount_iter = 0;
    for i = 1:1:size(record,2)
        state = record(:,i);
        if isequal(attractor,state)
            break;
        end
        amount_iter = amount_iter + 1;
    end
    fprintf('The amount of iterations needed: %d.\n',amount_iter);
end
grid on;
legend('initial state','time evolution','attractor','Location', 'northeast');
title('Time evolution in the phase space of 3d Hopfield model','fontsize',font,'fontweight','bold');
set(gca,'fontsize',font_ax,'fontweight','bold')
set(gca,'fontsize',font_ax,'fontweight','bold')
xlabel('coördinate 1 [-]','fontsize',font,'fontweight','bold')
ylabel('coördinate 2 [-]','fontsize',font,'fontweight','bold')
zlabel('coördinate 3 [-]','fontsize',font,'fontweight','bold')

