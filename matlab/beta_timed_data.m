%% Initialization
clear ; close all; clc

%% =================== Part 1:Create data and plot cost J ===================
%create data
data = betarnd(4,3,1000,1);
%sort data
data=sort(data,'ascend');

%true distribution with all data
[phat ci]=betafit(data,0.01)
y=betapdf(0:0.1:1,phat(1),phat(2));
hold on
plot(0:0.1:1,y,'--','LineWidth',2)

% ====== plot the cost functionGrid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-10, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];    
	  J_vals(i,j) = computeCost(t, data(1:200), betaPar, numFold);
    end
end

% Because of the way meshgrids work in the surf command, we need to 
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

%% =================== Part 2: Gradient descent ===================
fprintf('Running Gradient Descent ...\n')

%%======M1: partition data into numFold folds ======================
% # using matlab fminunc to optimize
%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 40000);
initial_theta = zeros(2, 1);
numFold =5;

%%===using the proportion of data in each fold to compute the real area

[theta, cost] = fminunc(@(t)(computeCost1(t, data(1:100), numFold)), initial_theta, options);
y=betapdf(0:0.1:1,exp(theta(1)),exp(theta(2)));
hold on
plot(0:0.1:1,y,'k')

[theta0, cost] = fminunc(@(t)(computeCost1(t, data(1:200),  numFold)), initial_theta, options);
y=betapdf(0:0.1:1,exp(theta0(1)),exp(theta0(2)));
hold on
plot(0:0.1:1,y,'r')

[theta1, cost] = fminunc(@(t)(computeCost1(t, data(1:300),  numFold)), initial_theta, options);
y=betapdf(0:0.1:1,exp(theta1(1)),exp(theta1(2)));
hold on
plot(0:0.1:1,y,'b')

[theta2, cost] = fminunc(@(t)(computeCost1(t, data(1:400),  numFold)), initial_theta, options);
y=betapdf(0:0.1:1,exp(theta2(1)),exp(theta2(2)));
hold on
plot(0:0.1:1,y,'c')

[theta3, cost] = fminunc(@(t)(computeCost1(t, data(1:500),  numFold)), initial_theta, options);
y=betapdf(0:0.1:1,exp(theta3(1)),exp(theta3(2)));
hold on
plot(0:0.1:1,y,'y')

[theta4, cost] = fminunc(@(t)(computeCost1(t, data(1:600), numFold)), initial_theta, options);
y=betapdf(0:0.1:1,exp(theta4(1)),exp(theta4(2)));
hold on
plot(0:0.1:1,y,'g')

% [theta5, cost] = fminunc(@(t)(computeCost1(t, data(1:700), numFold)), initial_theta, options);
% y=betapdf(0:0.1:1,exp(theta5(1)),exp(theta5(2)));
% hold on
% test = plot(0:0.1:1,y,'LineWidth',1);
% set(test,'Color',[0.6 0.2 0.6])

[theta6, cost] = fminunc(@(t)(computeCost1(t, data(1:800), numFold)), initial_theta, options);
y=betapdf(0:0.1:1,exp(theta6(1)),exp(theta6(2)));
hold on
test = plot(0:0.1:1,y,'LineWidth',1.5);
set(test,'Color',[0.8 0.2 0.84])
%%========= partition fixed intervals into numFold folds ============
clear ; close all; clc
%create data
data = betarnd(4,3,20000,1);
%sort data
data=sort(data,'ascend');
% split/partition data [0,1] into ten equally distributed intervals 
partition = 0:0.1:1;
CountPerInterval_contain_last = histc(data, partition);
CountPerInterval= CountPerInterval_contain_last(1:(end-1));
CountPerInterval(end) = CountPerInterval(end) + CountPerInterval_contain_last(end);
CumulativeRange = cumsum(fliplr(CountPerInterval));

numFold =5; % five fold for each of the 10 partitions

% ##### plot the cost function Grid over which we will calculate J
numOfBatch = 1;
x = data(1:CumulativeRange(numOfBatch)); % data up to the numOfBatch interval
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-10, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];    
	  J_vals(i,j) = computeCost2(t, x, numFold, numOfBatch, partition);
    end
end
% Because of the way meshgrids work in the surf command, we need to 
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

%true distribution with all data
[phat ci]=betafit(data,0.01)
y=betapdf(0:0.1:1,phat(1),phat(2));
figure
plot(0:0.1:1,y,'--','LineWidth',2)
%% ###### using matlab fminunc ########################################
% it seems that in this case, theta(2) is always 0 due to which  the
% denstity is always 0
options = optimset('GradObj', 'on', 'MaxIter', 40000);
initial_theta = zeros(2, 1);

numOfBatch=1;
[theta, cost] = fminunc(@(t)(computeCost2(t, data(1:CumulativeRange(numOfBatch)), numFold, numOfBatch, partition)), initial_theta, options);
y=betapdf(0:0.1:1,exp(theta(1)),exp(theta(2)));
hold on
plot(0:0.1:1,y,'k')

numOfBatch=2;
[theta0, cost] = fminunc(@(t)(computeCost2(t, data(1:CumulativeRange(numOfBatch)), numFold, numOfBatch, partition)), initial_theta, options);
y=betapdf(0:0.1:1,exp(theta0(1)),exp(theta0(2)));
hold on
plot(0:0.1:1,y,'r')

numOfBatch=3;
[theta1, cost] = fminunc(@(t)(computeCost2(t, data(1:CumulativeRange(numOfBatch)), numFold, numOfBatch, partition)), initial_theta, options);
y=betapdf(0:0.1:1,exp(theta1(1)),exp(theta1(2)));
hold on
plot(0:0.1:1,y,'g')

numOfBatch=4;
[theta2, cost] = fminunc(@(t)(computeCost2(t, data(1:CumulativeRange(numOfBatch)), numFold, numOfBatch, partition)), initial_theta, options);
y=betapdf(0:0.1:1,exp(theta2(1)),exp(theta2(2)));
hold on
plot(0:0.1:1,y,'b')

numOfBatch=5;
[theta3, cost] = fminunc(@(t)(computeCost2(t, data(1:CumulativeRange(numOfBatch)), numFold, numOfBatch, partition)), initial_theta, options);
y=betapdf(0:0.1:1,exp(theta3(1)),exp(theta3(2)));
hold on
plot(0:0.1:1,y,'y')

numOfBatch=6;
[theta4, cost] = fminunc(@(t)(computeCost2(t, data(1:CumulativeRange(numOfBatch)), numFold, numOfBatch, partition)), initial_theta, options);
y=betapdf(0:0.1:1,exp(theta4(1)),exp(theta4(2)));
hold on
plot(0:0.1:1,y,'c')

