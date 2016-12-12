%% Initialization
clear ; close all; clc

data = betarnd(4,3,20000,1);
%sort data
data=sort(data,'ascend');
% split/partition data [0,1] into ten equally distributed intervals 
partition = 0:0.1:1;
CountPerInterval_contain_last = histc(data, partition);
CountPerInterval= CountPerInterval_contain_last(1:(end-1));
CountPerInterval(end) = CountPerInterval(end) + CountPerInterval_contain_last(end);
CumulativeRange = cumsum(fliplr(CountPerInterval));

numOfBatch = 1;
x = data(1:CumulativeRange(numOfBatch)); % data up to the numOfBatch interval
numFold =5; % five fold for each of the 10 partitions


% test gradient
% [0.5;1] 
[J, grad] =computeCost2([0.5;1], x, numFold, numOfBatch, partition); % when theta(1) != 1 not working, why?
grad
[J1, grad1] =computeCost2([0.5 + 1e-6;1], x, numFold, numOfBatch, partition);
[J2, grad2] =computeCost2([0.5 - 1e-6;1], x, numFold, numOfBatch, partition);
(J1 - J2)/(2e-6)
[J3, grad3] =computeCost2([0.5 ;1+ 1e-6], x, numFold, numOfBatch, partition);
[J4, grad4] =computeCost2([0.5 ;1- 1e-6], x, numFold, numOfBatch, partition);
(J3 - J4)/(2e-6)

% [0.5;0.5] 
[J, grad] =computeCost2([0.5;0.5], x,numFold, numOfBatch, partition); 
grad
[J1, grad1] =computeCost2([0.5 + 1e-6;0.5], x, numFold, numOfBatch, partition);
[J2, grad2] =computeCost2([0.5 - 1e-6;0.5], x, numFold, numOfBatch, partition);
(J1 - J2)/(2e-6)
[J3, grad3] =computeCost2([0.5 ;0.5+ 1e-6], x, numFold, numOfBatch, partition);
[J4, grad4] =computeCost2([0.5 ;0.5- 1e-6], x, numFold, numOfBatch, partition);
(J3 - J4)/(2e-6)
% [0;0] always work
[J, grad] =computeCost2([0;0], x, numFold, numOfBatch, partition); % when theta!0 not work, why?
grad
[J1, grad1] =computeCost2([0 + 1e-6;0], x, numFold, numOfBatch, partition);
[J2, grad2] =computeCost2([0 - 1e-6;0], x, numFold, numOfBatch, partition);
(J1 - J2)/(2e-6)
[J3, grad3] =computeCost2([0 ;0+ 1e-6], x, numFold, numOfBatch, partition);
[J4, grad4] =computeCost2([0 ;0- 1e-6], x, numFold, numOfBatch, partition);
(J3 - J4)/(2e-6)

