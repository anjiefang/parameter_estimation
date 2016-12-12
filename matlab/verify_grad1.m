%% Initialization
clear ; close all; clc

%create data
data = betarnd(4,3,1000,1);
%sort data
data=sort(data,'ascend');

x = data(1:100); % data in the first interval
numFold =5;

% test gradient
% [0;0] 
[J, grad] =computeCost1([0;0], x, numFold); 
grad
[J1, grad1] =computeCost1([0 + 1e-6;0], x, numFold);
[J2, grad2] =computeCost1([0 - 1e-6;0], x,  numFold);
(J1 - J2)/(2e-6)
[J3, grad3] =computeCost1([0 ;0+ 1e-6], x, numFold);
[J4, grad4] =computeCost1([0 ;0- 1e-6], x,  numFold);
(J3 - J4)/(2e-6)
% [0.5;0.5]
[J, grad] =computeCost1([0.5;0.5], x, numFold); 
grad
[J1, grad1] =computeCost1([0.5 + 1e-6;0.5], x,  numFold);
[J2, grad2] =computeCost1([0.5 - 1e-6;0.5], x,  numFold);
(J1 - J2)/(2e-6)
[J3, grad3] =computeCost1([0.5 ;0.5+ 1e-6], x,  numFold);
[J4, grad4] =computeCost1([0.5 ;0.5- 1e-6], x,  numFold);
(J3 - J4)/(2e-6)

% [1;0.5] 
[J, grad] =computeCost1([1;0.5], x, numFold); 
grad
[J1, grad1] =computeCost1([1 + 1e-6;0.5], x,  numFold);
[J2, grad2] =computeCost1([1 - 1e-6;0.5], x,  numFold);
(J1 - J2)/(2e-6)
[J3, grad3] =computeCost1([1 ;0.5+ 1e-6], x, numFold);
[J4, grad4] =computeCost1([1 ;0.5- 1e-6], x,  numFold);
(J3 - J4)/(2e-6)


