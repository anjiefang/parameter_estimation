function [J, GradTheta] = computeCost1(theta, x, numFold)
%COMPUTECOST Compute cost for beta density estimation
%   J = COMPUTECOST(x, theta, betaPar, numFold) computes the cost of using theta as the
%   parameter for lbeta density estimation to fit the data points in x simulated from sort(betarnd(4,3,1000,1), 'ascend');

% Initialize some useful values
numPerFold = floor(length(x)/numFold);
numInLastFold = numPerFold + mod(length(x), numFold); % number of training examples in the last fold
% real area of fold i
reAreaFold = zeros(numFold, 1);
%% compute the real area for each fold
reAreaFold(1:(end-1)) = repelem(numPerFold/length(x), numFold -1);
reAreaFold(end) = numInLastFold/length(x);
% estimated area up to fold i
esAreaUpToFold = zeros(numFold, 1);
% estimated area of fold i
esAreaFold = zeros(numFold, 1);

% derivatives for the cost of each fold
gradFold = zeros(numFold, 2);

% initial value of esAreaFold(1)
esAreaFold(1) =  x(1)*(x(1)/2)^(exp(theta(1))-1)*(1-(x(1)/2))^(exp(theta(2))-1);
% initial derivative for cost of fold 1
gradFold(1,1) =  x(1)*(x(1)/2)^(exp(theta(1))-1)*(1-(x(1)/2))^(exp(theta(2))-1) * log(x(1)/2) * exp(theta(1));
gradFold(1,2) =  x(1)*(x(1)/2)^(exp(theta(1))-1)*(1-(x(1)/2))^(exp(theta(2))-1) * log(1-(x(1)/2))* exp(theta(2));

for j = 2:numPerFold 
        % compute the estimated area of fold 1
        esAreaFold(1) = esAreaFold(1)+(x(j)-x(j-1))*((x(j)+x(j-1))/2)^(exp(theta(1))-1)*(1-(x(j)+x(j-1))/2)^(exp(theta(2))-1);
        % derivative for cost of fold 1
        gradFold(1,1) = gradFold(1,1)+(x(j)-x(j-1))*((x(j)+x(j-1))/2)^(exp(theta(1))-1)*(1-(x(j)+x(j-1))/2)^(exp(theta(2))-1)...
        * log((x(j)+x(j-1))/2) * exp(theta(1)) ;
        gradFold(1,2) = gradFold(1,2)+(x(j)-x(j-1))*((x(j)+x(j-1))/2)^(exp(theta(1))-1)*(1-(x(j)+x(j-1))/2)^(exp(theta(2))-1)...
        * log(1-(x(j)+x(j-1))/2) * exp(theta(2));
end

% estimated area up to fold 1
esAreaUpToFold(1) = esAreaFold(1);

for i = 2:(numFold-1)
    temp = (i-1) * numPerFold; % index of the value befor the first value of fold i
    % compute the estimated area of fold i
    for j = (temp +1):(temp +numPerFold) 
        
        esAreaFold(i) = esAreaFold(i)+(x(j)-x(j-1))*((x(j)+x(j-1))/2)^(exp(theta(1))-1)*(1-(x(j)+x(j-1))/2)^(exp(theta(2))-1);
        gradFold(i,1) = gradFold(i,1)+ (x(j)-x(j-1))*((x(j)+x(j-1))/2)^(exp(theta(1))-1)*(1-(x(j)+x(j-1))/2)^(exp(theta(2))-1)...
        * log((x(j)+x(j-1))/2) * exp(theta(1)) ;
        gradFold(i,2) = gradFold(i,2)+(x(j)-x(j-1))*((x(j)+x(j-1))/2)^(exp(theta(1))-1)*(1-(x(j)+x(j-1))/2)^(exp(theta(2))-1)...
        * log(1-(x(j)+x(j-1))/2) * exp(theta(2));
    end
    % compute the estimated area up to fold i
    esAreaUpToFold(i) = esAreaUpToFold(i-1)+ esAreaFold(i);
      
end

% estimated area of last fold 
 temp1 = (numFold-1) * numPerFold; % index of the value befor the first value of last fold 
    % compute the estimated area of last fold 
    for j = (temp1 +1):(temp1 + numInLastFold) 
        
        esAreaFold(numFold) = esAreaFold(numFold)+(x(j)-x(j-1))*((x(j)+x(j-1))/2)^(exp(theta(1))-1)*(1-(x(j)+x(j-1))/2)^(exp(theta(2))-1);
        gradFold(numFold,1) = gradFold(numFold,1)+ (x(j)-x(j-1))*((x(j)+x(j-1))/2)^(exp(theta(1))-1)*(1-(x(j)+x(j-1))/2)^(exp(theta(2))-1)...
        * log((x(j)+x(j-1))/2) * exp(theta(1)) ;
        gradFold(numFold,2) = gradFold(numFold,2)+(x(j)-x(j-1))*((x(j)+x(j-1))/2)^(exp(theta(1))-1)*(1-(x(j)+x(j-1))/2)^(exp(theta(2))-1)...
        * log(1-(x(j)+x(j-1))/2) * exp(theta(2));
    end
    % compute the estimated area up to last fold 
    esAreaUpToFold(numFold) = esAreaUpToFold(numFold-1)+ esAreaFold(numFold);
    
% =========================================================================
   J =0.5 * (1/numFold) * (esAreaFold/esAreaUpToFold(numFold)- reAreaFold)' * ...
       (esAreaFold/esAreaUpToFold(numFold)- reAreaFold);
   
 % estimated real grad for each fold 
  reGradFold = zeros(numFold, 2);
  totalGrad = sum(gradFold,1);
  for i = 1:numFold
      reGradFold(i,1) = (1/numFold) * (esAreaFold(i)/esAreaUpToFold(numFold)- reAreaFold(i)) ...
          * (gradFold(i,1)*esAreaUpToFold(numFold) - totalGrad(1)*esAreaFold(i))/(esAreaUpToFold(numFold))^2;
      reGradFold(i,2) = (1/numFold) * (esAreaFold(i)/esAreaUpToFold(numFold)- reAreaFold(i)) ...
          * (gradFold(i,2)*esAreaUpToFold(numFold) - totalGrad(2)*esAreaFold(i))/(esAreaUpToFold(numFold))^2;
  end
  GradTheta = sum(reGradFold ,1)';
end








