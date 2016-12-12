function [J, GradTheta] = computeCost2(theta, x,  numFold, numOfBatch, partition)
%COMPUTECOST Compute cost for beta density estimation
%   J = COMPUTECOST(x, theta, betaPar, numFold) computes the cost of using theta as the
%   parameter for lbeta density estimation to fit the data points in x simulated from sort(betarnd(4,3,1000,1), 'ascend');

% Initialize some useful values

interval = partition(1):((partition(numOfBatch+1)-partition(1))/numFold):partition(numOfBatch+1);
NumPerInterval_contain_last = histc(x, interval);
NumPerInterval= NumPerInterval_contain_last(1:(end-1));
NumPerInterval(end) = NumPerInterval(end) + NumPerInterval_contain_last(end);
reAreaFold = NumPerInterval/length(x);

% estimated area of fold i
esAreaFold = zeros(numFold, 1);

% derivatives for the cost of each fold
gradFold = zeros(numFold, 2);

numPerfold=100*numOfBatch;
%theta = [0.5;0.5];
for i = 1:numFold
    
    lenOfGrid = (interval(i+1) - interval(i))/numPerfold;
    dataTemp = interval(i):lenOfGrid :interval(i+1);
    dataTemp = dataTemp';     
    esAreaFold(i) =  esAreaFold(i)+ repelem(lenOfGrid, numPerfold)* betaPdfForVec(dataTemp(2:end), theta);
    gradFold(i,1) = gradFold(i,1) + repelem(lenOfGrid, numPerfold)* deriOfFirstPar_BetaPdfForVec(dataTemp(2:end), theta);
    gradFold(i,2) = gradFold(i,2) + repelem(lenOfGrid, numPerfold)* deriOfSecondPar_BetaPdfForVec(dataTemp(2:end), theta);         
end

totEstArea = sum(esAreaFold);

J =0.5 * (1/numFold) * (esAreaFold/totEstArea- reAreaFold)' * ...
       (esAreaFold/totEstArea- reAreaFold);

 % estimated real grad for each fold 
  reGradFold = zeros(numFold, 2);
  totalGrad = sum(gradFold,1);
  for i = 1:numFold
         
      reGradFold(i,1) = (1/numFold) * (esAreaFold(i)/totEstArea- reAreaFold(i)) ...
          * (gradFold(i,1)*totEstArea - totalGrad(1)*esAreaFold(i))/totEstArea^2;
      reGradFold(i,2) = (1/numFold) * (esAreaFold(i)/totEstArea- reAreaFold(i)) ...
          * (gradFold(i,2)*totEstArea - totalGrad(2)*esAreaFold(i))/totEstArea^2;
  end
  GradTheta = sum(reGradFold ,1)';

end








