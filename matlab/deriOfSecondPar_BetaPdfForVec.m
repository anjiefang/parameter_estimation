% compute the first derivative of theta(2) beta density (without the denominator Beta(theta(1), theta(2)) for a vector

function deriOfSecondPar_BetaPdfForVec = deriOfSecondPar_BetaPdfForVec(x,theta)

m = length(x);

for i = 1:m
    x(i) = (x(i))^(exp(theta(1))-1)*(1-x(i))^(exp(theta(2))-1) * ...
        log(1 - x(i)) * exp(theta(2));
end


deriOfSecondPar_BetaPdfForVec = x;


end